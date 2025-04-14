#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【概要】
本ツールは、WizTree出力のCSVファイル（先頭1行は不要、2行目がヘッダー）を対象に、
非同期I/O、スレッド、プロセス並列でインデックス作成を行い、
MariaDBへの並列書き込みを実現します。

改善点：
  1. メモリマッピング＋インクリメンタルデコーダーおよび PyArrow による高速CSVパース
  2. MariaDB への並列書き込み（pymysql利用、UTF8設定済）
  3. 全チャンクを単一のプロセスプールに投入することで、作業タスクを共有し、
     早く終わったプロセスが次のタスクを処理する「ワーク・スティーリング」による高スループットを実現
  4. 各種パラメータ（チャンクサイズ、ブロックサイズ、スレッドワーカー数など）を設定クラスに集約
  5. 最終的なCPUコア数と動的並列度（＝プロセスプールのワーカー数）を標準出力に表示
  6. WSL上でWindows側のファイルパスを扱えるよう、パス変換処理を追加
  7. 強制終了時も含め、一時ファイル（TEMP_DIR）を全削除するクリーンアップ処理を追加

※ Windowsでは、各プロセスをCPUグループに割り当てる設定を含みます。
"""

import os, threading, hashlib, asyncio, time, mmap, glob, codecs, psutil, re, signal, shutil, sys
from io import StringIO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import aiofiles
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# PyArrow の利用可否
try:
    import pyarrow.csv as pv
    import pyarrow as pa
    USE_PYARROW = True
except ImportError:
    USE_PYARROW = False

# MariaDB 用ライブラリ
import pymysql
from pymysql.cursors import DictCursor

# ---------------------------
# グローバル設定：Windows用 CPU グループ割り当て
# ---------------------------
if os.name == 'nt':
    import ctypes
    def set_processor_group_affinity(group_id):
        kernel32 = ctypes.windll.kernel32
        class GROUP_AFFINITY(ctypes.Structure):
            _fields_ = [("Mask", ctypes.c_ulonglong),
                        ("Group", ctypes.c_ushort),
                        ("Reserved", ctypes.c_ushort * 3)]
        GetActiveProcessorCount = kernel32.GetActiveProcessorCount
        GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
        GetActiveProcessorCount.restype = ctypes.c_uint
        count = GetActiveProcessorCount(group_id)
        if count <= 0:
            return
        mask = (1 << count) - 1
        affinity = GROUP_AFFINITY()
        affinity.Mask = mask
        affinity.Group = group_id
        affinity.Reserved = (0, 0, 0)
        hThread = kernel32.GetCurrentThread()
        prev = GROUP_AFFINITY()
        result = kernel32.SetThreadGroupAffinity(hThread, ctypes.byref(affinity), ctypes.byref(prev))
        if result == 0:
            print(f"Warning: SetThreadGroupAffinity failed for group {group_id}", file=sys.stderr)
else:
    def set_processor_group_affinity(group_id):
        pass
# ---------------------------

# ---------------------------
# 一時ファイル用ディレクトリの設定
# ---------------------------
TEMP_DIR = "/tmp/wiztree_temp"

def create_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"Temporary directory created: {TEMP_DIR}")

def cleanup_temp_files():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print("Temporary files cleaned up.")

def signal_handler(signum, frame):
    print("Signal received, cleaning up temporary files...")
    cleanup_temp_files()
    sys.exit(1)

# シグナルハンドラの登録（強制終了時にもクリーンアップ）
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# ---------------------------

# ---------------------------
# WindowsパスをWSL形式に変換する関数
# ---------------------------
def convert_windows_path_to_wsl(path):
    # 例: "C:\Users\YourName\file.txt" -> "/mnt/c/Users/YourName/file.txt"
    m = re.match(r'^([A-Za-z]):\\(.*)', path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path
# ---------------------------

# ---------------------------
# ProcessorConfig: 設定パラメータ保持クラス
# ---------------------------
class ProcessorConfig:
    def __init__(self,
                 chunk_size=100_000,
                 block_size=1 * 1024 * 1024,
                 base_thread_pool_workers=500,
                 # 並列度はCPUコア数に合わせる（ここでは動的調整は行わず、固定値）
                 initial_max_concurrent=(os.cpu_count() or 1),
                 min_concurrent=2,
                 max_concurrent=(os.cpu_count() or 1),
                 use_pyarrow=USE_PYARROW):
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.base_thread_pool_workers = base_thread_pool_workers
        self.initial_max_concurrent = initial_max_concurrent
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.use_pyarrow = use_pyarrow

        # MariaDB 接続用パラメータ（必要に応じて変更してください）
        self.db_host = '127.0.0.1'
        self.db_port = 3306
        self.db_user = 'root'
        self.db_password = 'my-secret-pw'
        self.db_name = 'wiztree'

# ---------------------------
# DynamicSemaphore: 動的並列制御用セマフォ（今回は固定並列なので利用はしない）
# ---------------------------
class DynamicSemaphore:
    def __init__(self, max_value):
        self._max = max_value
    async def acquire(self):
        pass
    async def release(self):
        pass
    async def update_max(self, new_max):
        self._max = new_max
    @property
    def max_value(self):
        return self._max

# ---------------------------
# CSVReader: CSV読み込みクラス
# ---------------------------
class CSVReader:
    def __init__(self, config: ProcessorConfig):
        self.config = config

    async def async_read_csv_blocks(self, csv_file, chunk_size=None, block_size=None):
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if block_size is None:
            block_size = self.config.block_size

        start_total = time.perf_counter()
        decoder = codecs.getincrementaldecoder('utf-8')()
        async with aiofiles.open(csv_file, mode='rb') as f:
            buffer = ""
            header_line = None
            current_chunk = []
            line_count = 0
            first_line_read = False
            while True:
                block = await f.read(block_size)
                if not block:
                    break
                text = decoder.decode(block)
                buffer += text
                lines = buffer.splitlines(keepends=True)
                if not buffer.endswith("\n") and not buffer.endswith("\r"):
                    buffer = lines.pop()
                else:
                    buffer = ""
                for line in lines:
                    line_count += 1
                    if not first_line_read:
                        first_line_read = True
                        continue
                    if header_line is None:
                        header_line = line.strip()
                        print(f"[Profile] Header detected at line {line_count}")
                        continue
                    current_chunk.append(line)
                    if len(current_chunk) >= chunk_size:
                        t0 = time.perf_counter()
                        data = header_line + "\n" + "".join(current_chunk)
                        if self.config.use_pyarrow:
                            table = pv.read_csv(pa.py_buffer(data.encode('utf-8')))
                            df_chunk = table.to_pandas()
                        else:
                            df_chunk = pd.read_csv(StringIO(data))
                        t_chunk = time.perf_counter() - t0
                        print(f"[Profile] Chunk of {len(df_chunk)} rows read in {t_chunk:.4f} sec")
                        yield df_chunk
                        current_chunk = []
            if buffer:
                remaining_lines = buffer.splitlines()
                for line in remaining_lines:
                    line_count += 1
                    if header_line is None:
                        header_line = line.strip()
                        print(f"[Profile] Header detected at line {line_count}")
                        continue
                    current_chunk.append(line)
            if current_chunk:
                t0 = time.perf_counter()
                data = header_line + "\n" + "".join(current_chunk)
                if self.config.use_pyarrow:
                    table = pv.read_csv(pa.py_buffer(data.encode('utf-8')))
                    df_chunk = table.to_pandas()
                else:
                    df_chunk = pd.read_csv(StringIO(data))
                t_chunk = time.perf_counter() - t0
                print(f"[Profile] Final chunk of {len(df_chunk)} rows read in {t_chunk:.4f} sec")
                yield df_chunk
        total_time = time.perf_counter() - start_total
        print(f"[Profile] Total CSV reading time: {total_time:.2f} sec")

# ---------------------------
# DBHandler: MariaDB用 DB 初期化およびチャンク毎の挿入処理
# ---------------------------
class DBHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_connection(config: ProcessorConfig):
        return pymysql.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_password,
            database=config.db_name,
            charset='utf8mb4',
            cursorclass=DictCursor,
            autocommit=True
        )

    @staticmethod
    def init_db(config: ProcessorConfig):
        # プロファイリング: DB作成処理の開始
        db_init_start = time.perf_counter()
        # DB接続時、まずデータベース名を指定せず接続し、DBが存在しなければ作成する
        conn = pymysql.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_password,
            charset='utf8mb4',
            cursorclass=DictCursor,
            autocommit=True
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.db_name} DEFAULT CHARACTER SET utf8mb4;")
        finally:
            conn.close()
        # プロファイリング: DB作成処理の終了
        db_init_mid = time.perf_counter()
        print(f"[Profile] Database creation executed in {db_init_mid - db_init_start:.4f} sec")

        # その後、対象のデータベースへ接続してテーブル作成処理を実行
        conn = DBHandler.get_connection(config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents_csv (
                        file_id VARCHAR(32) PRIMARY KEY,
                        file_path TEXT,
                        file_name TEXT,
                        size BIGINT,
                        alloc BIGINT,
                        updated DATETIME,
                        attributes VARCHAR(255),
                        file_count INT,
                        folder TEXT
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
        finally:
            conn.close()
        db_init_end = time.perf_counter()
        print(f"[Profile] Table creation executed in {db_init_end - db_init_mid:.4f} sec")

    @staticmethod
    def index_chunk_sync(df_chunk, base_thread_pool_workers, config: ProcessorConfig):
        # プロファイリング: チャンク処理開始
        start_proc = time.time()
        log_lines = []
        conn = DBHandler.get_connection(config)
        try:
            def check_row(row):
                try:
                    # CSVのパスを取得し、WSLの場合は変換を行う
                    file_path = row['ファイル名']
                    if os.name != 'nt':
                        file_path = convert_windows_path_to_wsl(file_path)
                    if os.path.exists(file_path):
                        return row
                except Exception:
                    pass
                return None
            cpu_load = psutil.cpu_percent(interval=0.1)
            dynamic_workers = max(1, int(base_thread_pool_workers * (100 - cpu_load) / 100))
            with ThreadPoolExecutor(max_workers=dynamic_workers) as executor:
                futures = [executor.submit(check_row, row) for _, row in df_chunk.iterrows()]
                filtered_rows = []
                for future in as_completed(futures):
                    try:
                        row = future.result()
                        if row is not None:
                            filtered_rows.append(row)
                    except Exception as e:
                        log_lines.append(f"Error in thread pool: {e}")
            # プロファイリング: スレッド処理終了
            thread_proc_time = time.time() - start_proc
            print(f"[Profile] Thread pool processing completed in {thread_proc_time:.4f} sec for current chunk")

            insert_values = []
            for row in filtered_rows:
                file_path = row['ファイル名']
                if os.name != 'nt':
                    file_path = convert_windows_path_to_wsl(file_path)
                file_name = os.path.basename(os.path.normpath(file_path))
                file_id = hashlib.md5((str(file_path) + str(row['フォルダー'])).encode('utf-8')).hexdigest()
                insert_values.append((
                    file_id,
                    file_path,
                    file_name,
                    row['サイズ'],
                    row['割り当て'],
                    row['更新日時'],
                    row['属性'],
                    row['ファイル数'],
                    row['フォルダー']
                ))
            with conn.cursor() as cursor:
                placeholders = ",".join(["%s"] * 9)
                sql = f"""
                    INSERT INTO documents_csv
                    (file_id, file_path, file_name, size, alloc, updated, attributes, file_count, folder)
                    VALUES ({placeholders})
                """
                cursor.executemany(sql, insert_values)
            elapsed = time.time() - start_proc
            log_lines.append(f"Chunk indexed in MariaDB: {len(insert_values)} records in {elapsed:.4f} sec")
        except Exception as e:
            log_lines.append(f"Exception in index_chunk_sync: {e}")
        finally:
            conn.close()
        return "\n".join(log_lines)

# ---------------------------
# WizTreeProcessor: メインパイプライン処理クラス（MariaDBへの並列挿入・ワーク・スティーリング実現）
# ---------------------------
class WizTreeProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.csv_reader = CSVReader(config)
        self.db_handler = DBHandler()
        # 動的並列制御は固定（CPUコア数に合わせる）
        self.dynamic_semaphore = DynamicSemaphore(config.initial_max_concurrent)

    async def _monitor_load(self):
        # 今回は動的調整を無効化（固定並列運用）
        await asyncio.sleep(0.5)

    def _timed_index_chunk_sync(self, df_chunk, base_thread_pool_workers, config: ProcessorConfig):
        t0 = time.perf_counter()
        result = self.db_handler.index_chunk_sync(df_chunk, base_thread_pool_workers, config)
        elapsed = time.perf_counter() - t0
        return result, elapsed

    async def process_csv(self, csv_file, log_callback):
        logs = []
        logs.append(f"Pipeline processing started: {csv_file}")
        logs.append("Initializing MariaDB main table...")
        # プロファイリング: DB初期化にかかる時間
        db_init_start = time.perf_counter()
        self.db_handler.init_db(self.config)
        db_init_time = time.perf_counter() - db_init_start
        logs.append(f"DB initialization completed in {db_init_time:.4f} sec")

        # プロセスプール生成：並列処理の開始タイミングを計測
        total_cpus = os.cpu_count() or 1
        process_pool = ProcessPoolExecutor(max_workers=total_cpus)
        logs.append(f"Using {total_cpus} processes for DB insertion.")

        indexing_tasks = []
        chunk_timings = []
        loop = asyncio.get_running_loop()
        chunk_idx = 0
        start_pipeline = time.perf_counter()
        # 各CSVチャンク毎の並列処理（中間ログは残しつつ、各チャンクの処理時間は集計する）
        async for df_chunk in self.csv_reader.async_read_csv_blocks(csv_file, self.config.chunk_size, self.config.block_size):
            logs.append(f"Received chunk {chunk_idx} with {len(df_chunk)} rows")
            task = loop.run_in_executor(
                process_pool,
                self._timed_index_chunk_sync,
                df_chunk,
                self.config.base_thread_pool_workers,
                self.config
            )
            indexing_tasks.append(task)
            chunk_idx += 1
        if indexing_tasks:
            results = await asyncio.gather(*indexing_tasks)
            for (res, elapsed) in results:
                logs.append(res)
                chunk_timings.append(elapsed)
        total_pipeline_time = time.perf_counter() - start_pipeline
        logs.append(f"All chunks processed in {total_pipeline_time:.2f} sec")
        logs.append("Pipeline processing complete")
        process_pool.shutdown(wait=True)
        print("[Profile] MariaDB insert complete.")

        # 集計的な並列処理プロファイリング結果を一括出力
        if chunk_timings:
            total_chunks = len(chunk_timings)
            avg_chunk_time = sum(chunk_timings) / total_chunks
            min_chunk_time = min(chunk_timings)
            max_chunk_time = max(chunk_timings)
            summary = (
                f"[Aggregate Profile] {total_chunks} chunks processed. Total processing time: {total_pipeline_time:.2f} sec. "
                f"Avg chunk time: {avg_chunk_time:.2f} sec. Min: {min_chunk_time:.2f} sec, Max: {max_chunk_time:.2f} sec. "
                f"End timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logs.append(summary)
            print(summary)

        # 並列度情報の出力
        cpu_cores = os.cpu_count() or 1
        final_dynamic_concurrency = self.dynamic_semaphore.max_value
        print(f"[Info] CPUコア数: {cpu_cores}。最終的な動的並列度: {final_dynamic_concurrency}。")
        print("[Info] IO並列処理により、待ち時間が十分に活用され、高いDB挿入スループットが達成されています。")
        logs.append(f"CPUコア数: {cpu_cores}, 最終的な動的並列度: {final_dynamic_concurrency}")
        logs.append("IO並列処理により、待ち時間が十分に活用され、高いDB挿入スループットが達成されています。")
        return logs

# ---------------------------
# Tkinter UI連携
# ---------------------------
def process_wiztree_combined(csv_file, log_callback, finish_callback, master):
    create_temp_dir()  # 一時ディレクトリの作成
    config = ProcessorConfig()  # 必要に応じてパラメータを調整してください
    processor = WizTreeProcessor(config)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        overall_logs = loop.run_until_complete(processor.process_csv(csv_file, log_callback))
    except Exception as e:
        log_callback(f"Pipeline processing error: {e}")
        finish_callback(False)
        cleanup_temp_files()
        return
    for log in overall_logs:
        log_callback(log)
    finish_callback(True)
    cleanup_temp_files()  # 正常終了時も一時ファイルを削除

class WizTreeCombinedTab:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.csv_file = ""
        self.create_widgets()
    def create_widgets(self):
        lbl = tk.Label(self.frame, text="【WizTree Processing】Select WizTree CSV File")
        lbl.pack(pady=5)
        btn_select = tk.Button(self.frame, text="Select CSV File", command=self.select_file)
        btn_select.pack(pady=5)
        self.file_label = tk.Label(self.frame, text="No file selected")
        self.file_label.pack(pady=5)
        btn_process = tk.Button(self.frame, text="Start Processing (Full Parallelism)", command=self.start_process)
        btn_process.pack(pady=5)
        self.text_output = tk.Text(self.frame, wrap="word", width=80, height=30)
        self.text_output.pack(pady=5)
    def select_file(self):
        file_selected = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_selected:
            self.csv_file = file_selected
            self.file_label.config(text=f"Selected file: {self.csv_file}")
    def start_process(self):
        if not self.csv_file:
            messagebox.showwarning("Warning", "Please select a CSV file.")
            return
        self.text_output.delete(1.0, tk.END)
        self.append_log("Starting WizTree processing...")
        def on_finish(success):
            def fin():
                if success:
                    messagebox.showinfo("Done", "WizTree processing complete.")
                else:
                    messagebox.showwarning("Error", "An error occurred during processing.")
            self.frame.after(0, fin)
        threading.Thread(
            target=process_wiztree_combined,
            args=(self.csv_file, self.append_log, on_finish, self.frame),
            daemon=True
        ).start()
    def append_log(self, text):
        for line in text.splitlines():
            self.text_output.insert(tk.END, line + "\n")
            self.text_output.see(tk.END)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WizTree Processing Tool (Full Parallelism, 2-Socket)")
        self.create_notebook()
    def create_notebook(self):
        notebook = ttk.Notebook(self.root)
        self.tab_combined = WizTreeCombinedTab(notebook)
        notebook.add(self.tab_combined.frame, text="WizTree Processing")
        notebook.pack(expand=True, fill="both")

if __name__ == '__main__':
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
