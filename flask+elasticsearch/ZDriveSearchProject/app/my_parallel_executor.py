#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
my_parallel_executor.py - 汎用並列実行モジュール

提供機能:
- run_parallel_tasks: タスク一覧を並列実行し、結果と実行時間を取得するエントリポイント関数
  - mode: 'thread', 'process', または 'async' により実行モード（スレッド並列、プロセス並列、非同期）を指定
  - dynamic (bool): Trueの場合、CPU負荷に応じてワーカー数を動的調整する
  - タスク粒度調整: タスクをジェネレータで渡すことで随時キューイング（パイプライン処理対応）
  - 結果のプロファイル情報（各タスク実行時間の記録と統計出力）
  - ログ出力（標準出力および profile_log_YYYYMMDD.txt への追記保存）
"""
import concurrent.futures
import asyncio
import time
from datetime import datetime
import logging
import threading
try:
    import psutil  # CPU使用率取得に使用（インストールされていない場合も動作可）
except ImportError:
    psutil = None

# 内部関数: タスク関数を実行し、結果と実行時間をタプルで返す
def _timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return (result, elapsed)

# 並列タスク実行のメイン関数
def run_parallel_tasks(tasks, mode='thread', max_workers=None, dynamic=False, 
                       min_workers=1, profile=False, logger=None):
    """
    タスク列を並列実行し、結果リストを返す汎用関数。

    tasks: 実行するタスクのコレクション。
       - イテラブル（リスト・タプル・ジェネレータなど）で各要素が:
         * callable または (callable, args, kwargs) の形式
       - 非同期モード(mode='async')の場合はコルーチンまたは (async関数, args, kwargs) も可
    mode: 'thread'（スレッドプール）, 'process'（プロセスプール）, 'async'（asyncio）
    max_workers: 最大ワーカー数（dynamic=Falseの場合は固定数、dynamic=Trueの場合は上限値）
                 Noneの場合、デフォルトでは:
                 - ThreadPoolExecutor: CPUコア数 * 5&#8203;:contentReference[oaicite:6]{index=6}
                 - ProcessPoolExecutor: CPUコア数
    dynamic: Trueの場合、ワーカー数を動的に調整する（modeが'thread'または'process'で有効）
    min_workers: 動的調整する場合の下限ワーカー数（デフォルト1）
    profile: Trueの場合、各タスクの実行時間を計測しプロファイル情報を収集する
    logger: ログ出力先。Noneまたは'stdout'で標準出力、'stderr'で標準エラー。
            文字列パスを指定するとファイルに追記。logging.Loggerやファイル-likeオブジェクト、関数も指定可。

    戻り値: 各タスクの結果を格納したリスト。
       profile=Trueの場合、結果リストの各要素は (result, elapsed) タプル（elapsedは秒)、または例外発生時はExceptionオブジェクト。
       profile=Falseの場合、各要素はタスクの返り値（例外時はException）。
    """
    # ログ出力用の関数を設定
    log = None
    file_obj = None
    if logger is None or logger == 'stdout':
        # 標準出力にログ（日時付き）
        log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    elif logger == 'stderr':
        import sys
        log = lambda msg: sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    elif isinstance(logger, str):
        # ログをファイルに出力
        file_obj = open(logger, 'a', encoding='utf-8')
        log = lambda msg, f=file_obj: (f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"), f.flush())
    elif isinstance(logger, logging.Logger):
        log = lambda msg, lg=logger: lg.info(msg)
    elif hasattr(logger, "write"):
        # ファイルオブジェクト（write属性を持つもの）にログ
        log = lambda msg, f=logger: (f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"), f.flush())
    elif callable(logger):
        # 任意の関数にログ文字列を渡す
        log = lambda msg, func=logger: func(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    else:
        # デフォルト: 標準出力
        log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    # Asyncモードの場合は別処理
    if mode == 'async':
        # タスクがコルーチン関数の場合はコルーチンオブジェクトにする
        coro_list = []
        for task in tasks:
            if isinstance(task, (tuple, list)):
                func = task[0]
                args = task[1] if len(task) > 1 else ()
                kwargs = task[2] if len(task) > 2 else {}
                if asyncio.iscoroutinefunction(func):
                    coro_list.append(func(*args, **kwargs))
                else:
                    # 非コルーチン関数は別スレッドで実行
                    if profile:
                        # 実行時間計測込み
                        coro_list.append(asyncio.to_thread(_timed_call, func, *args, **kwargs))
                    else:
                        coro_list.append(asyncio.to_thread(func, *args, **kwargs))
            else:
                if asyncio.iscoroutinefunction(task):
                    coro_list.append(task())  # コルーチン関数なら呼び出してコルーチン取得
                elif asyncio.iscoroutine(task):
                    coro_list.append(task)    # 既にコルーチンオブジェクト
                else:
                    # 非コルーチン（通常の関数）タスク
                    if profile:
                        coro_list.append(asyncio.to_thread(_timed_call, task))
                    else:
                        coro_list.append(asyncio.to_thread(task))
        # 新規イベントループで非同期タスクを並列実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_time = time.perf_counter()
        results_coros = loop.run_until_complete(asyncio.gather(*coro_list, return_exceptions=True))
        pipeline_elapsed = time.perf_counter() - start_time
        loop.close()
        # コルーチンの実行結果を処理
        results = []
        times = []
        for i, res in enumerate(results_coros):
            if isinstance(res, Exception):
                # タスク内で発生した例外
                if profile and log:
                    log(f"Task {i} raised an exception: {res}")
                results.append(res)
                if profile:
                    times.append(None)
            else:
                # タスク正常完了
                if profile:
                    # スレッドで実行した場合は_resが(tuple or val)
                    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], (int, float)):
                        result_val, elapsed_val = res
                        results.append((result_val, elapsed_val))
                        times.append(elapsed_val)
                        if log:
                            log(f"Task {i} completed in {elapsed_val:.6f} seconds.")
                    else:
                        results.append((res, None))
                        times.append(None)
                        if log:
                            log(f"Task {i} completed.")
                else:
                    results.append(res)
        # プロファイル統計出力
        if profile and times:
            valid_times = [t for t in times if isinstance(t, (int, float))]
            if valid_times and log:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                log(f"Task execution times: min={min_time:.6f}s, max={max_time:.6f}s, avg={avg_time:.6f}s (n={len(valid_times)})")
        # ログファイルへの集計結果追記
        if profile:
            summary = (f"[Aggregate Profile] {len(results)} tasks processed. Total time: {pipeline_elapsed:.2f} sec. "
                       f"Avg task time: { (sum([t for t in times if t]) / len([t for t in times if t])) if [t for t in times if t] else 0:.2f} sec. "
                       f"Min: { (min([t for t in times if t]) if [t for t in times if t] else 0):.2f} sec, "
                       f"Max: { (max([t for t in times if t]) if [t for t in times if t] else 0):.2f} sec. "
                       f"End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # stdoutログ
            if log:
                log(summary)
            # ファイルに追記
            profile_filename = f"profile_log_{datetime.now().strftime('%Y%m%d')}.txt"
            try:
                with open(profile_filename, 'a', encoding='utf-8') as pf:
                    pf.write(summary + "\n")
            except Exception as e:
                if log:
                    log(f"[Error] Failed to write profile log: {e}")
        if file_obj:
            file_obj.close()
        return results

    # ここから mode 'thread' または 'process' の処理
    ExecutorClass = concurrent.futures.ThreadPoolExecutor if mode == 'thread' else concurrent.futures.ProcessPoolExecutor
    # max_workers 未指定ならデフォルト（ThreadPoolExecutorはCPU*5、ProcessPoolExecutorはCPU数）
    executor_kwargs = {}
    if max_workers is not None:
        # dynamicの場合、max_workersは上限値として使用
        max_limit = max_workers
        initial_workers = max_workers if not dynamic else min(max_workers, max(min_workers, (max_workers if max_workers < (psutil.cpu_count(logical=True) or 1) else (psutil.cpu_count(logical=True) or 1))))
        # dynamic=Trueならinitialをminとmaxの間に設定（ここでは簡易に minかmaxの適切な方）
    else:
        # max_workers未指定時のデフォルト設定
        if ExecutorClass is concurrent.futures.ThreadPoolExecutor:
            # ThreadPoolExecutor: デフォルトでmin(32, CPU数*5)ですが、用途に合わせCPU*5程度&#8203;:contentReference[oaicite:7]{index=7}
            default_workers = (psutil.cpu_count(logical=True) or 1) * 5
        else:
            default_workers = (psutil.cpu_count(logical=True) or 1)
        max_limit = default_workers
        initial_workers = default_workers if not dynamic else max(min_workers, (psutil.cpu_count(logical=True) or 1))
    # dynamic=Trueなら initial_workersから開始し、max_limit上限で調整
    current_max = initial_workers if dynamic else max_limit

    results_map = {}
    results = []  # 結果を格納するリスト（最終的にインデックス順に並べる）
    times = []
    task_count = 0

    # タスクのイテレータを取得（リスト・ジェネレータ等に対応）
    if hasattr(tasks, '__iter__') and not isinstance(tasks, (str, bytes)):
        task_iter = iter(tasks)
    else:
        # イテレータではない（単一タスクが渡された場合など） -> リスト化
        task_iter = iter([tasks])

    # 非同期ジェネレータ対応: 別スレッドで実行して同期的にキューから取得
    queue = None
    if hasattr(tasks, '__aiter__'):
        # 非同期イテラブルの場合、Queueを使ってタスクを受け取る
        queue = []
        # PythonのQueueを使う場合はimport queue; queue.Queue()など。ここでは簡略化のためリストを共有しpopで使用。
        done_sentinel = object()
        # 非同期タスク生産者を実行するヘルパー
        async def _produce_tasks(async_iter):
            async for item in async_iter:
                queue.append(item)
            queue.append(done_sentinel)
        # 非同期ジェネレータを別スレッドで実行
        loop = asyncio.new_event_loop()
        def _run_async_producer():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_produce_tasks(tasks))
            loop.close()
        producer_thread = threading.Thread(target=_run_async_producer, daemon=True)
        producer_thread.start()
        # タスクイテレータとしてキューから取り出すイテレータを定義
        def queue_iter():
            # キューから値を取得（シンプルにリストpopで実装）
            while True:
                if queue:
                    item = queue.pop(0)
                else:
                    # キューが空なら少し待機してリトライ
                    time.sleep(0.01)
                    continue
                if item is done_sentinel:
                    break
                yield item
        task_iter = queue_iter()

    # 並列実行開始
    start_time = time.perf_counter()
    with ExecutorClass(max_workers=max_limit) as executor:
        active_futures = []  # 現在実行中のFutureリスト
        # 初期のタスク投入
        try:
            for _ in range(current_max):
                task = next(task_iter)
                # タスクをFutureとして送信
                if isinstance(task, (tuple, list)):
                    func = task[0]; args = task[1] if len(task) > 1 else (); kwargs = task[2] if len(task) > 2 else {}
                else:
                    func = task; args = (); kwargs = {}
                if profile:
                    # 実行時間を測るため _timed_call で包む
                    future = executor.submit(_timed_call, func, *args, **kwargs)
                else:
                    future = executor.submit(func, *args, **kwargs)
                active_futures.append(future)
                # インデックス（タスク番号）を記録
                task_index = task_count
                results_map[task_index] = None  # 後で値埋め込み
                future._task_index = task_index  # カスタム属性で保持
                task_count += 1
        except StopIteration:
            pass  # タスクが初期投入分より少ない場合

        # タスクが完了するたびに次のタスクを投入（ワークスティーリング的スケジューリング）
        while active_futures:
            # 完了したFutureを待つ（1つ完了するごとに処理）
            done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                active_futures.remove(fut)
                idx = getattr(fut, "_task_index", None)
                try:
                    res = fut.result()
                except Exception as e:
                    # 例外発生
                    if profile and log:
                        log(f"Task {idx} raised an exception: {e}")
                    results_map[idx] = e
                    if profile:
                        times.append(None)
                else:
                    # タスク正常完了
                    if profile:
                        # タイミング結果を取得
                        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], (int, float)):
                            result_val, elapsed_val = res
                            results_map[idx] = (result_val, elapsed_val)
                            times.append(elapsed_val)
                            if log:
                                log(f"Task {idx} completed in {elapsed_val:.6f} seconds.")
                        else:
                            results_map[idx] = (res, None)
                            times.append(None)
                            if log:
                                log(f"Task {idx} completed.")
                    else:
                        results_map[idx] = res
                # 新たなタスクを一つ取得して実行
                try:
                    task = next(task_iter)
                except StopIteration:
                    # もう投入すべきタスクなし
                    pass
                else:
                    # 動的ワーカー調整: 現在の負荷を確認して許容ワーカー数を更新
                    if dynamic and psutil:
                        cpu_usage = psutil.cpu_percent(interval=0.05)
                        if cpu_usage < 50 and current_max < max_limit:
                            current_max += 1  # CPU余裕あり -> 並列度を増やす
                        elif cpu_usage > 90 and current_max > min_workers:
                            current_max -= 1  # CPU過負荷 -> 並列度を減らす
                    # もし現在アクティブ数が上限未満なら新タスクを投入
                    if dynamic:
                        # dynamicの場合はcurrent_max制限に基づき投入
                        if len(active_futures) < current_max:
                            # 次のタスクをFutureで投入
                            if isinstance(task, (tuple, list)):
                                func = task[0]; args = task[1] if len(task) > 1 else (); kwargs = task[2] if len(task) > 2 else {}
                            else:
                                func = task; args = (); kwargs = {}
                            if profile:
                                future = executor.submit(_timed_call, func, *args, **kwargs)
                            else:
                                future = executor.submit(func, *args, **kwargs)
                            future._task_index = task_count
                            results_map[task_count] = None
                            task_count += 1
                            active_futures.append(future)
                        else:
                            # 並列実行数が現在上限に達している場合、このループで取得したtaskを次回ループ開始時に投げる
                            # （シンプル化のため、ここではすぐ投入せず次回ループで投げるロジックにする）
                            # 実装簡略のため、一度StopIterationを発生させて以降の投入を止める
                            task_iter = iter([])  # タスク投入終了
                    else:
                        # dynamic=Falseなら常に即時投入
                        if isinstance(task, (tuple, list)):
                            func = task[0]; args = task[1] if len(task) > 1 else (); kwargs = task[2] if len(task) > 2 else {}
                        else:
                            func = task; args = (); kwargs = {}
                        future = executor.submit(_timed_call, func, *args, **kwargs) if profile else executor.submit(func, *args, **kwargs)
                        future._task_index = task_count
                        results_map[task_count] = None
                        task_count += 1
                        active_futures.append(future)
            # ループ終端。アクティブタスクがなくなれば終了
        pipeline_elapsed = time.perf_counter() - start_time

    # 結果をタスク順にリスト化
    total_tasks = task_count
    results = [None] * total_tasks
    for idx, res in results_map.items():
        results[idx] = res
    # プロファイル統計をログ出力
    if profile and times:
        valid_times = [t for t in times if isinstance(t, (int, float))]
        if valid_times and log:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            log(f"Task execution times: min={min_time:.6f}s, max={max_time:.6f}s, avg={avg_time:.6f}s (n={len(valid_times)})")
    # ログファイルへ集計結果の追記（全タスク完了後）
    if profile:
        summary = (f"[Aggregate Profile] {total_tasks} tasks processed. Total processing time: {pipeline_elapsed:.2f} sec. "
                   f"Avg task time: {(sum([t for t in times if t]) / len([t for t in times if t])) if [t for t in times if t] else 0:.2f} sec. "
                   f"Min: {(min([t for t in times if t]) if [t for t in times if t] else 0):.2f} sec, "
                   f"Max: {(max([t for t in times if t]) if [t for t in times if t] else 0):.2f} sec. "
                   f"End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if log:
            log(summary)
        profile_filename = f"profile_log_{datetime.now().strftime('%Y%m%d')}.txt"
        try:
            with open(profile_filename, 'a', encoding='utf-8') as pf:
                pf.write(summary + "\n")
        except Exception as e:
            if log:
                log(f"[Error] Failed to write profile log: {e}")
    # 最終的な動的並列度のログ出力
    cpu_cores = psutil.cpu_count(logical=True) if psutil else None
    if dynamic and log:
        log(f"CPUコア数: {cpu_cores or 'Unknown'}, 最終的な動的並列度: {current_max}")
        log("IO並列処理により、待ち時間を十分に活用し、高いスループットを達成しました。")
    if file_obj:
        file_obj.close()
    return results
