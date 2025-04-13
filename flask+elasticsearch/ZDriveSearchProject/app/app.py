import os
import time
import threading
import json
import re
import concurrent.futures
from datetime import datetime
import requests
from flask import Flask, request, escape, render_template_string

app = Flask(__name__)

# =========================================
# 環境変数や設定項目
# =========================================
ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))

# Windows 形式の "Z:" をホストの /mnt/z にマップするための環境変数
FILE_MOUNT_POINT = os.environ.get("FILE_MOUNT_POINT", "/mnt/z")

# 対象とするテキストファイルの拡張子セット
TEXT_EXTENSIONS = {
    ".txt", ".md", ".log", ".csv", ".json", ".py", ".java",
    ".c", ".cpp", ".html", ".htm", ".mhtml"
}

# =========================================
# Elasticsearch 関連
# =========================================
def ensure_elasticsearch_ready():
    """Elasticsearch が利用可能になるまで待機する"""
    for _ in range(30):
        try:
            r = requests.get(ES_URL, timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False

def ensure_index():
    """インデックス（INDEX_NAME）が存在しなければ作成し、マッピングを設定する"""
    r = requests.head(f"{ES_URL}/{INDEX_NAME}")
    if r.status_code != 200:
        mapping = {
            "mappings": {
                "properties": {
                    "path": {"type": "text"},
                    "filename": {"type": "text"},
                    "extension": {"type": "keyword"},
                    "size": {"type": "long"},
                    "modified": {"type": "date"},
                    "content": {"type": "text", "analyzer": "kuromoji"}
                }
            }
        }
        res = requests.put(f"{ES_URL}/{INDEX_NAME}", json=mapping)
        if not res.ok:
            print(f"Failed to create index: {res.text}")

def process_file(file_path, fname):
    """個々のファイルを読み込み、内容などを返す。インデックス作成用。"""
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
        # テキストファイルのエンコーディング判定
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("cp932")
            except UnicodeDecodeError:
                content = raw.decode("utf-8", errors="ignore")
        size = os.path.getsize(file_path)
        mtime = os.path.getmtime(file_path)
        modified_iso = datetime.fromtimestamp(mtime).isoformat()

        # /mnt/zdrive 以下のパス -> Windows 形式 "Z:\..." に変換
        # file_path 例: "/mnt/zdrive/dir/file.txt"
        # → "Z:\dir\file.txt"
        #   ※ '/mnt/zdrive' の文字数は 10
        #   ※ slash を backslash に
        win_path = "Z:" + file_path[len("/mnt/zdrive"):].replace("/", "\\")

        doc = {
            "path": win_path,
            "filename": fname,
            "extension": os.path.splitext(fname)[1].lower(),
            "size": size,
            "modified": modified_iso,
            "content": content
        }
        return (win_path, doc)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def index_files():
    """/mnt/zdrive を走査し、テキストファイルを全て Elasticsearch に並列処理でインデックス化する"""
    files_indexed = []
    bulk_actions = []

    # ファイル一覧を取得
    files_to_process = []
    for root, dirs, files in os.walk("/mnt/zdrive"):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                continue
            full_path = os.path.join(root, fname)
            files_to_process.append((full_path, fname))

    # 並列でファイルの読み込み処理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(process_file, fp, name) for fp, name in files_to_process]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    # Bulk インデックス用のアクションを構築
    for win_path, doc in results:
        action = {"index": {"_index": INDEX_NAME, "_id": win_path}}
        bulk_actions.append(action)
        bulk_actions.append(doc)
        files_indexed.append(win_path)

    if not bulk_actions:
        return

    # Elasticsearch へ一括送信
    bulk_body = "\n".join(json.dumps(item, ensure_ascii=False) for item in bulk_actions) + "\n"
    res = requests.post(f"{ES_URL}/_bulk", data=bulk_body.encode("utf-8"),
                        headers={"Content-Type": "application/x-ndjson"})
    if not res.ok:
        print(f"Bulk indexing error: {res.text}")

    # スクロール検索を使って既存の不要ドキュメントを削除
    try:
        scroll_res = requests.post(f"{ES_URL}/{INDEX_NAME}/_search",
                                   params={"scroll": "1m", "size": 1000},
                                   json={"_source": True, "query": {"match_all": {}}})
        if scroll_res.ok:
            data = scroll_res.json()
            scroll_id = data.get("_scroll_id")
            hits = data.get("hits", {}).get("hits", [])
            indexed_ids = [h["_id"] for h in hits]
            while scroll_id and hits:
                scroll_res = requests.post(f"{ES_URL}/_search/scroll",
                                           json={"scroll": "1m", "scroll_id": scroll_id})
                data = scroll_res.json()
                scroll_id = data.get("_scroll_id")
                hits = data.get("hits", {}).get("hits", [])
                indexed_ids.extend([h["_id"] for h in hits])
            to_delete = set(indexed_ids) - set(files_indexed)
            if to_delete:
                del_actions = []
                for fid in to_delete:
                    del_actions.append({"delete": {"_index": INDEX_NAME, "_id": fid}})
                del_body = "\n".join(json.dumps(item) for item in del_actions) + "\n"
                del_res = requests.post(f"{ES_URL}/_bulk", data=del_body,
                                        headers={"Content-Type": "application/x-ndjson"})
                if not del_res.ok:
                    print(f"Bulk delete error: {del_res.text}")
            if scroll_id:
                requests.delete(f"{ES_URL}/_search/scroll", json={"scroll_id": [scroll_id]})
    except Exception as e:
        print(f"Error during delete check: {e}")

def periodic_index_task():
    """定期的に index_files を実行するバックグラウンドタスク"""
    if not ensure_elasticsearch_ready():
        print("Elasticsearch not available, aborting indexing.")
        return
    ensure_index()
    while True:
        index_files()
        time.sleep(SCAN_INTERVAL)

# アプリ起動時にバックグラウンドでインデックス更新開始
threading.Thread(target=periodic_index_task, daemon=True).start()

# =========================================
# Flask ルーティング
# =========================================
@app.route("/", methods=["GET"])
def search():
    """
    検索画面。クエリを受け取り、Elasticsearch の検索結果を表示。
    Elasticsearch が生成するハイライトの <em>...</em> タグに対して、CSS で背景を黄色にしています。
    """
    query = request.args.get("q", "").strip()
    html = [
        "<html><head><meta charset='UTF-8'><title>File Search</title>",
        "<style> em { background-color: yellow; font-style: normal; } </style>",
        "</head><body>"
    ]
    html.append("<h1>File Search</h1>")
    html.append("<form method='GET' action='/'><input type='text' name='q' value='{}'/>".format(escape(query)))
    html.append("<input type='submit' value='Search'/></form><hr/>")

    if query:
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "filename", "path"]
                }
            },
            "highlight": {
                "fields": {
                    "content": {"fragment_size": 100, "number_of_fragments": 3},
                    "filename": {"fragment_size": 50, "number_of_fragments": 1}
                }
            }
        }
        try:
            es_res = requests.post(f"{ES_URL}/{INDEX_NAME}/_search", json=search_query)
            if es_res.ok:
                data = es_res.json()
                total_hits = data.get("hits", {}).get("total", {}).get("value", 0)
                hits = data.get("hits", {}).get("hits", [])
                html.append(f"<p>Found {total_hits} result(s) for '<strong>{escape(query)}</strong>'.</p>")
                html.append("<ul>")
                for hit in hits:
                    source = hit.get("_source", {})
                    score = hit.get("_score", 0)
                    filename = source.get("filename", "")
                    size = source.get("size", "")
                    modified = source.get("modified", "")
                    path = source.get("path", "")
                    if modified:
                        try:
                            modified = modified.split(".")[0].replace("T", " ")
                        except Exception:
                            pass

                    # リンク先を /open にし、target="_blank" で別タブを開く
                    # q=クエリ を付けることで /open 側でハイライト用に利用できる
                    html.append("<li>")
                    html.append(
                        f"<strong><a href='/open?file={escape(path)}&q={escape(query)}' target='_blank'>"
                        f"{escape(filename)}</a></strong> ( {size} bytes, {escape(modified)} )"
                        f" - Score: {score}<br/>"
                    )
                    html.append(f"<small>Path: {escape(path)}</small><br/>")

                    highlight = hit.get("highlight", {})
                    if highlight:
                        for field, snippets in highlight.items():
                            html.append(f"<div style='margin-top:5px;'>"
                                        f"<em>{escape(field)} のヒット個所 ({len(snippets)}) :</em>")
                            html.append("<ul>")
                            for snippet in snippets:
                                # snippet はすでに ES が <em>...</em> タグを入れてくれる
                                html.append(f"<li>{snippet}</li>")
                            html.append("</ul></div>")
                    html.append("</li>")
                html.append("</ul>")
            else:
                html.append(f"<p>Error searching: {escape(es_res.text)}</p>")
        except Exception as e:
            html.append(f"<p>Search error: {escape(str(e))}</p>")
    html.append("</body></html>")
    return render_template_string("".join(html))

@app.route("/open", methods=["GET"])
def open_file():
    """
    ファイルパスと任意の行番号、及びハイライト用の検索クエリ (q) を受け取り、
    該当ファイルの内容を行番号付きで表示し、行全体や一致文字列を黄色で強調表示。
    """
    file_param = request.args.get("file", "")
    try:
        line_number = int(request.args.get("line", "0"))
    except ValueError:
        line_number = 0
    query_hl = request.args.get("q", None)
    if not file_param:
        return "File parameter is missing.", 400

    # --- Windowsパス -> Linuxパス変換 ---
    # 例: "Z:\folder\file.html" -> "/mnt/z/folder/file.html"
    # 全角コロン "：" をファイル名に含むケースにも対応し、一旦 replace でハイフンへ変換するなど。
    if file_param.startswith("Z:"):
        file_path = FILE_MOUNT_POINT + file_param[2:].replace("\\", "/")
        if not os.path.exists(file_path):
            # 全角コロンをハイフンに置換して再試行
            alt_file_path = file_path.replace("：", "-")
            if os.path.exists(alt_file_path):
                file_path = alt_file_path
            else:
                return f"Error: File not found at '{escape(file_path)}'", 404
    else:
        file_path = file_param

    # --- ファイルの読み込み ---
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {escape(str(e))}", 500

    # --- ファイル内容を行番号付きで表示し、さらに q パラメータがあれば該当文字列をハイライト ---
    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>File Viewer - {escape(file_param)}</title>
        <style>
          pre {{
            counter-reset: linenumber;
          }}
          pre span.line:before {{
            counter-increment: linenumber;
            content: counter(linenumber) ". ";
            color: #888;
          }}
          .highlight {{
            background-color: yellow;
          }}
        </style>
      </head>
      <body>
        <h1>{escape(file_param)}</h1>
        <pre>
    """
    for idx, line in enumerate(lines, start=1):
        escaped_line = escape(line)
        # 正規表現で検索文字列をハイライト
        if query_hl:
            try:
                pattern = re.compile(re.escape(query_hl), re.IGNORECASE)
                display_line = pattern.sub(lambda m: f'<span class="highlight">{m.group(0)}</span>',
                                           escaped_line)
            except Exception:
                display_line = escaped_line
        else:
            display_line = escaped_line

        # 指定行はさらに強調
        if line_number and idx == line_number:
            html += f"<span class='line highlight'>{display_line}</span>"
        else:
            html += f"<span class='line'>{display_line}</span>"

    html += """
        </pre>
        <script>
          // 最初の highlight クラスの要素へスクロール
          var firstHighlighted = document.querySelector('.highlight');
          if (firstHighlighted) {
            firstHighlighted.scrollIntoView();
          }
        </script>
      </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
