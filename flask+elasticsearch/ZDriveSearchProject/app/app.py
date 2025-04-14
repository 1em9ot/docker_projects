import os
import time
import threading
import json
from datetime import datetime
import requests
from flask import Flask, request, escape, render_template_string, send_file
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# 環境変数から設定（Docker Compose の environment により上書き可能）
ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))
# 解析対象から除外するための文字数閾値（例: 1,000,000文字を超える場合）
SKIP_LENGTH_THRESHOLD = int(os.environ.get("SKIP_LENGTH_THRESHOLD", "1000000"))
# 対象とするテキストファイルの拡張子セット
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".json", ".py", ".java", ".c", ".cpp", ".html", ".htm", ".mhtml"}

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
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
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
        # /mnt/zdrive 以下のパスを Windows 形式の "Z:\" パスに変換
        win_path = "Z:" + file_path[len("/mnt/zdrive"):].replace("/", "\\")
        if len(content) > SKIP_LENGTH_THRESHOLD:
            # 長大な内容の場合、解析対象外としプレースホルダ文字列を設定する
            print(f"解析対象外: {file_path}（内容が長すぎるため、解析をスキップ）")
            content = "[このファイルは内容が長すぎるため、解析対象外です。]"
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
        print(f"Failed to index {file_path}: {e}")
        return None

def index_files():
    files_indexed = []
    bulk_actions = []
    tasks = []
    # CPUコア数に合わせたスレッド数でファイルを並列処理
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for root, dirs, files in os.walk("/mnt/zdrive"):
            for fname in files:
                file_path = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                tasks.append(executor.submit(process_file, file_path, fname))
        for future in as_completed(tasks):
            result = future.result()
            if result:
                win_path, doc = result
                files_indexed.append(win_path)
                action = {"index": {"_index": INDEX_NAME, "_id": win_path}}
                bulk_actions.append(action)
                bulk_actions.append(doc)
    if not bulk_actions:
        return
    bulk_body = "\n".join(json.dumps(item, ensure_ascii=False) for item in bulk_actions) + "\n"
    res = requests.post(f"{ES_URL}/_bulk", data=bulk_body.encode("utf-8"),
                        headers={"Content-Type": "application/x-ndjson"})
    if not res.ok:
        print(f"Bulk indexing error: {res.text}")
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

# バックグラウンドスレッドでインデックス更新を開始
threading.Thread(target=periodic_index_task, daemon=True).start()

@app.route("/", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    html = [
        "<html>",
          "<head>",
            "<meta charset='UTF-8'>",
            "<title>File Search</title>",
            "<style>",
              "body { font-family: Arial, sans-serif; }",
              "em { background-color: transparent; font-style: italic; }",
              "a { text-decoration: none; color: blue; }",
              "a:hover { text-decoration: underline; }",
            "</style>",
          "</head>",
          "<body>"
    ]
    html.append("<h1>File Search</h1>")
    html.append("<form method='GET' action='/'><input type='text' name='q' value='{}'/>".format(escape(query)))
    html.append("<input type='submit' value='Search'/></form><hr/>")
    if query:
        # Elasticsearch への検索クエリ
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
            res = requests.post(f"{ES_URL}/{INDEX_NAME}/_search", json=search_query)
            if res.ok:
                data = res.json()
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
                    html.append("<li>")
                    html.append(f"<strong>{escape(filename)}</strong> ( {size} bytes, {escape(modified)} ) - Score: {score}<br/>")
                    # /file エンドポイントを利用して、新タブでファイルを開く
                    html.append(f"<small>Path: <a href='/file?path={quote(path)}' target='_blank' rel='noopener noreferrer'>{escape(path)}</a></small><br/>")
                    highlight = hit.get("highlight", {})
                    if highlight:
                        for field, snippets in highlight.items():
                            html.append(f"<div style='margin-top:5px;'><em>{escape(field)} のヒット個所 ({len(snippets)}):</em>")
                            html.append("<ul>")
                            for snippet in snippets:
                                html.append(f"<li>{snippet}</li>")
                            html.append("</ul></div>")
                    html.append("</li>")
                html.append("</ul>")
            else:
                html.append(f"<p>Error searching: {escape(res.text)}</p>")
        except Exception as e:
            html.append(f"<p>Search error: {escape(str(e))}</p>")
    html.append("</body></html>")
    return render_template_string("".join(html))

@app.route("/file", methods=["GET"])
def serve_file():
    """/file エンドポイントでは、渡された Windows 形式のパスを /mnt/zdrive に変換し、
    HTML ファイルの場合は text/html で返す"""
    file_path = request.args.get("path", "")
    if not file_path:
        return "No path provided", 400
    if not file_path.startswith("Z:"):
        return "Invalid file path", 400
    actual_path = "/mnt/zdrive" + file_path[2:].replace("\\", "/")
    if not os.path.exists(actual_path):
        return "File not found", 404
    ext = os.path.splitext(actual_path)[1].lower()
    if ext in {".html", ".htm", ".mhtml"}:
        return send_file(actual_path, mimetype="text/html")
    else:
        return send_file(actual_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
