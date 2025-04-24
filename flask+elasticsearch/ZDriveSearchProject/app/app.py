# Flask + Elasticsearch full-text search application (app.py)
# Modified to meet requirements:
# 1. All original features (Elasticsearch connection, indexing, scroll search, highlight display, file listing) are retained.
# 2. HTML files in search results are served with Content-Type text/html for browser rendering (using send_file).
# 3. Search results retrieve the total hit count and display all results at once (no pagination, all hits shown).

import os
import time
import threading
import json
from datetime import datetime
import requests
from urllib.parse import quote
from flask import Flask, request, escape, render_template_string, send_file

app = Flask(__name__)

# 環境変数から設定（Docker Compose の environment により上書き可能）
ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))
# 対象とするテキストファイルの拡張子セット
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".json", ".py", ".java", ".c",
                  ".cpp", ".html", ".htm", ".mhtml"}
# インデックスされているファイル情報のリスト（ファイル一覧表示用）
indexed_files_info = []

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

def index_files():
    """/mnt/zdrive を走査し、テキストファイルをすべて Elasticsearch にインデックス化する"""
    global indexed_files_info
    files_info_list = []
    files_indexed = []
    bulk_actions = []
    for root, dirs, files in os.walk("/mnt/zdrive"):
        for fname in files:
            file_path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                continue
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
                # /mnt/zdrive 以下のパスを Windows 形式の "Z:\\" パスに変換
                win_path = "Z:" + file_path[len("/mnt/zdrive"):].replace("/", "\\")
                action = {"index": {"_index": INDEX_NAME, "_id": win_path}}
                doc = {
                    "path": win_path,
                    "filename": fname,
                    "extension": ext,
                    "size": size,
                    "modified": modified_iso,
                    "content": content
                }
                bulk_actions.append(action)
                bulk_actions.append(doc)
                files_indexed.append(win_path)
                files_info_list.append({
                    "path": win_path,
                    "filename": fname,
                    "size": size,
                    "modified": modified_iso
                })
            except Exception as e:
                print(f"Failed to index {file_path}: {e}")
                continue
    # 新規または更新されたファイルがあれば一括インデックスを実行
    if bulk_actions:
        bulk_body = "\n".join(json.dumps(item, ensure_ascii=False) for item in bulk_actions) + "\n"
        res = requests.post(f"{ES_URL}/_bulk", data=bulk_body.encode("utf-8"),
                            headers={"Content-Type": "application/x-ndjson"})
        if not res.ok:
            print(f"Bulk indexing error: {res.text}")
    # インデックス上の存在しないファイルのドキュメントを削除（新規ファイルがない場合も実行）
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
    # グローバルなファイル一覧を更新
    indexed_files_info = files_info_list

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
    """全文検索（ページングなし全件表示）"""
    query = request.args.get("q", "").strip()
    html_parts = [
        "<html><head><meta charset='UTF-8'><title>File Search</title></head><body>",
        "<h1>File Search</h1>",
        f"""<form method='GET' action='/' style='margin-bottom:1em'>
               <input type='text' name='q' value='{escape(query)}' size='40'/>
               <input type='submit' value='Search'/>
             </form>""",
        "<hr/>"
    ]
    # 検索実行
    if query:
        # Elasticsearchから全件の検索結果を取得（スクロール検索を使用）
        try:
            res = requests.post(f"{ES_URL}/{INDEX_NAME}/_search",
                                params={"scroll": "1m", "size": 1000},
                                json={
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
                                },
                                timeout=10)
            res.raise_for_status()
            data = res.json()
            scroll_id = data.get("_scroll_id")
            hits = data.get("hits", {}).get("hits", [])
            results = []
            results.extend(hits)
            while scroll_id and hits:
                scroll_res = requests.post(f"{ES_URL}/_search/scroll",
                                           json={"scroll": "1m", "scroll_id": scroll_id},
                                           timeout=10)
                scroll_res.raise_for_status()
                data = scroll_res.json()
                scroll_id = data.get("_scroll_id")
                hits = data.get("hits", {}).get("hits", [])
                results.extend(hits)
            total_hits = len(results)
            # 検索結果の件数を表示
            if scroll_id:
                requests.delete(f"{ES_URL}/_search/scroll", json={"scroll_id": [scroll_id]})
            if total_hits == 0:
                html_parts.append(f"<p>Found 0 result(s) for 『<strong>{escape(query)}</strong>』.</p>")
            else:
                html_parts.append(f"<p>Found {total_hits} result(s) for 『<strong>{escape(query)}</strong>』 — showing 1-{total_hits}.</p>")
                html_parts.append("<ul>")
                for h in results:
                    src = h["_source"]
                    score = h["_score"]
                    path = src["path"]
                    size = src["size"]
                    mod = src["modified"].split(".")[0].replace("T", " ")
                    html_parts.append(
                        f"<li><a href='/view?file={quote(path)}' target='_blank'><strong>{escape(src['filename'])}</strong></a> "
                        f"({size} bytes, {mod}) — Score {score:.2f}<br/><small>{escape(path)}</small>"
                    )
                    # ハイライト表示
                    if "highlight" in h:
                        for fld, snippets in h["highlight"].items():
                            html_parts.append(f"<div><em>{fld}:</em><ul>")
                            for s in snippets:
                                html_parts.append(f"<li>{s}</li>")
                            html_parts.append("</ul></div>")
                    html_parts.append("</li>")
                html_parts.append("</ul>")
        except requests.RequestException as e:
            html_parts.append(f"<p>Search error: {escape(str(e))}</p>")
    else:
        # クエリ未指定時はインデックスされた全ファイルを表示
        total_files = len(indexed_files_info)
        if total_files == 0:
            html_parts.append("<p>No files found.</p>")
        else:
            html_parts.append(f"<p>Found {total_files} file(s) in index — showing 1-{total_files}.</p>")
            html_parts.append("<ul>")
            for info in indexed_files_info:
                path = info["path"]
                fname = info["filename"]
                size = info["size"]
                mod = info["modified"].split(".")[0].replace("T", " ")
                html_parts.append(
                    f"<li><a href='/view?file={quote(path)}' target='_blank'><strong>{escape(fname)}</strong></a> "
                    f"({size} bytes, {mod})<br/><small>{escape(path)}</small></li>"
                )
            html_parts.append("</ul>")
    html_parts.append("</body></html>")
    return render_template_string("".join(html_parts))

@app.route("/view", methods=["GET"])
def view_file():
    """指定されたファイルをブラウザで表示する"""
    file_path_param = request.args.get("file", "")
    if not file_path_param:
        return "No file specified.", 400
    # "Z:\\" 形式のパスを実際のファイルパス "/mnt/zdrive/..." に変換
    if not file_path_param.startswith("Z:"):
        return "Invalid file parameter.", 400
    rel_path = file_path_param[2:]
    if rel_path.startswith("\\") or rel_path.startswith("/"):
        rel_path = rel_path[1:]
    full_path = os.path.normpath(os.path.join("/mnt/zdrive", rel_path.replace("\\", "/")))
    if not full_path.startswith("/mnt/zdrive"):
        return "Invalid file path.", 400
    if not os.path.isfile(full_path):
        return "File not found.", 404
    ext = os.path.splitext(full_path)[1].lower()
    # HTMLファイルは Content-Type: text/html で送信してブラウザでレンダリング
    if ext in (".html", ".htm", ".mhtml"):
        return send_file(full_path, mimetype="text/html")
    else:
        return send_file(full_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

