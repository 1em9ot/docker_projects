import os
import time
import threading
import json
from datetime import datetime
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 環境変数から設定を取得（docker-compose.ymlで設定）
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))

# 拡張子によるフィルタリング
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".json", ".py", ".java",
                   ".c", ".cpp", ".html", ".htm", ".mhtml"}

def ensure_elasticsearch_ready():
    """Elasticsearchが利用可能になるまで待機する"""
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
    """インデックスが存在しなければ作成し、マッピングを設定する"""
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

def index_all_files():
    """Zドライブ内のテキストファイルを全てインデックス（フルスキャン）"""
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
                # /mnt/zdrive 以下のパスを Windows形式 "Z:\..." に変換
                win_path = "Z:" + file_path[len("/mnt/zdrive"):].replace("/", "\\")
                bulk_actions.append({ "index": {"_index": INDEX_NAME, "_id": win_path} })
                bulk_actions.append({
                    "path": win_path,
                    "filename": fname,
                    "extension": ext,
                    "size": size,
                    "modified": modified_iso,
                    "content": content
                })
                files_indexed.append(win_path)
            except Exception as e:
                print(f"Failed to index {file_path}: {e}")
                continue
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
                                   json={"_source": False, "query": {"match_all": {}}})
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
                    del_actions.append({ "delete": {"_index": INDEX_NAME, "_id": fid} })
                del_body = "\n".join(json.dumps(item) for item in del_actions) + "\n"
                del_res = requests.post(f"{ES_URL}/_bulk", data=del_body,
                                        headers={"Content-Type": "application/x-ndjson"})
                if not del_res.ok:
                    print(f"Bulk delete error: {del_res.text}")
            if scroll_id:
                requests.delete(f"{ES_URL}/_search/scroll", json={"scroll_id": [scroll_id]})
    except Exception as e:
        print(f"Error during cleanup: {e}")

def periodic_index_task():
    if not ensure_elasticsearch_ready():
        print("Elasticsearch not available, indexing aborted.")
        return
    ensure_index()
    while True:
        index_all_files()
        time.sleep(SCAN_INTERVAL)

threading.Thread(target=periodic_index_task, daemon=True).start()

@app.route("/search", methods=["GET"])
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": [], "count": 0})
    search_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content", "filename", "path"]
            }
        },
        "highlight": {
            "fields": {
                "content": {"fragment_size": 100, "number_of_fragments": 1}
            }
        }
    }
    try:
        res = requests.post(f"{ES_URL}/{INDEX_NAME}/_search", json=search_query)
        if not res.ok:
            return jsonify({"error": res.text}), 500
        data = res.json()
        hits = data.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            filename = src.get("filename", "")
            size = src.get("size", 0)
            modified = src.get("modified", "")
            path = src.get("path", "")
            snippet = ""
            highlight = hit.get("highlight", {})
            if highlight:
                snippets = highlight.get("content", [])
                if snippets:
                    snippet = snippets[0]
            if modified:
                try:
                    modified = modified.split(".")[0].replace("T", " ")
                except Exception:
                    pass
            results.append({
                "filename": filename,
                "size": size,
                "modified": modified,
                "path": path,
                "snippet": snippet
            })
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/move", methods=["POST"])
def api_move():
    data = request.get_json(force=True)
    src_path_win = data.get("source", "") or ""
    dest_path_win = data.get("dest", "") or ""
    if not src_path_win or not dest_path_win:
        return jsonify({"error": "source and dest paths are required"}), 400
    try:
        if src_path_win.startswith("Z:"):
            src_internal = "/mnt/zdrive" + src_path_win[2:].replace("\\", "/")
        else:
            src_internal = src_path_win
        if dest_path_win.startswith("Z:"):
            dest_internal = "/mnt/zdrive" + dest_path_win[2:].replace("\\", "/")
        else:
            dest_internal = dest_path_win
        dest_dir = os.path.dirname(dest_internal)
        if dest_dir and not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(dest_internal):
            return jsonify({"error": "destination file already exists"}), 409
        os.replace(src_internal, dest_internal)
    except Exception as e:
        return jsonify({"error": f"Failed to move file: {e}"}), 500
    old_id = src_path_win
    new_id = dest_path_win
    requests.delete(f"{ES_URL}/{INDEX_NAME}/_doc/{old_id}")
    try:
        with open(dest_internal, "rb") as f:
            raw = f.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("cp932")
            except UnicodeDecodeError:
                content = raw.decode("utf-8", errors="ignore")
        size = os.path.getsize(dest_internal)
        mtime = os.path.getmtime(dest_internal)
        modified_iso = datetime.fromtimestamp(mtime).isoformat()
        doc = {
            "path": new_id,
            "filename": os.path.basename(dest_internal),
            "extension": os.path.splitext(dest_internal)[1].lower(),
            "size": size,
            "modified": modified_iso,
            "content": content
        }
        res = requests.post(f"{ES_URL}/{INDEX_NAME}/_doc/{new_id}", json=doc)
        if not res.ok:
            print(f"Failed to index moved file: {res.text}")
    except Exception as e:
        print(f"Warning: could not index moved file content: {e}")
    return jsonify({"result": "moved", "newPath": dest_path_win})

@app.route("/delete", methods=["POST"])
def api_delete():
    data = request.get_json(force=True)
    path_win = data.get("path", "") or ""
    if not path_win:
        return jsonify({"error": "path is required"}), 400
    try:
        internal_path = "/mnt/zdrive" + path_win[2:].replace("\\", "/") if path_win.startswith("Z:") else path_win
        os.remove(internal_path)
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {e}"}), 500
    requests.delete(f"{ES_URL}/{INDEX_NAME}/_doc/{path_win}")
    return jsonify({"result": "deleted", "path": path_win})

@app.route("/index", methods=["POST"])
def api_reindex():
    threading.Thread(target=index_all_files, daemon=True).start()
    return jsonify({"result": "reindex_started"})

@app.route("/metadata", methods=["GET", "POST"])
def api_metadata():
    path_win = None
    if request.method == "GET":
        path_win = request.args.get("path", "")
    else:
        data = request.get_json(force=True)
        path_win = data.get("path", "")
    if not path_win:
        return jsonify({"error": "path is required"}), 400
    internal_path = "/mnt/zdrive" + path_win[2:].replace("\\", "/") if path_win.startswith("Z:") else path_win
    if not os.path.exists(internal_path):
        return jsonify({"error": "file not found"}), 404
    try:
        stat = os.stat(internal_path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({
            "filename": os.path.basename(internal_path),
            "size": size,
            "modified": mtime,
            "created": ctime,
            "path": path_win
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get metadata: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
