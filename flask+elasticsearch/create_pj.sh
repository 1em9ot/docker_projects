#!/bin/bash
# create_and_run_zdrive_search_project.sh
# このスクリプトは、Zドライブ全文検索エンジン（Flask + Elasticsearch）の
# Docker Compose プロジェクトをまるごと生成し、既存のプロジェクトがあれば削除（一部永続化フォルダは保持）、
# Docker イメージをビルドして全サービスを自動で起動します。
#
# ※ホストの Z ドライブを利用するため、Docker Desktop の設定で /mnt/z (または Z ドライブ共有) が有効であることをご確認ください。
#
# 運用時は、ホスト側の「persistent」フォルダ内にデータが永続化されるので、
# 検証用に中身を編集することができます。また、本番環境でも初回生成時に永続化フォルダが自動生成され、
# 既存のデータが保持される設計となっています。
#
set -e

PROJECT_DIR="ZDriveSearchProject"

# 既存のプロジェクトフォルダが存在する場合、永続化フォルダ(persistent)があれば退避しておく
if [ -d "$PROJECT_DIR" ]; then
    echo "既存の '$PROJECT_DIR' フォルダが見つかりました。永続化フォルダは保持します..."
    if [ -d "$PROJECT_DIR/persistent" ]; then
         echo "永続化フォルダが検出されたため、一時退避します..."
         mv "$PROJECT_DIR/persistent" /tmp/persistent_backup_$$
    fi
    rm -rf "$PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    if [ -d /tmp/persistent_backup_$$ ]; then
         mv /tmp/persistent_backup_$$ "$PROJECT_DIR/persistent"
    fi
else
    mkdir -p "$PROJECT_DIR"
fi

# 必要なサブフォルダを作成（既存の永続化フォルダはそのまま保持）
mkdir -p "$PROJECT_DIR/elasticsearch"
mkdir -p "$PROJECT_DIR/app"
mkdir -p "$PROJECT_DIR/persistent"

########################################
# docker-compose.yml の生成（Elasticsearch のデータはホスト側 persistent フォルダに永続化、
# また、app サービスはホストの app フォルダをバインドマウントしてソース変更がすぐ反映される）
########################################
echo "docker-compose.yml を作成します…"
cat > "$PROJECT_DIR/docker-compose.yml" << 'EOF'
services:
  elasticsearch:
    build: ./elasticsearch
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - bootstrap.memory_lock=true
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - './persistent:/usr/share/elasticsearch/data'
    ports:
      - '9200:9200'
  app:
    build: ./app
    container_name: flask_app
    depends_on:
      - elasticsearch
    volumes:
      - './app:/app'
      - '/mnt/z:/mnt/zdrive:ro'
    ports:
      - '5000:5000'
EOF

########################################
# elasticsearch/Dockerfile の生成
########################################
echo "elasticsearch/Dockerfile を作成します…"
cat > "$PROJECT_DIR/elasticsearch/Dockerfile" << 'EOF'
FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.4
RUN bin/elasticsearch-plugin install --batch analysis-kuromoji
EOF

########################################
# app/Dockerfile の生成
########################################
echo "app/Dockerfile を作成します…"
cat > "$PROJECT_DIR/app/Dockerfile" << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]
EOF

########################################
# app/requirements.txt の生成
########################################
echo "app/requirements.txt を作成します…"
cat > "$PROJECT_DIR/app/requirements.txt" << 'EOF'
Flask==2.3.2
requests==2.31.0
EOF

########################################
# app/app.py の生成（検索結果にヒット件数、スコア、全ハイライトを表示するように改修）
########################################
echo "app/app.py を作成します…"
cat > "$PROJECT_DIR/app/app.py" << 'EOF'
import os
import time
import threading
import json
from datetime import datetime
import requests
from flask import Flask, request, escape, render_template_string

app = Flask(__name__)

# 環境変数から設定（Docker Compose の environment により上書き可能）
ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))
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

def index_files():
    """/mnt/zdrive を走査し、テキストファイルをすべて Elasticsearch にインデックス化する"""
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
                # /mnt/zdrive 以下のパスを Windows 形式の "Z:\" パスに変換
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
    html = ["<html><head><meta charset='UTF-8'><title>File Search</title></head><body>"]
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
                    html.append(f"<small>Path: {escape(path)}</small><br/>")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
EOF

echo "プロジェクトフォルダ '$PROJECT_DIR' の生成が完了しました。"
echo "生成されたディレクトリ構造:"
find "$PROJECT_DIR" -print

########################################
# プロジェクトフォルダへ移動してビルド・起動
########################################
echo "プロジェクトフォルダに移動します…"
cd "$PROJECT_DIR"

echo "Docker イメージをビルドします…"
docker compose build --no-cache --pull

echo "全サービスをバックグラウンド起動します…"
docker compose up -d

echo "プロジェクトのセットアップと起動が完了しました。"
echo "Webブラウザで http://localhost:5000 にアクセスしてください。"