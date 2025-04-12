#!/bin/bash
# createpj.sh
# このスクリプトは、Pleasanter と Zドライブ全文検索API (Flask + Elasticsearch)
# を Docker Compose による統合構築し起動します。
# ※ ホストの Z ドライブは WSL 上の /mnt/z にマウントされていることを前提とします。
# ※ nginx の外部公開ポートは 8882 を使用します。
# ※ CodeDefiner 用接続文字列のパスワード部分は "pwd=" に修正しています。
#
# :contentReference[oaicite:1]{index=1}

set -e

PROJECT_DIR="ZDriveSearchPleasanterProject"

# 既存のプロジェクトフォルダが存在する場合は削除する
if [ -d "$PROJECT_DIR" ]; then
    echo "既存のディレクトリ $PROJECT_DIR を削除します..."
    rm -rf "$PROJECT_DIR"
fi

echo "プロジェクトディレクトリ $PROJECT_DIR を作成します..."
mkdir -p "$PROJECT_DIR/app"
mkdir -p "$PROJECT_DIR/elasticsearch"
mkdir -p "$PROJECT_DIR/nginx"
mkdir -p "$PROJECT_DIR/codedefiner"

########################################
# docker-compose.yml の生成
########################################
echo "docker-compose.yml を作成します..."
cat > "$PROJECT_DIR/docker-compose.yml" <<'EOF'
version: "3.8"
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
      - esdata:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  flask_api:
    build: ./app
    container_name: flask_api
    depends_on:
      - elasticsearch
    volumes:
      - "/mnt/z:/mnt/zdrive:ro"
    environment:
      ES_URL: "http://elasticsearch:9200"
      INDEX_NAME: "files"
      SCAN_INTERVAL: "300"
    ports:
      - "5000:5000"

  db:
    image: postgres:16-alpine
    container_name: pleasanter_db
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=postgres
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  pleasanter:
    image: implem/pleasanter:latest
    container_name: pleasanter
    depends_on:
      db:
        condition: service_healthy
    expose:
      - "8080"
    environment:
      ASPNETCORE_ENVIRONMENT: "Development"
      ASPNETCORE_PATHBASE: "/myhomesite"
      # 接続文字列は "pwd=" キーを使用する形式に変更
      Implem.Pleasanter_Rds_PostgreSQL_SaConnectionString: "Server=db;Port=5432;Database=postgres;Uid=postgres;pwd=postgres;"
      Implem.Pleasanter_Rds_PostgreSQL_OwnerConnectionString: "Server=db;Port=5432;Database=Implem.Pleasanter;Uid=Pleasanter_Owner;pwd=P@ssw0rd;"
      Implem.Pleasanter_Rds_PostgreSQL_UserConnectionString: "Server=db;Port=5432;Database=Implem.Pleasanter;Uid=Pleasanter_User;pwd=P@ssw0rd;"
  
  codedefiner:
    image: implem/pleasanter:codedefiner
    container_name: codedefiner
    depends_on:
      db:
        condition: service_healthy
    environment:
      Implem.Pleasanter_Rds_PostgreSQL_SaConnectionString: "Server=db;Port=5432;Database=postgres;Uid=postgres;pwd=postgres;"
      Implem.Pleasanter_Rds_PostgreSQL_OwnerConnectionString: "Server=db;Port=5432;Database=Implem.Pleasanter;Uid=Pleasanter_Owner;pwd=P@ssw0rd;"
      Implem.Pleasanter_Rds_PostgreSQL_UserConnectionString: "Server=db;Port=5432;Database=Implem.Pleasanter;Uid=Pleasanter_User;pwd=P@ssw0rd;"
    # codedefiner は一時実行用

  nginx:
    image: nginx:alpine
    container_name: nginx_proxy
    depends_on:
      - pleasanter
      - flask_api
    ports:
      - "8882:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro

networks:
  default:
    name: zdrivesearchpleasanter_default

volumes:
  esdata:
  db_data:
EOF

########################################
# Elasticsearch 用 Dockerfile の生成
########################################
echo "elasticsearch/Dockerfile を作成します..."
cat > "$PROJECT_DIR/elasticsearch/Dockerfile" <<'EOF'
FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.4
RUN bin/elasticsearch-plugin install --batch analysis-kuromoji
EOF

########################################
# Flask API 用 Dockerfile の生成
########################################
echo "app/Dockerfile を作成します..."
cat > "$PROJECT_DIR/app/Dockerfile" <<'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]
EOF

########################################
# Flask API 用 requirements.txt の生成
########################################
echo "app/requirements.txt を作成します..."
cat > "$PROJECT_DIR/app/requirements.txt" <<'EOF'
Flask==2.3.2
requests==2.31.0
EOF

########################################
# Flask API アプリケーション app.py の生成
########################################
echo "app/app.py を作成します..."
cat > "$PROJECT_DIR/app/app.py" <<'EOF'
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
EOF

########################################
# nginx リバースプロキシ設定ファイルの生成
########################################
echo "nginx/nginx.conf を作成します..."
cat > "$PROJECT_DIR/nginx/nginx.conf" <<'EOF'
user  nginx;
worker_processes  1;
error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    sendfile      on;
    keepalive_timeout  65;

    upstream pleasanter_upstream {
        server pleasanter:8080;
    }
    upstream flask_upstream {
        server flask_api:5000;
    }

    server {
        listen       80;
        server_name  localhost;

        # Flask API (Zドライブ検索API) へのプロキシ設定（/zdrive/ 以下）
        location /zdrive/ {
            proxy_pass         http://flask_upstream/;
            proxy_set_header   Host              $host;
            proxy_set_header   X-Real-IP         $remote_addr;
            proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        }

        # Pleasanter アプリへのプロキシ設定
        location / {
            proxy_pass         http://pleasanter_upstream/;
            proxy_set_header   Host              $host;
            proxy_set_header   X-Real-IP         $remote_addr;
            proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        }
    }
}
EOF

echo "プロジェクト構成の生成が完了しました。"
echo "生成されたディレクトリ構造:"
find . -type d

########################################
# Docker イメージのビルド（キャッシュクリア）
########################################
echo "Docker イメージをビルドしています…"
docker compose build --no-cache --pull

########################################
# PostgreSQL コンテナの起動
########################################
echo "PostgreSQL データベースコンテナを起動します..."
docker compose up -d db

echo "PostgreSQL の起動を待機しています..."
retry=0
until docker exec pleasanter_db pg_isready -U postgres > /dev/null 2>&1; do
    retry=$((retry+1))
    if [ $retry -ge 30 ]; then
        echo "PostgreSQL の起動待ちがタイムアウトしました。"
        exit 1
    fi
    sleep 2
done

########################################
# CodeDefiner コンテナによる DB 初期化（日本語/JST設定）
########################################
echo "CodeDefiner コンテナでデータベースを初期化します（日本語/JST設定）..."
docker compose run --rm codedefiner _rds /l "ja" /z "Asia/Tokyo" /y

########################################
# 全サービスの起動
########################################
echo "Pleasanter、Elasticsearch、Flask API、nginx コンテナを起動します..."
docker compose up -d elasticsearch flask_api pleasanter nginx

echo "セットアップが完了しました。"
echo "Pleasanter（管理UI）は http://localhost:8882 にアクセスしてください。"
EOF

To cite the file, include :contentReference[oaicite:2]{index=2} in your response.
