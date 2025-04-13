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
# 設定
# =========================================
ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "files")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "300"))

# ※ インデックス作成では /mnt/zdrive を走査しているので、
# /open では変換時に /mnt/zdrive を利用する
FILE_MOUNT_PREFIX = "/mnt/zdrive"

# 対象とするテキストファイルの拡張子セット
TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".json", ".py",
                   ".java", ".c", ".cpp", ".html", ".htm", ".mhtml"}

# =========================================
# Elasticsearch 関連
# =========================================
def ensure_elasticsearch_ready():
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
        # ここでは、元のスキャンディレクトリ "/mnt/zdrive" を除いて、
        # Windows 形式 "Z:\..." に変換（※逆変換用情報として）
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
    files_indexed = []
    bulk_actions = []

    files_to_process = []
    for root, dirs, files in os.walk("/mnt/zdrive"):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                continue
            full_path = os.path.join(root, fname)
            files_to_process.append((full_path, fname))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(process_file, fp, name) for fp, name in files_to_process]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    for win_path, doc in results:
        action = {"index": {"_index": INDEX_NAME, "_id": win_path}}
        bulk_actions.append(action)
        bulk_actions.append(doc)
        files_indexed.append(win_path)

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
    if not ensure_elasticsearch_ready():
        print("Elasticsearch not available, aborting indexing.")
        return
    ensure_index()
    while True:
        index_files()
        time.sleep(SCAN_INTERVAL)

threading.Thread(target=periodic_index_task, daemon=True).start()

# =========================================
# ハイライト処理用関数
# =========================================
def highlight_line(line, query):
    """
    元の行のテキスト（そのまま）に対して、検索クエリに一致した部分のみを
    HTMLエスケープした上で <span class="highlight"> で囲み、連結して返す。
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    parts = []
    last = 0
    for m in pattern.finditer(line):
        # 非マッチ部分をエスケープして追加
        parts.append(escape(line[last:m.start()]))
        # マッチ部分はエスケープした上で span タグで囲む
        parts.append(f'<span class="highlight">{escape(m.group(0))}</span>')
        last = m.end()
    parts.append(escape(line[last:]))
    return "".join(parts)

# =========================================
# Flask ルーティング
# =========================================
@app.route("/", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    html = [
        "<html><head><meta charset='UTF-8'><title>File Search</title>",
        # Elasticsearch のハイライト結果（<em>タグ）に対して背景黄色を適用
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

                    # リンク先は /open エンドポイントへ。target="_blank" で新規タブ表示。
                    html.append("<li>")
                    html.append(
                        f"<strong><a href='/open?file={escape(path)}&q={escape(query)}' target='_blank'>"
                        f"{escape(filename)}</a></strong> ( {size} bytes, {escape(modified)} ) - Score: {score}<br/>"
                    )
                    html.append(f"<small>Path: {escape(path)}</small><br/>")
                    highlight = hit.get("highlight", {})
                    if highlight:
                        for field, snippets in highlight.items():
                            html.append(f"<div style='margin-top:5px;'><em>{escape(field)} のヒット個所 ({len(snippets)}) :</em>")
                            html.append("<ul>")
                            for snippet in snippets:
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
    該当ファイルの内容を行番号付きで表示し、行内の検索文字列部分を黄色で強調表示します。
    """
    file_param = request.args.get("file", "")
    try:
        line_number = int(request.args.get("line", "0"))
    except ValueError:
        line_number = 0
    query_hl = request.args.get("q", None)
    if not file_param:
        return "File parameter is missing.", 400

    # ここで、file_param (例:"Z:\ルドラ - Wikipedia (2025_3_4 12：28：21).html") を
    # 元のスキャンで使用した形式 "/mnt/zdrive/..." に変換する
    if file_param.startswith("Z:"):
        file_path = "/mnt/zdrive" + file_param[2:].replace("\\", "/")
        # 万一、全角コロンの問題などで存在しなければ再試行
        if not os.path.exists(file_path):
            alt_file_path = file_path.replace("：", "-")
            if os.path.exists(alt_file_path):
                file_path = alt_file_path
            else:
                return f"Error: File not found at '{escape(file_path)}'", 404
    else:
        file_path = file_param

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {escape(str(e))}", 500

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
        if query_hl:
            # ハイライト用関数で一致部分を <span class="highlight"> で囲む
            display_line = highlight_line(line, query_hl)
        else:
            display_line = escape(line)
        if line_number and idx == line_number:
            html += f"<span class='line highlight'>{display_line}</span>"
        else:
            html += f"<span class='line'>{display_line}</span>"
    html += """
        </pre>
        <script>
          var firstHighlighted = document.querySelector('.highlight');
          if (firstHighlighted) { firstHighlighted.scrollIntoView(); }
        </script>
      </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
