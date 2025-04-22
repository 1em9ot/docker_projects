#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# 事前準備: ホストの UID/GID を取得、防止 .pyc 自動生成
# -----------------------------------------------------------------------------
HOST_UID=$(id -u)
HOST_GID=$(id -g)
export PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# 変数定義
# -----------------------------------------------------------------------------
BASE_DIR="$(pwd)"
PROJECT_NAME="health_dashboard"
PROJECT_ROOT="${BASE_DIR}/${PROJECT_NAME}"

HOST_TEACHER_DIR="${BASE_DIR}/teacher_data"
CNTR_TEACHER_DIR="/myhealth/teacher_data"
CNTR_MODEL_DIR="/myhealth/models"
CNTR_DATA_DIR="/myhealth/data"
CNTR_DASHBOARD_DIR="/myhealth/dashboard"

VOLUMES=(
  "${HOST_TEACHER_DIR}:${CNTR_TEACHER_DIR}"
  "${PROJECT_ROOT}/models:${CNTR_MODEL_DIR}"
  "${PROJECT_ROOT}/data:${CNTR_DATA_DIR}"
  "${PROJECT_ROOT}/dashboard:${CNTR_DASHBOARD_DIR}"
)

# -----------------------------------------------------------------------------
# 1. teacher_data フォルダの存在チェック
# -----------------------------------------------------------------------------
if [ ! -d "$HOST_TEACHER_DIR" ]; then
  echo "‼️ Error: $HOST_TEACHER_DIR が見つかりません。教師データを配置してください。"
  exit 1
fi

# -----------------------------------------------------------------------------
# 2. 既存プロジェクト削除
# -----------------------------------------------------------------------------
if [ -d "$PROJECT_ROOT" ]; then
  echo "▶ 既存プロジェクトを削除: $PROJECT_ROOT"
  (cd "$PROJECT_ROOT" && docker compose down --volumes) || true
  echo "▶ __pycache__ 対策: 削除時の Permission denied を無視します"
  rm -rf "$PROJECT_ROOT" 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# 3. プロジェクト構造作成
# -----------------------------------------------------------------------------
echo "▶ プロジェクト構造作成: $PROJECT_ROOT"
mkdir -p \
  "${PROJECT_ROOT}/data_sources" \
  "${PROJECT_ROOT}/features" \
  "${PROJECT_ROOT}/strategies" \
  "${PROJECT_ROOT}/models" \
  "${PROJECT_ROOT}/dashboard"

# -----------------------------------------------------------------------------
# 4. .env ファイル生成
# -----------------------------------------------------------------------------
cat > "${PROJECT_ROOT}/.env" <<EOF
DB_HOST=pleasanter_postgres
DB_PORT=5432
DB_NAME=Implem.Pleasanter
DB_USER=postgres
DB_PASS=MyStrongPostgresPass!
SITE_ID=3
STAYFREE_DIR=${CNTR_TEACHER_DIR}/stayfree_exports
TWITTER_DIR=${CNTR_TEACHER_DIR}/twitter_exports
MODEL_DIR=${CNTR_MODEL_DIR}
EOF

# -----------------------------------------------------------------------------
# 5. 外部ネットワークの存在確認＆作成
# -----------------------------------------------------------------------------
NETWORK_NAME="pleasanterDocker_default"
if ! docker network ls --filter name="^${NETWORK_NAME}$" --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
  echo "▶ Docker network ${NETWORK_NAME} が存在しないため作成します。"
  docker network create "${NETWORK_NAME}"
fi

# -----------------------------------------------------------------------------
# 6. コード生成セクション
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# 2-5. requirements.txt
cat > requirements.txt <<'EOF'
dash
pandas
numpy
scikit-learn
psycopg2-binary
python-dotenv
textblob
plotly
plotly-calplot
scipy
matplotlib
wordcloud
# ── transformers 依存 ───────────────────────────
torch                 # CPU 版 wheel
transformers
sentencepiece
accelerate
protobuf>=3.20,<4.0
# ── ★ 追加：MeCab tokenizer 用 ──────────────────
fugashi[unidic-lite]  # Pure‑Python版辞書。同梱でOK
EOF

# 2-6. data_sources/pleasanter.py
cat > data_sources/pleasanter.py << 'EOF'
#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd, psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

def fetch_daily_entries(start_date=None, end_date=None):
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS')
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        site_id = os.getenv('SITE_ID', '3')
        sql = 'SELECT * FROM "Implem.Pleasanter"."Results" WHERE "SiteId" = %s'
        params = [site_id]
        if start_date and end_date:
            sql += ' AND "UpdatedTime" BETWEEN %s AND %s'
            params += [f"{start_date} 00:00:00", f"{end_date} 23:59:59"]
        cur.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:
        print(f"Error fetching DB entries: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()
    df = pd.DataFrame(rows)
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['UpdatedTime'], errors='coerce')
        df['date'] = df['created_at'].dt.normalize()
        df.rename(columns={'Body': 'content', 'Title': 'title'}, inplace=True)
    return df
EOF

# 2-7. data_sources/stayfree_loader.py
cat > data_sources/stayfree_loader.py << 'EOF'
#!/usr/bin/env python3
import os, sys, glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd

def load_stayfree_data():
    data_dir = os.getenv('STAYFREE_DIR', './teacher_data/stayfree_exports')
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    dfs = []
    for f in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(f)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
EOF

# 2-8. data_sources/twitter_loader.py
cat > data_sources/twitter_loader.py << 'EOF'
#!/usr/bin/env python3
"""
Twitter エクスポート ZIP から tweets.js / tweets‑part*.js を抽出し、
Dash で使いやすい 4 列（date / content / title / created_at）を返すローダー。
"""

import os
import sys
import zipfile
import json
import re
from datetime import datetime

import pandas as pd

# ── パス設定 ─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

# tweets.js / tweets-partN.js を検出する正規表現
PAT_JS = re.compile(r'(?:^|/)tweets?(-part\d+)?\.js$', re.I)

# created_at の代表的なフォーマット
DT_FMT = "%a %b %d %H:%M:%S %z %Y"   # 例: Tue Mar 18 08:07:10 +0000 2025


# ── ヘルパ関数 ─────────────────────────────────────
def _strip_wrapper(raw: str) -> str:
    """window.YTD ... = [...] ; というヘッダを外して JSON 部分だけ返す"""
    return raw.partition('=')[-1].strip().rstrip(';') if raw.lstrip().startswith('window.YTD') else raw


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """候補のうち DataFrame に存在する最初の列名を返す（無ければ None）"""
    return next((c for c in candidates if c in df.columns), None)


# ── メイン関数 ─────────────────────────────────────
def load_twitter_data() -> pd.DataFrame:
    """
    teacher_data/twitter_exports 配下の ZIP をすべて走査し、
    tweets.js / tweets-part*.js を読み込んで DataFrame を返す。
    取得列:
        - date         : 日付（0時正規化）
        - content      : ツイート本文
        - title        : 'Twitter' 固定（Dash のフィルタ用）
        - created_at   : 元の投稿日付 (datetime64[ns, UTC])
    行が 0 件なら空 DataFrame を返す。
    """
    root = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
    if not os.path.isdir(root):
        return pd.DataFrame()

    tweets: list[dict] = []

    # ZIP → tweets*.js を抽出
    for zname in filter(lambda f: f.lower().endswith('.zip'), os.listdir(root)):
        zpath = os.path.join(root, zname)
        with zipfile.ZipFile(zpath) as zf:
            for member in filter(PAT_JS.search, zf.namelist()):
                raw = zf.read(member).decode('utf-8', errors='ignore')
                try:
                    tweets.extend(json.loads(_strip_wrapper(raw)))
                except json.JSONDecodeError as e:
                    print(f"[twitter_loader] JSON decode error in {member}: {e}")

    if not tweets:
        return pd.DataFrame()

    # フラット化
    df = pd.json_normalize(tweets)

    # ── created_at / date 列作成 ──────────────────
    created_col = _pick_first(df, [
        'created_at', 'tweet.created_at', 'legacy.created_at'
    ])

    if created_col:
        # ① まず UTC で読む      （Twitter Export は +0000 付き）
        created_utc = pd.to_datetime(
            df[created_col], format=DT_FMT, errors='coerce', utc=True
        )

        # ② JST (Asia/Tokyo) に変換
        df['created_at'] = created_utc.dt.tz_convert('Asia/Tokyo')

        # ③ 0 時切り捨てで date 列
        df['date'] = df['created_at'].dt.normalize()

    # ── content 列作成 ────────────────────────────
    content_col = _pick_first(
        df,
        [
            'content', 'full_text', 'text',
            'tweet.content', 'tweet.full_text', 'tweet.text',
            'legacy.content', 'legacy.full_text', 'legacy.text',
        ]
    )
    if not content_col:
        # object 型列のうち最初を本文に充当（最後の保険）
        content_col = next((c for c in df.columns if df[c].dtype == object), None)
    if content_col:
        df['content'] = df[content_col]

    # Dash 側フィルタ用
    df['title'] = 'Twitter'

    # 不要列を落として返す
    return df[['date', 'content', 'title', 'created_at']].copy()


EOF

# 2-9. features/featurize.py
cat > features/featurize.py << 'EOF'
#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd
from textblob import TextBlob

from data_sources.pleasanter import fetch_daily_entries
from data_sources.stayfree_loader import load_stayfree_data
from data_sources.twitter_loader import load_twitter_data

def create_feature_set():
    df_posts = fetch_daily_entries()
    df_sf = load_stayfree_data()
    df_tw = load_twitter_data()
    df = pd.concat([df_posts, df_sf, df_tw], ignore_index=True)
    if 'content' in df.columns:
        df['sentiment'] = df['content'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
    else:
        df['sentiment'] = 0
    return df
EOF

# 2-10. models/predictor.py
cat > models/predictor.py << 'EOF'
#!/usr/bin/env python3
import os, sys, pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd
from sklearn.linear_model import LinearRegression

from features.featurize import create_feature_set

MODEL_PATH = os.getenv('MODEL_DIR', './models') + '/health_model.pkl'

def train_model():
    df = create_feature_set()
    if df.empty:
        print("No data available for training.")
        return
    X = df.index.values.reshape(-1, 1)
    y = df['sentiment']
    model = LinearRegression().fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved.")

def predict():
    if not os.path.isfile(MODEL_PATH):
        train_model()
    if not os.path.isfile(MODEL_PATH):
        return pd.DataFrame()
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    df = create_feature_set()
    if df.empty:
        return df
    X = df.index.values.reshape(-1, 1)
    pred = model.predict(X)
    df['pred_sentiment'] = pred
    df['pred_energy'] = pred
    df['energy_state'] = df['pred_energy'].apply(lambda x: 'Active' if x >= 0 else 'Low')
    return df
EOF

# 2-11. strategies/train.py
cat > strategies/train.py << 'EOF'
#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.predictor import train_model

if __name__ == '__main__':
    train_model()
EOF

# 2-12. strategies/predict.py
cat > strategies/predict.py << 'EOF'
#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.predictor import predict

if __name__ == '__main__':
    df = predict()
    print(df.to_csv(index=False))
EOF

# 2-13. dashboard/app.py
cat > dashboard/app.py << 'EOF'
#!/usr/bin/env python3
"""
Twitter 可視化ダッシュボード  ‑‑ transformers(BERT) による日本語感情分析版

 - Word Cloud（URL / RT / @ID などを除去、CJK フォント対応）
 - 時間帯別ツイート数＋平均感情
 - 感情カテゴリ分布
 - ツイート一覧（感情別ハイライト）

※ transformers と torch が未インストールの場合は ──────────────────
    pip install --upgrade transformers torch
   （初回実行時にモデル daigo/bert-base-japanese-sentiment を自動 DL）
"""

import os
import sys
import re
import json
import base64
from io import BytesIO
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar
from wordcloud import WordCloud, STOPWORDS
import dash
from dash import dcc, html, dash_table
import plotly.express as px
from plotly.subplots import make_subplots

# ―― transformers で事前学習済み日本語 BERT を利用 ──────────────────
from transformers import pipeline, Pipeline

# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
from data_sources.twitter_loader import load_twitter_data  # 自作ローダー

# ── フォント（Noto Sans CJK）が入っている前提 ────────────
JP_FONT = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

# ── Word Cloud 用のクリーニング正規表現 ───────────────
URL_RE      = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE  = re.compile(r'@\w+')
RT_RE       = re.compile(r'(?i)(?:\bRT\b|ＲＴ)')
ASCII_ID_RE = re.compile(r'^[a-zA-Z0-9_]{3,}$')  # 英数字3文字以上
EXTRA_STOP  = {'https', 'http', 't', 'co', 'amp', 'rt'}

# 代表的な強い怒りワード（Negative→anger 判定）
ANGER_WORDS_RE = re.compile(r'(怒|殺|死ね|最悪|クソ|政府|ふざけ|許さ|ムカつ|嫌い)')

# ★ ここを追加・更新
MODEL_ID = os.getenv(
    "HF_MODEL_ID",                      # docker‑compose.yml で上書き可
    "jarvisx17/japanese-sentiment-analysis"
)

# ──────────────────────── ヘルパ関数群 ───────────────────────

def _clean_text(text: str) -> str:
    """URL・@ID・RT を除去して返す"""
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = RT_RE.sub(' ', text)
    return text


def _token_filter(token: str) -> bool:
    """WordCloud 用 Stopwords 判定（True なら捨てる）"""
    return token in EXTRA_STOP or ASCII_ID_RE.match(token)


def sanitize_records(df: pd.DataFrame):
    """DataTable 用に JSON 変換・NaN 空文字化"""
    records: list[dict[str, Any]] = []
    for row in df.to_dict('records'):
        rec: dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = ''
                continue
            if isinstance(v, (list, dict)):
                rec[k] = json.dumps(v, ensure_ascii=False)
                continue
            if not is_scalar(v):
                try:
                    rec[k] = json.dumps(v.tolist() if hasattr(v, 'tolist') else list(v), ensure_ascii=False)
                except Exception:
                    rec[k] = str(v)
                continue
            rec[k] = v
        records.append(rec)
    return records


def generate_wordcloud(series: pd.Series) -> str | None:
    raw = " ".join(series.dropna().astype(str))
    if not raw.strip():
        return None
    cleaned = _clean_text(raw)

    freq = WordCloud(stopwords=set(), regexp=r"[\wぁ-んァ-ン一-龥]+")\
             .process_text(cleaned)
    freq = {tok: cnt for tok, cnt in freq.items() if not _token_filter(tok)}
    if not freq:
        return None

    stops = STOPWORDS.union(EXTRA_STOP)
    try:
        wc = WordCloud(width=800, height=400, background_color='white',
                       font_path=JP_FONT, stopwords=stops).generate_from_frequencies(freq)
    except OSError:
        # サーバに日本語フォントが無い場合はデフォルトフォントで描画
        wc = WordCloud(width=800, height=400, background_color='white',
                       stopwords=stops).generate_from_frequencies(freq)

    buf = BytesIO(); wc.to_image().save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ─────────────────── transformers 感情分類パイプライン ───────────────────
try:
    SENTIMENT_PL: Pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device_map="auto"
    )
except Exception as e:
    print("[WARNING] transformers pipeline 初期化に失敗:", e)
    SENTIMENT_PL = None
    
def _map_label(lbl: str, text: str) -> str:
    """positive / negative → calm / sad・anger"""
    l = lbl.lower()
    if l.startswith("pos"):
        return "calm"
    # negative → 怒語チェックで anger / sad
    return "anger" if ANGER_WORDS_RE.search(text) else "sad"

def classify_emotion(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or SENTIMENT_PL is None:
        return "neutral"
    result = SENTIMENT_PL(text[:128])[0]   # {'label':'negative','score':0.93}
    return _map_label(result["label"], text)


# ─────────────────── Dash アプリ本体 ───────────────────

def main():
    df = load_twitter_data()
    if df.empty or 'content' not in df.columns:
        print("No Twitter data.")
        return

    # 感情付与（transformers）
    df['emotion'] = df['content'].apply(classify_emotion)

    # 日時 & 時間帯
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['hour'] = df['created_at'].dt.hour
    score_map = {'anger': -1, 'sad': -0.5, 'neutral': 0, 'calm': 1}
    df['score'] = df['emotion'].map(score_map)

    # 時間帯集計
    hour_stats = (df.groupby('hour')
                    .agg(tweet_count=('content', 'count'),
                         avg_sentiment=('score', 'mean'))
                    .reindex(range(24), fill_value=0)
                    .reset_index())

    # 感情分布
    emo_cnt = df['emotion'].value_counts().reset_index()
    emo_cnt.columns = ['emotion', 'count']

    # ワードクラウド
    wc_uri = generate_wordcloud(df['content'])

    # グラフ: 時間帯別
    fig_hour = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hour.add_bar(x=hour_stats['hour'], y=hour_stats['tweet_count'],
                     name='ツイート数', marker_color='steelblue')
    fig_hour.add_scatter(x=hour_stats['hour'], y=hour_stats['avg_sentiment'],
                         name='平均感情値', mode='lines+markers',
                         line=dict(color='orangered'), secondary_y=True)
    fig_hour.update_xaxes(title='時間帯 (時)')
    fig_hour.update_yaxes(title_text='ツイート数', secondary_y=False)
    fig_hour.update_yaxes(title_text='平均感情値', range=[-1, 1], secondary_y=True)
    fig_hour.update_layout(title='時間帯別ツイート数と平均感情値')

    # グラフ: 感情割合
    fig_emotion = px.pie(
        emo_cnt, names='emotion', values='count', title='感情カテゴリ分布',
        color='emotion',
        color_discrete_map={'anger': '#ff6666', 'sad': '#66b3ff',
                            'calm': '#99ff99', 'neutral': '#cccccc'}
    )

    # Dash Layout
    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={'padding': '20px'},
        children=[
            html.H1("Health Dashboard"),
            html.H2("Twitterワードクラウド"),
            html.Img(src=wc_uri, style={'width': '100%', 'maxHeight': '400px'})
                if wc_uri else html.P("※ツイートデータがありません"),
            html.H2("時間帯別投稿数と平均感情値"),
            dcc.Graph(figure=fig_hour),
            html.H2("感情カテゴリ分布"),
            dcc.Graph(figure=fig_emotion),
            html.H2("ツイート一覧"),
            dash_table.DataTable(
                columns=[{'name': '日時',   'id': 'created_at'},
                         {'name': '内容',   'id': 'content'},
                         {'name': 'emotion','id': 'emotion'}],
                data=sanitize_records(df[['created_at', 'content', 'emotion']]),
                page_action='none',  # ページネーション無効
                style_table={'height': '500px', 'overflowY': 'auto', 'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '4px'},
                style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                hidden_columns=['emotion'],
                style_data_conditional=[
                    {'if': {'filter_query': '{emotion} = "anger"',  'column_id': 'content'}, 'backgroundColor': '#ffcccc'},
                    {'if': {'filter_query': '{emotion} = "sad"',    'column_id': 'content'}, 'backgroundColor': '#cce5ff'},
                    {'if': {'filter_query': '{emotion} = "calm"',   'column_id': 'content'}, 'backgroundColor': '#ccffcc'},
                ],
                virtualization=True  # 行数が多くても軽快
            )
        ]
    )

    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__ == '__main__':
    main()

EOF

# 2-14. entrypoint.sh
# entrypoint.sh を書き出し
cat > entrypoint.sh <<'EOF'
#!/bin/sh
set -e
export MPLCONFIGDIR=/tmp/.matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "▶ Running train.py..."
python /myhealth/strategies/train.py

echo "▶ Starting Dash..."
exec python /myhealth/dashboard/app.py
EOF

# CR(\r) を除去して実行権限を付与
sed -i 's/\r$//' entrypoint.sh
chmod +x entrypoint.sh

# 2-15. Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim

# ── 基本セットアップ ─────────────────────────
WORKDIR /myhealth
COPY requirements.txt .

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 日本語フォント & LFS など追加パッケージ
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates \
      fonts-noto-cjk libgl1 libglib2.0-0 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# アプリ一式をコピー
COPY . .

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/myhealth/dashboard/app.py"]
EOF

# -----------------------------------------------------------------------------
# 7. docker-compose.yml 生成＆起動
# -----------------------------------------------------------------------------
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  app:
    build: .
    container_name: ${PROJECT_NAME}
    ports:
      - "8060:8060"
    volumes:
$(printf '      - %s\n' "${VOLUMES[@]}")
    user: "${HOST_UID}:${HOST_GID}"
    env_file:
      - .env
    environment:
      - HF_HOME=/tmp/hfcache
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
        - STOPWORD_PATH=/teacher_data/stopwords.txt
    networks:
      - pleasanter-net

networks:
  pleasanter-net:
    external: true
    name: ${NETWORK_NAME}
EOF

echo "▶ Building Docker image..."
docker compose build --no-cache --pull

echo "▶ Starting containers..."
docker compose up -d

echo "✅ Ready: http://localhost:8060"
