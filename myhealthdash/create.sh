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
cat > requirements.txt <<EOF
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
import os, sys, zipfile, json
from datetime import datetime
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

def load_twitter_data():
    data_dir = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    all_tweets = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(os.path.join(data_dir, fname)) as zp:
                    for member in zp.namelist():
                        if 'tweet.js' in member:
                            raw = zp.read(member).decode('utf-8')
                            js = raw.partition('=')[-1].strip().rstrip(';')
                            all_tweets += json.loads(js)
            except Exception:
                continue
    if not all_tweets:
        return pd.DataFrame()
    df = pd.json_normalize(all_tweets)
    if 'created_at' in df.columns:
        df['created_at'] = df['created_at'].apply(
            lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y") if isinstance(x, str) else pd.NaT
        )
        df['date'] = df['created_at'].dt.normalize()
    df.rename(columns={'full_text': 'content', 'text': 'content'}, inplace=True)
    return df
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
import os, sys, json
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import dash
from dash import dcc, html, dash_table
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategies.predict import predict

def sanitize_records(df):
    records = []
    for row in df.to_dict('records'):
        new = {}
        for k,v in row.items():
            if pd.isna(v):
                new[k] = ''
            else:
                new[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v
        records.append(new)
    return records

def cluster_emotion_keywords(df, top_k=50):
    df_tw = df[df['title'].str.contains("Twitter", na=False)]
    if df_tw.empty: return {'anger':[], 'sad':[], 'calm':[]}
    cnt = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=top_k)
    X = cnt.fit_transform(df_tw['content'].dropna().astype(str))
    words = cnt.get_feature_names_out()
    counts = X.toarray().sum(axis=0).reshape(-1,1)
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(counts)
    clusters = {i:[] for i in range(3)}
    for w,l in zip(words, labels): clusters[l].append(w)
    return {'anger':clusters[0], 'sad':clusters[1], 'calm':clusters[2]}

def classify_emotion(text, emo_dict):
    for label, words in emo_dict.items():
        if any(w in text for w in words): return label
    return 'neutral'

def main():
    df = predict()
    if df.empty:
        print("No data to display."); exit

    emo_dict = cluster_emotion_keywords(df)
    df['emotion'] = df['content'].apply(lambda t: classify_emotion(t, emo_dict))

    df['date'] = pd.to_datetime(df['date'])
    last_month = df['date'].dt.month.max()
    dm = df[df['date'].dt.month==last_month].copy()
    dm['day'] = dm['date'].dt.day
    daily = dm.groupby('day')['pred_energy'].mean().reset_index()

    fig1 = px.bar(daily, x='day', y='pred_energy',
                  title=f'Avg Energy Month {last_month}', range_y=[-1,1])
    table = sanitize_records(df)
    styles = [
      {'if':{'filter_query':f'{{emotion}}="{e}"','column_id':'content'},
       'backgroundColor':c}
      for e,c in zip(['anger','sad','calm','neutral'],
                     ['#ffcccc','#cce5ff','#ccffcc','#ffffff'])
    ]

    app = dash.Dash(__name__)
    app.layout = html.Div([
      html.H1("Health Dashboard"),
      dcc.Graph(figure=fig1),
      html.H2("Entries"),
      dash_table.DataTable(
        columns=[{'name':col,'id':col} for col in df.columns],
        data=table, page_size=10,
        style_data_conditional=styles
      )
    ], style={'padding':'20px'})

    app.run(host='0.0.0.0', port=8060, debug=True)

if __name__=='__main__':
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
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /myhealth
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python","/myhealth/dashboard/app.py"]
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
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
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
