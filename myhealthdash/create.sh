#!/usr/bin/env bash
# create.sh ―― health_dashboard プロジェクト＋Docker構成を一括スキャフォールド＆起動
# フルコード／エラー対策付き

set -euo pipefail

########################################
# 1. 変数定義（ホスト／コンテナ パス）
########################################
BASE_DIR=$(pwd)
PROJECT_NAME="health_dashboard"
PROJECT_ROOT="${BASE_DIR}/${PROJECT_NAME}"

# ホスト側ディレクトリ
HOST_TEACHER_DIR="${BASE_DIR}/teacher_data"

# コンテナ内ディレクトリ
CNTR_TEACHER_DIR="/myhealth/teacher_data"
CNTR_MODEL_DIR="/myhealth/models"
CNTR_DATA_DIR="/myhealth/data"
CNTR_DASHBOARD_DIR="/myhealth/dashboard"

# Docker Compose volumes 定義 (ホスト:コンテナ)
VOLUMES=(
  "${HOST_TEACHER_DIR}:${CNTR_TEACHER_DIR}"
  "./models:${CNTR_MODEL_DIR}"
  "./data:${CNTR_DATA_DIR}"
  "./dashboard:${CNTR_DASHBOARD_DIR}"
)

########################################
# 2. プログラム生成セクション
########################################
# 2-1. teacher_data フォルダ確認
if [ ! -d "${HOST_TEACHER_DIR}" ]; then
  echo "‼️ Error: ${HOST_TEACHER_DIR} が見つかりません。配置を確認してください。"
  exit 1
fi

# 2-2. 既存プロジェクトクリア
if [ -d "${PROJECT_ROOT}" ]; then
  echo "▶ 既存プロジェクトを削除: ${PROJECT_ROOT}"
  (cd "${PROJECT_ROOT}" && docker compose down --volumes) || true
  rm -rf "${PROJECT_ROOT}"
fi

# 2-3. プロジェクト構造作成
echo "▶ プロジェクト構造作成: ${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

for pkg in data_sources features strategies models dashboard; do
  mkdir -p "$pkg"
  touch "$pkg/__init__.py"
done

# 2-4. .env
cat > .env <<EOF
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
        df.rename(columns={'Body':'content','Title':'title'}, inplace=True)
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

# 2-8. data_sources/twitter_loader.py (エラー対策追加)
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
    df.rename(columns={'full_text':'content','text':'content'}, inplace=True)
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
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import dash
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px

from strategies.predict import predict

def main():
    df = predict()
    if df.empty:
        return
    df = df.sort_values('date')
    fig1 = px.line(
        df, x='date', y='pred_sentiment', markers=True,
        hover_data=['date','pred_sentiment','ResultId','title'], title='Sentiment over Time'
    )
    fig2 = px.line(
        df, x='date', y='pred_energy', markers=True,
        hover_data=['date','pred_energy','ResultId','title'], title='Energy over Time'
    )
    try:
        import plotly_calplot
        fig3 = plotly_calplot.calplot(
            df, x='date', y='pred_energy', colorscale='RdYlGn', title='Energy Calendar'
        )
    except ImportError:
        df['m'], df['d'] = df['date'].dt.month, df['date'].dt.day
        pivot = df.pivot_table(
            index='m', columns='d', values='pred_energy', aggfunc='mean'
        )
        fig3 = px.imshow(pivot, origin='lower', labels={'color':'Energy'}, title='Energy Heatmap')
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Health Dashboard"),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3),
        html.H2("Raw Entries"),
        dash_table.DataTable(
            columns=[{'name':c,'id':c} for c in df.columns],
            data=df.to_dict('records'),
            page_size=10,
            style_table={'overflowX':'auto'},
            style_cell={'textAlign':'left','padding':'4px'},
            style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'}
        )
    ], style={'padding':'20px'})
    app.run(host='0.0.0.0', port=8060, debug=True)

if __name__ == '__main__':
    main()
EOF

# 2-14. entrypoint.sh
cat > entrypoint.sh << 'EOF'
#!/bin/sh
set -e
echo "▶ Running train.py..."
python /myhealth/strategies/train.py
echo "▶ Starting Dash on 8060..."
exec python /myhealth/dashboard/app.py
EOF
chmod +x entrypoint.sh

# 2-15. Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /myhealth
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8060
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python","/myhealth/dashboard/app.py"]
EOF

########################################
# 3. 操作セクション（セットアップ & 起動）
########################################
# docker-compose.yml を生成
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  app:
    build: .
    container_name: ${PROJECT_NAME}
    ports:
      - "8060:8060"
    volumes:
EOF
for vol in "${VOLUMES[@]}"; do
  echo "      - \"$vol\"" >> docker-compose.yml
done
cat >> docker-compose.yml <<EOF
    environment:
      DB_HOST: "pleasanter_postgres"
      DB_PORT: "5432"
      DB_NAME: "Implem.Pleasanter"
      DB_USER: "postgres"
      DB_PASS: "MyStrongPostgresPass!"
      SITE_ID: "3"
      STAYFREE_DIR: "${CNTR_TEACHER_DIR}/stayfree_exports"
      TWITTER_DIR: "${CNTR_TEACHER_DIR}/twitter_exports"
      MODEL_DIR: "${CNTR_MODEL_DIR}"
    networks:
      - pleasanter-net
networks:
  pleasanter-net:
    external: true
    name: pleasanterDocker_default
EOF

# ビルド & 起動

echo "▶ Docker イメージをビルド"
docker compose build --no-cache --pull
echo "▶ コンテナを起動"
docker compose up -d

echo "✅ Setup 完了: http://localhost:8060"