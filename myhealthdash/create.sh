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
  "${PROJECT_ROOT}/dashboard" \
  "${PROJECT_ROOT}/data"
chown -R "$HOST_UID:$HOST_GID" "${PROJECT_ROOT}/data"

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
torch
transformers
sentencepiece
accelerate
protobuf>=3.20,<4.0
fugashi[unidic-lite]
torchmetrics>=1.4
EOF

# ───────────────────────────────────────────────────────────────
# 2-6. data_sources/pleasanter.py を生成
# ───────────────────────────────────────────────────────────────
cat > data_sources/pleasanter.py << 'EOF'
#!/usr/bin/env python3
import os
import pandas as pd
import psycopg2
import pytz
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()
JST = pytz.timezone('Asia/Tokyo')

def fetch_daily_entries(start_date=None, end_date=None) -> pd.DataFrame:
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
        print(f"[Error] fetching Pleasanter entries: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # UpdatedTime を UTC→JST に変換し、ミリ秒以下を切り捨て
    df['created_at'] = (
        pd.to_datetime(df['UpdatedTime'], utc=True, errors='coerce')
          .dt.tz_convert(JST)
          .dt.floor('s')        # ミリ秒以下を 00 に
    )
    df['date'] = df['created_at'].dt.normalize()
    df.rename(columns={'Body': 'content', 'Title': 'title'}, inplace=True)
    return df
EOF
chmod +x data_sources/pleasanter.py


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

        # ② JST (Asia/Tokyo) に変換し、ミリ秒以下を落とす（秒単位に丸め）
        df['created_at'] = (
            created_utc
            .dt.tz_convert('Asia/Tokyo')
            .dt.floor('s')        # ミリ秒以下を切り捨て
        )
        

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
"""
ニューラルネットによるヘルススコア回帰モデル
"""

import os, json, random, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchmetrics import MeanAbsoluteError

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("/myhealth/models/health_model.pt")

# ── データ読み込み（例として CSV） ─────────────────────
def load_feature_matrix():
    """
    返り値:
        X: np.ndarray [n_samples, n_features]
        y: np.ndarray [n_samples]
    """
    import pandas as pd
    df = pd.read_csv("/myhealth/data/health_features.csv")
    y = df["target"].values.astype(np.float32)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    return X, y

# ── MLP モデル ───────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── 学習関数 ───────────────────────────────────────
def train_model(num_epochs: int = 300, batch_size: int = 64) -> None:
    X, y = load_feature_matrix()
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
    except ValueError as e:
        print(f"[WARN] train_test_split でエラー ({e}) → 学習をスキップします")
        return


    tr_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)

    model = MLP(in_dim=X.shape[1]).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()
    metric = MeanAbsoluteError().to(DEVICE)

    best_loss, patience, PATIENCE = float("inf"), 0, 15

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward(); opt.step()

        # ---- validate ----
        model.eval(); metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += crit(pred, yb).item() * xb.size(0)
                metric.update(pred, yb)
        val_loss /= len(val_loader.dataset)
        mae = metric.compute().item()
        print(f"[{epoch:03d}] val_loss={val_loss:.4f}  MAE={mae:.4f}")

        # early stopping
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"Best val_loss={best_loss:.4f}  → saved to {MODEL_PATH}")

# ── 予測関数 ─────────────────────────────────────────
_model_cache = None
def predict(X: np.ndarray) -> np.ndarray:
    global _model_cache
    if _model_cache is None:
        in_dim = X.shape[1]
        model = MLP(in_dim); model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        _model_cache = model
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return _model_cache(X_t).cpu().numpy()

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

# 2-11. strategies/auto_generate_features.py
cat > strategies/auto_generate_features.py << 'EOF'
#!/usr/bin/env python3
import os, sys, glob, json
from datetime import datetime, timedelta, timezone

# Directories for input data (from environment variables or defaults)
TWITTER_DIR = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
STAYFREE_DIR = os.getenv('STAYFREE_DIR', './teacher_data/stayfree_exports')

# Try to initialize a sentiment analysis pipeline (Japanese BERT model)
sentiment_pipeline = None
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="daigo/bert-base-japanese-sentiment"
    )
except Exception as e:
    print(f"[WARNING] Could not load transformers model: {e}")
    sentiment_pipeline = None

# Define a simple fallback sentiment classifier (keyword-based) if pipeline is not available
positive_keywords = ["嬉しい", "楽しい", "最高", "大好き", "最高", "感謝", "happy", "joy", "love"]
negative_keywords = ["悲しい", "辛い", "苦しい", "嫌い", "最悪", "怒り", "死にたい", "疲れた", "angry", "sad"]

def classify_sentiment(text: str) -> str:
    """Classify sentiment of the given text as 'positive', 'negative', or 'neutral'."""
    # Use the BERT sentiment pipeline if available
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text[:512])  # truncate to 512 tokens if very long
            if result:
                # Get the label and normalize to lowercase
                label = result[0]['label'].lower()
                if label in ["positive", "negative", "neutral"]:
                    return label
                # Some models might output labels in Japanese or different format; handle if needed
                if label in ["ポジティブ", "ポジティブだね"]:  # just an example if Japanese output
                    return "positive"
                if label in ["ネガティブ", "ネガティブだね"]:
                    return "negative"
                # If label not recognized, fall through to keyword method
        except Exception as e:
            print(f"[WARNING] Sentiment pipeline failed on text: {e}")
            # fallback to keyword method if pipeline fails for this text

    # Fallback keyword-based sentiment detection:
    text_lower = text.lower()
    pos = any(word in text for word in positive_keywords)
    neg = any(word in text for word in negative_keywords)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"

# Helper to convert epoch ms to local datetime
JST = timezone(timedelta(hours=9))  # Japan Standard Time
def to_jst_datetime(ms: int) -> datetime:
    """Convert a timestamp in milliseconds to a timezone-aware datetime in JST."""
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(JST)

# 1. Parse StayFree usage backup to get sleep duration per day
sleep_hours_by_date = {}  # store estimated sleep hours for each date (string)
base_target_by_date = {}  # store base target (60,70,80) for each date

# Find usage backup file(s) in STAYFREE_DIR
for filepath in glob.glob(os.path.join(STAYFREE_DIR, "*.usage_backup")):
    try:
        # Read the file as binary and extract JSON content (skipping any non-JSON header bytes)
        with open(filepath, "rb") as bf:
            data = bf.read()
            # Find the first curly brace to locate JSON start
            start_idx = data.find(b'{')
            json_str = data[start_idx:].decode('utf-8')
            usage_data = json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Could not parse StayFree backup file {filepath}: {e}")
        continue

    # The usage_data is expected to contain a 'stores' dict with 'sessions_2' list
    sessions = usage_data.get("stores", {}).get("sessions_2", [])
    # Split sessions by day boundaries and record usage intervals per day
    usage_intervals_by_date = {}  # dict of date -> list of (start, end) datetimes in that date
    for sess in sessions:
        start_dt = to_jst_datetime(sess.get("startedAt"))
        end_dt = to_jst_datetime(sess.get("endedAt"))
        # Ensure start_dt <= end_dt
        if end_dt < start_dt:
            end_dt = start_dt
        # If session spans multiple days, split at midnight of start_dt's day
        start_date = start_dt.date()
        end_date = end_dt.date()
        if start_date == end_date:
            usage_intervals_by_date.setdefault(str(start_date), []).append((start_dt, end_dt))
        else:
            # Session goes into the next day
            # End of start_date at 23:59:59...
            end_of_start_day = datetime(start_date.year, start_date.month, start_date.day, 23, 59, 59, tzinfo=JST)
            usage_intervals_by_date.setdefault(str(start_date), []).append((start_dt, end_of_start_day))
            # Beginning of next day at 00:00:00 to end_dt
            usage_intervals_by_date.setdefault(str(end_date), []).append((datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=JST), end_dt))
            # Note: Assuming sessions don't span more than 2 days continuously.

    # Now compute longest gap (in hours) for each day
    for date_str, intervals in usage_intervals_by_date.items():
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        day_start = datetime.fromisoformat(date_str).replace(tzinfo=JST)
        day_end = (day_start + timedelta(days=1))
        longest_gap = 0.0
        prev_end = day_start  # start of day
        for (s_dt, e_dt) in intervals:
            # gap from prev_end to current start
            gap = (s_dt - prev_end).total_seconds() / 3600.0
            if gap > longest_gap:
                longest_gap = gap
            # move prev_end forward if this session ends later
            if e_dt > prev_end:
                prev_end = e_dt
        # gap from last usage end to end of day
        final_gap = (day_end - prev_end).total_seconds() / 3600.0
        if final_gap > longest_gap:
            longest_gap = final_gap

        sleep_hours_by_date[date_str] = longest_gap
        # Determine base target from sleep_hours
        if longest_gap >= 7.0:
            base_target_by_date[date_str] = 80
        elif longest_gap < 6.0:
            base_target_by_date[date_str] = 60
        else:
            base_target_by_date[date_str] = 70

# 2. Parse Twitter tweets.js data to get tweets per day and sentiment
tweets_by_date = {}   # dict of date_str -> list of (tweet_text, sentiment_label)
# Find all tweets.js files in TWITTER_DIR (in case of multiple parts)
for filepath in glob.glob(os.path.join(TWITTER_DIR, "**", "tweets*.js"), recursive=True):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read Twitter data file {filepath}: {e}")
        continue
    # Remove the JavaScript assignment part if present (e.g., "window.YTD.tweets.part0 = ")
    json_start = raw.find('[')
    json_end = raw.rfind(']')
    if json_start == -1 or json_end == -1:
        print(f"[WARNING] No JSON array found in {filepath}")
        continue
    tweets_json_str = raw[json_start:json_end+1]
    try:
        tweets = json.loads(tweets_json_str)
    except Exception as e:
        print(f"[ERROR] JSON parse failed for {filepath}: {e}")
        continue

    # Iterate through tweets
    for entry in tweets:
        tweet = entry.get("tweet", {})
        text = tweet.get("full_text", "")
        created_at = tweet.get("created_at")
        if not created_at or text is None:
            continue
        # Parse created_at (e.g. "Tue Mar 18 08:07:10 +0000 2025") and convert to JST date
        try:
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            # If format unexpected, skip
            continue
        dt_jst = dt.astimezone(JST)
        date_str = dt_jst.strftime("%Y-%m-%d")
        # Classify sentiment of the tweet content
        sentiment_label = classify_sentiment(text)
        # Store result
        tweets_by_date.setdefault(date_str, []).append((text, sentiment_label))

# 3. Determine sentiment adjustment per day and prepare CSV rows
rows = []
for date_str, base_target in base_target_by_date.items():
    # Determine overall day sentiment adjustment
    adjust = 0
    tweet_list = tweets_by_date.get(date_str, [])
    if tweet_list:
        # Count if any positive and any negative tweets on that day
        any_positive = any(lbl == "positive" for (_, lbl) in tweet_list)
        any_negative = any(lbl == "negative" for (_, lbl) in tweet_list)
        if any_positive and not any_negative:
            adjust = 5
        elif any_negative and not any_positive:
            adjust = -5
        else:
            adjust = 0
    else:
        # No tweets that day -> no sentiment adjustment
        adjust = 0
    final_target = base_target + adjust

    # Add a row for the StayFree (sleep) data of that day
    sleep_hours = sleep_hours_by_date.get(date_str, None)
    if sleep_hours is not None:
        # Format sleep hours to one decimal place
        content = f"睡眠時間（推定）: {sleep_hours:.1f}時間"
        rows.append((date_str, content, "stayfree", "neutral", final_target))
    # Add rows for each tweet on that day
    for (text, sent) in tweets_by_date.get(date_str, []):
        rows.append((date_str, text, "twitter", sent, final_target))

# 4. Write to CSV file
output_path = "/myhealth/data/health_features.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
try:
    with open(output_path, "w", encoding="utf-8") as csvfile:
        # Write header
        csvfile.write("date,content,source,sentiment,target\n")
        for (date_str, content, source, sentiment, target) in rows:
            # Escape quotes in content
            content_escaped = content.replace('"', '""')
            # Enclose content in quotes if it contains comma
            if ',' in content_escaped or '\n' in content_escaped:
                content_escaped = f'"{content_escaped}"'
            csvfile.write(f"{date_str},{content_escaped},{source},{sentiment},{target}\n")
    print(f"✅ Successfully generated CSV at {output_path}")
except Exception as e:
    print(f"[ERROR] Failed to write CSV: {e}")
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
# ── sys.path を最初に通す ───────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
# ここから下で data_sources.* を import しても OK
from data_sources.twitter_loader import load_twitter_data
from data_sources.pleasanter import fetch_daily_entries
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


# ---------------------------------------------------------------------
#  ストップワード定義ブロック  ★ここを丸ごと追加 / 置換してください
# ---------------------------------------------------------------------
from wordcloud import STOPWORDS   # 英語既定 STOPWORDS

# ① URL・RT など既定で除外したい“英数字系”トークン
DEFAULT_STOP = {
    'https', 'http', 't', 'co', 'amp', 'rt'
}

# ② 追加ストップワードをファイルから読み込む（1 行 1 語）
STOPWORD_FILE = os.getenv("STOPWORD_PATH", "/teacher_data/stopwords.txt")
EXTRA_STOP: set[str] = set()
if os.path.isfile(STOPWORD_FILE):
    with open(STOPWORD_FILE, encoding="utf-8") as f:
        EXTRA_STOP = {
            ln.strip() for ln in f
            if ln.strip() and not ln.startswith('#')
        }

# ③ すべて合体させた最終セット（英語既定 STOPWORDS も加える）
STOP_SET = DEFAULT_STOP | EXTRA_STOP | set(STOPWORDS)

# ④ STOP_SET の語が連続しただけのトークン（例: ｗｗｗ, あああ）も除外
RE_STOP = re.compile(r'^(?:' + '|'.join(map(re.escape, STOP_SET)) + r')+$')

def _token_filter(tok: str) -> bool:
    """WordCloud で除外するトークン判定"""
    return ASCII_ID_RE.match(tok) or tok in STOP_SET or RE_STOP.match(tok)
# ---------------------------------------------------------------------

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
    # ───────────────── Pleasanter データ ─────────────────
    df_pl = fetch_daily_entries()
    if not df_pl.empty and 'content' in df_pl.columns:
        # JST → hour 抽出（twitter_loader と揃える）
        df_pl['hour'] = df_pl['created_at'].dt.hour
        pl_hour_stats = (df_pl.groupby('hour')
                           .size()
                           .reindex(range(24), fill_value=0)
                           .reset_index(name='record_count'))

        # ★ グラフ（時間帯別レコード数）
        fig_pl_hour = px.bar(
            pl_hour_stats, x='hour', y='record_count',
            title='Pleasanter 時間帯別レコード数',
            labels={'hour':'時間帯 (時)', 'record_count':'レコード数'},
            color='record_count', color_continuous_scale='Blues'
        )
    else:
        fig_pl_hour = None

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
            # ------ Pleasanter セクション -----------------
            html.H2("Pleasanter 時間帯別レコード数"),
            dcc.Graph(figure=fig_pl_hour) if fig_pl_hour else html.P("※レコードなし"),

            html.H2("Pleasanter レコード一覧"),
            dash_table.DataTable(
                columns=[{'name':'日時','id':'created_at'},
                         {'name':'タイトル','id':'title'},
                         {'name':'内容','id':'content'}],
                data=sanitize_records(
                    df_pl[['created_at','title','content']] if not df_pl.empty else pd.DataFrame()
                ),
                page_action='none',
                style_table={'height':'500px','overflowY':'auto','overflowX':'auto'},
                style_cell={'textAlign':'left','padding':'4px'},
                style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'},
                virtualization=True
            ),
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

# 2-15. entrypoint.sh  (modify this section to run feature generation)
cat > entrypoint.sh <<'EOF'
#!/bin/sh
set -e
export MPLCONFIGDIR=/tmp/.matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "▶ Generating training features..."
python /myhealth/strategies/auto_generate_features.py || echo "⚠️ Feature generation failed"

echo "▶ Running train.py..."
python /myhealth/strategies/train.py                    # Proceed with model training

echo "▶ Starting Dash..."
exec python /myhealth/dashboard/app.py                  # Launch the Dash web app
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
