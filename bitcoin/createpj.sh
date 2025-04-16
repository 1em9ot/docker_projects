#!/bin/bash
# create_bitintel.sh
# グローバルなイベント（Twitter投稿、ニュース記事、経済イベントなど）とそのセンチメントを収集し、
# Bitcoin価格に紐づけて Dash で可視化する Docker プロジェクト「bitintel」を生成・起動するスクリプト。
#
# ※ ビットフライヤーライトニング取引用：BITFLYER_API_KEY と BITFLYER_API_SECRET 両方が必須です。
# ※ 情報収集用については、TWITTER_BEARER_TOKEN または NEWSAPI_KEY のいずれか１つ以上があれば採用します。
#
# このスクリプトは既存のプロジェクトがあれば削除（persistent フォルダはバックアップ保存）し、
# 必要なファイル（docker-compose.yml、app/Dockerfile、app/requirements.txt、app/app.py）を生成後、
# 同層にある .env を自動配置し、Docker イメージをビルドしてサービスを起動します。
#
# ※ WSL 環境ではプロジェクトフォルダを Linux 側に置くとパフォーマンスが向上します。
#    Windows 環境の場合は Docker Desktop のファイル共有設定をご確認ください。
# ※ ポート 8050 が既に使用中の場合はエラー表示して終了します。
# ※ 同層に .env ファイルが存在しない場合はエラーメッセージを出して中止します。

set -e

# --- ポート使用状況チェック ---
if lsof -i :8050 >/dev/null 2>&1; then
    echo "Error: Port 8050 is already in use. Please free the port or change the port mapping in docker-compose.yml."
    exit 1
fi

# --- .env ファイルの存在確認 ---
if [ ! -f .env ]; then
    echo "Error: .env file not found in current directory. Please create one with your API credentials. Aborting."
    exit 1
fi

# プロジェクトディレクトリ名（小文字で統一）
PROJECT_DIR="trade"

# 既存のプロジェクトフォルダ処理（存在する場合はpersistentを退避して削除）
if [ -d "$PROJECT_DIR" ]; then
    echo "既存の '$PROJECT_DIR' フォルダが見つかりました。永続データフォルダをバックアップします..."
    if [ -d "$PROJECT_DIR/persistent" ]; then
        echo "persistent フォルダを一時退避します..."
        mv "$PROJECT_DIR/persistent" "/tmp/persistent_backup_$$"
    fi
    rm -rf "$PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    if [ -d "/tmp/persistent_backup_$$" ]; then
        mv "/tmp/persistent_backup_$$" "$PROJECT_DIR/persistent"
    fi
else
    mkdir -p "$PROJECT_DIR"
fi

# 必要なサブフォルダの作成
mkdir -p "$PROJECT_DIR/app"
mkdir -p "$PROJECT_DIR/persistent"

########################################
# docker-compose.yml の生成（サービス・コンテナ・イメージ名：bitintel）
########################################
echo "docker-compose.yml を作成します..."
cat > "$PROJECT_DIR/docker-compose.yml" << 'EOF'
services:
  bitintel:
    build: ./app
    container_name: bitintel
    image: bitintel
    env_file: .env
    volumes:
      - './app:/app'
      - './persistent:/persistent'
    ports:
      - '8050:8050'
EOF

########################################
# app/Dockerfile の生成（タイムゾーン JST 設定付き Python 3.11-slim）
########################################
echo "app/Dockerfile を作成します..."
cat > "$PROJECT_DIR/app/Dockerfile" << 'EOF'
FROM python:3.11-slim
WORKDIR /app
# タイムゾーンを JST (Asia/Tokyo) に設定
RUN apt-get update && apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo 'Asia/Tokyo' > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Tokyo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8050
CMD ["python", "app.py"]
EOF

########################################
# app/requirements.txt の生成（必要なPythonライブラリ群）
########################################
echo "app/requirements.txt を作成します..."
cat > "$PROJECT_DIR/app/requirements.txt" << 'EOF'
dash==2.11.1
requests==2.31.0
tweepy==4.14.0
newsapi-python==0.2.7
transformers==4.30.2
torch==2.0.0
sentencepiece==0.1.99
googletrans==4.0.2
EOF

########################################
# app/app.py の生成（Dash アプリ：BTC価格と世界イベントの収集・可視化）
########################################
echo "app/app.py を作成します..."
cat > "$PROJECT_DIR/app/app.py" << 'EOF'
import os, logging, time, hmac, hashlib
from datetime import datetime, timedelta
import requests
import tweepy
from newsapi import NewsApiClient
from transformers import pipeline
from googletrans import Translator
from dash import Dash, dcc, html, Input, Output, callback_context

# --- API キーのロード ---
# BitFlyer Lightning API には KEY と SECRET の両方が必須
BITFLYER_API_KEY = os.environ.get("BITFLYER_API_KEY")
BITFLYER_API_SECRET = os.environ.get("BITFLYER_API_SECRET")
if not BITFLYER_API_KEY or not BITFLYER_API_SECRET:
    print("ERROR: BitFlyer Lightning API のキーとシークレットは両方設定してください。")
    exit(1)

# 情報収集用 API キーについては、Twitter または NewsAPI のいずれか１つがあれば利用開始
TWITTER_BEARER = os.environ.get("TWITTER_BEARER_TOKEN")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not TWITTER_BEARER and not NEWSAPI_KEY:
    print("ERROR: 情報収集用の API キー (TWITTER_BEARER_TOKEN または NEWSAPI_KEY) のいずれかを設定してください。")
    exit(1)

# ログ設定（/persistent/app.log に出力）
logging.basicConfig(filename="/persistent/app.log", level=logging.INFO)
logging.info(f"BitIntel Dash app started at {datetime.now()}")

# Bitflyer 公開APIで BTC_JPY の最新価格を取得
BITFLYER_API_URL = "https://api.bitflyer.com"
def get_ticker(product_code="BTC_JPY"):
    try:
        resp = requests.get(f"{BITFLYER_API_URL}/v1/ticker?product_code={product_code}")
        if resp.ok:
            data = resp.json()
            return data.get("ltp") or data.get("best_bid")
    except Exception as e:
        logging.error(f"Failed to fetch ticker: {e}")
    return None

# CoinGecko から過去のBTC価格（参考：過去24hのヒストリカルデータ）を取得
def get_historical_prices(days=1):
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        )
        data = resp.json() if resp.status_code == 200 else None
        prices, times = [], []
        if data and 'prices' in data:
            for ts, price in data['prices']:
                dt = datetime.fromtimestamp(ts/1000)
                times.append(dt)
                prices.append(price)
        logging.info(f"Loaded {len(prices)} historical price points.")
        return times, prices
    except Exception as e:
        logging.error(f"Failed to fetch historical prices: {e}")
        return [], []

# --- クライアント初期化 --- 
# Twitter: キーがあれば初期化
if TWITTER_BEARER:
    twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER)
else:
    twitter_client = None
# NewsAPI: キーがあれば初期化
if NEWSAPI_KEY:
    news_client = NewsApiClient(api_key=NEWSAPI_KEY)
else:
    news_client = None

# 翻訳エンジンとセンチメント分析パイプラインの初期化
translator = Translator()
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
except Exception as e:
    logging.error(f"Sentiment model load failed: {e}")
    sentiment_pipeline = None

# グローバル変数：価格とイベントの記録
price_times = []
price_values = []
events = []  # 各イベントは{'time': datetime, 'text': str, 'sentiment': str, 'source': str}
seen_tweet_ids = set()
seen_news_urls = set()

# Dash アプリの構築
app = Dash(__name__)
app.layout = html.Div([
    html.H1("BitIntel - Bitcoin Price & Global Events Dashboard"),
    dcc.Graph(id="price-graph"),
    dcc.Interval(id="price-interval", interval=5000, n_intervals=0),
    dcc.Interval(id="event-interval", interval=60000, n_intervals=0)
])

# グラフ更新コールバック
@app.callback(
    Output("price-graph", "figure"),
    Input("price-interval", "n_intervals"),
    Input("event-interval", "n_intervals")
)
def update_graph(price_tick, event_tick):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # 初回時に過去24hの価格データをロード
    if not price_times:
        hist_times, hist_prices = get_historical_prices(days=1)
        if hist_times and hist_prices:
            price_times.extend(hist_times)
            price_values.extend(hist_prices)

    # 価格更新
    if triggered_id == "price-interval" or triggered_id is None:
        current_price = get_ticker("BTC_JPY")
        if current_price is not None:
            now = datetime.now()
            price_times.append(now)
            price_values.append(current_price)
            cutoff = datetime.now() - timedelta(days=1)
            while price_times and price_times[0] < cutoff:
                price_times.pop(0)
                price_values.pop(0)
            logging.info(f"Price updated: {current_price} JPY at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # イベント更新（Twitter と NewsAPI：キーがあるものだけ実行）
    if triggered_id == "event-interval" or triggered_id is None:
        try:
            # Twitter: キーがあれば最新ツイート収集
            if twitter_client:
                query = "bitcoin -is:retweet"
                tweet_params = {
                    "query": query,
                    "max_results": 20,
                    "tweet_fields": ["created_at", "lang"]
                }
                if event_tick == 0:
                    start_time = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    tweet_params["start_time"] = start_time
                    tweet_params["max_results"] = 50
                tweet_resp = twitter_client.search_recent_tweets(**tweet_params)
                if tweet_resp.data:
                    for tweet in tweet_resp.data:
                        tid = tweet.id
                        if tid not in seen_tweet_ids:
                            seen_tweet_ids.add(tid)
                            text = tweet.text
                            ttime = tweet.created_at.replace(tzinfo=None) if tweet.created_at else datetime.now()
                            lang = tweet.lang or ""
                            display_text = text
                            if lang.lower() != "ja":
                                try:
                                    trans = translator.translate(text, dest='ja')
                                    display_text = trans.text
                                except Exception as e:
                                    logging.error(f"Tweet translation failed for {tid}: {e}")
                            sentiment = "不明"
                            if sentiment_pipeline:
                                try:
                                    res = sentiment_pipeline(text)[0]
                                    label = res["label"].lower()
                                    if "positive" in label:
                                        sentiment = "ポジティブ"
                                    elif "negative" in label:
                                        sentiment = "ネガティブ"
                                    else:
                                        sentiment = "ニュートラル"
                                except Exception as e:
                                    logging.error(f"Tweet sentiment analysis failed for {tid}: {e}")
                            events.append({
                                "time": ttime,
                                "text": display_text,
                                "sentiment": sentiment,
                                "source": "ツイート"
                            })
            # NewsAPI: キーがあれば Bitcoin 関連ニュース取得
            if news_client:
                news_params = {
                    "q": "Bitcoin",
                    "language": "en",
                    "sort_by": "publishedAt",
                    "page_size": 5
                }
                if event_tick == 0:
                    from_time = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    news_params["from_param"] = from_time
                    news_params["page_size"] = 20
                news_resp = news_client.get_everything(**news_params)
                if news_resp.get("articles"):
                    for article in news_resp["articles"]:
                        url = article.get("url", "")
                        title = article.get("title", "")
                        if not url:
                            url = title
                        if url and url not in seen_news_urls:
                            seen_news_urls.add(url)
                            text = title if title else article.get("description", "No Title")
                            time_str = article.get("publishedAt", "")
                            try:
                                atime = datetime.fromisoformat(time_str.replace("Z", "+00:00")).replace(tzinfo=None)
                            except Exception:
                                atime = datetime.now()
                            lang = article.get("language", "") or ""
                            display_text = text
                            if lang.lower() != "ja":
                                try:
                                    trans = translator.translate(text, dest='ja')
                                    display_text = trans.text
                                except Exception as e:
                                    logging.error(f"News translation failed: {e}")
                            sentiment = "不明"
                            if sentiment_pipeline:
                                try:
                                    res = sentiment_pipeline(text)[0]
                                    label = res["label"].lower()
                                    if "positive" in label:
                                        sentiment = "ポジティブ"
                                    elif "negative" in label:
                                        sentiment = "ネガティブ"
                                    else:
                                        sentiment = "ニュートラル"
                                except Exception as e:
                                    logging.error(f"News sentiment analysis failed: {e}")
                            events.append({
                                "time": atime,
                                "text": display_text,
                                "sentiment": sentiment,
                                "source": "ニュース"
                            })
        except Exception as e:
            logging.error(f"Event fetch error: {e}")

        cutoff = datetime.now() - timedelta(days=1)
        events[:] = [ev for ev in events if ev["time"] >= cutoff]
        events.sort(key=lambda ev: ev["time"])
        logging.info(f"Events updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (count: {len(events)})")

    # グラフ描画用データ生成
    price_x = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in price_times]
    price_y = price_values.copy()
    event_x = []
    event_y = []
    event_texts = []
    for ev in events:
        etime = ev["time"]
        price_at_event = None
        for i in range(len(price_times)-1, -1, -1):
            if price_times[i] <= etime:
                price_at_event = price_values[i]
                break
        if price_at_event is None:
            price_at_event = price_values[0] if price_values else 0
        event_x.append(etime.strftime("%Y-%m-%d %H:%M:%S"))
        event_y.append(price_at_event)
        hover_text = f"{ev['source']}: {ev['text']} (感情: {ev['sentiment']})"
        event_texts.append(hover_text)

    figure = {
        "data": [
            {"x": price_x, "y": price_y, "type": "line", "name": "BTC Price"},
            {"x": event_x, "y": event_y, "mode": "markers", "name": "Events",
             "marker": {"size": 8, "symbol": "diamond"}, "text": event_texts, "hoverinfo": "text"}
        ],
        "layout": {"title": "BTC価格とイベント (Bitcoin Price and Events)"}
    }
    return figure

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
EOF

########################################
# .env ファイルの配置（同層に存在する .env をプロジェクトフォルダへコピー）
########################################
if [ -f .env ]; then
    echo ".env ファイルが見つかったので、プロジェクトフォルダにコピーします..."
    cp .env "$PROJECT_DIR/.env"
else
    echo "Error: 同層に .env ファイルが見つかりません。プロジェクトの実行を中止します。"
    exit 1
fi

# APIキーのプレースホルダチェック
if grep -q "YOUR_TWITTER_BEARER_TOKEN_HERE" "$PROJECT_DIR/.env" || \
   grep -q "YOUR_NEWSAPI_KEY_HERE" "$PROJECT_DIR/.env" || \
   grep -q "YOUR_BITFLYER_API_KEY_HERE" "$PROJECT_DIR/.env" || \
   grep -q "YOUR_BITFLYER_API_SECRET_HERE" "$PROJECT_DIR/.env"; then
    echo "ERROR: 必須の API キーが設定されていません。.env ファイルを編集して必要な API キー (TWITTER_BEARER_TOKEN, NEWSAPI_KEY, BITFLYER_API_KEY, BITFLYER_API_SECRET) を入力してください。"
    echo "プロジェクトは作成されましたが、キー設定完了までサービスは起動しません。"
    echo "キー設定後、以下のコマンドでサービスを起動できます:"
    echo "  docker compose -f $PROJECT_DIR/docker-compose.yml up -d"
    exit 1
fi

# 古いコンテナ（オーファン）の削除
echo "古いコンテナ（オーファン）を削除します..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" down --remove-orphans

# Docker Compose によるサービスのビルド・起動
docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d

echo "プロジェクト bitintel のセットアップと起動が完了しました。"
echo "Webブラウザで http://localhost:8050 にアクセスしてください。"
echo "ソースコードは '$PROJECT_DIR/app/' 内にあり、編集するとコンテナ内に即時反映されます。"
