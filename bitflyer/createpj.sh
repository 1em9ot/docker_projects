#!/bin/bash
# create_bitflyerlightningapp.sh
# BitFlyer API を用いた Dash デイトレード Web アプリケーションの Docker プロジェクトを
# "bitflyerlightningapp" という名称で生成・起動するスクリプト
#
# このスクリプトは、既存のプロジェクトがあれば削除（persistent フォルダはバックアップ）し、
# 必要なファイル（docker-compose.yml、app/Dockerfile、app/requirements.txt、app/app.py）を生成後、
# 同層にある .env を自動配置し、Docker イメージをビルドしてサービスを起動します。
#
# ※ WSL 環境では、プロジェクトフォルダを Linux 側に置くとパフォーマンスが向上します。
#   Windows 側のフォルダを利用する場合は Docker Desktop のファイル共有設定をご確認ください。
# ※ ポート8050が既に使用中の場合はエラーを出してプロジェクト実行を中止します。
# ※ 同層に .env ファイルが存在しない場合はエラーメッセージを出して中止します。

set -e

# --- ポートチェック ---
if lsof -i :8050 >/dev/null; then
    echo "Error: Port 8050 is already in use. Please free the port or change the port mapping in docker-compose.yml."
    exit 1
fi

# --- .env ファイルの存在確認 ---
if [ ! -f .env ]; then
    echo "Error: .env file not found in current directory. Please create one with your API credentials. Aborting."
    exit 1
fi

# プロジェクトディレクトリ名（フォルダ名は任意、Docker内のサービス名は小文字で統一）
PROJECT_DIR="bitflyerlightningapp"

# 既存のプロジェクトフォルダが存在する場合、persistent フォルダがあれば退避して削除
if [ -d "$PROJECT_DIR" ]; then
    echo "既存の '$PROJECT_DIR' フォルダが見つかりました。persistent フォルダは保持します..."
    if [ -d "$PROJECT_DIR/persistent" ]; then
        echo "persistent フォルダを一時退避します..."
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

# 必要なサブフォルダの作成（persistent フォルダは復元済みの場合あり）
mkdir -p "$PROJECT_DIR/app"
mkdir -p "$PROJECT_DIR/persistent"

########################################
# docker-compose.yml の生成（サービス・コンテナ・イメージ名：bitflyerlightningapp）
########################################
echo "docker-compose.yml を作成します..."
cat > "$PROJECT_DIR/docker-compose.yml" << 'EOF'
services:
  bitflyerlightningapp:
    build: ./app
    container_name: bitflyerlightningapp
    image: bitflyerlightningapp
    env_file: .env
    volumes:
      - './app:/app'
      - './persistent:/persistent'
    ports:
      - '8050:8050'
EOF

########################################
# app/Dockerfile の生成（システム時刻を JST に設定）
########################################
echo "app/Dockerfile を作成します..."
cat > "$PROJECT_DIR/app/Dockerfile" << 'EOF'
FROM python:3.11-slim
WORKDIR /app
# システム時刻を JST に設定（tzdata インストール）
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
# app/requirements.txt の生成
########################################
echo "app/requirements.txt を作成します..."
cat > "$PROJECT_DIR/app/requirements.txt" << 'EOF'
dash==2.11.1
requests==2.31.0
EOF

########################################
# app/app.py の生成
########################################
echo "app/app.py を作成します..."
cat > "$PROJECT_DIR/app/app.py" << 'EOF'
import os, requests, logging, time, hmac, hashlib
from datetime import datetime
from dash import Dash, dcc, html, Input, Output

# 環境変数から API キー・シークレットを取得
API_KEY = os.environ.get("BITFLYER_API_KEY")
API_SECRET = os.environ.get("BITFLYER_API_SECRET")
API_URL = "https://api.bitflyer.com"
if not API_KEY or not API_SECRET:
    print("WARNING: APIキーが設定されていません。公開モードで起動します。")

# ログ設定（persistent フォルダに出力）
logging.basicConfig(filename="/persistent/app.log", level=logging.INFO)
logging.info(f"Dash app started at {datetime.now()}")

# Bitflyer 公開API：最新ティッカー情報を取得する関数
def get_ticker(product_code="BTC_JPY"):
    try:
        resp = requests.get(f"{API_URL}/v1/ticker?product_code={product_code}")
        return resp.json() if resp.ok else None
    except Exception as e:
        logging.error(f"Failed to fetch ticker: {e}")
        return None

# Bitflyer 認証API：アカウント残高を取得する関数
def get_balance():
    if not API_KEY or not API_SECRET:
        return None
    timestamp = str(time.time())
    text = timestamp + "GET" + "/v1/me/getbalance"
    sign = hmac.new(API_SECRET.encode('utf-8'), text.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": sign,
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(f"{API_URL}/v1/me/getbalance", headers=headers)
        if not resp.ok:
            logging.error(f"Balance request failed: HTTP {resp.status_code} {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        logging.error(f"Failed to fetch balance: {e}")
        return None

# Dash アプリのセットアップ
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Bitflyer Trading Dashboard"),
    html.Div(id="price-text", style={"fontWeight": "bold", "fontSize": "1.2em"}),
    dcc.Graph(id="price-graph"),
    dcc.Interval(id="interval-component", interval=5*1000, n_intervals=0),
    html.Hr(),
    html.H2("アカウント残高"),
    html.Button("残高更新", id="refresh-button", n_clicks=0),
    html.Div(id="balance-info")
])

# 価格情報の定期更新コールバック（5秒間隔）
price_times = []
price_values = []

@app.callback(
    Output("price-text", "children"),
    Output("price-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_price(n):
    data = get_ticker("BTC_JPY")
    price = None
    if data:
        price = data.get("ltp") or data.get("best_bid")
    if price is not None:
        price_times.append(datetime.now().strftime("%H:%M:%S"))
        price_values.append(price)
        if len(price_times) > 20:
            price_times.pop(0)
            price_values.pop(0)
        logging.info(f"Fetched price {price} at {price_times[-1]}")
    figure = {
        "data": [{"x": price_times, "y": price_values, "type": "line", "name": "BTC/JPY"}],
        "layout": {"title": "BTC/JPY Price"}
    }
    text = f"現在のBTC/JPY価格: {price} 円" if price is not None else "現在の価格を取得できません"
    return text, figure

# 残高情報の更新コールバック（ボタン押下時）
@app.callback(
    Output("balance-info", "children"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True
)
def update_balance(n_clicks):
    if not API_KEY or not API_SECRET:
        return "APIキーが設定されていません"
    bal_data = get_balance()
    if bal_data is None:
        return "残高取得に失敗しました"
    return [html.Div(f"{entry['currency_code']}: {entry['amount']} （利用可能: {entry['available']}）")
            for entry in bal_data]

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
EOF

########################################
# .env ファイルの配置（同層に存在する .env をプロジェクトフォルダにコピー）
########################################
if [ -f .env ]; then
    echo ".env ファイルが見つかったので、プロジェクトフォルダにコピーします..."
    cp .env "$PROJECT_DIR/.env"
else
    echo "Error: 同層に .env ファイルが見つかりません。プロジェクトの実行を中止します。"
    exit 1
fi

# 古いコンテナ（オーファン）があれば削除
echo "古いコンテナ（オーファン）を削除します..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" down --remove-orphans

# Docker Compose によるサービス起動
docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d

echo "プロジェクトのセットアップと起動が完了しました。"
echo "Webブラウザで http://localhost:8050 にアクセスしてください。"
echo "ホスト側の '$PROJECT_DIR/app/' 内のソースコードを編集すると、コンテナ内に即時反映されます。"
