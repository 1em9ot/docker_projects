#!/bin/bash
# create_bitflyerlightningapp.sh
# BitFlyer API を用いた Dash デイトレード Web アプリケーションの Docker プロジェクト
# "bitflyerlightningapp" という名称で生成・起動するスクリプト
#
# このスクリプトは、既存のプロジェクトがあれば削除（persistent フォルダはバックアップ）し、
# 必要なファイル（docker-compose.yml、app/Dockerfile、app/requirements.txt、app/app.py）を生成後、
# 同層にある .env を自動配置し、Docker コンテナ・イメージ・ボリュームを完全にリセットした上で
# キャッシュ無視でビルド、サービスを起動します。
#
# ※ WSL 環境では、プロジェクトフォルダを Linux 側に置くとパフォーマンスが向上します。
# ※ ポート8050が既に使用中の場合はエラーを出して中止します。
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

# プロジェクトディレクトリ名（Docker内のサービス名は小文字に統一）
PROJECT_DIR="bitflyerlightningapp"

# 既存のプロジェクトフォルダが存在する場合、persistent フォルダを退避して削除
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
# docker-compose.yml の生成
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
# app/Dockerfile の生成
########################################
echo "app/Dockerfile を作成します..."
cat > "$PROJECT_DIR/app/Dockerfile" << 'EOF'
FROM python:3.11-slim
WORKDIR /app
# システム時刻を JST (Asia/Tokyo) に設定（tzdata インストール）
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
dash_bootstrap_components==1.4.1
EOF

########################################
# app/app.py の生成（注文モーダル・シミュレーション拡張版）
########################################
echo "app/app.py を作成します..."
cat > "$PROJECT_DIR/app/app.py" << 'EOF'
import os, requests, logging, time, hmac, hashlib
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback_context, State
from dash.exceptions import PreventUpdate

# 環境変数から API キー・シークレットを取得
API_KEY = os.environ.get("BITFLYER_API_KEY")
API_SECRET = os.environ.get("BITFLYER_API_SECRET")
API_URL = "https://api.bitflyer.com"
if not API_KEY or not API_SECRET:
    print("WARNING: APIキーが設定されていません。公開モードで起動します。")

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

logging.basicConfig(filename="/persistent/app.log", level=logging.INFO)
logging.info(f"Dash app started at {datetime.now()}")

def get_ticker(product_code="BTC_JPY"):
    try:
        resp = requests.get(f"{API_URL}/v1/ticker?product_code={product_code}")
        return resp.json() if resp.ok else None
    except Exception as e:
        logging.error(f"Failed to fetch ticker: {e}")
        return None

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

def get_news_events():
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    from_time = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"
    params = {
        "q": "Bitcoin",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "from": from_time,
        "apiKey": NEWSAPI_KEY
    }
    try:
        resp = requests.get(url, params=params)
        if resp.ok:
            data = resp.json()
            events = []
            for article in data.get("articles", []):
                published_at = article.get("publishedAt")
                title = article.get("title", "")
                if published_at:
                    event_time = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    events.append({"time": event_time, "title": title})
            return events
        else:
            logging.error("News API response not OK: " + str(resp.text))
            return []
    except Exception as e:
        logging.error(f"Failed to fetch news events: {e}")
        return []

price_times = []
price_values = []
news_events = []

def get_price_at(event_time):
    if not price_times or not price_values:
        return None
    for i in range(len(price_times) - 1, -1, -1):
        if price_times[i] <= event_time:
            return price_values[i]
    return price_values[0]

def build_balance_content():
    ticker_data = get_ticker("BTC_JPY")
    try:
        current_price = float(ticker_data.get("ltp") or ticker_data.get("best_bid"))
    except:
        current_price = None
    if not API_KEY or not API_SECRET:
        return "APIキーが設定されていません"
    bal_data = get_balance()
    if bal_data is None:
        return "残高取得に失敗しました"
    price_div = html.Div(
        f"現在のBTC/JPY価格: {current_price} 円" if current_price else "BTC/JPY価格取得不可",
        style={"fontWeight": "bold", "marginRight": "20px"}
    )
    balance_divs = []
    if bal_data:
        for entry in bal_data:
            code = entry.get("currency_code", "")
            amount = entry.get("amount", "")
            available = entry.get("available", "")
            if code.upper() == "BTC" and current_price:
                try:
                    av_val = float(available)
                    av_jpy = av_val * current_price
                    available_str = f"{av_jpy:,.0f} 円"
                except Exception as e:
                    logging.error(f"Conversion error: {e}")
                    available_str = available
            else:
                available_str = available
            balance_divs.append(html.Div(f"{code}: {amount} （利用可能: {available_str}）"))
    return [price_div] + balance_divs

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("Bitflyer Trading Dashboard"),
    html.Div(id="price-text", style={"fontWeight": "bold", "fontSize": "1.2em"}),
    dcc.Graph(id="price-graph"),
    dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
    html.Hr(),
    html.H2("アカウント残高"),
    dbc.Button("残高更新", id="refresh-button", n_clicks=0),
    html.Div(id="balance-info", style={"display": "flex", "alignItems": "center", "gap": "20px"}),
    html.Hr(),
    html.H2("注文"),
    dbc.Button("注文", id="open-modal-btn", n_clicks=0, color="primary", className="mb-2"),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("注文")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="order-amount", type="number", min=0, step=1, placeholder="入金金額を入力"),
                dbc.InputGroupText("円")
            ], className="mb-3"),
            dbc.Button("買い", id="buy-button", color="success", className="me-2"),
            dbc.Button("売り", id="sell-button", color="danger", className="me-2")
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-modal-btn", color="secondary"))
    ], id="order-modal", is_open=False),
    dcc.Interval(id="reset-interval", interval=10000, n_intervals=0, disabled=True),
    dcc.Store(id="stored-balance")
], fluid=True)

@app.callback(
    Output("price-text", "children"),
    Output("price-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_price(n):
    global news_events
    data = get_ticker("BTC_JPY")
    price = None
    if data:
        price = data.get("ltp") or data.get("best_bid")
    now = datetime.now()
    if price is not None:
        price_times.append(now)
        price_values.append(price)
        if len(price_times) > 60:
            price_times.pop(0)
            price_values.pop(0)
        logging.info(f"Fetched price {price} at {now.strftime('%H:%M:%S')}")
    if n % 12 == 0:
        news_events = get_news_events()
    x_values = [dt.strftime("%H:%M:%S") for dt in price_times]
    traces = [{
        "x": x_values,
        "y": price_values,
        "type": "line",
        "name": "BTC/JPY"
    }]
    if price_times:
        start_time = price_times[0]
        end_time = price_times[-1]
        event_x, event_y, event_texts = [], [], []
        for ev in news_events:
            if start_time <= ev["time"] <= end_time:
                event_x.append(ev["time"].strftime("%H:%M:%S"))
                event_y.append(price if price is not None else 0)
                event_texts.append(ev["title"])
        if event_x:
            traces.append({
                "x": event_x,
                "y": event_y,
                "mode": "markers",
                "marker": {"size": 10, "symbol": "diamond"},
                "name": "News Events",
                "text": event_texts,
                "hoverinfo": "text"
            })
    figure = {"data": traces, "layout": {"title": "BTC/JPY Price with News Events"}}
    text = f"現在のBTC/JPY価格: {price} 円" if price is not None else "現在の価格を取得できません"
    return text, figure

@app.callback(
    Output("balance-info", "children"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True
)
def refresh_balance(n):
    return build_balance_content()

@app.callback(
    Output("order-modal", "is_open"),
    [Input("open-modal-btn", "n_clicks"), Input("close-modal-btn", "n_clicks")],
    [State("order-modal", "is_open")]
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("balance-info", "children", allow_duplicate=True),
    Output("stored-balance", "data", allow_duplicate=True),
    Output("reset-interval", "disabled", allow_duplicate=True),
    Input("buy-button", "n_clicks"),
    Input("sell-button", "n_clicks"),
    State("order-amount", "value"),
    State("balance-info", "children"),
    prevent_initial_call=True
)
def simulate_order(buy_clicks, sell_clicks, order_amount, current_balance):
    ctx = callback_context
    if not ctx.triggered or order_amount is None or order_amount <= 0:
        raise PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    original_balance = current_balance
    ticker_data = get_ticker("BTC_JPY")
    try:
        current_price = float(ticker_data.get("ltp") or ticker_data.get("best_bid"))
    except:
        current_price = None
    if not current_price or current_price == 0:
        raise PreventUpdate
    fee_rate = 0.002  # 手数料【0.2%】
    fee = order_amount * fee_rate
    effective_amount = order_amount - fee
    btc_quantity = effective_amount / current_price
    now_str = datetime.now().strftime("%H:%M:%S")
    if button_id == "buy-button":
        sim_text = (f"{now_str} 買い注文シミュレーション: 入金 {order_amount:,}円 "
                    f"[手数料: {fee:,.0f}円], BTC取得量: {btc_quantity:.8f} BTC")
    else:
        sim_text = (f"{now_str} 売り注文シミュレーション: 入金 {order_amount:,}円 "
                    f"[手数料: {fee:,.0f}円], BTC売却量: {btc_quantity:.8f} BTC")
    sim_div = html.Div(sim_text, style={"color": "green", "fontWeight": "bold"})
    updated_balance = build_balance_content()
    if isinstance(updated_balance, list):
        updated_balance.append(sim_div)
    else:
        updated_balance = [updated_balance, sim_div]
    return updated_balance, original_balance, False

@app.callback(
    Output("balance-info", "children", allow_duplicate=True),
    Output("reset-interval", "disabled", allow_duplicate=True),
    Input("reset-interval", "n_intervals"),
    State("stored-balance", "data"),
    prevent_initial_call=True
)
def revert_balance(n_intervals, original_balance):
    if not original_balance:
        raise PreventUpdate
    return original_balance, True

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

# 完全にクリーンな状態にするため、既存のコンテナ、イメージ、ボリュームを全削除
echo "古いコンテナ、イメージ、ボリュームを全削除します..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" down --rmi all --volumes --remove-orphans

# キャッシュを無視してビルドし、コンテナ起動
echo "Docker イメージをキャッシュ無視でビルドし、サービスを起動します..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" build --no-cache
docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d

echo "プロジェクトのセットアップと起動が完了しました。"
echo "Webブラウザで http://localhost:8050 にアクセスしてください。"
echo "ホスト側の '$PROJECT_DIR/app/' 内のソースコードを編集すると、コンテナ内に即時反映されます。"
