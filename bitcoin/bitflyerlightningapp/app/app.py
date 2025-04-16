import os, requests, logging, time, hmac, hashlib
from datetime import datetime
from dash import Dash, dcc, html, Input, Output

# 環境変数から API キー・シークレットを取得
API_KEY = os.environ.get("BITFLYER_API_KEY")
API_SECRET = os.environ.get("BITFLYER_API_SECRET")
API_URL = "https://api.bitflyer.com"
if not API_KEY or not API_SECRET:
    print("WARNING: APIキーが設定されていません。公開モードで起動します。")

# ログ設定（/persistent/app.log に出力）
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
        now_str = datetime.now().strftime("%H:%M:%S")
        price_times.append(now_str)
        price_values.append(price)
        if len(price_times) > 20:
            price_times.pop(0)
            price_values.pop(0)
        logging.info(f"Fetched price {price} at {now_str}")
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
