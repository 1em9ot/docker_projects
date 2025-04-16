import os
import time
import hmac
import hashlib
import logging
import requests
import pandas as pd
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc

# ——————————————————————————————————
# 環境変数
# ——————————————————————————————————
API_URL     = "https://api.bitflyer.com"
API_KEY     = os.environ.get("BITFLYER_API_KEY")
API_SECRET  = os.environ.get("BITFLYER_API_SECRET")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

logging.basicConfig(filename="/persistent/app.log", level=logging.INFO)
logging.info(f"Dash app started at {datetime.now()}")

# 履歴用グローバル
price_times = []
price_vals  = []

# ——————————————————————————————————
# データ取得関数
# ——————————————————————————————————
def get_ticker(product_code="BTC_JPY"):
    try:
        r = requests.get(f"{API_URL}/v1/ticker?product_code={product_code}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"get_ticker error: {e}")
        return {}

def get_news_events():
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    frm = (datetime.utcnow() - pd.Timedelta(minutes=10)).isoformat() + "Z"
    params = {
        "q":        "Bitcoin",
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 10,
        "from":     frm,
        "apiKey":   NEWSAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        logging.error(f"get_news_events error: {e}")
        return []

def get_positions():
    """Bitflyer Lightning API: 現在のポジション一覧を取得"""
    if not API_KEY or not API_SECRET:
        return []
    ts = str(time.time())
    method = "GET"
    path   = "/v1/me/getpositions"
    text = ts + method + path
    sign = hmac.new(API_SECRET.encode(), text.encode(), hashlib.sha256).hexdigest()
    headers = {
        "ACCESS-KEY":       API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN":      sign,
        "Content-Type":     "application/json"
    }
    try:
        r = requests.get(API_URL + path, headers=headers, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"get_positions error: {e}")
        return []

# ——————————————————————————————————
# Dash アプリ定義
# ——————————————————————————————————
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(fluid=True, children=[

    html.H1("BitFlyer + News + Position Dashboard"),

    # 自動更新
    dcc.Interval(id="interval", interval=3_000, n_intervals=0),

    # ティッカー
    html.H2("BTC/JPY リアルタイム価格"),
    html.Div(id="ticker-text", style={"fontWeight":"bold"}),
    dcc.Graph(id="price-chart"),

    html.Hr(),

    # ニュース
    html.H2("最新ニュース"),
    dash_table.DataTable(
        id="news-table",
        columns=[
            {"name":"Published At","id":"publishedAt"},
            {"name":"Title","id":"title"}
        ],
        style_cell={"textAlign":"left","whiteSpace":"normal","height":"auto"},
        style_table={"overflowY":"auto","maxHeight":"300px"}
    ),

    html.Hr(),

    # シミュレーションモード
    dbc.Button("シミュレーションモード", id="open-sim-btn", color="warning", className="mb-3"),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("シミュレーションモード")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="sim-invest-jpy", type="number", min=0, placeholder="投資金額 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.Input(id="sim-target-jpy", type="number", min=0, placeholder="目標利益 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            html.Div(id="sim-exit-price",
                     style={"fontWeight":"bold","fontSize":"1.2em","marginTop":"1rem"})
        ]),
        dbc.ModalFooter(dbc.Button("閉じる", id="close-sim-btn", className="ms-auto"))
    ], id="sim-modal", is_open=False),

    html.Hr(),

    # My Positions テーブル
    html.H2("My Positions"),
    dash_table.DataTable(
        id="position-table",
        columns=[
            {"name":"Product","id":"product_code"},
            {"name":"Side","id":"side"},
            {"name":"Size","id":"size"},
            {"name":"Entry Price","id":"price"},
            {"name":"Unrealized P/L","id":"pnl"}
        ],
        style_cell={"textAlign":"center"},
        style_header={"fontWeight":"bold"},
        style_table={"overflowX":"auto"}
    ),

])

# ——————————————————————————————————
# コールバック: 価格・ニュース・ポジション更新
# ——————————————————————————————————
@app.callback(
    Output("ticker-text","children"),
    Output("price-chart","figure"),
    Output("news-table","data"),
    Output("position-table","data"),
    Input("interval","n_intervals")
)
def update_all(n):
    # ティッカー取得＆履歴更新
    tk    = get_ticker()
    price = tk.get("ltp") or tk.get("best_bid") or None
    now   = datetime.now()
    if price is not None:
        price_times.append(now)
        price_vals.append(price)
        if len(price_times)>100:
            price_times.pop(0); price_vals.pop(0)

    txt = f"現在のBTC/JPY: {price:,} 円" if price else "価格取得エラー"

    # チャート作成
    traces = [{"x":price_times, "y":price_vals, "type":"line", "name":"BTC/JPY"}]
    layout = {"title":"BTC/JPY Price","margin":{"l":40,"r":20,"t":30,"b":30}}
    price_fig = {"data":traces, "layout":layout}

    # NewsAPI
    arts = get_news_events()
    news_data = [
        {"publishedAt": art.get("publishedAt","")[:19].replace("T"," "),
         "title": art.get("title","")}
        for art in arts
    ]

    # Positions
    positions = get_positions()
    pos_data = []
    for pos in positions:
        pos_data.append({
            "product_code": pos.get("product_code",""),
            "side":         pos.get("side",""),
            "size":         pos.get("size",0),
            "price":        pos.get("price",0),
            "pnl":          pos.get("pnl",0)
        })

    return txt, price_fig, news_data, pos_data

# ——————————————————————————————————
# コールバック: シミュレーションモード開閉
# ——————————————————————————————————
@app.callback(
    Output("sim-modal","is_open"),
    [Input("open-sim-btn","n_clicks"), Input("close-sim-btn","n_clicks")],
    [State("sim-modal","is_open")]
)
def toggle_sim_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# ——————————————————————————————————
# コールバック: シミュレーション計算
# ——————————————————————————————————
@app.callback(
    Output("sim-exit-price","children"),
    Input("sim-invest-jpy","value"),
    Input("sim-target-jpy","value"),
    Input("interval","n_intervals")
)
def update_simulation(invest_jpy, target_jpy, n):
    if not invest_jpy or invest_jpy <= 0 or target_jpy is None or target_jpy < 0:
        return "投資金額と目標利益を入力してください。"
    tk    = get_ticker("BTC_JPY")
    price = tk.get("ltp") or tk.get("best_bid")
    if not price:
        return "価格取得失敗"
    exit_price = price * (invest_jpy + target_jpy) / invest_jpy
    return f"売却推奨価格: {exit_price:,.0f} JPY／BTC  (現在価格: {price:,.0f} JPY)"

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
