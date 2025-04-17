import os
import time
import hmac
import hashlib
import logging
import json
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
from requests.exceptions import HTTPError

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# ————————————————————
# 環境変数
# ————————————————————
API_URL    = "https://api.bitflyer.com"
API_KEY    = os.environ.get("BITFLYER_API_KEY")
API_SECRET = os.environ.get("BITFLYER_API_SECRET")

logging.basicConfig(level=logging.INFO)
logging.info(f"App started at {datetime.now()}")

# ————————————————————
# HMAC署名付きリクエスト
# ————————————————————
def _sign(ts: str, method: str, path: str, body=""):
    msg = ts + method + path + body
    return hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()

def _private(path: str, method="GET", params=None, body=None):
    ts = str(time.time())
    body_json = json.dumps(body) if body else ""
    headers = {
        "ACCESS-KEY":       API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN":      _sign(ts, method, path, body_json),
        "Content-Type":     "application/json"
    }
    url = API_URL + path
    r = requests.get(url, headers=headers, params=params) if method=="GET" else requests.post(url, headers=headers, data=body_json)
    r.raise_for_status()
    return r.json()

# ————————————————————
# BitFlyer API ヘルパー
# ————————————————————
def get_ticker(product="BTC_JPY"):
    r = requests.get(f"{API_URL}/v1/ticker?product_code={product}")
    r.raise_for_status()
    return r.json()

from requests.exceptions import HTTPError

_cached_child_orders = []
_last_orders_fetch  = 0.0

def get_child_orders(limit=100):
    """
    直近 60 秒以内はキャッシュを返し、
    429 エラー時はキャッシュでリカバーします。
    """
    global _cached_child_orders, _last_orders_fetch
    now = time.time()
    if now - _last_orders_fetch < 60 and _cached_child_orders:
        return _cached_child_orders

    path = f"/v1/me/getchildorders?count={limit}&product_code=BTC_JPY"
    try:
        data = _private(path)
        _cached_child_orders = data
        _last_orders_fetch  = now
        return data
    except HTTPError as e:
        if e.response.status_code == 429:
            logging.warning("429 Too Many Requests — using cached child orders")
            return _cached_child_orders
        raise


# ————————————————————
# get_executions：キャッシュ＆429対策
# ————————————————————
_cached_executions = []
_last_exec_fetch = 0.0
def get_executions(limit=100):
    global _cached_executions, _last_exec_fetch
    now = time.time()
    if now - _last_exec_fetch < 60 and _cached_executions:
        return _cached_executions
    path = f"/v1/me/getexecutions?count={limit}&product_code=BTC_JPY"
    try:
        data = _private(path)
        _cached_executions = data
        _last_exec_fetch = now
        return data
    except HTTPError as e:
        if e.response.status_code == 429:
            logging.warning("429 Too Many Requests — using cached executions")
            return _cached_executions
        raise

def get_balances():
    return _private("/v1/me/getbalance")

def send_childorder_limit(jpy_amt: float, side: str):
    p = get_ticker()["ltp"]
    size = round(jpy_amt / p, 8)
    body = {
        "product_code":     "BTC_JPY",
        "child_order_type": "LIMIT",
        "side":             side,
        "price":            int(p),
        "size":             size,
        "minute_to_expire": 43200,
        "time_in_force":    "GTC"
    }
    return _private("/v1/me/sendchildorder", "POST", body=body)

# ——————————————————————————————
# Binance API から BTC/JPY 日次終値を取得
# （ダッシュボード.py の仕組みを活用） :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
# ——————————————————————————————
def get_historical_btc_jpy(interval="1d", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','num_trades','tbav','tqav','ignore'
    ])
    df['open_time']  = pd.to_datetime(df['open_time'], unit='ms')
    df['close']      = pd.to_numeric(df['close'], errors='coerce')
    # USD→JPY 為替レートを取得
    fx = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5).json()['rates']['JPY']
    df['close_jpy']  = df['close'] * fx
    return list(zip(df['open_time'], df['close_jpy']))

# ————————————————————
# P/L 履歴バッファ
# ————————————————————
pl_history = deque(maxlen=500)

# ————————————————————
# Dash アプリ & レイアウト
# ————————————————————
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.layout = dbc.Container(fluid=True, children=[

    html.H1("BitFlyer JPY目標売買 + 履歴 / 評価損益 / 資産集計"),
    dcc.Interval(id="interval", interval=5000, n_intervals=0),

    # JPY目標売却／購入ボタン
    dbc.Button("JPY目標で売却", id="open-sell", color="danger", className="me-2"),
    dbc.Button("JPY目標で購入", id="open-buy",  color="success"),

    # 売却モーダル
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("JPY目標売却")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="sell-jpy", type="number", min=0, placeholder="売却目標 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            html.Div(id="sell-info", className="fw-bold"),
            dbc.Button("注文実行", id="sell-exec", color="primary"),
            html.Div(id="sell-result", className="mt-2 text-success")
        ]),
        dbc.ModalFooter(dbc.Button("閉じる", id="close-sell", className="ms-auto"))
    ], id="modal-sell", is_open=False),

    # 購入モーダル
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("JPY目標購入")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="buy-jpy", type="number", min=0, placeholder="購入目標 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            html.Div(id="buy-info", className="fw-bold"),
            dbc.Button("注文実行", id="buy-exec", color="primary"),
            html.Div(id="buy-result", className="mt-2 text-success")
        ]),
        dbc.ModalFooter(dbc.Button("閉じる", id="close-buy", className="ms-auto"))
    ], id="modal-buy", is_open=False),

    html.Hr(),

    # 取引履歴テーブル
    html.H2("取引履歴 (全件)"),
    dash_table.DataTable(
        id="order-table",
        columns=[
            {"name":"受付ID","id":"child_order_acceptance_id"},
            {"name":"売買","id":"side"},
            {"name":"価格","id":"price"},
            {"name":"数量","id":"size"},
            {"name":"状態","id":"child_order_state"},
            {"name":"約定数","id":"executions"},
            {"name":"約定額 (JPY)","id":"amount_jpy"}
        ],
        data=[], page_size=20,
        style_table={"overflowY":"auto","maxHeight":"400px"}
    ),

    html.Hr(),

    # 評価損益時系列面グラフ
    html.H2("評価損益 (2024/08/02〜現在)"),
    dcc.Graph(id="pl-graph", config={"displayModeBar": False}),

    html.H2("通貨別評価損益"),
    dcc.Graph(id="asset-pl-bar"),

    html.Hr(),

    # 実現損益 + 評価資産サマリー
    html.H2("実現損益 + 評価資産 (JPY)"),
    html.Div(id="pl-summary", className="fw-bold")
])

# ————————————————————
# モーダル開閉コールバック
# ————————————————————
@app.callback(
    Output("modal-sell","is_open"),
    [Input("open-sell","n_clicks"), Input("close-sell","n_clicks")],
    [State("modal-sell","is_open")]
)
def toggle_sell(o, c, is_open):
    return not is_open if o or c else is_open

@app.callback(
    Output("modal-buy","is_open"),
    [Input("open-buy","n_clicks"), Input("close-buy","n_clicks")],
    [State("modal-buy","is_open")]
)
def toggle_buy(o, c, is_open):
    return not is_open if o or c else is_open

# ————————————————————
# 数量表示コールバック
# ————————————————————
@app.callback(
    Output("sell-info","children"),
    [Input("sell-jpy","value"), Input("interval","n_intervals")]
)
def show_sell(jpy, _):
    if not jpy or jpy <= 0:
        return "売却目標を入力してください"
    p = get_ticker()["ltp"]
    return f"売却予定: {round(jpy/p,8):.8f} BTC ＠ {p:,} JPY"

@app.callback(
    Output("buy-info","children"),
    [Input("buy-jpy","value"), Input("interval","n_intervals")]
)
def show_buy(jpy, _):
    if not jpy or jpy <= 0:
        return "購入目標を入力してください"
    p = get_ticker()["ltp"]
    return f"購入予定: {round(jpy/p,8):.8f} BTC ＠ {p:,} JPY"

# ————————————————————
# 注文実行コールバック
# ————————————————————
@app.callback(
    Output("sell-result","children"),
    Input("sell-exec","n_clicks"),
    State("sell-jpy","value")
)
def exec_sell(n, jpy):
    if not n or not jpy:
        return ""
    res = send_childorder_limit(jpy, "SELL")
    return "注文成功: "+res.get("child_order_acceptance_id","") if "error" not in res else f"注文失敗: {res['error']}"

@app.callback(
    Output("buy-result","children"),
    Input("buy-exec","n_clicks"),
    State("buy-jpy","value")
)
def exec_buy(n, jpy):
    if not n or not jpy:
        return ""
    res = send_childorder_limit(jpy, "BUY")
    return "注文成功: "+res.get("child_order_acceptance_id","") if "error" not in res else f"注文失敗: {res['error']}"

# ————————————————————
# 更新＆グラフ描画コールバック
# ————————————————————
@app.callback(
    [Output("order-table","data"),
     Output("pl-summary","children"),
     Output("pl-graph","figure"),
     Output("asset-pl-bar","figure")],
    Input("interval","n_intervals")
)
def refresh_all(_):
    # 1) 約定履歴→実現損益・保有BTC量の時系列イベント
    executions = get_executions()
    executions.sort(key=lambda e: datetime.fromisoformat(e["exec_date"].replace("Z","+00:00")))
    cum_realized, holdings = 0.0, 0.0
    events = []
    for ex in executions:
        t = datetime.fromisoformat(ex["exec_date"].replace("Z","+00:00"))
        s = ex["size"]
        if ex["side"] == "BUY":
            cum_realized -= (ex["price"]*s + ex.get("commission",0))
            holdings      += s
        else:
            cum_realized += (ex["price"]*s - ex.get("commission",0))
            holdings      -= s
        events.append((t, cum_realized, holdings))

    # 2) 取引履歴テーブル
    orders = get_child_orders()
    ex_map = defaultdict(list)
    for ex in executions:
        ex_map[ex["child_order_acceptance_id"]].append(ex)
    rows = []
    for o in orders:
        exs = ex_map[o["child_order_acceptance_id"]]
        amt = sum((e["price"]*e["size"] - e.get("commission",0)) if e["side"]=="SELL"
                  else -(e["price"]*e["size"] + e.get("commission",0)) for e in exs)
        rows.append({
            "child_order_acceptance_id": o["child_order_acceptance_id"],
            "side":                      o["side"],
            "price":                     o["price"],
            "size":                      o["size"],
            "child_order_state":         o["child_order_state"],
            "executions":                len(exs),
            "amount_jpy":                round(amt,0)
        })

    # 3) 評価資産
    bals = get_balances()
    asset_map = {}
    for b in bals:
        c,a = b["currency_code"], b["amount"]
        if a <= 0: continue
        asset_map[c] = a if c=="JPY" else a * get_ticker(f"{c}_JPY")["ltp"]
    asset_total = sum(asset_map.values())

    # 4) 過去BTC/JPY価格取得＆評価損益計算
    price_series = get_historical_btc_jpy()
    times, totals = [], []
    ei = 0
    for t_price, price in price_series:
        while ei < len(events) and events[ei][0] <= t_price:
            cum_realized, holdings = events[ei][1], events[ei][2]
            ei += 1
        times.append(t_price)
        totals.append(cum_realized + holdings * price)

    # 5) 面グラフ
    fig_pl = go.Figure([
        go.Scatter(
            x=times, y=[max(0,v)   for v in totals], mode="lines",
            fill="tozeroy", fillcolor="rgba(0,200,0,0.2)",
            line=dict(color="rgba(0,200,0,1)")
        ),
        go.Scatter(
            x=times, y=[min(0,v)   for v in totals], mode="lines",
            fill="tozeroy", fillcolor="rgba(200,0,0,0.2)",
            line=dict(color="rgba(200,0,0,1)")
        )
    ])
    fig_pl.update_layout(
        title="評価損益推移",
        xaxis=dict(range=[datetime(2024,8,2), datetime.now()]),
        yaxis_title="JPY", showlegend=False
    )

    # 6) 通貨別評価損益棒グラフ
    fig_bar = go.Figure([go.Bar(x=list(asset_map.keys()), y=list(asset_map.values()))])
    fig_bar.update_layout(title="通貨別評価損益", xaxis_title="通貨", yaxis_title="JPY")

    # 7) サマリー
    summary = f"実現損益: {cum_realized:,.0f} JPY ／ 評価資産: {asset_total:,.0f} JPY ／ 合計: {cum_realized+asset_total:,.0f} JPY"

    return rows, summary, fig_pl, fig_bar

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
