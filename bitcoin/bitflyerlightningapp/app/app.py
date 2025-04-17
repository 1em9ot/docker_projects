import os
import time
import hmac
import hashlib
import logging
import json
import requests
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ——————————————————————————————————
# 環境変数
# ——————————————————————————————————
API_URL    = "https://api.bitflyer.com"
API_KEY    = os.environ.get("BITFLYER_API_KEY")
API_SECRET = os.environ.get("BITFLYER_API_SECRET")

logging.basicConfig(level=logging.INFO)
logging.info(f"App started at {datetime.now()}")

# ——————————————————————————————————
# BitFlyer API ヘルパー
# ——————————————————————————————————
def _get_signature(ts: str, method: str, path: str, body: str = "") -> str:
    text = ts + method + path + body
    return hmac.new(API_SECRET.encode(), text.encode(), hashlib.sha256).hexdigest()


def get_ticker(product_code: str = "BTC_JPY") -> dict:
    resp = requests.get(f"{API_URL}/v1/ticker?product_code={product_code}")
    resp.raise_for_status()
    return resp.json()


def send_childorder_limit(jpy_amount: float, side: str) -> dict:
    tk = get_ticker()
    price = tk.get("ltp") or tk.get("best_bid")
    if not price:
        return {"error": "価格取得失敗"}
    size = round(jpy_amount / price, 8)
    ts = str(time.time())
    method = "POST"
    path   = "/v1/me/sendchildorder"
    body = {
        "product_code":     "BTC_JPY",
        "child_order_type": "LIMIT",
        "side":             side,
        "price":            int(price),
        "size":             size,
        "minute_to_expire": 43200,
        "time_in_force":    "GTC"
    }
    body_json = json.dumps(body)
    sign = _get_signature(ts, method, path, body_json)
    headers = {
        "ACCESS-KEY":       API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN":      sign,
        "Content-Type":     "application/json"
    }
    resp = requests.post(API_URL + path, headers=headers, data=body_json)
    resp.raise_for_status()
    return resp.json()


def get_child_orders(limit: int = 20) -> list:
    ts = str(time.time())
    method = "GET"
    path = f"/v1/me/getchildorders?count={limit}&product_code=BTC_JPY"
    sign = _get_signature(ts, method, path)
    headers = {"ACCESS-KEY":API_KEY, "ACCESS-TIMESTAMP":ts, "ACCESS-SIGN":sign}
    resp = requests.get(API_URL + path, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_executions(limit: int = 100) -> list:
    ts = str(time.time())
    method = "GET"
    path = f"/v1/me/getexecutions?count={limit}&product_code=BTC_JPY"
    sign = _get_signature(ts, method, path)
    headers = {"ACCESS-KEY":API_KEY, "ACCESS-TIMESTAMP":ts, "ACCESS-SIGN":sign}
    resp = requests.get(API_URL + path, headers=headers)
    resp.raise_for_status()
    return resp.json()

# ——————————————————————————————————
# Dash アプリ定義
# ——————————————————————————————————
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.layout = dbc.Container(fluid=True, children=[
    html.H1("BitFlyer JPY目標売買 + 履歴 & P/L集計"),
    dcc.Interval(id="interval", interval=5000, n_intervals=0),

    # JPY目標売却／購入ボタン
    dbc.Button("JPY目標で売却", id="open-sell-jpy-btn", color="danger", className="me-2"),
    dbc.Button("JPY目標で購入", id="open-buy-jpy-btn", color="success"),

    # 売却モーダル
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("JPY目標売却")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="sell-jpy-input", type="number", min=0, placeholder="売却目標 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            html.Div(id="sell-jpy-qty", style={"fontWeight":"bold","marginBottom":"1rem"}),
            dbc.Button("注文実行", id="sell-jpy-execute-btn", color="primary"),
            html.Div(id="sell-jpy-result", style={"marginTop":"1rem","color":"green"})
        ]),
        dbc.ModalFooter(dbc.Button("閉じる", id="close-sell-jpy-btn", className="ms-auto"))
    ], id="sell-jpy-modal", is_open=False),

    # 購入モーダル
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("JPY目標購入")),
        dbc.ModalBody([
            dbc.InputGroup([
                dbc.Input(id="buy-jpy-input", type="number", min=0, placeholder="購入目標 (JPY)"),
                dbc.InputGroupText("JPY")
            ], className="mb-2"),
            html.Div(id="buy-jpy-qty", style={"fontWeight":"bold","marginBottom":"1rem"}),
            dbc.Button("注文実行", id="buy-jpy-execute-btn", color="primary"),
            html.Div(id="buy-jpy-result", style={"marginTop":"1rem","color":"green"})
        ]),
        dbc.ModalFooter(dbc.Button("閉じる", id="close-buy-jpy-btn", className="ms-auto"))
    ], id="buy-jpy-modal", is_open=False),

    html.Hr(),
    html.H2("取引履歴 (直近)"),
    dash_table.DataTable(
        id="order-history",
        columns=[
            {"name":"受付ID","id":"child_order_acceptance_id"},
            {"name":"売買","id":"side"},
            {"name":"価格","id":"price"},
            {"name":"数量","id":"size"},
            {"name":"状態","id":"child_order_state"},
            {"name":"約定数","id":"executions"},
            {"name":"約定額 (JPY)","id":"amount_jpy"}
        ], page_action="native", page_size=20,
        style_table={"overflowY":"auto","maxHeight":"300px"}
    ),

    html.Hr(),
    html.H2("実現損益 (JPY)"),
    html.Div(id="pnl-summary", style={"fontWeight":"bold"})
])

# ——————————————————————————————————
# コールバック: モーダル開閉
# ——————————————————————————————————
@app.callback(
    Output("sell-jpy-modal","is_open"),
    [Input("open-sell-jpy-btn","n_clicks"), Input("close-sell-jpy-btn","n_clicks")],
    [State("sell-jpy-modal","is_open")]
)
def toggle_sell(o, c, is_open):
    if o or c:
        return not is_open
    return is_open

@app.callback(
    Output("buy-jpy-modal","is_open"),
    [Input("open-buy-jpy-btn","n_clicks"), Input("close-buy-jpy-btn","n_clicks")],
    [State("buy-jpy-modal","is_open")]
)
def toggle_buy(o, c, is_open):
    if o or c:
        return not is_open
    return is_open

# ——————————————————————————————————
# コールバック: JPY→BTC量表示
# ——————————————————————————————————
@app.callback(
    Output("sell-jpy-qty","children"),
    [Input("sell-jpy-input","value"), Input("interval","n_intervals")]
)
def sell_qty(jpy, _):
    if not jpy or jpy <= 0:
        return "目標JPYを入力してください"
    price = get_ticker().get("ltp")
    return f"売却予定: {round(jpy/price,8):.8f} BTC @ {price:,} JPY"

@app.callback(
    Output("buy-jpy-qty","children"),
    [Input("buy-jpy-input","value"), Input("interval","n_intervals")]
)
def buy_qty(jpy, _):
    if not jpy or jpy <= 0:
        return "目標JPYを入力してください"
    price = get_ticker().get("ltp")
    return f"購入予定: {round(jpy/price,8):.8f} BTC @ {price:,} JPY"

# ——————————————————————————————————
# コールバック: 注文実行
# ——————————————————————————————————
@app.callback(
    Output("sell-jpy-result","children"),
    Input("sell-jpy-execute-btn","n_clicks"),
    State("sell-jpy-input","value")
)
def exec_sell(n, jpy):
    if not n or not jpy:
        return ""
    res = send_childorder_limit(jpy, "SELL")
    return ("注文成功: " + res.get("child_order_acceptance_id")) if not res.get("error") else f"注文失敗: {res['error']}"

@app.callback(
    Output("buy-jpy-result","children"),
    Input("buy-jpy-execute-btn","n_clicks"),
    State("buy-jpy-input","value")
)
def exec_buy(n, jpy):
    if not n or not jpy:
        return ""
    res = send_childorder_limit(jpy, "BUY")
    return ("注文成功: " + res.get("child_order_acceptance_id")) if not res.get("error") else f"注文失敗: {res['error']}"

# ——————————————————————————————————
# コールバック: 履歴 & P/L 更新
# ——————————————————————————————————
@app.callback(
    [Output("order-history","data"), Output("pnl-summary","children")],
    Input("interval","n_intervals")
)
def update_history(n):
    orders = get_child_orders()
    data = []
    for o in orders:
        amt = sum(ex.get("price",0)*ex.get("size",0) for ex in o.get("executions",[]))
        data.append({
            "child_order_acceptance_id": o.get("child_order_acceptance_id"),
            "side": o.get("side"),
            "price": o.get("price"),
            "size": o.get("size"),
            "child_order_state": o.get("child_order_state"),
            "executions": len(o.get("executions",[])),
            "amount_jpy": round(amt,0)
        })
    exs = get_executions()
    buy = sum(ex.get("price",0)*ex.get("size",0)+ex.get("commission",0) for ex in exs if ex.get("side")=="BUY")
    sell = sum(ex.get("price",0)*ex.get("size",0)-ex.get("commission",0) for ex in exs if ex.get("side")=="SELL")
    comm = sum(ex.get("commission",0) for ex in exs)
    pnl = sell - buy
    summary = f"総買付額: {buy:,.0f} JPY ／ 総売却額: {sell:,.0f} JPY ／ 実現損益: {pnl:,.0f} JPY (手数料合計 {comm:,.0f} JPY)"
    return data, summary

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)