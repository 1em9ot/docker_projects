import os
import time
import hmac
import hashlib
import logging
import json
import requests
from datetime import datetime
from collections import defaultdict, deque

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
        "ACCESS-KEY": API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN": _sign(ts, method, path, body_json),
        "Content-Type": "application/json"
    }
    url = API_URL + path
    resp = requests.get(url, headers=headers, params=params) if method=="GET" else \
           requests.post(url, headers=headers, data=body_json)
    resp.raise_for_status()
    return resp.json()

# ————————————————————
# API ヘルパー
# ————————————————————
def get_ticker(product="BTC_JPY"):
    r = requests.get(f"{API_URL}/v1/ticker?product_code={product}"); r.raise_for_status(); return r.json()

def get_child_orders(limit=100):
    return _private(f"/v1/me/getchildorders?count={limit}&product_code=BTC_JPY")

def get_executions(limit=500):
    return _private(f"/v1/me/getexecutions?count={limit}&product_code=BTC_JPY")

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

# ————————————————————
# P/L 履歴バッファ
# ————————————————————
pl_history = deque(maxlen=500)

# ————————————————————
# Dash アプリ & レイアウト
# ————————————————————
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.layout = dbc.Container(fluid=True, children=[

    html.H1("BitFlyer JPY目標売買 + 履歴 / P&L / 資産集計"),
    dcc.Interval(id="interval", interval=5000, n_intervals=0),

    # ボタン
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
        data=[],
        page_size=20,
        style_table={"overflowY":"auto","maxHeight":"400px"}
    ),

    html.Hr(),

    # 評価損益グラフ
    html.H2("評価損益 (時系列)"),
    dcc.Graph(id="pl-graph"),

    # 通貨別評価損益棒グラフ
    html.H2("通貨別評価損益"),
    dcc.Graph(id="asset-pl-bar"),

    html.Hr(),

    # 実現損益 + 評価資産
    html.H2("実現損益 + 評価資産 (JPY)"),
    html.Div(id="pl-summary", className="fw-bold")

])

# ————————————————————
# コールバック：モーダル開閉
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
# コールバック：量表示
# ————————————————————
@app.callback(
    Output("sell-info","children"),
    [Input("sell-jpy","value"), Input("interval","n_intervals")]
)
def show_sell(jpy, _):
    if not jpy or jpy<=0:
        return "目標JPYを入力してください"
    p = get_ticker()["ltp"]
    return f"{round(jpy/p,8):.8f} BTC ＠ {p:,} JPY"

@app.callback(
    Output("buy-info","children"),
    [Input("buy-jpy","value"), Input("interval","n_intervals")]
)
def show_buy(jpy, _):
    if not jpy or jpy<=0:
        return "目標JPYを入力してください"
    p = get_ticker()["ltp"]
    return f"{round(jpy/p,8):.8f} BTC ＠ {p:,} JPY"

# ————————————————————
# コールバック：注文実行
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
# コールバック：履歴 / グラフ / P&L 更新
# ————————————————————
@app.callback(
    [Output("order-table","data"),
     Output("pl-summary","children"),
     Output("pl-graph","figure"),
     Output("asset-pl-bar","figure")],
    Input("interval","n_intervals")
)
def refresh_all(_):
    try:
        # 取引履歴 & 実現損益
        orders     = get_child_orders()
        executions = get_executions()
        ex_map = defaultdict(list)
        for ex in executions:
            ex_map[ex["child_order_acceptance_id"]].append(ex)

        rows, realized = [], 0
        for o in orders:
            exs = ex_map[o["child_order_acceptance_id"]]
            amt = 0
            for ex in exs:
                delta = (ex["price"]*ex["size"] - ex.get("commission",0)) if ex["side"]=="SELL" \
                        else -(ex["price"]*ex["size"] + ex.get("commission",0))
                amt      += delta
                realized += delta
            rows.append({
                "child_order_acceptance_id": o["child_order_acceptance_id"],
                "side":                      o["side"],
                "price":                     o["price"],
                "size":                      o["size"],
                "child_order_state":         o["child_order_state"],
                "executions":                len(exs),
                "amount_jpy":                round(amt,0)
            })

        # 評価資産
        bal       = get_balances()
        asset_map = {}
        for b in bal:
            c, a = b["currency_code"], b["amount"]
            if a<=0: continue
            asset_map[c] = a if c=="JPY" else a * get_ticker(f"{c}_JPY")["ltp"]
        asset_total = sum(asset_map.values())

        # P/L 履歴に蓄積
        now = datetime.now()
        pl_history.append({"time":now, "realized":realized, "unrealized":asset_total, "total":realized+asset_total})

        # 評価損益時系列グラフ
        times = [p["time"] for p in pl_history]
        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(x=times, y=[p["realized"]   for p in pl_history], name="実現損益"))
        fig_pl.add_trace(go.Scatter(x=times, y=[p["unrealized"] for p in pl_history], name="評価資産"))
        fig_pl.add_trace(go.Scatter(x=times, y=[p["total"]      for p in pl_history], name="合計損益"))
        fig_pl.update_layout(title="評価損益推移", xaxis_title="時間", yaxis_title="JPY")

        # 通貨別棒グラフ
        fig_bar = go.Figure([go.Bar(x=list(asset_map.keys()), y=list(asset_map.values()))])
        fig_bar.update_layout(title="通貨別評価損益", xaxis_title="通貨", yaxis_title="JPY")

        # 合計サマリー
        summary = f"実現損益: {realized:,.0f} JPY ／ 評価資産: {asset_total:,.0f} JPY ／ 合計: {realized+asset_total:,.0f} JPY"
        return rows, summary, fig_pl, fig_bar

    except Exception as e:
        logging.error("refresh_all error: %s", e)
        return no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
