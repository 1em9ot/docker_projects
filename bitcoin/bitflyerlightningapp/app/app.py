# app.py — bitFlyer DCA／スイング統合ダッシュボード
# Lightning約定 + 販売所（残高履歴）をマージして全履歴表示

import os
import time
import hmac
import hashlib
import logging
from urllib.parse import urlencode

import pandas as pd
import requests

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.graph_objs as go

# ---------- 設定 ----------
API_URL       = "https://api.bitflyer.com"
API_KEY       = os.getenv("BITFLYER_API_KEY")
API_SECRET    = os.getenv("BITFLYER_API_SECRET")
DCA_THRESHOLD = 50_000  # JPY：DCA判断閾値

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- bitFlyer Private API ヘルパ ----------
def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    return hmac.new(
        API_SECRET.encode(),
        (ts + method + path + body).encode(),
        hashlib.sha256
    ).hexdigest()


def _private_get(path: str, params: dict | None = None):
    """
    GET リクエスト (プライベート API)
    """
    ts = str(int(time.time()))
    query = f"?{urlencode(params)}" if params else ""
    req_path = path + query
    sign = _sign(ts, "GET", req_path, "")
    headers = {
        "ACCESS-KEY":       API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN":      sign,
        "Content-Type":     "application/json"
    }
    url = API_URL + req_path
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_ltp() -> float:
    """現在価格(LTP)を取得"""
    try:
        return _private_get("/v1/ticker", {"product_code": "BTC_JPY"})["ltp"]
    except Exception:
        return 0.0

# ---------- Lightning 約定履歴 ----------
def fetch_executions() -> list[dict]:
    recs = []
    before = None
    while True:
        params = {"product_code": "BTC_JPY", "count": 500}
        if before:
            params["before"] = before
        data = _private_get("/v1/me/getexecutions", params)
        if not data:
            break

        for e in data:
            ts = pd.to_datetime(e["exec_date"])
            jpy = e["price"] * e["size"]
            kind = "買い" if e["side"] == "BUY" else "売り"
            strat = "DCA" if (kind == "買い" and jpy <= DCA_THRESHOLD) else "スイング"
            recs.append({
                "id":          e["id"],
                "日時":        ts.strftime("%Y-%m-%d %H:%M"),
                "取引種別":     kind,
                "金額(JPY)":   round(jpy),
                "数量(BTC)":   e["size"],
                "戦略分類":     strat,
                "備考":        "Lightning"
            })
        if len(data) < 500:
            break
        before = data[-1]["id"]

    recs.sort(key=lambda x: x["日時"])
    return recs

# ---------- 販売所・残高履歴（入出金＋Spot取引） ----------
def fetch_balance_history() -> list[dict]:
    recs = []
    before = None
    while True:
        params = {"count": 500, "currency_code": "BTC"}
        if before:
            params["before"] = before
        data = _private_get("/v1/me/getbalancehistory", params)
        if not data:
            break

        for e in data:
            typ = e.get("trade_type")
            if typ in ("BUY", "SELL"):
                ts = pd.to_datetime(e.get("trade_date") or e.get("event_date"))
                kind = "買い" if typ == "BUY" else "売り"
                jpy = abs(e.get("amount", 0))
                strat = "DCA" if (kind == "買い" and jpy <= DCA_THRESHOLD) else "スイング"
                recs.append({
                    "id":          e.get("id"),
                    "日時":        ts.strftime("%Y-%m-%d %H:%M"),
                    "取引種別":     kind,
                    "金額(JPY)":   round(jpy),
                    "数量(BTC)":   abs(e.get("quantity", 0)),
                    "戦略分類":     strat,
                    "備考":        "BalanceHistory"
                })
        if len(data) < 500:
            break
        before = data[-1].get("id")

    recs.sort(key=lambda x: x["日時"])
    return recs

# ---------- 全取引履歴をマージ ----------
def fetch_all_trades() -> list[dict]:
    execs = fetch_executions()
    spots = fetch_balance_history()
    alltx = execs + spots
    alltx.sort(key=lambda x: x["日時"])
    return alltx

# ---------- データ準備 ----------
all_trades   = fetch_all_trades()
ltp          = get_ltp()
dca_trades   = [r for r in all_trades if r["戦略分類"] == "DCA" and r["取引種別"] == "買い"]

# 累積投資＆評価額
times, invest_vals, eval_vals, point_ids = [], [], [], []
cum_jpy = cum_btc = 0.0
for rec in dca_trades:
    cum_jpy += rec["金額(JPY)"]
    cum_btc += rec["数量(BTC)"]
    times.append(pd.to_datetime(rec["日時"]))
    invest_vals.append(cum_jpy)
    eval_vals.append(cum_btc * ltp)
    point_ids.append(rec["id"])

# ---------- Dash レイアウト ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(fluid=True, children=[
    html.H1("bitFlyer 戦略ダッシュボード", className="my-3"),
    dbc.Row([
        dbc.Col([
            html.H3("DCA戦略（累積投資 vs 評価額）"),
            dcc.Graph(
                id="dca-graph",
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=times, y=invest_vals,
                            mode="lines+markers", name="累積投資額",
                            marker=dict(size=8), customdata=point_ids
                        ),
                        go.Scatter(
                            x=times, y=eval_vals,
                            mode="lines+markers", name="評価額",
                            marker=dict(size=8), customdata=point_ids
                        )
                    ],
                    layout=go.Layout(
                        title="DCA取引ベース" if times else "購入履歴なし",
                        xaxis_title="日時", yaxis_title="JPY"
                    )
                )
            )
        ], width=6),
        dbc.Col([
            html.H3("全取引履歴"),
            dash_table.DataTable(
                id="trade-table",
                columns=[{"name":c, "id":c} for c in [
                    "日時","取引種別","金額(JPY)","数量(BTC)","戦略分類","備考"
                ]],
                data=all_trades,
                row_selectable="single",
                selected_rows=[],
                filter_action="native",
                sort_action="native",
                style_table={"maxHeight":"500px","overflowY":"auto"},
                style_cell={"textAlign":"center"}
            )
        ], width=6)
    ])
])

# ---------- コールバック: グラフ→テーブル ----------
@app.callback(
    Output("trade-table", "selected_rows"),
    Input("dca-graph", "hoverData"),
    Input("dca-graph", "clickData")
)
def graph_to_table(hover, click):
    pt = None
    if click and click.get("points"): pt = click["points"][0]
    elif hover and hover.get("points"): pt = hover["points"][0]
    if pt:
        pid = pt.get("customdata")
        for idx, row in enumerate(all_trades):
            if row["id"] == pid:
                return [idx]
    return []

# ---------- コールバック: テーブル→グラフ ----------
@app.callback(
    Output("dca-graph", "figure"),
    Input("trade-table", "selected_rows"),
    State("dca-graph", "figure")
)
def table_to_graph(selected, fig):
    base_colors = ["royalblue","darkorange"]
    base_size   = 8
    hi_color    = "red"
    hi_size     = 14
    for t in (0,1):
        npts = len(fig["data"][t]["x"])
        fig["data"][t]["marker"]["color"] = [base_colors[t]]*npts
        fig["data"][t]["marker"]["size"]  = [base_size]*npts
    if selected:
        sel = selected[0]
        if sel < len(all_trades) and all_trades[sel]["id"] in point_ids:
            j = point_ids.index(all_trades[sel]["id"])
            for t in (0,1):
                fig["data"][t]["marker"]["color"][j] = hi_color
                fig["data"][t]["marker"]["size"][j]  = hi_size
    return fig

# ---------- サーバ起動 ----------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)