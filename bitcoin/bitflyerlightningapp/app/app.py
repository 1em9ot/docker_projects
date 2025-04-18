#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, hmac, hashlib, logging
from urllib.parse import urlencode

import pandas as pd
import requests

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table
import plotly.graph_objs as go

# ── ロギング設定 ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ── bitFlyer API 認証付き GET（転用：bitflyerAPI金額計算.py） :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
API_URL    = "https://api.bitflyer.com"
API_KEY    = os.getenv("BITFLYER_API_KEY", "")
API_SECRET = os.getenv("BITFLYER_API_SECRET", "")

def _api(path, params=None):
    ts     = str(int(time.time()))
    q      = urlencode(params or {})
    target = f"{path}?{q}" if q else path
    sign   = hmac.new(API_SECRET.encode(),
                      (ts + "GET" + target).encode(),
                      hashlib.sha256).hexdigest()
    hdr    = {"ACCESS-KEY":API_KEY,
              "ACCESS-TIMESTAMP":ts,
              "ACCESS-SIGN":sign}
    try:
        r = requests.get(API_URL + path, params=params, headers=hdr, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("%s : %s", path, e)
        return []

# ── 対象通貨設定 ─────────────────────────────────────────────
CRYPTOS = ["BTC", "ETH", "XRP", "JPY"]

# ── 全取引履歴取得 & 整形（Lightning＋Spot＋入出金＋手数料） ── :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
def fetch_api_trades():
    rows, seen = [], set()
    def add(rec):
        tid = rec[-1]
        if tid in seen: return
        seen.add(tid)
        rows.append(rec)

    # 1) BTC Lightning executions
    before = None
    while True:
        params = {"product_code":"BTC_JPY","count":500}
        if before: params["before"] = before
        data = _api("/v1/me/getexecutions", params)
        if not data: break
        for e in data:
            ts    = pd.to_datetime(e["exec_date"])
            code  = "BTC"
            qty   = e["size"]
            price = e["price"]          # 当時の価格が取れる
            val   = round(price * qty)
            op    = "BUY" if e["side"]=="BUY" else "SELL"
            add([ts, code, price, qty, val, op, e["id"]])
        if len(data)<500: break
        before = data[-1]["id"]

    # 2) Spot trades & Deposit/Withdraw for BTC, ETH, XRP
    for code in ["BTC","ETH","XRP"]:
        before = None
        while True:
            params = {"count":500, "currency_code":code}
            if before: params["before"] = before
            data = _api("/v1/me/getbalancehistory", params)
            if not data: break
            for e in data:
                ts  = pd.to_datetime(e.get("trade_date") or e.get("event_date"))
                typ = e.get("trade_type")            # BUY/SELL or None
                qty = abs(e.get("quantity",0))
                amt = round(e.get("amount",0))       # API が返す JPY 金額
                rid = e.get("id") or f"{code}_{ts.value}"

                if typ in ("BUY","SELL"):
                    # 取引価格
                    price = e.get("price") or (amt/qty if qty else 0)
                    val   = round(price * qty)
                    op    = "BUY" if typ=="BUY" else "SELL"
                    add([ts, code, price, qty, val, op, rid])
                else:
                    # 暗号資産の入出金：API の amt をそのまま使う
                    if qty>0 and amt!=0:
                        price = amt/qty
                        val   = amt
                        op    = "DEPOSIT" if amt>0 else "WITHDRAW"
                        add([ts, code, price, qty, val, op, rid])
                # 手数料
                fee = e.get("fee",0) or e.get("commission",0)
                if fee:
                    fee_amt = -abs(round(fee))
                    add([ts, code, None, 0.0, fee_amt, "FEE", f"{rid}_fee"])
            if len(data)<500: break
            before = data[-1].get("id")

    # 3) JPY 入出金 & 手数料
    before = None
    while True:
        params = {"count":500, "currency_code":"JPY"}
        if before: params["before"] = before
        data = _api("/v1/me/getbalancehistory", params)
        if not data: break
        for e in data:
            ts  = pd.to_datetime(e.get("trade_date") or e.get("event_date"))
            amt = round(e.get("amount",0))
            rid = e.get("id") or f"JPY_{ts.value}"
            if amt!=0:
                op = "DEPOSIT" if amt>0 else "WITHDRAW"
                add([ts, "JPY", None, 0.0, amt, op, rid])
            fee = e.get("fee",0) or e.get("commission",0)
            if fee:
                fee_amt = -abs(round(fee))
                add([ts, "JPY", None, 0.0, fee_amt, "FEE", f"{rid}_fee"])
        if len(data)<500: break
        before = data[-1].get("id")

    # DataFrame 化
    df = pd.DataFrame(rows, columns=[
        "日時","通貨","価格(JPY)","数量","金額(JPY)","操作","id"
    ]).sort_values("日時").reset_index(drop=True)

    # 残高(数量 or JPY) 列
    balances = {c:0.0 for c in ["BTC","ETH","XRP","JPY"]}
    bal_list = []
    for _, r in df.iterrows():
        code = r["通貨"]
        if code=="JPY":
            balances["JPY"] += r["金額(JPY)"]
            bal = balances["JPY"]
        else:
            balances[code] += r["数量"]
            bal = balances[code]
        bal_list.append(bal)
    df["残高"] = bal_list

    return df

# ── データ準備 ───────────────────────────────────────────────
df = fetch_api_trades()
records = df.to_dict("records")

# ── 時系列 ウォーターフォールチャート作成 ───────────────────
wf = go.Figure(go.Waterfall(
    orientation="v",
    x=df["日時"].dt.strftime("%Y-%m-%d %H:%M"),
    y=df["金額(JPY)"],
    measure=["relative"]*len(df),
    text=df["金額(JPY)"].map(lambda v: f"{v:,}"),
    textposition="outside"
))
wf.update_layout(
    title="時系列 資産フロー ウォーターフォール",
    xaxis_title="日時", yaxis_title="金額"
)

# ── Dash アプリケーション定義 ───────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(fluid=True, children=[
    html.H1("Crypto 資産フロー & 残高推移", className="my-3"),
    dbc.Row([
        dbc.Col([html.H3("時系列 ウォーターフォール"), dcc.Graph(figure=wf)], width=6),
        dbc.Col([
            html.H3("全取引履歴"),
            dash_table.DataTable(
                id="flow-table",
                columns=[{"name":c,"id":c} for c in
                    ["日時","通貨","価格(JPY)","数量","金額(JPY)","操作","残高"]
                ],
                data=records,
                filter_action="native", sort_action="native", page_size=25,
                style_table={"maxHeight":"600px","overflowY":"auto"},
                style_cell={"textAlign":"center","whiteSpace":"nowrap"}
            )
        ], width=6)
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
