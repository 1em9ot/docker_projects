#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_crypto_treemap.py
S&P500 と主要暗号資産を左右 50:50 で並べたヒートマップを描画。
必要パッケージ:
  pip install pandas yfinance requests plotly==5.* numpy lxml
実行:
  python stock_crypto_treemap.py
"""

import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import yfinance as yf


# ----------------------------------------------------------------------
# 1. データ取得
# ----------------------------------------------------------------------
def get_sp500_tickers() -> list[str]:
    """Wikipedia から S&P500 ティッカー一覧をスクレイピング"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].tolist()


def fetch_stock_snapshot(ticker: str) -> dict | None:
    """yfinance で 1 銘柄ずつ snapshot (高速化のため情報最小限)"""
    try:
        info = yf.Ticker(ticker).fast_info  # fast_info は v0.2 以降
        return dict(
            symbol=ticker,
            name=ticker,
            parent="Stocks",
            market_cap=info["marketCap"],
            change=float(info["lastPrice"])
            / float(info["previousClose"])
            * 100.0
            - 100.0,
        )
    except Exception:
        return None


def fetch_crypto_market(page_size: int = 250) -> list[dict]:
    """CoinGecko API から暗号資産のマーケットデータを取得"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = dict(
        vs_currency="usd",
        order="market_cap_desc",
        per_page=page_size,
        price_change_percentage="24h",
        page=1,
    )
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    out = []
    for row in res.json():
        out.append(
            dict(
                symbol=row["symbol"].upper(),
                name=row["name"],
                parent="Crypto",
                market_cap=row["market_cap"],
                change=row["price_change_percentage_24h_in_currency"],
            )
        )
    return out


def gather_all_data() -> pd.DataFrame:
    """株+暗号資産をまとめて DataFrame 化"""
    print("Fetching S&P500 tickers …")
    tickers = get_sp500_tickers()

    print("Downloading stock snapshots …")
    stocks = []
    for t in tickers:
        snap = fetch_stock_snapshot(t)
        if snap:
            stocks.append(snap)

    print("Downloading crypto market data …")
    cryptos = fetch_crypto_market(250)

    df = pd.DataFrame(stocks + cryptos)
    # 欠損処理
    df["market_cap"] = df["market_cap"].fillna(0)
    df["change"] = df["change"].fillna(0)
    return df


# ----------------------------------------------------------------------
# 2. 比率調整 ― 左右を 50:50 に割る
# ----------------------------------------------------------------------
def scale_halves(df: pd.DataFrame) -> pd.DataFrame:
    cap_sum = df.groupby("parent")["market_cap"].sum()
    max_half = cap_sum.max()
    def _scale(row):
        weight = max_half / cap_sum[row["parent"]]
        return row["market_cap"] * weight
    df["scaled_cap"] = df.apply(_scale, axis=1)
    return df


# ----------------------------------------------------------------------
# 3. 描画
# ----------------------------------------------------------------------
def draw_treemap(df: pd.DataFrame, outfile: str | None = None):
    fig = px.treemap(
        df,
        path=["parent", "symbol"],
        values="scaled_cap",
        color="change",
        color_continuous_scale=["red", "black", "green"],
        range_color=[-5, 5],
        hover_data=dict(
            market_cap=":.3s",
            change=":.2f",
            parent=False,  # hover 表示不要
            scaled_cap=False,
        ),
    )
    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        title=f"Stocks vs Crypto Heat-Treemap  (UTC {dt.datetime.utcnow():%Y-%m-%d %H:%M})",
    )
    if outfile:
        fig.write_html(outfile)
        print(f"Saved → {outfile}")
    else:
        fig.show()


# ----------------------------------------------------------------------
# 4. メイン
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = gather_all_data()
    df = scale_halves(df_raw)
    out_file = None
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    draw_treemap(df, out_file)
