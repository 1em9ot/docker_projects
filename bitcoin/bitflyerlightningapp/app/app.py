#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC SMA Causal & Explainable Strategy Dashboard

Features:
- SMA クロスオーバー (短期20, 長期50) + 統計的有意性フィルタ (t検定)
- ランダムフォレストによる次足上昇確率予測
- 因果推論フィルタ (CausalForestDML)
- SHAP 値によるモデル説明ログ
- 表示期間をカレンダーで指定
"""
import os, logging, requests
from datetime import date, timedelta
import pandas as pd, numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from scipy.stats import ttest_1samp
from sklearn.ensemble import RandomForestClassifier
import shap
from econml.dml import CausalForestDML
from ta.momentum import rsi

# --- API キー設定 ---
API_KEY     = os.getenv("BITFLYER_API_KEY", "")
API_SECRET  = os.getenv("BITFLYER_API_SECRET", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# --- ロギング設定 ---
logging.basicConfig(filename="/persistent/app.log", level=logging.INFO)
logging.info("Starting BTC causal & explainable dashboard")

# --------------------------------------------------
# データ取得
# --------------------------------------------------
def fetch_ohlcv(symbol='BTC', interval='1h', limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': limit}
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    cols = [
        'open_time','open','high','low','close','volume',
        'close_time','qav','count','taker_base','taker_quote','ignore'
    ]
    df = pd.DataFrame(data, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['open','high','low','close','volume']]

# --------------------------------------------------
# バックテスト＋有意性テスト
# --------------------------------------------------
def backtest_with_significance(df, short_w=20, long_w=50):
    df = df.copy()
    df['sma_short'] = df['close'].rolling(short_w).mean()
    df['sma_long']  = df['close'].rolling(long_w).mean()
    df['signal'] = (df['sma_short'] > df['sma_long']).astype(int)
    df['signal_change'] = df['signal'].diff().fillna(0)
    df['future_return'] = df['close'].shift(-1)/df['close'] - 1
    buys = df.loc[df['signal_change']==1, 'future_return'].dropna()
    if len(buys) >= 2:
        _, pval = ttest_1samp(buys, popmean=0, alternative='greater')
    else:
        pval = 1.0
    df['return'] = df['close'].pct_change().fillna(0)
    df['strategy_return'] = df['return'] * df['signal'].shift(1).fillna(0)
    df['equity_curve'] = (1 + df['strategy_return']).cumprod()
    return df, pval

# --------------------------------------------------
# 次足上昇確率モデル訓練（特徴量とともに返す）
# --------------------------------------------------
def train_rf(df):
    d = df.copy()
    d['SMA20'] = d['close'].rolling(20).mean()
    d['EMA20'] = d['close'].ewm(span=20).mean()
    d['RSI14'] = rsi(d['close'], window=14)
    d.dropna(inplace=True)
    d['future_up'] = (d['close'].shift(-1) > d['close']).astype(int)
    X = d[['SMA20','EMA20','RSI14']]
    y = d['future_up']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X

def predict_next_prob(model, X):
    if X.empty:
        return None
    return model.predict_proba(X.iloc[-1:])[0][1]

# --------------------------------------------------
# 因果フィルタ
# --------------------------------------------------
def causal_filter(df):
    d = df.dropna().copy()
    T = (d['sma_short'] > d['sma_long']).astype(int)
    Y = d['close'].pct_change().shift(-1).fillna(0)
    X = d[['volume','close']]
    cf = CausalForestDML(n_estimators=100, random_state=0)
    cf.fit(Y, T, X=X)
    te = cf.effect(X)
    return float(te.mean()) > 0

# --------------------------------------------------
# Dash アプリケーション
# --------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
today = date.today()
two_weeks_ago = today - timedelta(weeks=2)

app.layout = dbc.Container(fluid=True, children=[
    html.H1("BTC Causal & Explainable Dashboard"),
    dbc.Row([
        dbc.Col([html.Label("時間足:"), dcc.Dropdown(
            id='interval',
            options=[
                {'label':'1分','value':'1m'},
                {'label':'5分','value':'5m'},
                {'label':'1時間','value':'1h'},
                {'label':'4時間','value':'4h'},
                {'label':'1日','value':'1d'},
            ],
            value='1h', clearable=False
        )], width=2),
        dbc.Col([html.Label("短期SMA期間:"), dcc.Input(id='short-window', type='number', value=20)], width=1),
        dbc.Col([html.Label("長期SMA期間:"), dcc.Input(id='long-window', type='number', value=50)], width=1),
        dbc.Col([html.Label("取得本数:"), dcc.Input(id='limit', type='number', value=500)], width=1),
        dbc.Col([html.Label("有意水準 α:"), dcc.Input(id='alpha', type='number', value=0.05, step=0.001)], width=1),
        dbc.Col([html.Label("表示範囲:"), dcc.DatePickerRange(
            id='date-range',
            start_date=two_weeks_ago,
            end_date=today,
            min_date_allowed=today - timedelta(days=365),
            max_date_allowed=today,
            display_format='YYYY-MM-DD'
        )], width=3),
    ], className="my-3"),
    dcc.Interval(id='refresh', interval=60*1000, n_intervals=0),
    html.Div(id='graphs')
])

@app.callback(
    Output('graphs','children'),
    [
        Input('interval','value'),
        Input('short-window','value'),
        Input('long-window','value'),
        Input('limit','value'),
        Input('alpha','value'),
        Input('date-range','start_date'),
        Input('date-range','end_date'),
        Input('refresh','n_intervals'),
    ]
)
def update_graphs(interval, short_w, long_w, limit, alpha, start_date, end_date, n):
    # データ取得 + バックテスト
    df = fetch_ohlcv('BTC', interval, limit)
    df_bt, pval = backtest_with_significance(df, short_w, long_w)

    # 因果推論フィルタ適用
    if not causal_filter(df_bt):
        df_bt['signal_change'] = 0

    # MLモデル訓練 + 次足確率予測 + SHAP
    rf, X_feat = train_rf(df)
    p_up = predict_next_prob(rf, X_feat)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_feat)
    logging.info(f"SHAP mean contributions: {np.mean(shap_vals, axis=1)}")

    # 日付範囲フィルタ
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    df_bt = df_bt.loc[(df_bt.index >= start) & (df_bt.index <= end)]

    # シグナル抽出
    buys  = df_bt[df_bt['signal_change'] == 1]
    sells = df_bt[df_bt['signal_change'] == -1]

    # 価格 & シグナル チャート
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
        x=df_bt.index, open=df_bt['open'], high=df_bt['high'],
        low=df_bt['low'], close=df_bt['close'], name='Price'
    ))
    fig1.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt['sma_short'], mode='lines',
        name=f'SMA{short_w}', line=dict(color='green', width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt['sma_long'], mode='lines',
        name=f'SMA{long_w}', line=dict(color='red', width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=buys.index, y=buys['close'], mode='markers',
        marker=dict(color='green', size=8), name='BUY'
    ))
    fig1.add_trace(go.Scatter(
        x=sells.index, y=sells['close'], mode='markers',
        marker=dict(color='red', size=8), name='SELL'
    ))
    prob_text = f"Next↑Prob: {p_up*100:.1f}%" if p_up is not None else "Next↑Prob: N/A"
    fig1.add_annotation(
        text=f"α={alpha:.3f}, p-value={pval:.3f}, {prob_text}",
        xref="paper", yref="paper", x=0.01, y=0.99,
        showarrow=False, bgcolor="rgba(255,255,255,0.7)"
    )
    fig1.update_xaxes(range=[start, end])
    fig1.update_layout(title="BTC Price & SMA Signals", yaxis_title="Price")

    # エクイティカーブ
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt['equity_curve'],
        mode='lines', name='Equity Curve'
    ))
    fig2.update_xaxes(range=[start, end])
    fig2.update_layout(title="Equity Curve", yaxis_title="Equity")

    return [dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
