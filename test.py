# quant_demo.py  -- EMAクロス＋出来高フィルタのBTC/USDデイリー戦略
import datetime as dt
import pandas as pd
import numpy as np
import ccxt        # pip install ccxt
import matplotlib.pyplot as plt

# --- 1. データ取得（過去2年） ---
exchange = ccxt.binance()
symbol   = 'BTC/USDT'
since    = exchange.parse8601((dt.datetime.utcnow() - dt.timedelta(days=730)).strftime('%Y-%m-%dT%H:%M:%SZ'))
bars     = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
cols     = ['timestamp','open','high','low','close','volume']
df       = pd.DataFrame(bars, columns=cols).set_index('timestamp')
df.index = pd.to_datetime(df.index, unit='ms')
df.sort_index(inplace=True)

# --- 2. テクニカル指標 ---
df['EMA_fast'] = df['close'].ewm(span=12).mean()
df['EMA_slow'] = df['close'].ewm(span=26).mean()
df['Volume_Z'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

# --- 3. シグナル生成 ---
df['signal'] = 0
long_cond  = (df['EMA_fast'] > df['EMA_slow']) & (df['Volume_Z'] > 0)
short_cond = (df['EMA_fast'] < df['EMA_slow']) & (df['Volume_Z'] > 0)
df.loc[long_cond,  'signal'] = 1
df.loc[short_cond, 'signal'] = -1
df['position'] = df['signal'].shift().fillna(0)   # 翌日寄付で建てる想定

# --- 4. バックテスト ---
df['ret']      = df['close'].pct_change()
df['strategy'] = df['position'] * df['ret']
df['eq_curve'] = (1 + df['strategy']).cumprod()

# --- 5. 成績サマリー ---
cagr = df['eq_curve'].iloc[-1] ** (365/len(df)) - 1
dd   = 1 - df['eq_curve'] / df['eq_curve'].cummax()
max_dd = dd.max()
sharpe = np.sqrt(365) * df['strategy'].mean() / df['strategy'].std()

summary = f"""
⚙  Strategy:   EMA 12/26 Cross + Vol Filter
📈  CAGR:       {cagr:.2%}
📉  Max DD:     {max_dd:.2%}
🔧  Sharpe:     {sharpe:.2f}
"""
print(summary)

# --- 6. プロット ---
plt.figure(figsize=(10,4))
plt.plot(df.index, df['eq_curve'], label='Strategy EQ', linewidth=1.2)
plt.title('Equity Curve')
plt.legend()
plt.grid(True)
plt.show()
