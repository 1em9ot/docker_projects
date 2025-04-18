import os
import datetime
import logging
import socket
import subprocess
import cProfile
import pstats

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests
import ta
from sklearn.ensemble import RandomForestClassifier
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------------------------------
# Logging andファイル出力関連ユーティリティ
# --------------------------------------------------
def setup_logging(output_dir: str) -> str:
    """
    ロギングの設定を行い、ログファイルのパスを返す

    Args:
        output_dir (str): ログ出力先ディレクトリのパス

    Returns:
        str: ログファイルのパス
    """
    log_path = os.path.join(output_dir, 'process.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    return log_path

def create_output_dir(base_dir: str, debug: bool = False) -> str:
    """
    出力ディレクトリを作成する

    Args:
        base_dir (str): ベースとなるディレクトリ
        debug (bool, optional): デバッグモードの場合は固定ディレクトリを作成。 Defaults to False.

    Returns:
        str: 作成されたディレクトリのパス
    """
    if debug:
        path = os.path.join(base_dir, 'debug_output')
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(base_dir, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

# --------------------------------------------------
# ネットワーク・プロセス管理関連ユーティリティ
# --------------------------------------------------
def find_free_port() -> int:
    """
    利用可能な空きポートを返す

    Returns:
        int: 空きポート番号
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def open_new_browser_window(url: str) -> int:
    """
    指定したURLを新しいChromeウィンドウで開き、プロセスIDを返す

    Args:
        url (str): 開くURL

    Returns:
        int: 起動したプロセスのID
    """
    chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"  # 必要に応じてパスを修正
    process = subprocess.Popen([chrome_path, '--new-window', url])
    return process.pid

def close_browser_window(pid: int) -> None:
    """
    指定したプロセスIDのブラウザウィンドウを閉じる

    Args:
        pid (int): プロセスID
    """
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def adjust_workers(current_workers: int, max_workers: int, cpu_threshold: float = 90, mem_threshold: float = 90) -> int:
    """
    システム負荷に応じてワーカー数を調整する

    Args:
        current_workers (int): 現在のワーカー数
        max_workers (int): 最大許容ワーカー数
        cpu_threshold (float, optional): CPU使用率の閾値. Defaults to 90.
        mem_threshold (float, optional): メモリ使用率の閾値. Defaults to 90.

    Returns:
        int: 調整後のワーカー数
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    logging.info(f"CPU使用率: {cpu_usage}%, メモリ使用率: {mem_usage}%")

    if cpu_usage > cpu_threshold or mem_usage > mem_threshold:
        return max(1, current_workers - 1)
    elif cpu_usage < 70 and mem_usage < 70 and current_workers < max_workers:
        return current_workers + 1
    return current_workers

# --------------------------------------------------
# データ取得・処理関連関数
# --------------------------------------------------
def get_crypto_data(symbol: str, interval: str = '1m', limit: int = 1000, currency: str = 'JPY') -> pd.DataFrame:
    """
    Binance APIから仮想通貨のデータを取得し、DataFrameで返す

    Args:
        symbol (str): 仮想通貨シンボル（例: 'BTC', 'ETH'）
        interval (str, optional): データの間隔. Defaults to '1m'.
        limit (int, optional): 取得するデータ数. Defaults to 1000.
        currency (str, optional): 価格通貨 ('JPY'または'USD'). Defaults to 'JPY'.

    Returns:
        pd.DataFrame: 取得したデータ（DataFrame形式）
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
    start_time = datetime.datetime.now()
    response = requests.get(url, timeout=5)
    response_time = (datetime.datetime.now() - start_time).total_seconds()

    if response_time > 1:
        logging.warning(f"Data retrieval took too long: {response_time} seconds")

    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    if currency == 'JPY':
        # タイムゾーン変換と為替レート適用
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
        rate = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5).json()['rates']['JPY']
        df['close'] *= rate

    df.set_index('timestamp', inplace=True)
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    テクニカル指標（SMA, EMA, Bollinger Bands, RSI, MACD）を計算してDataFrameに追加する

    Args:
        df (pd.DataFrame): 価格データを含むDataFrame

    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA'] = ta.trend.ema_indicator(df['close'], window=20)
    df['Upper_BB'] = ta.volatility.bollinger_hband(df['close'])
    df['Lower_BB'] = ta.volatility.bollinger_lband(df['close'])
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['close'])
    return df

def create_features_and_labels(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    特徴量とラベルを作成する。テクニカル指標の計算、将来のリターンとシグナルの生成を行う。

    Args:
        df (pd.DataFrame): 仮想通貨の価格データ

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 特徴量DataFrameとシグナルラベルSeries
    """
    df = calculate_technical_indicators(df)
    df['future_return'] = df['close'].shift(-1) - df['close']
    df['signal'] = (df['future_return'] > 0).astype(int)
    df.dropna(inplace=True)
    features = df[['SMA', 'EMA', 'Upper_BB', 'Lower_BB', 'RSI', 'MACD', 'MACD_signal']]
    labels = df['signal']
    return features, labels

def train_model(features: pd.DataFrame, labels: pd.Series) -> RandomForestClassifier:
    """
    ランダムフォレストモデルを訓練する

    Args:
        features (pd.DataFrame): 特徴量
        labels (pd.Series): ラベル

    Returns:
        RandomForestClassifier: 訓練済みモデル
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model

def predict_signals(model: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """
    モデルによるシグナル予測をDataFrameに追加する

    Args:
        model (RandomForestClassifier): 訓練済みモデル
        df (pd.DataFrame): 入力のDataFrame

    Returns:
        pd.DataFrame: 予測シグナルを追加したDataFrame
    """
    feature_cols = ['SMA', 'EMA', 'Upper_BB', 'Lower_BB', 'RSI', 'MACD', 'MACD_signal']
    features = df[feature_cols].dropna()
    df = df.loc[features.index]
    df['predicted_signal'] = model.predict(features)
    return df

def create_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """
    仮想通貨のチャート（ローソク足、各種テクニカル指標、シグナル）を作成する

    Args:
        df (pd.DataFrame): データを含むDataFrame
        title (str): チャートタイトル（例: 'BTC', 'ETH'）

    Returns:
        go.Figure: 作成されたFigureオブジェクト
    """
    fig = go.Figure()
    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=f'{title} Candlestick'
    ))
    # テクニカル指標
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name=f'{title} SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name=f'{title} EMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], mode='lines', name=f'{title} Upper Bollinger Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], mode='lines', name=f'{title} Lower Bollinger Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'{title} RSI', yaxis='y2'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name=f'{title} MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name=f'{title} MACD Signal'))

    # シグナルプロット
    buy_signals = df[df['predicted_signal'] == 1]
    sell_signals = df[df['predicted_signal'] == 0]
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['close'],
        mode='markers',
        marker=dict(color='green', size=10),
        name='Buy Signal'
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['close'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Sell Signal'
    ))

    fig.update_layout(
        title=f"{title} Price Data",
        yaxis_title='Price',
        xaxis_title='Date',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False
    )
    return fig

# --------------------------------------------------
# Dashアプリ設定とコールバック
# --------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Cryptocurrency Dashboard"), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='btc-graph'), width=6),
        dbc.Col(dcc.Graph(id='eth-graph'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id='currency-switch',
            options=[
                {'label': 'JPY', 'value': 'JPY'},
                {'label': 'USD', 'value': 'USD'}
            ],
            value='JPY',
            labelStyle={'display': 'inline-block'}
        ), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Interval(
            id='interval-component',
            interval=60000,  # 1分ごとに更新
            n_intervals=0
        )),
    ]),
])

@app.callback(
    [Output('btc-graph', 'figure'), Output('eth-graph', 'figure')],
    [Input('interval-component', 'n_intervals'), Input('currency-switch', 'value')]
)
def update_graph(n_intervals: int, currency: str) -> (go.Figure, go.Figure):
    """
    Dashコールバック：定期的にBTC/ETHのグラフを更新する

    Args:
        n_intervals (int): インターバルコンポーネントからの更新カウント
        currency (str): 選択された通貨 ('JPY'または 'USD')

    Returns:
        Tuple[go.Figure, go.Figure]: BTCおよびETHのグラフ
    """
    btc_df = get_crypto_data('BTC', currency=currency)
    eth_df = get_crypto_data('ETH', currency=currency)

    btc_features, btc_labels = create_features_and_labels(btc_df)
    eth_features, eth_labels = create_features_and_labels(eth_df)

    btc_model = train_model(btc_features, btc_labels)
    eth_model = train_model(eth_features, eth_labels)

    btc_df = predict_signals(btc_model, btc_df)
    eth_df = predict_signals(eth_model, eth_df)

    btc_fig = create_figure(btc_df, 'BTC')
    eth_fig = create_figure(eth_df, 'ETH')

    return btc_fig, eth_fig

# --------------------------------------------------
# メイン実行部
# --------------------------------------------------
def main() -> None:
    """
    アプリケーションのエントリーポイント
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_mode = True
    output_dir = create_output_dir(script_dir, debug=debug_mode)
    log_path = setup_logging(output_dir)

    # プロファイリング開始
    profiler = cProfile.Profile()
    profiler.enable()

    # 空いているポートを取得し、ブラウザでアプリを起動
    free_port = find_free_port()
    url = f"http://127.0.0.1:{free_port}"
    pid = open_new_browser_window(url)

    app.run_server(debug=True,use_reloader=False, port=free_port)

    # プロファイリング終了・結果保存
    profiler.disable()
    profile_path = os.path.join(output_dir, 'profile.prof')
    with open(profile_path, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()

    logging.info("Application finished successfully.")

    # アプリ終了時にブラウザウィンドウを閉じる
    close_browser_window(pid)

if __name__ == '__main__':
    main()
