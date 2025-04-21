#!/usr/bin/env python3
"""
Twitter データ専用の Dash ダッシュボード。
ワードクラウド / 時間帯別投稿数＋平均感情 / 感情カテゴリ分布 / ツイート一覧 を表示。
"""

import os
import sys
import json
import base64
from io import BytesIO
import pandas as pd
from pandas.api.types import is_scalar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── パス設定 ─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from data_sources.twitter_loader import load_twitter_data


# ── ユーティリティ ──────────────────────────────
def sanitize_records(df: pd.DataFrame):
    """DataTable 用に NaN→空文字・非スカラー→JSON 文字列化"""
    records = []
    for row in df.to_dict('records'):
        rec = {}
        for k, v in row.items():
            # NaN→''
            try:
                na = pd.isna(v)
                if (hasattr(na, "all") and na.all()) or (isinstance(na, bool) and na):
                    rec[k] = ''
                    continue
            except Exception:
                pass
            # list / dict → JSON
            if isinstance(v, (list, dict)):
                rec[k] = json.dumps(v, ensure_ascii=False)
                continue
            # 非スカラー → JSON or str
            if not is_scalar(v):
                try:
                    rec[k] = json.dumps(v.tolist() if hasattr(v, 'tolist') else list(v), ensure_ascii=False)
                except Exception:
                    rec[k] = str(v)
                continue
            # そのまま
            rec[k] = v
        records.append(rec)
    return records


def cluster_emotion_keywords(df: pd.DataFrame, top_k: int = 50) -> dict:
    """ツイート本文から単語頻度上位 top_k を 3 クラスタに分け、怒り / 悲しみ / 落ち着き に割当"""
    texts = df['content'].dropna().astype(str)
    if texts.empty:
        return {'anger': [], 'sad': [], 'calm': []}

    vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=top_k)
    X = vec.fit_transform(texts)
    words = vec.get_feature_names_out()
    counts = X.toarray().sum(axis=0).reshape(-1, 1)

    labels = KMeans(n_clusters=3, random_state=42).fit_predict(counts)
    clusters = {0: [], 1: [], 2: []}
    for w, l in zip(words, labels):
        clusters[l].append(w)

    return {'anger': clusters[0], 'sad': clusters[1], 'calm': clusters[2]}


def classify_emotion(text: str, emo_dict: dict) -> str:
    """本文に感情語彙が含まれていればカテゴリ名を返す。なければ neutral"""
    if not isinstance(text, str):
        return 'neutral'
    for label, words in emo_dict.items():
        if any(w in text for w in words):
            return label
    return 'neutral'


def generate_wordcloud_image(text_series: pd.Series) -> str | None:
    """Series を結合してワードクラウドを生成。フォント読めない環境は自動フォールバック"""
    text = " ".join(text_series.dropna().astype(str))
    if not text.strip():
        return None

    # 1st: DejaVu
    try:
        wc = WordCloud(
            width=800, height=400, background_color='white',
            stopwords=STOPWORDS, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ).generate(text)
    except OSError:
        # fallback: default font
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)

    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── メイン処理 ────────────────────────────────
def main():
    df = load_twitter_data()
    if df.empty or 'content' not in df.columns:
        print("No Twitter data.")
        return

    # 感情カテゴリ付与
    emo_dict = cluster_emotion_keywords(df)
    df['emotion'] = df['content'].apply(lambda t: classify_emotion(t, emo_dict))

    # 日時処理
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['hour'] = df['created_at'].dt.hour

    # 時間帯別集計
    score_map = {'anger': -1, 'sad': -0.5, 'neutral': 0, 'calm': 1}
    df['score'] = df['emotion'].map(score_map)
    hour_stats = (df.groupby('hour')
                    .agg(tweet_count=('content', 'count'),
                         avg_sentiment=('score', 'mean'))
                    .reindex(range(24), fill_value=0)
                    .reset_index())

    # 感情カテゴリ分布
    emo_cnt = df['emotion'].value_counts().reset_index()
    emo_cnt.columns = ['emotion', 'count']

    # ワードクラウド
    wc_uri = generate_wordcloud_image(df['content'])

    # --- グラフ ---
    fig_hour = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hour.add_bar(
        x=hour_stats['hour'], y=hour_stats['tweet_count'],
        name='ツイート数', marker_color='steelblue'
    )
    fig_hour.add_scatter(
        x=hour_stats['hour'], y=hour_stats['avg_sentiment'],
        name='平均感情値', mode='lines+markers', line=dict(color='orangered'),
        secondary_y=True
    )
    fig_hour.update_xaxes(title='時間帯 (時)')
    fig_hour.update_yaxes(title='ツイート数', secondary_y=False)
    fig_hour.update_yaxes(title='平均感情値', range=[-1, 1], secondary_y=True)
    fig_hour.update_layout(title='時間帯別ツイート数と平均感情値')

    fig_emotion = px.pie(
        emo_cnt, names='emotion', values='count', title='感情カテゴリ分布',
        color='emotion',
        color_discrete_map={'anger': '#ff6666', 'sad': '#66b3ff',
                            'calm': '#99ff99', 'neutral': '#cccccc'}
    )

    # --- Dash Layout ---
    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={'padding': '20px'},
        children=[
            html.H1("Health Dashboard"),
            html.H2("Twitterワードクラウド"),
            html.Img(
                src=wc_uri,
                style={'width': '100%', 'maxHeight': '400px', 'objectFit': 'contain'}
            ) if wc_uri else html.P("※ツイートデータがありません"),
            html.H2("時間帯別投稿数と平均感情値"),
            dcc.Graph(figure=fig_hour),
            html.H2("感情カテゴリ分布"),
            dcc.Graph(figure=fig_emotion),
            html.H2("ツイート一覧"),
            dash_table.DataTable(
                columns=[
                    {'name': '日時', 'id': 'created_at'},
                    {'name': '内容', 'id': 'content'},
                    {'name': 'emotion', 'id': 'emotion'}
                ],
                data=sanitize_records(df[['created_at', 'content', 'emotion']]),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '4px'},
                style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{emotion} = "anger"',   'column_id': 'content'}, 'backgroundColor': '#ffcccc'},
                    {'if': {'filter_query': '{emotion} = "sad"',     'column_id': 'content'}, 'backgroundColor': '#cce5ff'},
                    {'if': {'filter_query': '{emotion} = "calm"',    'column_id': 'content'}, 'backgroundColor': '#ccffcc'},
                ],
                hidden_columns=['emotion']
            )
        ]
    )

    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__ == '__main__':
    main()
