#!/usr/bin/env python3
"""
Twitter 可視化ダッシュボード
 - Word Cloud（URL / RT / @ID などを除去、CJK フォント対応）
 - 時間帯別ツイート数＋平均感情
 - 感情カテゴリ分布
 - ツイート一覧（感情別ハイライト）
"""

import os
import sys
import re
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

# ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
from data_sources.twitter_loader import load_twitter_data  # 自作ローダー

# ── フォント（Noto Sans CJK）が入っている前提 ────────────
JP_FONT = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

# ── Word Cloud 用のクリーニング正規表現 ───────────────
URL_RE      = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE  = re.compile(r'@\w+')
RT_RE       = re.compile(r'(?i)(?:\bRT\b|ＲＴ)')
ASCII_ID_RE = re.compile(r'^[a-zA-Z0-9_]{3,}$')  # 英数字3文字以上

EXTRA_STOP = {'https', 'http', 't', 'co', 'amp', 'rt'}

def _clean_text(text: str) -> str:
    """URL・@ID・RT を除去して返す"""
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = RT_RE.sub(' ', text)
    return text

def _token_filter(token: str) -> bool:
    """Stopwords 判定関数（True なら捨てる）"""
    return token in EXTRA_STOP or ASCII_ID_RE.match(token)

# ── DataTable 用ヘルパ ────────────────────────────────
def sanitize_records(df: pd.DataFrame):
    records = []
    for row in df.to_dict('records'):
        rec = {}
        for k, v in row.items():
            # NaN → 空文字
            try:
                if pd.isna(v):
                    rec[k] = ''
                    continue
            except Exception:
                pass
            # list/dict → JSON
            if isinstance(v, (list, dict)):
                rec[k] = json.dumps(v, ensure_ascii=False)
                continue
            # 非スカラー → JSON / str
            if not is_scalar(v):
                try:
                    rec[k] = json.dumps(v.tolist() if hasattr(v, 'tolist') else list(v), ensure_ascii=False)
                except Exception:
                    rec[k] = str(v)
                continue
            rec[k] = v
        records.append(rec)
    return records

# ── 感情クラスタリング ──────────────────────────────
def cluster_emotion_keywords(df: pd.DataFrame, top_k=50) -> dict:
    txt = df['content'].dropna().astype(str)
    if txt.empty:
        return {'anger': [], 'sad': [], 'calm': []}
    vec = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=top_k)
    X = vec.fit_transform(txt)
    words = vec.get_feature_names_out()
    counts = X.toarray().sum(axis=0).reshape(-1, 1)
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(counts)
    cl = {0: [], 1: [], 2: []}
    for w, l in zip(words, labels):
        cl[l].append(w)
    return {'anger': cl[0], 'sad': cl[1], 'calm': cl[2]}

def classify_emotion(text: str, emo_dict: dict) -> str:
    if not isinstance(text, str):
        return 'neutral'
    for lab, words in emo_dict.items():
        if any(w in text for w in words):
            return lab
    return 'neutral'

# ── WordCloud 生成 ────────────────────────────────────
def generate_wordcloud(series: pd.Series) -> str | None:
    raw = " ".join(series.dropna().astype(str))
    if not raw.strip():
        return None
    cleaned = _clean_text(raw)

    # WordCloud.tokenize の代わりに自前で頻度辞書を生成してフィルタ
    wc_tmp = WordCloud(stopwords=set(), regexp=r"[\wぁ-んァ-ン一-龥]+")
    freq = wc_tmp.process_text(cleaned)
    freq = {tok: cnt for tok, cnt in freq.items() if not _token_filter(tok)}
    if not freq:
        return None

    stops = STOPWORDS.union(EXTRA_STOP)
    try:
        wc = WordCloud(width=800, height=400, background_color='white',
                       font_path=JP_FONT, stopwords=stops).generate_from_frequencies(freq)
    except OSError:
        wc = WordCloud(width=800, height=400, background_color='white',
                       stopwords=stops).generate_from_frequencies(freq)

    buf = BytesIO(); wc.to_image().save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Dash アプリ ───────────────────────────────────────
def main():
    df = load_twitter_data()
    if df.empty or 'content' not in df.columns:
        print("No Twitter data.")
        return

    # 感情付与
    emo_dict = cluster_emotion_keywords(df)
    df['emotion'] = df['content'].apply(lambda x: classify_emotion(x, emo_dict))

    # 日時 & 時間帯
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['hour'] = df['created_at'].dt.hour
    score_map = {'anger': -1, 'sad': -0.5, 'neutral': 0, 'calm': 1}
    df['score'] = df['emotion'].map(score_map)

    # 時間帯集計
    hour_stats = (df.groupby('hour')
                    .agg(tweet_count=('content', 'count'),
                         avg_sentiment=('score', 'mean'))
                    .reindex(range(24), fill_value=0)
                    .reset_index())

    # 感情分布
    emo_cnt = df['emotion'].value_counts().reset_index()
    emo_cnt.columns = ['emotion', 'count']

    # ワードクラウド
    wc_uri = generate_wordcloud(df['content'])

    # グラフ
    fig_hour = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hour.add_bar(x=hour_stats['hour'], y=hour_stats['tweet_count'],
                     name='ツイート数', marker_color='steelblue')
    fig_hour.add_scatter(x=hour_stats['hour'], y=hour_stats['avg_sentiment'],
                         name='平均感情値', mode='lines+markers',
                         line=dict(color='orangered'), secondary_y=True)
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

    # Dash Layout
    app = dash.Dash(__name__)
    app.layout = html.Div(
        style={'padding': '20px'},
        children=[
            html.H1("Health Dashboard"),
            html.H2("Twitterワードクラウド"),
            html.Img(src=wc_uri, style={'width': '100%', 'maxHeight': '400px'})
                if wc_uri else html.P("※ツイートデータがありません"),
            html.H2("時間帯別投稿数と平均感情値"),
            dcc.Graph(figure=fig_hour),
            html.H2("感情カテゴリ分布"),
            dcc.Graph(figure=fig_emotion),
            html.H2("ツイート一覧"),
            dash_table.DataTable(
                columns=[{'name': '日時', 'id': 'created_at'},
                         {'name': '内容', 'id': 'content'},
                         {'name': 'emotion', 'id': 'emotion'}],
                data=sanitize_records(df[['created_at', 'content', 'emotion']]),
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '4px'},
                style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                hidden_columns=['emotion'],
                style_data_conditional=[
                    {'if': {'filter_query': '{emotion} = \"anger\"', 'column_id': 'content'}, 'backgroundColor': '#ffcccc'},
                    {'if': {'filter_query': '{emotion} = \"sad\"',   'column_id': 'content'}, 'backgroundColor': '#cce5ff'},
                    {'if': {'filter_query': '{emotion} = \"calm\"',  'column_id': 'content'}, 'backgroundColor': '#ccffcc'},
                ]
            )
        ]
    )

    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__ == '__main__':
    main()

