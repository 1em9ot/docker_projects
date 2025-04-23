#!/usr/bin/env python3
"""
Twitter 可視化ダッシュボード  ‑‑ transformers(BERT) による日本語感情分析版

 - Word Cloud（URL / RT / @ID などを除去、CJK フォント対応）
 - 時間帯別ツイート数＋平均感情
 - 感情カテゴリ分布
 - ツイート一覧（感情別ハイライト）

※ transformers と torch が未インストールの場合は ──────────────────
    pip install --upgrade transformers torch
   （初回実行時にモデル daigo/bert-base-japanese-sentiment を自動 DL）
"""

import os
import sys
import re
import json
import base64
from io import BytesIO
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar
from wordcloud import WordCloud, STOPWORDS
import dash
from dash import dcc, html, dash_table
import plotly.express as px
from plotly.subplots import make_subplots

# ―― transformers で事前学習済み日本語 BERT を利用 ──────────────────
from transformers import pipeline, Pipeline
# ── sys.path を最初に通す ───────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
# ここから下で data_sources.* を import しても OK
from data_sources.twitter_loader import load_twitter_data
from data_sources.pleasanter import fetch_daily_entries
# ── フォント（Noto Sans CJK）が入っている前提 ────────────
JP_FONT = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

# ── Word Cloud 用のクリーニング正規表現 ───────────────
URL_RE      = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE  = re.compile(r'@\w+')
RT_RE       = re.compile(r'(?i)(?:\bRT\b|ＲＴ)')
ASCII_ID_RE = re.compile(r'^[a-zA-Z0-9_]{3,}$')  # 英数字3文字以上
EXTRA_STOP  = {'https', 'http', 't', 'co', 'amp', 'rt'}

# 代表的な強い怒りワード（Negative→anger 判定）
ANGER_WORDS_RE = re.compile(r'(怒|殺|死ね|最悪|クソ|政府|ふざけ|許さ|ムカつ|嫌い)')

# ★ ここを追加・更新
MODEL_ID = os.getenv(
    "HF_MODEL_ID",                      # docker‑compose.yml で上書き可
    "jarvisx17/japanese-sentiment-analysis"
)

# ──────────────────────── ヘルパ関数群 ───────────────────────

def _clean_text(text: str) -> str:
    """URL・@ID・RT を除去して返す"""
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = RT_RE.sub(' ', text)
    return text


# ---------------------------------------------------------------------
#  ストップワード定義ブロック  ★ここを丸ごと追加 / 置換してください
# ---------------------------------------------------------------------
from wordcloud import STOPWORDS   # 英語既定 STOPWORDS

# ① URL・RT など既定で除外したい“英数字系”トークン
DEFAULT_STOP = {
    'https', 'http', 't', 'co', 'amp', 'rt'
}

# ② 追加ストップワードをファイルから読み込む（1 行 1 語）
STOPWORD_FILE = os.getenv("STOPWORD_PATH", "/teacher_data/stopwords.txt")
EXTRA_STOP: set[str] = set()
if os.path.isfile(STOPWORD_FILE):
    with open(STOPWORD_FILE, encoding="utf-8") as f:
        EXTRA_STOP = {
            ln.strip() for ln in f
            if ln.strip() and not ln.startswith('#')
        }

# ③ すべて合体させた最終セット（英語既定 STOPWORDS も加える）
STOP_SET = DEFAULT_STOP | EXTRA_STOP | set(STOPWORDS)

# ④ STOP_SET の語が連続しただけのトークン（例: ｗｗｗ, あああ）も除外
RE_STOP = re.compile(r'^(?:' + '|'.join(map(re.escape, STOP_SET)) + r')+$')

def _token_filter(tok: str) -> bool:
    """WordCloud で除外するトークン判定"""
    return ASCII_ID_RE.match(tok) or tok in STOP_SET or RE_STOP.match(tok)
# ---------------------------------------------------------------------

def sanitize_records(df: pd.DataFrame):
    """DataTable 用に JSON 変換・NaN 空文字化"""
    records: list[dict[str, Any]] = []
    for row in df.to_dict('records'):
        rec: dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                rec[k] = ''
                continue
            if isinstance(v, (list, dict)):
                rec[k] = json.dumps(v, ensure_ascii=False)
                continue
            if not is_scalar(v):
                try:
                    rec[k] = json.dumps(v.tolist() if hasattr(v, 'tolist') else list(v), ensure_ascii=False)
                except Exception:
                    rec[k] = str(v)
                continue
            rec[k] = v
        records.append(rec)
    return records


def generate_wordcloud(series: pd.Series) -> str | None:
    raw = " ".join(series.dropna().astype(str))
    if not raw.strip():
        return None
    cleaned = _clean_text(raw)

    freq = WordCloud(stopwords=set(), regexp=r"[\wぁ-んァ-ン一-龥]+")\
             .process_text(cleaned)
    freq = {tok: cnt for tok, cnt in freq.items() if not _token_filter(tok)}
    if not freq:
        return None

    stops = STOPWORDS.union(EXTRA_STOP)
    try:
        wc = WordCloud(width=800, height=400, background_color='white',
                       font_path=JP_FONT, stopwords=stops).generate_from_frequencies(freq)
    except OSError:
        # サーバに日本語フォントが無い場合はデフォルトフォントで描画
        wc = WordCloud(width=800, height=400, background_color='white',
                       stopwords=stops).generate_from_frequencies(freq)

    buf = BytesIO(); wc.to_image().save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ─────────────────── transformers 感情分類パイプライン ───────────────────
try:
    SENTIMENT_PL: Pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device_map="auto"
    )
except Exception as e:
    print("[WARNING] transformers pipeline 初期化に失敗:", e)
    SENTIMENT_PL = None
    
def _map_label(lbl: str, text: str) -> str:
    """positive / negative → calm / sad・anger"""
    l = lbl.lower()
    if l.startswith("pos"):
        return "calm"
    # negative → 怒語チェックで anger / sad
    return "anger" if ANGER_WORDS_RE.search(text) else "sad"

def classify_emotion(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or SENTIMENT_PL is None:
        return "neutral"
    result = SENTIMENT_PL(text[:128])[0]   # {'label':'negative','score':0.93}
    return _map_label(result["label"], text)


# ─────────────────── Dash アプリ本体 ───────────────────

def main():
    df = load_twitter_data()
    # ───────────────── Pleasanter データ ─────────────────
    df_pl = fetch_daily_entries()
    if not df_pl.empty and 'content' in df_pl.columns:
        # JST → hour 抽出（twitter_loader と揃える）
        df_pl['hour'] = df_pl['created_at'].dt.hour
        pl_hour_stats = (df_pl.groupby('hour')
                           .size()
                           .reindex(range(24), fill_value=0)
                           .reset_index(name='record_count'))

        # ★ グラフ（時間帯別レコード数）
        fig_pl_hour = px.bar(
            pl_hour_stats, x='hour', y='record_count',
            title='Pleasanter 時間帯別レコード数',
            labels={'hour':'時間帯 (時)', 'record_count':'レコード数'},
            color='record_count', color_continuous_scale='Blues'
        )
    else:
        fig_pl_hour = None

    if df.empty or 'content' not in df.columns:
        print("No Twitter data.")
        return

    # 感情付与（transformers）
    df['emotion'] = df['content'].apply(classify_emotion)

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

    # グラフ: 時間帯別
    fig_hour = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hour.add_bar(x=hour_stats['hour'], y=hour_stats['tweet_count'],
                     name='ツイート数', marker_color='steelblue')
    fig_hour.add_scatter(x=hour_stats['hour'], y=hour_stats['avg_sentiment'],
                         name='平均感情値', mode='lines+markers',
                         line=dict(color='orangered'), secondary_y=True)
    fig_hour.update_xaxes(title='時間帯 (時)')
    fig_hour.update_yaxes(title_text='ツイート数', secondary_y=False)
    fig_hour.update_yaxes(title_text='平均感情値', range=[-1, 1], secondary_y=True)
    fig_hour.update_layout(title='時間帯別ツイート数と平均感情値')

    # グラフ: 感情割合
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
            # ------ Pleasanter セクション -----------------
            html.H2("Pleasanter 時間帯別レコード数"),
            dcc.Graph(figure=fig_pl_hour) if fig_pl_hour else html.P("※レコードなし"),

            html.H2("Pleasanter レコード一覧"),
            dash_table.DataTable(
                columns=[{'name':'日時','id':'created_at'},
                         {'name':'タイトル','id':'title'},
                         {'name':'内容','id':'content'}],
                data=sanitize_records(
                    df_pl[['created_at','title','content']] if not df_pl.empty else pd.DataFrame()
                ),
                page_action='none',
                style_table={'height':'500px','overflowY':'auto','overflowX':'auto'},
                style_cell={'textAlign':'left','padding':'4px'},
                style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'},
                virtualization=True
            ),
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
                columns=[{'name': '日時',   'id': 'created_at'},
                         {'name': '内容',   'id': 'content'},
                         {'name': 'emotion','id': 'emotion'}],
                data=sanitize_records(df[['created_at', 'content', 'emotion']]),
                page_action='none',  # ページネーション無効
                style_table={'height': '500px', 'overflowY': 'auto', 'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '4px'},
                style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                hidden_columns=['emotion'],
                style_data_conditional=[
                    {'if': {'filter_query': '{emotion} = "anger"',  'column_id': 'content'}, 'backgroundColor': '#ffcccc'},
                    {'if': {'filter_query': '{emotion} = "sad"',    'column_id': 'content'}, 'backgroundColor': '#cce5ff'},
                    {'if': {'filter_query': '{emotion} = "calm"',   'column_id': 'content'}, 'backgroundColor': '#ccffcc'},
                ],
                virtualization=True  # 行数が多くても軽快
            )
        ]
    )

    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__ == '__main__':
    main()


