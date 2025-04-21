#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime

import pandas as pd
import collections
from pandas.api.types import is_scalar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import base64
from io import BytesIO
from wordcloud import WordCloud

# プロジェクトルートをパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import dash
from dash import dcc, html, dash_table
import plotly.express as px

from strategies.predict import predict


def sanitize_records(df: pd.DataFrame):
    """
    DataTable用に各セルをJSON-シリアライズし、
    NAは空文字に、配列やシリーズにも安全に対応します。
    """
    records = []
    for row in df.to_dict('records'):
        new_row = {}
        for k, v in row.items():
            # 全体がNAか判定（配列はall(), 単一値はbool）
            try:
                na_mask = pd.isna(v)
                if hasattr(na_mask, 'all') and na_mask.all():
                    new_row[k] = ''
                    continue
                if isinstance(na_mask, bool) and na_mask:
                    new_row[k] = ''
                    continue
            except Exception:
                pass
            # dict/listをJSON化
            if isinstance(v, (dict, list)):
                new_row[k] = json.dumps(v, ensure_ascii=False)
                continue
            # 非スカラーをtolist()→JSON
            if not is_scalar(v):
                try:
                    seq = v.tolist() if hasattr(v, 'tolist') else list(v)
                    new_row[k] = json.dumps(seq, ensure_ascii=False)
                except Exception:
                    new_row[k] = str(v)
                continue
            # その他はそのまま
            new_row[k] = v
        records.append(new_row)
    return records


def cluster_emotion_keywords(df: pd.DataFrame, top_k=50):
    df_tw = df[df['title'].str.contains("Twitter", na=False)]
    if df_tw.empty:
        return {'anger': [], 'sad': [], 'calm': []}
    texts = df_tw['content'].dropna().astype(str)
    cnt = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=top_k)
    X = cnt.fit_transform(texts)
    words = cnt.get_feature_names_out()
    counts = X.toarray().sum(axis=0).reshape(-1, 1)
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(counts)
    clusters = {i: [] for i in range(3)}
    for w, l in zip(words, labels):
        clusters[l].append(w)
    return {'anger': clusters[0], 'sad': clusters[1], 'calm': clusters[2]}


def classify_emotion(text: str, emo_dict: dict) -> str:
    if not isinstance(text, str):
        return 'neutral'
    for label, words in emo_dict.items():
        if any(w in text for w in words):
            return label
    return 'neutral'


def generate_wordcloud_image(df: pd.DataFrame) -> str:
    df_tw = df[df['title'].str.contains("Twitter", na=False)]
    if df_tw.empty:
        return None
    text = ' '.join(df_tw['content'].dropna().astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white',
                   font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    img = wc.generate(text)
    buf = BytesIO()
    img.to_image().save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{data}'


def main():
    df = predict()
    if df.empty:
        print("No data available to display.")
        return

    emo_dict = cluster_emotion_keywords(df)
    df['emotion'] = df['content'].apply(lambda t: classify_emotion(t, emo_dict))
    table = sanitize_records(df)

    colors = {'anger':'#ffcccc','sad':'#cce5ff','calm':'#ccffcc','neutral':'#ffffff'}
    style_cond = [
        {'if':{'filter_query':f'{{emotion}} = "{emo}"','column_id':'content'},
         'backgroundColor':col}
        for emo, col in colors.items()
    ]

    df['date'] = pd.to_datetime(df['date'])
    last_m = int(df['date'].dt.month.max())
    df_m = df[df['date'].dt.month == last_m].copy()
    df_m.loc[:, 'day'] = df_m['date'].dt.day
    bar = df_m.groupby('day')['pred_energy'].mean().reset_index()
    fig_bar = px.bar(bar, x='day', y='pred_energy',
                      labels={'day':'Day','pred_energy':'Energy'},
                      title=f'Month {last_m} Daily Energy',
                      range_y=[bar['pred_energy'].min(), bar['pred_energy'].max()])

    try:
        import plotly_calplot
        cal = df_m.copy()
        cal['date'] = cal['date'].dt.floor('D')
        cal = cal.groupby('date')['pred_energy'].mean().reset_index()
        fig_cal = plotly_calplot.calplot(cal, x='date', y='pred_energy',
                                         colorscale='RdYlGn', gap=1, showscale=True)
    except ImportError:
        piv = df.pivot_table(index=df['date'].dt.month,
                              columns=df['date'].dt.day,
                              values='pred_energy',
                              aggfunc='mean')
        fig_cal = px.imshow(piv, origin='lower', labels={'color':'Energy'}, title='Heatmap')

    wc_url = generate_wordcloud_image(df)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Health Dashboard"),
        dcc.Graph(figure=fig_bar),
        dcc.Graph(figure=fig_cal),
        html.H2("Raw Entries"),
        dash_table.DataTable(
            columns=[{'name':c,'id':c} for c in df.columns],
            data=table,
            page_size=10,
            style_table={'overflowX':'auto'},
            style_cell={'textAlign':'left','padding':'4px'},
            style_header={'backgroundColor':'#f0f0f0','fontWeight':'bold'},
            style_data_conditional=style_cond
        ),
        html.H2("Twitter Word Cloud"),
        html.Img(src=wc_url, style={'width':'100%','maxHeight':'400px','objectFit':'contain'})
        if wc_url else html.P("No Twitter data available")
    ], style={'padding':'20px'})

    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__=='__main__':
    main()
