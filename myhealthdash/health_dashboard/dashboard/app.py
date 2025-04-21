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
    DataTable用に各セルをJSON-シリアライズし、NaNを空文字に変換。
    非スカラー型(配列など)も安全に文字列化します。
    """
    records = []
    for row in df.to_dict('records'):
        new_row = {}
        for k, v in row.items():
            # リストや辞書はJSON文字列に変換
            if isinstance(v, (list, dict)):
                new_row[k] = json.dumps(v, ensure_ascii=False)
            # 非スカラーはtolist()可能ならリスト化、不可なら文字列化
            elif not is_scalar(v):
                try:
                    serializable = v.tolist() if hasattr(v, 'tolist') else v
                    new_row[k] = json.dumps(serializable, ensure_ascii=False)
                except Exception:
                    new_row[k] = str(v)
            # スカラーかつNaNなら空文字
            elif pd.isna(v):
                new_row[k] = ''
            else:
                new_row[k] = v
        records.append(new_row)
    return records


def cluster_emotion_keywords(df: pd.DataFrame, top_k=50):
    """
    Twitter投稿から頻出単語を抽出し、3クラスタに分けてemotionカテゴリ語彙を生成
    """
    df_tw = df[df['title'].str.contains("Twitter", na=False)]
    if df_tw.empty:
        return {'anger': [], 'sad': [], 'calm': []}
    texts = df_tw['content'].dropna().astype(str)
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=top_k)
    X = vectorizer.fit_transform(texts)
    word_counts = X.toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()

    # KMeansで3クラスタ
    labels = KMeans(n_clusters=3, n_init='auto', random_state=42).fit_predict(word_counts.reshape(-1, 1))
    clusters = collections.defaultdict(list)
    for w, lbl in zip(words, labels):
        clusters[lbl].append(w)
    # 暫定的にmapping
    return {'anger': clusters.get(0, []), 'sad': clusters.get(1, []), 'calm': clusters.get(2, [])}


def classify_emotion(text: str, emotion_dict: dict) -> str:
    """
    テキスト中にemotion_dict内の単語があれば該当ラベルを返す
    """
    if not isinstance(text, str):
        return 'neutral'
    for label, words in emotion_dict.items():
        for w in words:
            if w and w in text:
                return label
    return 'neutral'


def generate_wordcloud_image(df: pd.DataFrame) -> str:
    """
    Twitter投稿の語彙からワードクラウド画像を生成し、base64データURLを返す
    """
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

    # 感情辞書を生成・適用
    emo_dict = cluster_emotion_keywords(df)
    df['emotion'] = df['content'].apply(lambda t: classify_emotion(t, emo_dict))
    table = sanitize_records(df)

    # テーブルのスタイル
    colors = {'anger':'#ffcccc','sad':'#cce5ff','calm':'#ccffcc','neutral':'#ffffff'}
    style_cond = [
        {'if':{'filter_query':f'{{emotion}} = "{emo}"','column_id':'content'},
         'backgroundColor':col}
        for emo,col in colors.items()
    ]

    # 月別エナジーバー
    df['date'] = pd.to_datetime(df['date'])
    last_m = int(df['date'].dt.month.max())
    df_m = df[df['date'].dt.month==last_m].copy()
    df_m['day'] = df_m['date'].dt.day
    bar = df_m.groupby('day')['pred_energy'].mean().reset_index()
    fig_bar = px.bar(bar,x='day',y='pred_energy',labels={'day':'Day','pred_energy':'Energy'},
                      title=f'Month {last_m} Daily Energy',range_y=[bar['pred_energy'].min(),bar['pred_energy'].max()])

    # カレンダーヒート
    try:
        import plotly_calplot
        cal = df_m.copy()
        cal['date']=cal['date'].dt.floor('D')
        cal = cal.groupby('date')['pred_energy'].mean().reset_index()
        fig_cal = plotly_calplot.calplot(cal,x='date',y='pred_energy',colorscale='RdYlGn',gap=1,showscale=True)
    except:
        piv = df.pivot_table(index=df['date'].dt.month,columns=df['date'].dt.day,values='pred_energy',aggfunc='mean')
        fig_cal=px.imshow(piv,origin='lower',labels={'color':'Energy'},title='Heatmap')

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
        html.Img(src=wc_url,style={'width':'100%','maxHeight':'400px','objectFit':'contain'})
        if wc_url else html.P("No Twitter data available")
    ],style={'padding':'20px'})

    app.run(host='0.0.0.0',port=8060,debug=True)

if __name__=='__main__':
    main()
