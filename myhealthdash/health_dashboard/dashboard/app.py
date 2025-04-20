#!/usr/bin/env python3
import os, sys, json
from datetime import datetime
import pandas as pd

# プロジェクトルートをパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import dash
from dash import dcc, html, dash_table
import plotly.express as px

from strategies.predict import predict


def sanitize_records(df: pd.DataFrame):
    """
    DataTable に渡す前に、list/dict 型のセルを JSON 文字列に変換し、NaN を空文字にする
    """
    records = []
    for row in df.to_dict('records'):
        new_row = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                new_row[k] = json.dumps(v, ensure_ascii=False)
            elif pd.isna(v):
                new_row[k] = ''
            else:
                new_row[k] = v
        records.append(new_row)
    return records


def main():
    # モデル予測データ取得
    df = predict()
    if df.empty:
        print("No data available to display.")
        return

    # 日付順にソート
    df = df.sort_values('date')

    # Sentiment 折れ線（ツールチップにレコードIDとタイトルを表示）
    fig1 = px.line(
        df,
        x='date',
        y='pred_sentiment',
        title='Sentiment over Time',
        markers=True,
        hover_data={
            'date': True,
            'pred_sentiment': ':.3f',
            'ResultId': True,
            'title': True
        }
    )

    # Energy 折れ線
    fig2 = px.line(
        df,
        x='date',
        y='pred_energy',
        title='Energy over Time',
        markers=True,
        hover_data={
            'date': False,
            'pred_energy': ':.3f',
            'ResultId': True,
            'title': True
        }
    )

    # カレンダーヒートマップ or フォールバック
    try:
        import plotly_calplot
        df_calendar = df.dropna(subset=['date']).copy()
        df_calendar['date'] = pd.to_datetime(df_calendar['date'])
        fig3 = plotly_calplot.calplot(
            df_calendar,
            x='date',
            y='pred_energy',
            colorscale='RdYlGn',
            title='Energy Calendar'
        )
    except (ImportError, TypeError, ValueError):
        df_alt = df.dropna(subset=['date']).copy()
        df_alt['m'] = df_alt['date'].dt.month.astype(int)
        df_alt['d'] = df_alt['date'].dt.day.astype(int)
        pivot = df_alt.pivot_table(
            index='m', columns='d', values='pred_energy', aggfunc='mean'
        )
        fig3 = px.imshow(
            pivot,
            origin='lower',
            labels={'color': 'Energy'},
            title='Energy Heatmap'
        )

    # データテーブルのレコードをサニタイズ
    table_records = sanitize_records(df)

    # Dash アプリ設定
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Health Dashboard"),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3),
        html.H2("Raw Entries"),
        dash_table.DataTable(
            id='entries-table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=table_records,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '4px'},
            style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'}
        )
    ], style={'padding': '20px'})

    # サーバー起動
    app.run(host='0.0.0.0', port=8060, debug=True)


if __name__ == '__main__':
    main()
