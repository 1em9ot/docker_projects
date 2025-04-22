#!/usr/bin/env python3
import os, sys, re, json, base64
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd
import psycopg2
import pytz                          # ← 正しい import
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

JST = pytz.timezone('Asia/Tokyo')

def fetch_daily_entries(start_date=None, end_date=None):
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS')
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        site_id = os.getenv('SITE_ID', '3')
        sql = 'SELECT * FROM "Implem.Pleasanter"."Results" WHERE "SiteId" = %s'
        params = [site_id]
        if start_date and end_date:
            sql += ' AND "UpdatedTime" BETWEEN %s AND %s'
            params += [f"{start_date} 00:00:00", f"{end_date} 23:59:59"]
        cur.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:
        print(f"Error fetching DB entries: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── 時刻処理 ─────────────────────────────────────────
    # DB が「JST の文字列 (タイムゾーンなし)」の場合:
    df['created_at'] = pd.to_datetime(df['UpdatedTime'], errors='coerce') \
                         .dt.tz_localize(JST)      # ← JST を付与するだけ

    # もし DB が UTC(+00) で保存されているなら ↓ を使う
    # df['created_at'] = pd.to_datetime(df['UpdatedTime'], utc=True, errors='coerce') \
    #                      .dt.tz_convert(JST)

    # 日付列
    df['date'] = df['created_at'].dt.normalize()

    # 列名リネーム
    df.rename(columns={'Body': 'content', 'Title': 'title'}, inplace=True)
    return df
