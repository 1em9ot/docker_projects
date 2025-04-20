#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd, psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

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
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['UpdatedTime'], errors='coerce')
        df['date'] = df['created_at'].dt.normalize()
        df.rename(columns={'Body':'content','Title':'title'}, inplace=True)
    return df
