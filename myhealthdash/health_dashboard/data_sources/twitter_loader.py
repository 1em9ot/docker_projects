#!/usr/bin/env python3
import os, sys, zipfile, json
from datetime import datetime
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

def load_twitter_data():
    data_dir = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    all_tweets = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(os.path.join(data_dir, fname)) as zp:
                    for member in zp.namelist():
                        if 'tweet.js' in member:
                            raw = zp.read(member).decode('utf-8')
                            js = raw.partition('=')[-1].strip().rstrip(';')
                            all_tweets += json.loads(js)
            except Exception:
                continue
    if not all_tweets:
        return pd.DataFrame()
    df = pd.json_normalize(all_tweets)
    if 'created_at' in df.columns:
        df['created_at'] = df['created_at'].apply(
            lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y") if isinstance(x, str) else pd.NaT
        )
        df['date'] = df['created_at'].dt.normalize()
    df.rename(columns={'full_text':'content','text':'content'}, inplace=True)
    return df
