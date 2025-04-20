#!/usr/bin/env python3
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd
from textblob import TextBlob

from data_sources.pleasanter import fetch_daily_entries
from data_sources.stayfree_loader import load_stayfree_data
from data_sources.twitter_loader import load_twitter_data

def create_feature_set():
    df_posts = fetch_daily_entries()
    df_sf = load_stayfree_data()
    df_tw = load_twitter_data()
    df = pd.concat([df_posts, df_sf, df_tw], ignore_index=True)
    if 'content' in df.columns:
        df['sentiment'] = df['content'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
    else:
        df['sentiment'] = 0
    return df
