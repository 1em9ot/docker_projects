#!/usr/bin/env python3
import os
import sys
import zipfile
import json
from datetime import datetime

import pandas as pd

# プロジェクトルートをインポートパスに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))


def load_twitter_data() -> pd.DataFrame:
    """
    Twitter 公式エクスポート（ZIP 形式）から tweet.js を抽出し、
    日付と本文だけの DataFrame を返す。
    Dash 側のフィルタで使うため title='Twitter' 列を付与する。
    """
    data_dir = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
    if not os.path.isdir(data_dir):
        return pd.DataFrame()

    all_tweets = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(data_dir, fname)
        try:
            with zipfile.ZipFile(zip_path) as zp:
                # Twitter エクスポートは tweet.js / tweet_activity.js などが入っている
                for member in zp.namelist():
                    if member.endswith('tweet.js'):
                        raw = zp.read(member).decode('utf-8')
                        # ファイルは "window.YTD.tweet.part0 = [ {...} ];" 形式なので = 以降を JSON とみなす
                        js = raw.partition('=')[-1].strip().rstrip(';')
                        all_tweets.extend(json.loads(js))
        except Exception as e:
            # 壊れた ZIP はスキップ
            print(f"[twitter_loader] skip {zip_path}: {e}")
            continue

    if not all_tweets:
        return pd.DataFrame()

    df = pd.json_normalize(all_tweets)

    # 日付パース
    if 'created_at' in df.columns:
        df['created_at'] = df['created_at'].apply(
            lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y")
            if isinstance(x, str) else pd.NaT
        )
        df['date'] = df['created_at'].dt.normalize()

    # 本文列 (full_text or text)
    df.rename(columns={'full_text': 'content', 'text': 'content'}, inplace=True)

    # Dash 側フィルタ用
    df['title'] = 'Twitter'

    return df
