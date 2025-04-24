#!/usr/bin/env python3
"""
Twitter エクスポート ZIP から tweets.js / tweets‑part*.js を抽出し、
Dash で使いやすい 4 列（date / content / title / created_at）を返すローダー。
"""

import os
import sys
import zipfile
import json
import re
from datetime import datetime

import pandas as pd

# ── パス設定 ─────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

# tweets.js / tweets-partN.js を検出する正規表現
PAT_JS = re.compile(r'(?:^|/)tweets?(-part\d+)?\.js$', re.I)

# created_at の代表的なフォーマット
DT_FMT = "%a %b %d %H:%M:%S %z %Y"   # 例: Tue Mar 18 08:07:10 +0000 2025


# ── ヘルパ関数 ─────────────────────────────────────
def _strip_wrapper(raw: str) -> str:
    """window.YTD ... = [...] ; というヘッダを外して JSON 部分だけ返す"""
    return raw.partition('=')[-1].strip().rstrip(';') if raw.lstrip().startswith('window.YTD') else raw


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """候補のうち DataFrame に存在する最初の列名を返す（無ければ None）"""
    return next((c for c in candidates if c in df.columns), None)


# ── メイン関数 ─────────────────────────────────────
def load_twitter_data() -> pd.DataFrame:
    """
    teacher_data/twitter_exports 配下の ZIP をすべて走査し、
    tweets.js / tweets-part*.js を読み込んで DataFrame を返す。
    取得列:
        - date         : 日付（0時正規化）
        - content      : ツイート本文
        - title        : 'Twitter' 固定（Dash のフィルタ用）
        - created_at   : 元の投稿日付 (datetime64[ns, UTC])
    行が 0 件なら空 DataFrame を返す。
    """
    root = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
    if not os.path.isdir(root):
        return pd.DataFrame()

    tweets: list[dict] = []

    # ZIP → tweets*.js を抽出
    for zname in filter(lambda f: f.lower().endswith('.zip'), os.listdir(root)):
        zpath = os.path.join(root, zname)
        with zipfile.ZipFile(zpath) as zf:
            for member in filter(PAT_JS.search, zf.namelist()):
                raw = zf.read(member).decode('utf-8', errors='ignore')
                try:
                    tweets.extend(json.loads(_strip_wrapper(raw)))
                except json.JSONDecodeError as e:
                    print(f"[twitter_loader] JSON decode error in {member}: {e}")

    if not tweets:
        return pd.DataFrame()

    # フラット化
    df = pd.json_normalize(tweets)

    # ── created_at / date 列作成 ──────────────────
    created_col = _pick_first(df, [
        'created_at', 'tweet.created_at', 'legacy.created_at'
    ])

    if created_col:
        # ① まず UTC で読む      （Twitter Export は +0000 付き）
        created_utc = pd.to_datetime(
            df[created_col], format=DT_FMT, errors='coerce', utc=True
        )

        # ② JST (Asia/Tokyo) に変換し、ミリ秒以下を落とす（秒単位に丸め）
        df['created_at'] = (
            created_utc
            .dt.tz_convert('Asia/Tokyo')
            .dt.floor('s')        # ミリ秒以下を切り捨て
        )
        

        # ③ 0 時切り捨てで date 列
        df['date'] = df['created_at'].dt.normalize()

    # ── content 列作成 ────────────────────────────
    content_col = _pick_first(
        df,
        [
            'content', 'full_text', 'text',
            'tweet.content', 'tweet.full_text', 'tweet.text',
            'legacy.content', 'legacy.full_text', 'legacy.text',
        ]
    )
    if not content_col:
        # object 型列のうち最初を本文に充当（最後の保険）
        content_col = next((c for c in df.columns if df[c].dtype == object), None)
    if content_col:
        df['content'] = df[content_col]

    # Dash 側フィルタ用
    df['title'] = 'Twitter'

    # 不要列を落として返す
    return df[['date', 'content', 'title', 'created_at']].copy()



