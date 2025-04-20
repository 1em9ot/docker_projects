#!/usr/bin/env python3
import os, sys, glob
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd

def load_stayfree_data():
    data_dir = os.getenv('STAYFREE_DIR', './teacher_data/stayfree_exports')
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    dfs = []
    for f in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(f)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
