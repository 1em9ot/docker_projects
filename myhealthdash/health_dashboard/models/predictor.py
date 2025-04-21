#!/usr/bin/env python3
import os, sys, pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import pandas as pd
from sklearn.linear_model import LinearRegression

from features.featurize import create_feature_set

MODEL_PATH = os.getenv('MODEL_DIR', './models') + '/health_model.pkl'

def train_model():
    df = create_feature_set()
    if df.empty:
        print("No data available for training.")
        return
    X = df.index.values.reshape(-1, 1)
    y = df['sentiment']
    model = LinearRegression().fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved.")

def predict():
    if not os.path.isfile(MODEL_PATH):
        train_model()
    if not os.path.isfile(MODEL_PATH):
        return pd.DataFrame()
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    df = create_feature_set()
    if df.empty:
        return df
    X = df.index.values.reshape(-1, 1)
    pred = model.predict(X)
    df['pred_sentiment'] = pred
    df['pred_energy'] = pred
    df['energy_state'] = df['pred_energy'].apply(lambda x: 'Active' if x >= 0 else 'Low')
    return df
