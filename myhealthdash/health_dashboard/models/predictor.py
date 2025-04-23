#!/usr/bin/env python3
"""
ニューラルネットによるヘルススコア回帰モデル
"""

import os, json, random, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchmetrics import MeanAbsoluteError

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("/myhealth/models/health_model.pt")

# ── データ読み込み（例として CSV） ─────────────────────
def load_feature_matrix():
    """
    返り値:
        X: np.ndarray [n_samples, n_features]
        y: np.ndarray [n_samples]
    """
    import pandas as pd
    df = pd.read_csv("/myhealth/data/health_features.csv")
    y = df["target"].values.astype(np.float32)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    return X, y

# ── MLP モデル ───────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── 学習関数 ───────────────────────────────────────
def train_model(num_epochs: int = 300, batch_size: int = 64) -> None:
    X, y = load_feature_matrix()
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
    except ValueError as e:
        print(f"[WARN] train_test_split でエラー ({e}) → 学習をスキップします")
        return


    tr_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)

    model = MLP(in_dim=X.shape[1]).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()
    metric = MeanAbsoluteError().to(DEVICE)

    best_loss, patience, PATIENCE = float("inf"), 0, 15

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward(); opt.step()

        # ---- validate ----
        model.eval(); metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += crit(pred, yb).item() * xb.size(0)
                metric.update(pred, yb)
        val_loss /= len(val_loader.dataset)
        mae = metric.compute().item()
        print(f"[{epoch:03d}] val_loss={val_loss:.4f}  MAE={mae:.4f}")

        # early stopping
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"Best val_loss={best_loss:.4f}  → saved to {MODEL_PATH}")

# ── 予測関数 ─────────────────────────────────────────
_model_cache = None
def predict(X: np.ndarray) -> np.ndarray:
    global _model_cache
    if _model_cache is None:
        in_dim = X.shape[1]
        model = MLP(in_dim); model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        _model_cache = model
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return _model_cache(X_t).cpu().numpy()

