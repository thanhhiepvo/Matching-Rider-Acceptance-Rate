"""Phần dùng chung giữa train_mlp.py (MLP thuần) và train_hybrid.py (GBDT+NN hybrid).

Tách ra để không lặp code TabularMLP/TabularDataset/tiền xử lý giữa 2 script — cùng tinh
thần với evaluation.py (1 chỗ sửa, mọi script train dùng chung).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Feature lệch phải nặng (đã thấy ở EDA: total_fee P99.9=1.2M vs median 53k, trip_distance_km
# tới 800km, fee_per_km tới 5,000,000) — cây quyết định không quan tâm scale nhưng NN rất nhạy.
# log1p trước khi chuẩn hoá giúp NN học tốt hơn.
LOG1P_COLS = ["total_fee", "trip_distance_km", "fee_per_km", "eta_seconds", "cust_orders_30d"]


def fit_numeric_scaler(X_tr: pd.DataFrame, numeric_features: list[str]) -> dict:
    """Tính median/mean/std TRÊN TRAIN (log1p trước với cột lệch phải) — trả về dict để lưu lại
    bằng joblib, dùng thống nhất cho cả predict sau này (không tính lại trên test)."""
    transformed = X_tr[numeric_features].copy()
    for c in LOG1P_COLS:
        if c in transformed.columns:
            transformed[c] = np.log1p(transformed[c].clip(lower=0))
    median = transformed.median()
    filled = transformed.fillna(median)
    mean, std = filled.mean(), filled.std().replace(0, 1)
    return {"numeric_features": numeric_features, "median": median, "mean": mean, "std": std}


def transform_numeric(Xsub: pd.DataFrame, scaler: dict) -> np.ndarray:
    numeric_features = scaler["numeric_features"]
    transformed = Xsub[numeric_features].copy()
    for c in LOG1P_COLS:
        if c in transformed.columns:
            transformed[c] = np.log1p(transformed[c].clip(lower=0))
    filled = transformed.fillna(scaler["median"])
    return ((filled - scaler["mean"]) / scaler["std"]).values.astype(np.float32)


def encode_categoricals(X_tr_cat: pd.DataFrame, X_te_cat: pd.DataFrame):
    """Map category -> int index theo vocab học từ TRAIN; giá trị lạ ở test -> 'unknown'."""
    encoders, tr_codes, te_codes = {}, {}, {}
    for col in X_tr_cat.columns:
        vocab = {v: i for i, v in enumerate(X_tr_cat[col].astype(str).unique())}
        unknown_idx = len(vocab)
        encoders[col] = (vocab, unknown_idx)
        tr_codes[col] = X_tr_cat[col].astype(str).map(vocab).fillna(unknown_idx).astype(int).values
        te_codes[col] = X_te_cat[col].astype(str).map(vocab).fillna(unknown_idx).astype(int).values
    return tr_codes, te_codes, encoders


def apply_categorical_encoders(Xsub_cat: pd.DataFrame, encoders: dict) -> np.ndarray:
    codes = {}
    for col, (vocab, unknown_idx) in encoders.items():
        codes[col] = Xsub_cat[col].astype(str).map(vocab).fillna(unknown_idx).astype(int).values
    return np.stack([codes[c] for c in encoders], axis=1)


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y, X_extra=None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.X_extra = torch.tensor(X_extra, dtype=torch.float32) if X_extra is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.X_extra is None:
            return self.X_num[i], self.X_cat[i], self.y[i]
        return self.X_num[i], self.X_cat[i], self.X_extra[i], self.y[i]


class TabularMLP(nn.Module):
    """MLP + embedding cho categorical. `extra_dim` > 0 để nhận thêm input (vd: leaf embedding
    của GBDT trong train_hybrid.py) — mặc định 0 thì hệt bản MLP thuần."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], embed_dim: int = 4,
                 extra_dim: int = 0, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities  # +1 cho giá trị lạ (unseen ở test)
        ])
        in_dim = n_numeric + embed_dim * len(cat_cardinalities) + extra_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat, x_extra=None):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        parts = [x_num] + embs
        if x_extra is not None:
            parts.append(x_extra)
        x = torch.cat(parts, dim=1)
        return self.net(x).squeeze(1)


class LeafEmbedding(nn.Module):
    """GBDT+NN hybrid: coi leaf index của mỗi cây LightGBM (đã train sẵn, KHÔNG train lại)
    như 1 categorical feature, embed rồi mean-pool qua các cây -> 1 vector đặc trưng, nhét
    làm `extra_dim` cho TabularMLP. Kỹ thuật kinh điển kiểu Facebook GBDT+LR / DeepFM.
    """

    def __init__(self, n_trees: int, num_leaves_cap: int, embed_dim: int = 8):
        super().__init__()
        self.n_trees = n_trees
        self.num_leaves_cap = num_leaves_cap
        self.embedding = nn.Embedding(n_trees * num_leaves_cap, embed_dim)

    def forward(self, leaf_idx: torch.Tensor) -> torch.Tensor:
        """leaf_idx: (batch, n_trees) long, mỗi giá trị trong [0, num_leaves_cap)."""
        offset = torch.arange(self.n_trees, device=leaf_idx.device) * self.num_leaves_cap
        global_idx = leaf_idx + offset  # broadcast theo batch
        embs = self.embedding(global_idx)  # (batch, n_trees, embed_dim)
        return embs.mean(dim=1)  # (batch, embed_dim)
