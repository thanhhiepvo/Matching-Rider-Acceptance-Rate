"""So sánh BCE vs FocalLoss cho MLP V2 và MambaTab V2 — tiếp nối phát hiện ở FT-Transformer
(`train_ft_transformer_v3.py --loss focal`): Focal loss cải thiện ROC-AUC/Cancel PR-AUC thật
nhưng phá calibration (ECE tăng vọt). Kiểm tra xem 2 model DL còn lại có cùng pattern không —
dùng ĐÚNG hyperparameter V2 đã tune (`metrics_{mlp,mambatab}_v2.json`, cùng nguồn với
error_analysis_dl.py/bootstrap_ci.py) để so sánh công bằng, chỉ đổi loss function.

    python3 Baseline/train_dl_focal_comparison.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluation import evaluate
from features import FEATURES, PRE_CATEGORICAL, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    TabularDataset,
    TabularMLP,
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from split import align_categories, time_split
from train import ART, RAW
from train_mambatab import MambaTab
from tune_mlp import HIDDEN_CHOICES

torch.set_num_threads(1)
MLP_NUMERIC = [f for f in FEATURES if f not in PRE_CATEGORICAL]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
MLP_EPOCHS, MLP_PATIENCE = 40, 5
MAMBATAB_EPOCHS, MAMBATAB_PATIENCE = 1000, 5

torch.manual_seed(SEED)
np.random.seed(SEED)


def train_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, loss_kind: str):
    bp = json.load(open(os.path.join(ART, "metrics_mlp_v2.json")))["best_params"]

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    X_te_cat = apply_categorical_encoders(X_te[PRE_CATEGORICAL], encoders)
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
    loader = DataLoader(train_ds, batch_size=bp["batch_size"], shuffle=True, drop_last=True)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(DEVICE)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long).to(DEVICE)
    X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
    X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)

    model = TabularMLP(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_cardinalities,
                        embed_dim=bp["embed_dim"], hidden=HIDDEN_CHOICES[bp["hidden"]],
                        dropout=bp["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    if loss_kind == "focal":
        loss_fn = FocalLoss(alpha=compute_focal_alpha(y_tr.values), gamma=2.0)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num, xb_cat, yb = xb_num.to(DEVICE), xb_cat.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb_num, xb_cat), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
    return evaluate(y_te, p_te, f"MLP V2 + {loss_kind}", verbose=False)


def train_mambatab(X_tr, y_tr, X_va, y_va, X_te, y_te, loss_kind: str):
    bp = json.load(open(os.path.join(ART, "metrics_mambatab_v2.json")))["best_params"]
    d_model, m_blocks, d_state, d_conv, dropout = (
        bp["d_model"], bp["m_blocks"], bp["d_state"], bp["d_conv"], bp["dropout"])
    lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    X_te_cat_raw = apply_categorical_encoders(X_te[PRE_CATEGORICAL], encoders)
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_te_cat = X_te_cat_raw.astype(np.float32)
    X_tr_flat = np.concatenate([X_tr_num, X_tr_cat], axis=1)
    X_va_flat = np.concatenate([X_va_num, X_va_cat], axis=1)
    X_te_flat = np.concatenate([X_te_num, X_te_cat], axis=1)
    n_features = X_tr_flat.shape[1]

    X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
    X_va_t = torch.tensor(X_va_flat, dtype=torch.float32).to(DEVICE)
    X_te_t = torch.tensor(X_te_flat, dtype=torch.float32).to(DEVICE)

    model = MambaTab(n_features=n_features, d_model=d_model, m_blocks=m_blocks,
                      d_state=d_state, d_conv=d_conv, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAMBATAB_EPOCHS)
    if loss_kind == "focal":
        loss_fn = FocalLoss(alpha=compute_focal_alpha(y_tr.values), gamma=2.0)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    n_train = len(X_tr_flat)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, MAMBATAB_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_t)).cpu().numpy()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= MAMBATAB_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_t)).cpu().numpy()
    return evaluate(y_te, p_te, f"MambaTab V2 + {loss_kind}", verbose=False)


def main():
    print("1 · Nạp data + chia train/valid/test (post-dispatch, khoá 13/07)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")

    rows = []
    for model_name, fn in [("MLP", train_mlp), ("MambaTab", train_mambatab)]:
        for loss_kind in ["bce", "focal"]:
            print(f"\n2 · {model_name} + {loss_kind}")
            m = fn(X_tr, y_tr, X_va, y_va, X_te, y_te, loss_kind)
            print(f"  ROC-AUC={m['roc_auc']:.4f}  Cancel PR-AUC={m['pr_auc_cancel']:.4f}  "
                  f"ECE={m['ece']:.4f}  Recall_huỷ={m['recall_cancel']:.4f}")
            rows.append({"model": model_name, "loss": loss_kind, **{
                k: m[k] for k in ["roc_auc", "pr_auc_cancel", "ece", "brier",
                                   "recall_cancel", "precision_cancel", "f1_cancel"]}})

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(ART, "dl_focal_comparison.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n{result_df.to_string(index=False)}")
    print(f"\n✓ -> {out_path}")


if __name__ == "__main__":
    main()
