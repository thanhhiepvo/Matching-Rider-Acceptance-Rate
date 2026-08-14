"""Phân tích lỗi theo segment cho 3 model DL (FT-Transformer V2, MambaTab V2, MLP V2) — tiếp
nối tinh thần `error_analysis.py`/báo cáo Tuần 2 (`reports/part1_shap.md`), nhưng thay vì so
sánh 2 architecture của CÙNG 1 model, so sánh 4 MODEL CLASS trên CÙNG 1 test set khoá (13/07,
post-dispatch, n=8.609) — thêm XGBoost V2 (GBDT tốt nhất theo mục 3b W3_results.ipynb) làm đối
chứng, để trả lời: 3 model DL yếu ở ĐÚNG cùng 1 kiểu phân khúc, hay mỗi model yếu 1 kiểu khác
nhau? Và chỗ nào DL sai mà GBDT KHÔNG sai (khác với "GBDT thắng chung chung")?

Dùng lại params V2 đã tune (Optuna) qua `metrics_<model>_v2.json` nếu có — CÙNG hyperparameter
đang dùng để chọn model trong `bootstrap_ci.py`/W3_results.ipynb mục 3b, không train lại từ
tay để tránh lệch so với con số đã báo cáo.

    python3 Baseline/error_analysis_dl.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, PRE_CATEGORICAL, build_features, load_raw
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
from train_ft_transformer import FTTransformer
from train_mambatab import MambaTab
from train_xgboost import PARAMS as XGB_PARAMS
from tune_mlp import HIDDEN_CHOICES

torch.set_num_threads(1)
MLP_NUMERIC = [f for f in FEATURES if f not in PRE_CATEGORICAL]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MLP_EPOCHS, MLP_PATIENCE = 40, 5
MLP_V1_PARAMS = {"embed_dim": 4, "hidden": "medium", "dropout": 0.3, "lr": 1e-3,
                  "weight_decay": 1e-5, "batch_size": 512}
SEED = 42

PEAK_HOURS = {7, 8, 9, 17, 18, 19}
LONG_ETA_THRESHOLD = 600


def _v2_params(name, fallback):
    path = os.path.join(ART, f"metrics_{name}_v2.json")
    if os.path.exists(path):
        return {**fallback, **json.load(open(path))["best_params"]}, f"{name} V2 (Optuna-tuned)"
    return dict(fallback), f"{name} V1 (hand-set params)"


# ---------------------------------------------------------------------------------------
# Mỗi hàm train_and_wrap_* trả về dict: {"label", "p_te" (xác suất accept trên X_te gốc),
# "predict_fn" (raw X DataFrame -> np.array xác suất accept, dùng lại được cho SHAP trên BẤT
# KỲ subset nào của X_te), "model" (object đã train, cho debug)}.
# ---------------------------------------------------------------------------------------

def train_and_wrap_xgboost(X_tr, y_tr, X_va, y_va, X_te):
    params, label = _v2_params("xgboost", XGB_PARAMS)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    booster = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, "valid")],
                         early_stopping_rounds=50, verbose_eval=False)

    def predict_fn(X_raw: pd.DataFrame) -> np.ndarray:
        X_raw = _coerce_raw(X_raw, X_tr)
        d = xgb.DMatrix(X_raw, enable_categorical=True)
        return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))

    p_te = predict_fn(X_te)
    return {"label": label, "p_te": p_te, "predict_fn": predict_fn, "model": booster}


def train_and_wrap_mlp(X_tr, y_tr, X_va, y_va, X_te):
    v2_path = os.path.join(ART, "metrics_mlp_v2.json")
    if os.path.exists(v2_path):
        params, label = json.load(open(v2_path))["best_params"], "MLP V2 (Optuna-tuned)"
    else:
        params, label = dict(MLP_V1_PARAMS), "MLP V1 (hand-set params)"

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num = (transform_numeric(d, scaler) for d in (X_tr, X_va))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(DEVICE)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long).to(DEVICE)

    model = TabularMLP(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_cardinalities,
                        embed_dim=params["embed_dim"], hidden=HIDDEN_CHOICES[params["hidden"]],
                        dropout=params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
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

    def predict_fn(X_raw: pd.DataFrame) -> np.ndarray:
        X_raw = _coerce_raw(X_raw, X_tr)
        X_num = transform_numeric(X_raw, scaler)
        X_cat = apply_categorical_encoders(X_raw[PRE_CATEGORICAL], encoders)
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.tensor(X_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_cat, dtype=torch.long).to(DEVICE),
            )).cpu().numpy()
        return p

    p_te = predict_fn(X_te)
    return {"label": label, "p_te": p_te, "predict_fn": predict_fn, "model": model, "encoders": encoders}


def train_and_wrap_ft_transformer(X_tr, y_tr, X_va, y_va, X_te):
    v2_path = os.path.join(ART, "metrics_ft_transformer_v2.json")
    if os.path.exists(v2_path):
        bp = json.load(open(v2_path))["best_params"]
        d_token, n_heads, n_layers, d_ffn, dropout = (
            bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
        label = "FT-Transformer V2 (Optuna-tuned)"
    else:
        d_token, n_heads, n_layers, d_ffn, dropout = 32, 4, 3, 64, 0.15
        label = "FT-Transformer V1 (mặc định, chưa tune)"

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num = (transform_numeric(d, scaler) for d in (X_tr, X_va))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    X_tr_num_t = torch.tensor(X_tr_num, dtype=torch.float32).to(DEVICE)
    X_tr_cat_t = torch.tensor(X_tr_cat, dtype=torch.long).to(DEVICE)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(DEVICE)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long).to(DEVICE)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)

    model = FTTransformer(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_cardinalities,
                           d_token=d_token, n_layers=n_layers, n_heads=n_heads,
                           d_ffn=d_ffn, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    n_train, batch_size = len(y_tr), 512

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, 41):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_num_t[idx], X_tr_cat_t[idx]), y_tr_t[idx])
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
            if no_improve >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict_fn(X_raw: pd.DataFrame) -> np.ndarray:
        X_raw = _coerce_raw(X_raw, X_tr)
        X_num = transform_numeric(X_raw, scaler)
        X_cat = apply_categorical_encoders(X_raw[PRE_CATEGORICAL], encoders)
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.tensor(X_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_cat, dtype=torch.long).to(DEVICE),
            )).cpu().numpy()
        return p

    p_te = predict_fn(X_te)
    return {"label": label, "p_te": p_te, "predict_fn": predict_fn, "model": model, "encoders": encoders}


MAMBATAB_EPOCHS, MAMBATAB_PATIENCE = 1000, 5
MAMBATAB_D_MODEL, MAMBATAB_M_BLOCKS, MAMBATAB_D_STATE, MAMBATAB_D_CONV = 32, 3, 32, 4


def train_and_wrap_mambatab(X_tr, y_tr, X_va, y_va, X_te):
    v2_path = os.path.join(ART, "metrics_mambatab_v2.json")
    if os.path.exists(v2_path):
        bp = json.load(open(v2_path))["best_params"]
        d_model, m_blocks, d_state, d_conv, dropout = (
            bp["d_model"], bp["m_blocks"], bp["d_state"], bp["d_conv"], bp["dropout"])
        lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]
        label = "MambaTab V2 (Optuna-tuned)"
    else:
        d_model, m_blocks, d_state, d_conv, dropout = (
            MAMBATAB_D_MODEL, MAMBATAB_M_BLOCKS, MAMBATAB_D_STATE, MAMBATAB_D_CONV, 0.0)
        lr, weight_decay, batch_size = 1e-4, 0.0, 512
        label = "MambaTab V1 (paper defaults, chưa tune)"

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num = (transform_numeric(d, scaler) for d in (X_tr, X_va))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_tr_flat = np.concatenate([X_tr_num, X_tr_cat], axis=1)
    X_va_flat = np.concatenate([X_va_num, X_va_cat], axis=1)
    n_features = X_tr_flat.shape[1]

    X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
    X_va_t = torch.tensor(X_va_flat, dtype=torch.float32).to(DEVICE)

    model = MambaTab(n_features=n_features, d_model=d_model, m_blocks=m_blocks,
                      d_state=d_state, d_conv=d_conv, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAMBATAB_EPOCHS)
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

    def predict_fn(X_raw: pd.DataFrame) -> np.ndarray:
        X_raw = _coerce_raw(X_raw, X_tr)
        X_num = transform_numeric(X_raw, scaler)
        X_cat = apply_categorical_encoders(X_raw[PRE_CATEGORICAL], encoders).astype(np.float32)
        X_flat = np.concatenate([X_num, X_cat], axis=1)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(X_flat, dtype=torch.float32).to(DEVICE))).cpu().numpy()
        return p

    p_te = predict_fn(X_te)
    return {"label": label, "p_te": p_te, "predict_fn": predict_fn, "model": model, "encoders": encoders}


def _coerce_raw(X_raw, X_ref: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá input về đúng dạng DataFrame(FEATURES) với categorical dtype khớp X_ref —
    dùng để predict_fn nhận được CẢ DataFrame gốc LẪN ndarray (SHAP permutation truyền
    ndarray khi mask/hoán vị hàng)."""
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(np.asarray(X_raw, dtype=object), columns=FEATURES)
    else:
        X_raw = X_raw[FEATURES].copy()
    for c in FEATURES:
        if c in CATEGORICAL_FEATURES:
            continue
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
    return align_categories(X_ref, X_raw)


if __name__ == "__main__":
    print("(chạy trực tiếp file này chỉ để sanity-check import — dùng notebook/script gọi "
          "các hàm train_and_wrap_* để phân tích đầy đủ, xem reports/W4_dl_error_analysis.ipynb)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")
