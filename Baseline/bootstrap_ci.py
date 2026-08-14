"""W4 prep (Tier 2 của backlog reliability, xem memory) — bootstrap confidence interval cho
bảng so sánh 6 model class ở `compare_models_table.csv`, trên ĐÚNG locked test day (13/07,
n=8.609) hiện có — KHÔNG cần rolling-fold retrain (việc lớn, để dành Tier 3).

Ý tưởng: train các model 1 lần (giống compare_models.py, dùng lại params V2 đã tune nếu có),
lấy xác suất thô trên test set, rồi RESAMPLE CÓ HOÀN LẠI (bootstrap) cùng 1 bộ chỉ số ngẫu
nhiên cho MỌI model ở mỗi vòng lặp (paired bootstrap — so sánh công bằng giữa các model ở
cùng resample) để tính khoảng tin cậy 95% cho ROC-AUC và Cancel PR-AUC. Trả lời câu hỏi: các
model "gần như hoà" ở compare_models.py có thật sự không phân biệt được về mặt thống kê, hay
chỉ là cách nói định tính?

Không train lại Mamba (giữ đúng quy ước --include-mamba của compare_models.py, ~30 phút nếu
train) — mặc định chỉ bootstrap 5 model nhanh; dùng --include-mamba nếu muốn thêm Mamba.

    python3 Baseline/bootstrap_ci.py                  # 5 model, nhanh
    python3 Baseline/bootstrap_ci.py --include-mamba   # 6 model, train lại Mamba (~30 phút)
    python3 Baseline/bootstrap_ci.py --n-boot 2000      # đổi số vòng bootstrap (mặc định 1000)
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

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
from train import ART, PARAMS as LGB_V1_PARAMS, RAW
from train_catboost import PARAMS as CATBOOST_PARAMS, to_catboost_categoricals
from train_ft_transformer import FTTransformer
from train_mamba import MambaTabular
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


def _v2_params(name, fallback):
    path = os.path.join(ART, f"metrics_{name}_v2.json")
    if os.path.exists(path):
        return {**fallback, **json.load(open(path))["best_params"]}, f"{name} V2 (Optuna-tuned)"
    return dict(fallback), f"{name} V1 (hand-set params)"


def predict_lightgbm(X_tr, y_tr, X_va, y_va, X_te):
    params, label = _v2_params("lightgbm", LGB_V1_PARAMS)
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                       valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return label, model.predict(X_te, num_iteration=model.best_iteration)


def predict_xgboost(X_tr, y_tr, X_va, y_va, X_te):
    params, label = _v2_params("xgboost", XGB_PARAMS)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dtest = xgb.DMatrix(X_te, enable_categorical=True)
    booster = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, "valid")],
                         early_stopping_rounds=50, verbose_eval=False)
    return label, booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))


def predict_catboost(X_tr, y_tr, X_va, y_va, X_te):
    params, label = _v2_params("catboost", CATBOOST_PARAMS)
    X_tr_cb, X_va_cb, X_te_cb = (to_catboost_categoricals(d) for d in (X_tr, X_va, X_te))
    model = CatBoostClassifier(**params, cat_features=CATEGORICAL_FEATURES, early_stopping_rounds=50)
    model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
    return label, model.predict_proba(X_te_cb)[:, 1]


def predict_mlp(X_tr, y_tr, X_va, y_va, X_te):
    v2_path = os.path.join(ART, "metrics_mlp_v2.json")
    if os.path.exists(v2_path):
        params, label = json.load(open(v2_path))["best_params"], "MLP V2 (Optuna-tuned)"
    else:
        params, label = dict(MLP_V1_PARAMS), "MLP V1 (hand-set params)"

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    _, te_codes, _ = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(DEVICE)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long).to(DEVICE)
    X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
    X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)

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
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
    return label, p_te


def _prep_token_inputs(X_tr, X_va, X_te):
    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    _, te_codes, _ = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    def to_t(x_num, x_cat):
        return (torch.tensor(x_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(x_cat, dtype=torch.long).to(DEVICE))

    return (to_t(X_tr_num, X_tr_cat), to_t(X_va_num, X_va_cat), to_t(X_te_num, X_te_cat), cat_cardinalities)


def _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va, epochs=40, batch_size=512, lr=1e-3,
                        weight_decay=1e-5, patience=5, grad_clip=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    X_tr_num_t, X_tr_cat_t = X_tr_t
    X_va_num_t, X_va_cat_t = X_va_t
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
    n_train = len(y_tr)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_num_t[idx], X_tr_cat_t[idx]), y_tr_t[idx])
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_ft_transformer(X_tr, y_tr, X_va, y_va, X_te):
    """Tự động dùng hyperparameter Optuna-tuned (`metrics_ft_transformer_v2.json`) nếu đã có —
    CÙNG pattern `_v2_params()`/predict_mambatab() — không còn cố định ở bản mặc định."""
    v2_path = os.path.join(ART, "metrics_ft_transformer_v2.json")
    if os.path.exists(v2_path):
        bp = json.load(open(v2_path))["best_params"]
        d_token, n_heads, n_layers, d_ffn, dropout = (
            bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
        label = "FT-Transformer V2 (Optuna-tuned)"
    else:
        d_token, n_heads, n_layers, d_ffn, dropout = 32, 4, 3, 64, 0.15
        label = "FT-Transformer V1 (mặc định, chưa tune)"

    (X_tr_t, X_va_t, X_te_t, cat_card) = _prep_token_inputs(X_tr, X_va, X_te)
    model = FTTransformer(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_card,
                           d_token=d_token, n_layers=n_layers, n_heads=n_heads,
                           d_ffn=d_ffn, dropout=dropout).to(DEVICE)
    model = _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va)
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(*X_te_t)).cpu().numpy()
    return label, p_te


def predict_mamba(X_tr, y_tr, X_va, y_va, X_te):
    (X_tr_t, X_va_t, X_te_t, cat_card) = _prep_token_inputs(X_tr, X_va, X_te)
    model = MambaTabular(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_card,
                          d_token=32, n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.15).to(DEVICE)
    model = _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va, lr=3e-4, grad_clip=1.0)
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(*X_te_t)).cpu().numpy()
    return "Mamba (selective state-space)", p_te


MAMBATAB_EPOCHS = 1000
MAMBATAB_PATIENCE = 5
MAMBATAB_D_MODEL = 32
MAMBATAB_M_BLOCKS = 3
MAMBATAB_D_STATE = 32
MAMBATAB_D_CONV = 4


def predict_mambatab(X_tr, y_tr, X_va, y_va, X_te):
    """Flatten toàn bộ feature (numeric scale + categorical ordinal-code) thành 1 vector, KHÔNG
    token hoá từng feature như FT-Transformer/Mamba (xem docstring train_mambatab.py). Tự động
    dùng hyperparameter Optuna-tuned (`metrics_mambatab_v2.json`, từ tune_mambatab.py) nếu đã
    có, else fallback về baseline mặc định — CÙNG pattern `_v2_params()` như 4 model kia."""
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
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    te_cat_arr = apply_categorical_encoders(X_te[PRE_CATEGORICAL], encoders)
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1).astype(np.float32)
    X_te_cat = te_cat_arr.astype(np.float32)
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
    return label, p_te


def bootstrap_ci(y_te: np.ndarray, preds: dict[str, np.ndarray], n_boot: int, seed: int = SEED):
    """Paired bootstrap — CÙNG bộ chỉ số resample cho mọi model ở mỗi vòng lặp, để so sánh
    model-đối-model công bằng (không phải mỗi model tự resample riêng)."""
    rng = np.random.default_rng(seed)
    n = len(y_te)
    names = list(preds.keys())
    roc_samples = {k: np.empty(n_boot) for k in names}
    prc_samples = {k: np.empty(n_boot) for k in names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == n:  # resample toàn 1 lớp -> AUC không định nghĩa được, bỏ qua
            for k in names:
                roc_samples[k][b] = np.nan
                prc_samples[k][b] = np.nan
            continue
        for k in names:
            pb = preds[k][idx]
            roc_samples[k][b] = roc_auc_score(yb, pb)
            prc_samples[k][b] = average_precision_score(1 - yb, 1 - pb)

    rows = []
    for k in names:
        roc, prc = roc_samples[k][~np.isnan(roc_samples[k])], prc_samples[k][~np.isnan(prc_samples[k])]
        rows.append({
            "model_class": k,
            "roc_auc_mean": round(float(roc.mean()), 4),
            "roc_auc_ci_low": round(float(np.percentile(roc, 2.5)), 4),
            "roc_auc_ci_high": round(float(np.percentile(roc, 97.5)), 4),
            "pr_auc_cancel_mean": round(float(prc.mean()), 4),
            "pr_auc_cancel_ci_low": round(float(np.percentile(prc, 2.5)), 4),
            "pr_auc_cancel_ci_high": round(float(np.percentile(prc, 97.5)), 4),
        })
    return pd.DataFrame(rows), roc_samples, prc_samples


def pairwise_win_rate(samples: dict[str, np.ndarray], metric_name: str):
    """% số vòng bootstrap model A > model B — proxy phi tham số cho ý nghĩa thống kê, thay vì
    chỉ nhìn khoảng CI có chồng lấn hay không (CI overlap là kiểm định bảo thủ hơn thực tế)."""
    names = list(samples.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mask = ~np.isnan(samples[a]) & ~np.isnan(samples[b])
            win_a = float((samples[a][mask] > samples[b][mask]).mean())
            rows.append({"metric": metric_name, "model_a": a, "model_b": b,
                         "p_a_beats_b": round(win_a, 3)})
    return pd.DataFrame(rows)


def plot_ci(df: pd.DataFrame, metric: str, ci_low: str, ci_high: str, mean: str, title: str, path: str):
    df = df.sort_values(mean, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(df))
    errs = np.vstack([df[mean] - df[ci_low], df[ci_high] - df[mean]])
    ax.errorbar(df[mean], y_pos, xerr=errs, fmt="o", capsize=4, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_class"])
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mamba", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("1 · Nạp data + chia train/valid/test THEO THỜI GIAN (split.py, post-dispatch only)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,} (locked audit day)")

    runners = [predict_lightgbm, predict_xgboost, predict_catboost, predict_mlp,
               predict_ft_transformer, predict_mambatab]
    if args.include_mamba:
        runners.append(predict_mamba)

    preds = {}
    print("2 · Train từng model 1 lần (dùng params V2 đã tune nếu có), lấy xác suất trên test")
    for runner in runners:
        t0 = time.perf_counter()
        label, p_te = runner(X_tr, y_tr, X_va, y_va, X_te)
        print(f"  {runner.__name__:24s} -> {label:35s} ({time.perf_counter() - t0:6.1f}s) "
              f"ROC-AUC point-estimate {roc_auc_score(y_te, p_te):.4f}")
        preds[label] = np.asarray(p_te)

    y_te_arr = y_te.values
    print(f"\n3 · Bootstrap {args.n_boot} vòng (paired — cùng resample cho mọi model)")
    ci_df, roc_samples, prc_samples = bootstrap_ci(y_te_arr, preds, args.n_boot)
    ci_df = ci_df.sort_values("pr_auc_cancel_mean", ascending=False)  # Cancel PR-AUC là metric quyết định chính
    print(ci_df.to_string(index=False))

    print("\n4 · Pairwise win-rate (proxy ý nghĩa thống kê, không phải p-value chính thức)")
    pw_roc = pairwise_win_rate(roc_samples, "roc_auc")
    pw_prc = pairwise_win_rate(prc_samples, "pr_auc_cancel")
    pw = pd.concat([pw_roc, pw_prc], ignore_index=True)
    print(pw.to_string(index=False))

    ci_csv = os.path.join(ART, "bootstrap_ci.csv")
    pw_csv = os.path.join(ART, "bootstrap_pairwise.csv")
    ci_df.to_csv(ci_csv, index=False)
    pw.to_csv(pw_csv, index=False)

    plot_ci(ci_df, "ROC-AUC", "roc_auc_ci_low", "roc_auc_ci_high", "roc_auc_mean",
            f"Bootstrap 95% CI — ROC-AUC (n_boot={args.n_boot}, test n={len(y_te):,})",
            os.path.join(ART, "bootstrap_ci_roc_auc.png"))
    plot_ci(ci_df, "PR-AUC (huỷ)", "pr_auc_cancel_ci_low", "pr_auc_cancel_ci_high", "pr_auc_cancel_mean",
            f"Bootstrap 95% CI — Cancel PR-AUC (n_boot={args.n_boot}, test n={len(y_te):,})",
            os.path.join(ART, "bootstrap_ci_pr_auc_cancel.png"))

    print(f"\n✓ -> {ci_csv}")
    print(f"✓ -> {pw_csv}")
    print(f"✓ -> {os.path.join(ART, 'bootstrap_ci_roc_auc.png')}")
    print(f"✓ -> {os.path.join(ART, 'bootstrap_ci_pr_auc_cancel.png')}")


if __name__ == "__main__":
    main()
