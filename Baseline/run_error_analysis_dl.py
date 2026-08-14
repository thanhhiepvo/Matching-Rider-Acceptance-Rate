"""W4 — Phân tích lỗi 3 model DL (FT-Transformer V2, MambaTab V2, MLP V2) so với GBDT (XGBoost
V2), tiếp nối tinh thần `error_analysis.py`/báo cáo Tuần 2 nhưng cho 4 model class thay vì 1.
Train từng model 1 lần (params V2 nếu có), lưu predictions + SHAP (mean(|SHAP|) trên mẫu FN)
ra `artifacts/dl_error_analysis/` — bước nặng nhất (train FT-Transformer/MambaTab + SHAP
Permutation explainer). Chạy `segment_analysis_dl.py` sau (nhẹ, không train lại) để ra bảng
segment + FN overlap.

    python3 Baseline/run_error_analysis_dl.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time

import numpy as np
import pandas as pd
import shap
import torch

from error_analysis_dl import (
    SEED,
    train_and_wrap_ft_transformer, train_and_wrap_mambatab, train_and_wrap_mlp,
    train_and_wrap_xgboost,
)
from evaluation import evaluate
from features import FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, RAW

OUT = os.path.join(ART, "dl_error_analysis")
MAX_EXPLAIN = 200
BACKGROUND_N = 40


def encode_categorical_for_shap(X_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """SHAP's Independent tabular masker gọi np.isclose() nội bộ -> cần TOÀN BỘ cột numeric
    (kể cả categorical). Map categorical string -> code (đúng vocab đã fit lúc train), numeric
    giữ nguyên. predict_fn thật sẽ decode ngược lại trước khi vào model (xem
    `make_decoding_predict_fn`) — nếu không sẽ lỗi TypeError khi trừ str-str."""
    X_enc = X_df.copy()
    for col, (vocab, unknown_idx) in encoders.items():
        X_enc[col] = X_df[col].astype(str).map(vocab).fillna(unknown_idx).astype(float)
    for col in FEATURES:
        if col not in encoders:
            X_enc[col] = pd.to_numeric(X_df[col], errors="coerce")
    return X_enc[FEATURES]


def make_decoding_predict_fn(predict_fn, encoders: dict):
    inv = {col: {v: k for k, v in vocab.items()} for col, (vocab, _) in encoders.items()}

    def wrapped(X_arr):
        X_df = pd.DataFrame(np.asarray(X_arr), columns=FEATURES)
        for col, mapping in inv.items():
            X_df[col] = X_df[col].round().astype(int).map(mapping)
        return predict_fn(X_df)

    return wrapped


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("1 · Nạp data + chia train/valid/test (post-dispatch, khoá 13/07)", flush=True)
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}", flush=True)

    runners = {
        "XGBoost": train_and_wrap_xgboost,
        "MLP": train_and_wrap_mlp,
        "FT-Transformer": train_and_wrap_ft_transformer,
        "MambaTab": train_and_wrap_mambatab,
    }

    results = {}
    print("\n2 · Train từng model (dùng params V2 nếu có)", flush=True)
    for name, fn in runners.items():
        t0 = time.perf_counter()
        res = fn(X_tr, y_tr, X_va, y_va, X_te)
        dt = time.perf_counter() - t0
        m = evaluate(y_te, res["p_te"], name, verbose=False)
        print(f"  {name:16s} -> {res['label']:32s} ({dt:6.1f}s) ROC-AUC={m['roc_auc']:.4f} "
              f"pr_auc_cancel={m['pr_auc_cancel']:.4f}", flush=True)
        results[name] = res

    pred_df = pd.DataFrame({name: r["p_te"] for name, r in results.items()}, index=X_te.index)
    pred_df["y_accept"] = y_te.values
    pred_df.to_csv(os.path.join(OUT, "predictions_test.csv"))
    labels = {name: r["label"] for name, r in results.items()}
    json.dump(labels, open(os.path.join(OUT, "model_labels.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ predictions saved -> {os.path.join(OUT, 'predictions_test.csv')}", flush=True)

    print("\n3 · SHAP (Permutation explainer, raw feature space) cho 3 model DL — giải thích "
          "trên mẫu FN (huỷ, đoán sai) của từng model, KHÔNG phải toàn bộ test (quá chậm)", flush=True)
    rng = np.random.default_rng(SEED)
    background = X_tr.sample(n=BACKGROUND_N, random_state=SEED)

    for name in ["MLP", "FT-Transformer", "MambaTab"]:
        predict_fn = results[name]["predict_fn"]
        encoders = results[name]["encoders"]
        p_te = results[name]["p_te"]
        is_fn = (y_te.values == 0) & (p_te >= 0.5)
        fn_idx = X_te.index[is_fn]
        if len(fn_idx) > MAX_EXPLAIN:
            fn_idx = pd.Index(rng.choice(fn_idx, size=MAX_EXPLAIN, replace=False))
        X_explain = X_te.loc[fn_idx]
        print(f"  {name}: {is_fn.sum()} FN tổng, giải thích {len(X_explain)} mẫu...", flush=True)

        background_enc = encode_categorical_for_shap(background, encoders)
        X_explain_enc = encode_categorical_for_shap(X_explain, encoders)
        wrapped_fn = make_decoding_predict_fn(predict_fn, encoders)

        try:
            t0 = time.perf_counter()
            explainer = shap.Explainer(wrapped_fn, background_enc, algorithm="permutation",
                                        feature_names=FEATURES)
            sv = explainer(X_explain_enc, max_evals=2 * len(FEATURES) + 1)
            dt = time.perf_counter() - t0
            print(f"    xong ({dt:6.1f}s)", flush=True)

            slug = name.replace("-", "_").lower()
            mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
            mean_abs.to_csv(os.path.join(OUT, f"shap_meanabs_{slug}.csv"))
            np.save(os.path.join(OUT, f"shap_values_{slug}.npy"), sv.values)
            X_explain.to_csv(os.path.join(OUT, f"shap_explained_rows_{slug}.csv"))
            print(mean_abs.head(8).to_string(), flush=True)
        except Exception as e:
            print(f"    LỖI SHAP cho {name}: {e!r} — bỏ qua model này", flush=True)

    print("\n  XGBoost: TreeExplainer thật (nhanh, chính xác) trên đúng FN sample cùng size", flush=True)
    booster = results["XGBoost"]["model"]
    p_te_xgb = results["XGBoost"]["p_te"]
    is_fn_xgb = (y_te.values == 0) & (p_te_xgb >= 0.5)
    fn_idx_xgb = X_te.index[is_fn_xgb]
    if len(fn_idx_xgb) > MAX_EXPLAIN:
        fn_idx_xgb = pd.Index(rng.choice(fn_idx_xgb, size=MAX_EXPLAIN, replace=False))
    X_explain_xgb = X_te.loc[fn_idx_xgb]
    tree_explainer = shap.TreeExplainer(booster)
    sv_xgb = tree_explainer(X_explain_xgb)
    mean_abs_xgb = pd.Series(np.abs(sv_xgb.values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    mean_abs_xgb.to_csv(os.path.join(OUT, "shap_meanabs_xgboost.csv"))
    np.save(os.path.join(OUT, "shap_values_xgboost.npy"), sv_xgb.values)
    X_explain_xgb.to_csv(os.path.join(OUT, "shap_explained_rows_xgboost.csv"))
    print(mean_abs_xgb.head(8).to_string(), flush=True)

    print("\n✓ ALL DONE", flush=True)


if __name__ == "__main__":
    main()
