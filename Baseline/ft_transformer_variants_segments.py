"""So sánh CÁC PHIÊN BẢN FT-Transformer (từ `reports/W4_dl_error_analysis.ipynb` mục 8) theo
segment — trả lời "mỗi phiên bản chủ yếu sai ở đâu", không chỉ so tổng ROC-AUC/Cancel PR-AUC.
Train lại từng biến thể 1 lần (dùng lại logic `train_ft_transformer_v3.py`), lưu p_te từng biến
thể + segment breakdown, cùng segment definition với `segment_analysis_dl.py` (W1/W2/W4).

    python3 Baseline/ft_transformer_variants_segments.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# lightgbm PHẢI import trước torch (thứ tự nạp OpenMP/BLAS runtime trên macOS Apple Silicon —
# đảo ngược thứ tự từng gây segfault (exit 139) ngay giữa lúc train LightGBM, xem
# train_ft_transformer_v3.py — nơi thứ tự này đã đúng và chạy ổn nhiều lần).
import lightgbm as lgb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from calibration import CALIBRATION_METHODS, apply_calibrator, fit_calibrator
from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import apply_categorical_encoders, encode_categoricals
from split import align_categories, time_split
from train import ART, RAW
from train import PARAMS as LGB_V1_PARAMS
from train_ft_transformer_v3 import N_BINS, FTTransformerV3, PiecewiseLinearEncoder
from train_ft_transformer_v3_calibrated import pick_threshold

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS, PATIENCE = 60, 7
PEAK_HOURS = {7, 8, 9, 17, 18, 19}
LONG_ETA_THRESHOLD = 600
OUT = os.path.join(ART, "dl_error_analysis")

torch.manual_seed(SEED)
np.random.seed(SEED)


def train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te, use_ple, use_leaf, loss_kind,
                   n_trees=0, num_leaves_cap=0, leaf_tr=None, leaf_va=None, leaf_te=None):
    v2_path = os.path.join(ART, "metrics_ft_transformer_v2.json")
    bp = json.load(open(v2_path))["best_params"]
    d_token, n_heads, n_layers, d_ffn, dropout = (
        bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
    lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]

    if use_ple:
        ple = PiecewiseLinearEncoder(N_BINS).fit(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_te_num = (ple.transform(d, NUMERIC_FEATURES) for d in (X_tr, X_va, X_te))
    else:
        from mlp_common import fit_numeric_scaler, transform_numeric
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))

    tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
    X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
    cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

    def to_t(x, dtype):
        return torch.tensor(x, dtype=dtype).to(DEVICE)

    X_tr_num_t, X_va_num_t, X_te_num_t = (to_t(a, torch.float32) for a in (X_tr_num, X_va_num, X_te_num))
    X_tr_cat_t, X_va_cat_t, X_te_cat_t = (to_t(a, torch.long) for a in (X_tr_cat, X_va_cat, X_te_cat))
    if use_leaf:
        leaf_tr_t, leaf_va_t, leaf_te_t = (to_t(a, torch.long) for a in (leaf_tr, leaf_va, leaf_te))
    y_tr_t = to_t(y_tr.values, torch.float32)

    model = FTTransformerV3(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                             n_trees=n_trees, num_leaves_cap=num_leaves_cap, d_token=d_token,
                             n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout,
                             use_ple=use_ple, use_leaf=use_leaf).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_kind in ("focal", "focal_calibrated"):
        loss_fn = FocalLoss(alpha=compute_focal_alpha(y_tr.values), gamma=2.0)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    n_train = len(X_tr)
    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            leaf_batch = leaf_tr_t[idx] if use_leaf else None
            loss = loss_fn(model(X_tr_num_t[idx], X_tr_cat_t[idx], leaf_batch), y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t, leaf_va_t if use_leaf else None)).cpu().numpy()
        auc = roc_auc_score(y_va, p_va)
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t, leaf_te_t if use_leaf else None)).cpu().numpy()

    if loss_kind != "focal_calibrated":
        return p_te

    # Focal + calibrated: cần thêm p trên calib để fit calibrator + chọn threshold
    if use_ple:
        X_ca_num = ple.transform(X_ca, NUMERIC_FEATURES)
    else:
        X_ca_num = transform_numeric(X_ca, scaler)
    X_ca_cat = apply_categorical_encoders(X_ca[CATEGORICAL_FEATURES], encoders)
    X_ca_num_t, X_ca_cat_t = to_t(X_ca_num, torch.float32), to_t(X_ca_cat, torch.long)
    with torch.no_grad():
        p_va_raw = torch.sigmoid(model(X_va_num_t, X_va_cat_t, None)).cpu().numpy()
        p_ca_raw = torch.sigmoid(model(X_ca_num_t, X_ca_cat_t, None)).cpu().numpy()

    best_method, best_ece, best_cal = None, np.inf, None
    for method in CALIBRATION_METHODS:
        cal = fit_calibrator(method, p_ca_raw, y_ca)
        p_va_cal = apply_calibrator(method, cal, p_va_raw)
        ece_va = evaluate(y_va, p_va_cal, f"{method}-valid", verbose=False)["ece"]
        if ece_va < best_ece:
            best_method, best_ece, best_cal = method, ece_va, cal
    p_ca_cal = apply_calibrator(best_method, best_cal, p_ca_raw)
    threshold = pick_threshold(y_ca, p_ca_cal)
    p_te_cal = apply_calibrator(best_method, best_cal, p_te)
    return p_te_cal, threshold


def main():
    print("1 · Nạp data + chia train/valid/calib/test (post-dispatch, khoá 13/07)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_ca, y_ca = align_categories(X_tr, split.calib[0]), split.calib[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    df_te = df.loc[X_te.index]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · calib {len(X_ca):,} · test {len(X_te):,}")

    # leaf-embedding nguồn (LightGBM V2) — dùng chung cho full/leaf_only
    print("2 · Train LightGBM V2 làm nguồn leaf-embedding (dùng cho full/leaf_only)")
    lgb_v2_params = json.load(open(os.path.join(ART, "metrics_lightgbm_v2.json")))["best_params"]
    lgb_params = {**LGB_V1_PARAMS, **lgb_v2_params}
    lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                           valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    n_trees, num_leaves_cap = lgb_model.num_trees(), lgb_params["num_leaves"]
    leaf_tr = lgb_model.predict(X_tr, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)
    leaf_va = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)
    leaf_te = lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)

    preds = {}
    existing = pd.read_csv(os.path.join(OUT, "predictions_test.csv"), index_col=0)
    assert (existing["y_accept"].values == y_te.values).all(), "y mismatch với predictions_test.csv gốc"
    preds["V2 gốc (BCE)"] = existing["FT-Transformer"].reindex(X_te.index).values
    print("  V2 gốc (BCE): dùng lại từ predictions_test.csv (không train lại)")

    print("\n3 · ple_only (BCE)")
    preds["ple_only (BCE)"] = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                             use_ple=True, use_leaf=False, loss_kind="bce")

    print("\n4 · leaf_only (BCE)")
    preds["leaf_only (BCE)"] = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                              use_ple=False, use_leaf=True, loss_kind="bce",
                                              n_trees=n_trees, num_leaves_cap=num_leaves_cap,
                                              leaf_tr=leaf_tr, leaf_va=leaf_va, leaf_te=leaf_te)

    print("\n5 · full = ple + leaf (BCE)")
    preds["full (PLE+leaf, BCE)"] = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                                    use_ple=True, use_leaf=True, loss_kind="bce",
                                                    n_trees=n_trees, num_leaves_cap=num_leaves_cap,
                                                    leaf_tr=leaf_tr, leaf_va=leaf_va, leaf_te=leaf_te)

    print("\n6 · V2 gốc + Focal (raw)")
    preds["V2 gốc + Focal"] = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                             use_ple=False, use_leaf=False, loss_kind="focal")

    print("\n7 · ple_only + Focal (raw)")
    preds["ple_only + Focal (raw)"] = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                                      use_ple=True, use_leaf=False, loss_kind="focal")

    print("\n8 · ple_only + Focal + Calibrated")
    p_cal, thr = train_variant(X_tr, y_tr, X_va, y_va, X_ca, y_ca, X_te, y_te,
                                use_ple=True, use_leaf=False, loss_kind="focal_calibrated")
    preds["ple_only + Focal + Calibrated"] = p_cal
    print(f"  threshold chọn được = {thr:.2f}")

    pred_df = pd.DataFrame(preds, index=X_te.index)
    pred_df["y_accept"] = y_te.values
    pred_df.to_csv(os.path.join(OUT, "ft_transformer_variants_predictions.csv"))
    print(f"\n✓ predictions -> {os.path.join(OUT, 'ft_transformer_variants_predictions.csv')}")

    print("\n9 · Bảng ROC-AUC / Cancel PR-AUC theo segment, từng phiên bản")
    is_new = (df_te.cust_orders_30d.isna() | (df_te.cust_orders_30d == 0)).values
    is_peak = df_te.hour_of_day.isin(PEAK_HOURS).values
    has_eta = df_te.eta_seconds.notna().values
    is_long_eta = has_eta & (df_te.eta_seconds > LONG_ETA_THRESHOLD).values
    is_short_eta = has_eta & (df_te.eta_seconds <= LONG_ETA_THRESHOLD).values
    segments = {
        "Toàn bộ post-dispatch": np.ones(len(df_te), dtype=bool),
        "Khách mới": is_new, "Khách quen": ~is_new,
        "Giờ cao điểm": is_peak, "Giờ thường": ~is_peak,
        "ETA dài (>600s)": is_long_eta, "ETA ngắn (≤600s)": is_short_eta,
    }
    variant_names = list(preds.keys())
    rows = []
    thresholds = {v: (thr if v == "ple_only + Focal + Calibrated" else 0.5) for v in variant_names}
    for seg_name, mask in segments.items():
        for variant in variant_names:
            p = pred_df[variant].values[mask]
            yv = pred_df["y_accept"].values[mask]
            m = evaluate(yv, p, f"{seg_name}-{variant}", threshold=thresholds[variant], verbose=False)
            rows.append({"segment": seg_name, "variant": variant, "n": m["n"],
                         "roc_auc": m["roc_auc"], "pr_auc_cancel": m["pr_auc_cancel"],
                         "recall_cancel": m["recall_cancel"]})
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(os.path.join(OUT, "ft_transformer_variants_segments.csv"), index=False)

    pivot = seg_df.pivot(index="segment", columns="variant", values="pr_auc_cancel")[variant_names]
    order = ["Toàn bộ post-dispatch", "Khách mới", "Khách quen", "Giờ cao điểm", "Giờ thường",
             "ETA dài (>600s)", "ETA ngắn (≤600s)"]
    print("\nCancel PR-AUC theo segment (mỗi cột 1 phiên bản):")
    print(pivot.loc[order].round(4).to_string())

    print("\n10 · Segment yếu nhất mỗi phiên bản")
    worst = (seg_df[seg_df.segment != "Toàn bộ post-dispatch"]
             .sort_values("pr_auc_cancel").groupby("variant").first()[["segment", "n", "pr_auc_cancel"]]
             .loc[variant_names])
    print(worst.to_string())

    print("\n✓ DONE")


if __name__ == "__main__":
    main()
