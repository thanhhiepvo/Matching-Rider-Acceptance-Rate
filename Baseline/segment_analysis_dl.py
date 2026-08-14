"""W4 — Bảng phân khúc + FN overlap + soi trực tiếp cho 4 model (XGBoost, MLP, FT-Transformer,
MambaTab) — chạy SAU `run_error_analysis_dl.py` (đọc lại `predictions_test.csv` đã lưu, KHÔNG
train lại — nhẹ, chạy vài giây). Cùng segment definition với `error_analysis.py` (W1/W2):
khách mới/quen, giờ cao điểm/thường, ETA dài/ngắn.

    python3 Baseline/segment_analysis_dl.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, RAW

OUT = os.path.join(ART, "dl_error_analysis")
PEAK_HOURS = {7, 8, 9, 17, 18, 19}
LONG_ETA_THRESHOLD = 600
MODEL_COLS = ["XGBoost", "MLP", "FT-Transformer", "MambaTab"]
DL_MODELS = ["MLP", "FT-Transformer", "MambaTab"]


def main():
    pred_df = pd.read_csv(os.path.join(OUT, "predictions_test.csv"), index_col=0)

    print("1 · Nạp lại data/features để lấy segment info (không train lại)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    df_te = df.loc[X_te.index]
    assert (pred_df["y_accept"].values == y_te.values).all(), \
        "y mismatch giữa predictions_test.csv và data hiện tại"
    print(f"  test {len(X_te):,} dòng — khớp y")

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

    print("\n2 · Bảng ROC-AUC / Cancel PR-AUC theo segment, cho từng model")
    rows = []
    for seg_name, mask in segments.items():
        for model in MODEL_COLS:
            p = pred_df[model].values[mask]
            yv = pred_df["y_accept"].values[mask]
            m = evaluate(yv, p, f"{seg_name}-{model}", verbose=False)
            rows.append({"segment": seg_name, "model": model, "n": m["n"], "base_rate": m["base_rate"],
                         "roc_auc": m["roc_auc"], "pr_auc_cancel": m["pr_auc_cancel"],
                         "recall_cancel": m["recall_cancel"], "precision_cancel": m["precision_cancel"]})
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(os.path.join(OUT, "segment_metrics.csv"), index=False)
    print(seg_df.to_string(index=False))

    print("\n3 · Segment yếu nhất mỗi model (Cancel PR-AUC thấp nhất, bỏ 'Toàn bộ')")
    worst = (seg_df[seg_df.segment != "Toàn bộ post-dispatch"]
             .sort_values("pr_auc_cancel").groupby("model").first()[["segment", "n", "pr_auc_cancel"]])
    print(worst.to_string())

    print("\n4 · FN sets (huỷ, đoán sai @0.5) từng model + overlap")
    fn_sets = {}
    for model in MODEL_COLS:
        p = pred_df[model].values
        yv = pred_df["y_accept"].values
        is_fn = (yv == 0) & (p >= 0.5)
        fn_sets[model] = set(pred_df.index[is_fn])
        print(f"  {model:16s}: {is_fn.sum():4d} FN / {(yv == 0).sum()} huỷ thật "
              f"({is_fn.sum()/(yv==0).sum():.1%} miss rate)")

    dl_fn_union = set.union(*[fn_sets[m] for m in DL_MODELS])
    dl_fn_all3 = set.intersection(*[fn_sets[m] for m in DL_MODELS])
    xgb_fn = fn_sets["XGBoost"]
    dl_only_wrong = dl_fn_all3 - xgb_fn
    print(f"\n  Union FN (>=1 trong 3 DL sai): {len(dl_fn_union)}")
    print(f"  Intersection FN (CẢ 3 DL đều sai): {len(dl_fn_all3)}")
    print(f"  XGBoost FN: {len(xgb_fn)}")
    print(f"  'DL-only-wrong' (cả 3 DL sai NHƯNG XGBoost đúng): {len(dl_only_wrong)}")
    print("  Overlap giữa 3 DL (Jaccard pairwise):")
    for i, a in enumerate(DL_MODELS):
        for b in DL_MODELS[i + 1:]:
            inter = len(fn_sets[a] & fn_sets[b])
            union = len(fn_sets[a] | fn_sets[b])
            print(f"    {a} vs {b}: {inter}/{union} = {inter/union:.1%}")

    overlap_summary = {
        "fn_counts": {m: len(fn_sets[m]) for m in MODEL_COLS},
        "dl_fn_union": len(dl_fn_union), "dl_fn_intersection": len(dl_fn_all3),
        "dl_only_wrong_vs_xgboost": len(dl_only_wrong),
    }
    json.dump(overlap_summary, open(os.path.join(OUT, "fn_overlap_summary.json"), "w"), indent=2)

    print(f"\n5 · Soi trực tiếp: 'DL-only-wrong' (n={len(dl_only_wrong)}) vs 'DL-all-correct TP' — median feature")
    dl_tp_all3 = set(pred_df.index[pred_df["y_accept"] == 0]) - dl_fn_union
    print(f"  DL-only-wrong (FN cả 3 DL, XGBoost đúng): n={len(dl_only_wrong)}")
    print(f"  DL-all-correct (TP cả 3 DL): n={len(dl_tp_all3)}")

    if len(dl_only_wrong) >= 5 and len(dl_tp_all3) >= 5:
        numeric_cols = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
        wrong_idx = pd.Index(dl_only_wrong)
        correct_idx = pd.Index(dl_tp_all3)
        med_wrong = df_te.loc[wrong_idx, numeric_cols].median()
        med_correct = df_te.loc[correct_idx, numeric_cols].median()
        pct_diff = (med_wrong - med_correct) / med_correct.replace(0, np.nan) * 100
        comp = pd.DataFrame({"DL-only-wrong (median)": med_wrong, "DL-all-correct TP (median)": med_correct,
                              "% lệch": pct_diff}).sort_values("% lệch", key=abs, ascending=False)
        comp.to_csv(os.path.join(OUT, "dl_only_wrong_vs_correct_numeric.csv"))
        print(comp.to_string())

        print("\n  Categorical:")
        cat_rows = []
        for col in CATEGORICAL_FEATURES:
            vc_wrong = df_te.loc[wrong_idx, col].value_counts(normalize=True)
            vc_correct = df_te.loc[correct_idx, col].value_counts(normalize=True)
            for val in set(vc_wrong.index) | set(vc_correct.index):
                cat_rows.append({"feature": col, "value": val,
                                  "wrong_%": vc_wrong.get(val, 0) * 100, "correct_%": vc_correct.get(val, 0) * 100,
                                  "diff_pp": vc_wrong.get(val, 0) * 100 - vc_correct.get(val, 0) * 100})
        cat_df = pd.DataFrame(cat_rows).sort_values("diff_pp", key=abs, ascending=False)
        cat_df.to_csv(os.path.join(OUT, "dl_only_wrong_vs_correct_categorical.csv"), index=False)
        print(cat_df.head(15).to_string(index=False))
    else:
        print("  QUÁ ÍT MẪU để so sánh median tin cậy — bỏ qua bước này.")

    print("\n✓ DONE")


if __name__ == "__main__":
    main()
