"""Phân tích lỗi theo phân khúc — khách mới / giờ cao điểm / ETA dài (kế hoạch tuần 2 trong
báo cáo W1: "phân tích lỗi theo phân khúc (khách mới, giờ cao điểm, ETA dài)").

Chỉ chạy trên tập test POST-DISPATCH (8,609 dòng, không phải 9,347 toàn bộ) — vì đây mới là
con số quan trọng theo báo cáo (0,7267→0,7342 khi có thêm data), và ETA/giờ cao điểm chỉ thật
sự có ý nghĩa nghiệp vụ sau khi đơn đã được dispatch (pre-dispatch không có eta_seconds).

Dùng lại model_v1.txt đã train sẵn (KHÔNG train lại) — chỉ predict rồi cắt theo phân khúc.

    python3 Baseline/error_analysis.py
"""
from __future__ import annotations

import os

import lightgbm as lgb
import pandas as pd

from evaluation import evaluate
from features import build_features, load_raw
from train import ART, RAW, TEST_DATE

PEAK_HOURS = {7, 8, 9, 17, 18, 19}      # giờ cao điểm sáng + chiều
LONG_ETA_THRESHOLD = 600                 # giây (10 phút) — cùng ngưỡng với rule fallback ban đầu của dự án


def main():
    print("1 · Nạp data + build feature + predict bằng model đã train sẵn (model_v1.txt)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily)
    is_test = df.order_date >= pd.Timestamp(TEST_DATE)
    df_te, X_te, y_te = df[is_test], X[is_test], y[is_test]

    booster = lgb.Booster(model_file=os.path.join(ART, "model_v1.txt"))
    p_te = booster.predict(X_te)

    # Chỉ xét post-dispatch — pre-dispatch không có eta_seconds, và AUC toàn bộ đã biết bị
    # is_post_dispatch thổi phồng nên không phản ánh đúng chất lượng model ở đây.
    post_mask = df_te.is_post_dispatch.astype(int).values == 1
    df_post, X_post, y_post, p_post = df_te[post_mask], X_te[post_mask], y_te[post_mask], p_te[post_mask]
    print(f"  test post-dispatch: {len(df_post):,} dòng")

    print("\n2 · Baseline post-dispatch (toàn bộ, không cắt phân khúc)")
    evaluate(y_post, p_post, "toàn bộ post")

    print("\n3 · Phân khúc: khách mới vs khách quen")
    is_new = df_post.cust_orders_30d.isna() | (df_post.cust_orders_30d == 0)
    print(f"  khách mới: {is_new.sum():,} ({is_new.mean():.1%}) · khách quen: {(~is_new).sum():,}")
    evaluate(y_post[is_new.values], p_post[is_new.values], "khách mới")
    evaluate(y_post[~is_new.values], p_post[~is_new.values], "khách quen")

    print("\n4 · Phân khúc: giờ cao điểm vs giờ thường")
    is_peak = df_post.hour_of_day.isin(PEAK_HOURS)
    print(f"  cao điểm ({sorted(PEAK_HOURS)}): {is_peak.sum():,} ({is_peak.mean():.1%}) · "
          f"thường: {(~is_peak).sum():,}")
    evaluate(y_post[is_peak.values], p_post[is_peak.values], "giờ cao điểm")
    evaluate(y_post[~is_peak.values], p_post[~is_peak.values], "giờ thường")

    print(f"\n5 · Phân khúc: ETA dài (>{LONG_ETA_THRESHOLD}s) vs ETA ngắn")
    has_eta = df_post.eta_seconds.notna()
    is_long_eta = has_eta & (df_post.eta_seconds > LONG_ETA_THRESHOLD)
    is_short_eta = has_eta & (df_post.eta_seconds <= LONG_ETA_THRESHOLD)
    print(f"  ETA dài: {is_long_eta.sum():,} ({is_long_eta.mean():.1%}) · "
          f"ETA ngắn: {is_short_eta.sum():,} · thiếu ETA: {(~has_eta).sum():,}")
    evaluate(y_post[is_long_eta.values], p_post[is_long_eta.values], "ETA dài")
    evaluate(y_post[is_short_eta.values], p_post[is_short_eta.values], "ETA ngắn")

    print("\n6 · Tổng hợp — phân khúc nào model đang YẾU nhất (ROC-AUC thấp nhất)?")
    segments = {
        "toàn bộ post": pd.Series(True, index=df_post.index),
        "khách mới": is_new, "khách quen": ~is_new,
        "giờ cao điểm": is_peak, "giờ thường": ~is_peak,
        "ETA dài": is_long_eta, "ETA ngắn": is_short_eta,
    }
    rows = []
    for name, mask in segments.items():
        m = evaluate(y_post[mask.values], p_post[mask.values], name, verbose=False)
        rows.append({"phân khúc": name, "n": m["n"], "base_rate": m["base_rate"],
                     "roc_auc": m["roc_auc"], "brier": m["brier"], "ece": m["ece"],
                     "recall_cancel": m["recall_cancel"]})
    tbl = pd.DataFrame(rows).sort_values("roc_auc")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
