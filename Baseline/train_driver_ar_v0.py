"""Baseline v0 - LightGBM trên 12 feature short-list.

    python3 src/train.py
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from features import FEATURES, build_features, load_raw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_driver_ar")  # data/raw giờ chứa data cho model Rider Cancellation
ART = os.path.join(ROOT, "artifacts")

TEST_DATE = "2026-07-13"   # chia THEO THỜI GIAN: train 06→12/07, test 13/07
PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "num_threads": 8,
    "seed": 42,
}


def evaluate(y, p, label):
    m = {
        "n": int(len(y)),
        "base_rate": round(float(np.mean(y)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }
    print(f"  [{label:5s}] n={m['n']:>7,} · AR nền {m['base_rate']:.4f} · ROC-AUC {m['roc_auc']:.4f}"
          f" · PR-AUC {m['pr_auc']:.4f} · Brier {m['brier']:.4f}")
    return m


def main():
    os.makedirs(ART, exist_ok=True)

    print("1 · Nạp 3 file raw")
    offers, orders, driver_daily = load_raw(RAW)
    print(f"  offers {len(offers):,} · orders {len(orders):,} · driver_daily {len(driver_daily):,}")

    print("2 · Tạo feature")
    df, X, y = build_features(offers, orders, driver_daily)

    print("3 · Chia train/test theo THỜI GIAN (không random)")
    is_test = df.order_date >= pd.Timestamp(TEST_DATE)
    X_tr, y_tr = X[~is_test], y[~is_test]
    X_te, y_te = X[is_test], y[is_test]
    # hạng mục của test phải dùng ĐÚNG bộ của train, nếu không LightGBM ánh xạ mã khác nhau
    for c in X_tr.columns:
        if str(X_tr[c].dtype) == "category":
            X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
    print(f"  train {len(X_tr):,} (→{TEST_DATE}) · test {len(X_te):,} ({TEST_DATE})")

    print("4 · Huấn luyện")
    model = lgb.train(
        PARAMS,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_te, y_te)],
        valid_names=["test"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  dừng ở vòng {model.best_iteration}")

    print("5 · Đánh giá")
    p_tr = model.predict(X_tr, num_iteration=model.best_iteration)
    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    metrics = {"train": evaluate(y_tr, p_tr, "train"), "test": evaluate(y_te, p_te, "test")}

    # Mốc so sánh — một con AUC đứng một mình không có nghĩa
    print("  ── mốc so sánh (test) ──")
    metrics["baselines"] = {}
    metrics["baselines"]["đoán hằng số"] = evaluate(
        y_te, np.full(len(y_te), y_tr.mean()), "hằng")
    eta = X_te.eta_seconds.fillna(X_tr.eta_seconds.median())
    metrics["baselines"]["chỉ ETA"] = evaluate(y_te, 1 / (1 + eta / 300), "ETA")
    m1 = lgb.train(PARAMS, lgb.Dataset(X_tr[["driver_ar_7d"]], y_tr), num_boost_round=100)
    metrics["baselines"]["chỉ AR lịch sử"] = evaluate(
        y_te, m1.predict(X_te[["driver_ar_7d"]]), "AR-ls")

    print("6 · Lưu artifact")
    model.save_model(os.path.join(ART, "model_v0.txt"), num_iteration=model.best_iteration)
    metrics["features"] = FEATURES
    metrics["best_iteration"] = int(model.best_iteration)
    json.dump(metrics, open(os.path.join(ART, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    imp = pd.DataFrame({"feature": model.feature_name(),
                        "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(ART, "feature_importance.csv"), index=False)
    print(imp.to_string(index=False))
    print(f"\n -> {ART}")


if __name__ == "__main__":
    main()
 