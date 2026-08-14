"""Task 3 (W3) — XGBoost trên CÙNG feature set & split với LightGBM (evaluation.py, split.py,
features.py dùng chung). Native categorical support (`enable_categorical=True` + `tree_method=
"hist"`) — ăn thẳng cột pandas `category` dtype đã có sẵn từ features.py, KHÔNG cần one-hot/
label-encode thêm.

    python3 Baseline/train_xgboost.py
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evaluation import evaluate
from features import FEATURES, build_features, load_raw
from imbalance import compute_scale_pos_weight
from pipeline_diagram import save_pipeline_diagram
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": 42,
    "nthread": 8,
}


def log_metrics(prefix: str, m: dict):
    for k, v in m.items():
        if k == "confusion_matrix":
            cm = v
            mlflow.log_metrics({
                f"{prefix}_cm_tp": cm["tp"], f"{prefix}_cm_fp": cm["fp"],
                f"{prefix}_cm_fn": cm["fn"], f"{prefix}_cm_tn": cm["tn"],
            })
        else:
            mlflow.log_metric(f"{prefix}_{k}", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", choices=["none", "weighted"], default="none",
                     help="weighted: áp scale_pos_weight (imbalance.py) để tăng trọng số lớp huỷ (y=0)")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "xgboost-v1" + (f"-{args.imbalance}" if args.imbalance != "none" else "")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "xgboost", "imbalance": args.imbalance})
        mlflow.log_param("imbalance", args.imbalance)
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", len(FEATURES))

        print("1 · Nạp data + build feature")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)

        print("2 · Chia train/valid/test THEO THỜI GIAN (post-dispatch only, split.py)")
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")
        mlflow.log_params({"n_train": len(X_tr), "n_valid": len(X_va), "n_test": len(X_te)})

        print("3 · Huấn luyện (early-stop trên valid, native categorical qua pandas category dtype)")
        run_params = dict(PARAMS)
        if args.imbalance == "weighted":
            spw = compute_scale_pos_weight(y_tr)
            run_params["scale_pos_weight"] = spw
            mlflow.log_param("scale_pos_weight", round(spw, 4))
            print(f"  imbalance=weighted -> scale_pos_weight={spw:.4f} (tăng trọng số lớp huỷ)")
            # Chỉ theo dõi auc để early-stop (không phải logloss) — như train.py: scale_pos_weight
            # làm logloss xấu đi đơn điệu dù AUC vẫn tốt lên, cần tách riêng metric chọn model.
            run_params["eval_metric"] = ["auc"]

        # Sơ đồ pipeline (sklearn estimator TƯƠNG ĐƯƠNG, chỉ để trực quan hoá — model thật vẫn
        # train qua xgb.train() bên dưới để early-stop đúng cách, xem pipeline_diagram.py).
        viz_pipeline = Pipeline([
            ("xgboost", XGBClassifier(**run_params, n_estimators=1000, enable_categorical=True)),
        ])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_xgboost.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_xgboost.html"))

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest = xgb.DMatrix(X_te, label=y_te, enable_categorical=True)

        t0 = time.perf_counter()
        booster = xgb.train(
            run_params, dtrain, num_boost_round=1000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50, verbose_eval=False,
        )
        train_seconds = time.perf_counter() - t0
        print(f"  dừng ở vòng {booster.best_iteration}, train n={len(X_tr):,}, {train_seconds:.1f}s")
        mlflow.log_param("best_iteration", booster.best_iteration)
        mlflow.log_metric("train_seconds", train_seconds)

        print("4 · Đánh giá trên test")
        t0 = time.perf_counter()
        p_te = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
        predict_seconds = time.perf_counter() - t0
        metrics = {"test": evaluate(y_te, p_te, "test")}
        mlflow.log_metric("predict_seconds", predict_seconds)
        log_metrics("test", metrics["test"])

        print("5 · Hệ thống đầy đủ (rule + model) — như train.py")
        post_mask_full = df.is_post_dispatch.astype(int) == 1
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_tr, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        dall_post = xgb.DMatrix(X_te_all[post_te_all], enable_categorical=True)
        p_te_all = pd.Series(0.0, index=X_te_all.index)
        p_te_all[post_te_all] = booster.predict(dall_post, iteration_range=(0, booster.best_iteration + 1))
        metrics["system_full"] = evaluate(y_te_all, p_te_all.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("6 · Feature importance (gain)")
        gain = booster.get_score(importance_type="gain")
        imp = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())}) \
            .sort_values("gain", ascending=False)
        imp_path = os.path.join(ART, "xgboost_feature_importance.csv")
        imp.to_csv(imp_path, index=False)
        print(imp.to_string(index=False))

        print("7 · Lưu artifact")
        booster.save_model(os.path.join(ART, "model_xgboost_v1.json"))
        metrics["best_iteration"] = int(booster.best_iteration)
        metrics["features"] = FEATURES
        metrics["train_seconds"] = train_seconds
        metrics["predict_seconds"] = predict_seconds
        metrics_path = os.path.join(ART, "metrics_xgboost.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(imp_path)
        mlflow.xgboost.log_model(booster, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
