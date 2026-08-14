"""Task 3 (W3) — CatBoost trên CÙNG feature set & split với LightGBM/XGBoost (evaluation.py,
split.py, features.py dùng chung). Native categorical support qua `cat_features` — CatBoost
KHÔNG dùng pandas `category` dtype như LightGBM/XGBoost, mà cần cột categorical ở dạng
string/object, nên 3 cột categorical được ép `astype(str)` (NaN -> chuỗi "nan", CatBoost coi
là 1 category riêng) trước khi đưa vào model — điểm khác biệt duy nhất so với train.py/
train_xgboost.py.

    python3 Baseline/train_catboost.py
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import compute_class_weights
from pipeline_diagram import save_pipeline_diagram
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW

PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "eval_metric": "AUC",
    "random_seed": 42,
    "thread_count": 8,
    "verbose": False,
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


def to_catboost_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost cần cột categorical ở dạng string, không phải pandas `category` dtype —
    NaN -> "nan" (CatBoost coi là 1 giá trị category riêng, không lỗi/không cần impute)."""
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", choices=["none", "weighted"], default="none",
                     help="weighted: áp class_weights (imbalance.py) để tăng trọng số lớp huỷ (y=0)")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "catboost-v1" + (f"-{args.imbalance}" if args.imbalance != "none" else "")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "catboost", "imbalance": args.imbalance})
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

        X_tr_cb, X_va_cb, X_te_cb = (to_catboost_categoricals(d) for d in (X_tr, X_va, X_te))

        print("3 · Huấn luyện (early-stop trên valid, native categorical qua cat_features)")
        run_params = dict(PARAMS)
        class_weights = None
        if args.imbalance == "weighted":
            class_weights = compute_class_weights(y_tr)
            mlflow.log_params({f"class_weight_{k}": round(v, 4) for k, v in class_weights.items()})
            print(f"  imbalance=weighted -> class_weights={class_weights} (tăng trọng số lớp huỷ)")
        model = CatBoostClassifier(**run_params, cat_features=CATEGORICAL_FEATURES,
                                    early_stopping_rounds=50,
                                    class_weights=class_weights)

        # Sơ đồ pipeline — CatBoostClassifier đã sklearn-compatible sẵn nên đây là ĐÚNG object
        # thật sẽ được .fit() bên dưới, không phải bản tương đương như LightGBM/XGBoost.
        viz_pipeline = Pipeline([
            ("to_str_categoricals", FunctionTransformer(to_catboost_categoricals)),
            ("catboost", model),
        ])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_catboost.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_catboost.html"))
        t0 = time.perf_counter()
        model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
        train_seconds = time.perf_counter() - t0
        print(f"  dừng ở vòng {model.get_best_iteration()}, train n={len(X_tr):,}, {train_seconds:.1f}s")
        mlflow.log_param("best_iteration", model.get_best_iteration())
        mlflow.log_metric("train_seconds", train_seconds)

        print("4 · Đánh giá trên test")
        t0 = time.perf_counter()
        p_te = model.predict_proba(X_te_cb)[:, 1]
        predict_seconds = time.perf_counter() - t0
        metrics = {"test": evaluate(y_te, p_te, "test")}
        mlflow.log_metric("predict_seconds", predict_seconds)
        log_metrics("test", metrics["test"])

        print("5 · Hệ thống đầy đủ (rule + model) — như train.py")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_tr, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_te_all_cb = to_catboost_categoricals(X_te_all[post_te_all])
        p_te_all = pd.Series(0.0, index=X_te_all.index)
        p_te_all[post_te_all] = model.predict_proba(X_te_all_cb)[:, 1]
        metrics["system_full"] = evaluate(y_te_all, p_te_all.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("6 · Feature importance")
        imp = pd.DataFrame({
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        imp_path = os.path.join(ART, "catboost_feature_importance.csv")
        imp.to_csv(imp_path, index=False)
        print(imp.to_string(index=False))

        print("7 · Lưu artifact")
        model.save_model(os.path.join(ART, "model_catboost_v1.cbm"))
        metrics["best_iteration"] = int(model.get_best_iteration())
        metrics["features"] = FEATURES
        metrics["train_seconds"] = train_seconds
        metrics["predict_seconds"] = predict_seconds
        metrics_path = os.path.join(ART, "metrics_catboost.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(imp_path)
        mlflow.catboost.log_model(model, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
