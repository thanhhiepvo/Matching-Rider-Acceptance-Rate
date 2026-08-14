"""Baseline v1 — Rider (customer) acceptance, LightGBM chỉ train/infer trên POST-DISPATCH.

is_post_dispatch giờ là 1 RULE if/else bên ngoài model, không phải feature:
  - is_post_dispatch = False -> trả thẳng "huỷ" (y_accept=0, đúng 100% theo định nghĩa dữ liệu),
    KHÔNG đi qua model.
  - is_post_dispatch = True  -> mới đi qua model (model chỉ thấy phần dữ liệu này khi train/test).
"""
from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")   # không có display khi chạy script — phải set trước khi import pyplot
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from evaluation import evaluate
from features import FEATURES, build_features, load_raw
from imbalance import compute_scale_pos_weight
from pipeline_diagram import save_pipeline_diagram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ART = os.path.join(ROOT, "artifacts")

MLFLOW_DB_PATH = os.path.join(ROOT, "mlflow.db")
EXPERIMENT_NAME = "rider-cancellation-prediction"

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
    # Tự dựng sqlite URI (path thô, không qua bước encode) — để mlflow tự suy luận default sẽ
    # percent-encode khoảng trắng trong đường dẫn project rồi tạo nhầm file "...%20..." bên
    # cạnh project thật. MLflow 3.x cũng không còn cho dùng file store ('./mlruns') nữa.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "baseline-v1-post-dispatch-only"
    if args.imbalance != "none":
        run_name += f"-{args.imbalance}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "lightgbm", "imbalance": args.imbalance})
        mlflow.log_param("test_date", TEST_DATE)
        mlflow.log_param("imbalance", args.imbalance)
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", len(FEATURES))

        print("1 · Nạp 2 file raw")
        orders, customer_daily = load_raw(RAW)
        print(f"  orders {len(orders):,} · customer_daily {len(customer_daily):,}")

        print("2 · Tạo feature")
        df, X, y = build_features(orders, customer_daily)

        print("3 · Áp rule is_post_dispatch: pre-dispatch trả thẳng huỷ, model chỉ thấy post-dispatch")
        is_test_full = df.order_date >= pd.Timestamp(TEST_DATE)
        post_mask = df.is_post_dispatch.astype(int) == 1
        print(f"  post-dispatch: {post_mask.sum():,}/{len(df):,} ({post_mask.mean():.1%}) — "
              f"phần còn lại ({(~post_mask).sum():,}) không qua model, luôn trả huỷ")

        print("4 · Chia train/test theo THỜI GIAN (trên phần post-dispatch)")
        X_tr, y_tr = X[~is_test_full & post_mask], y[~is_test_full & post_mask]
        X_te, y_te = X[is_test_full & post_mask], y[is_test_full & post_mask]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
        print(f"  train {len(X_tr):,} (→{TEST_DATE}) · test {len(X_te):,} ({TEST_DATE})")
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_test", len(X_te))

        print("5 · Huấn luyện (chỉ trên post-dispatch)")
        run_params = dict(PARAMS)
        # first_metric_only: mặc định early_stopping đòi CẢ auc lẫn binary_logloss cùng cải
        # thiện mới reset kiên nhẫn — khi scale_pos_weight lệch mạnh, xác suất dự đoán mất
        # calibration (logloss XẤU ĐI đều đặn dù AUC/xếp hạng vẫn TỐT LÊN — bằng chứng thực
        # nghiệm: đã thấy binary_logloss tăng đơn điệu từ vòng 1 trong khi AUC vẫn tăng), khiến
        # model dừng NGAY vòng 1 nếu theo dõi cả 2. Đây chính xác là lý do calibration.py tồn
        # tại như bước RIÊNG sau imbalance handling — early-stop theo đúng AUC (metric chọn
        # model), để calibration xử lý phần xác suất lệch sau.
        first_metric_only = args.imbalance == "weighted"
        if args.imbalance == "weighted":
            spw = compute_scale_pos_weight(y_tr)
            run_params["scale_pos_weight"] = spw
            mlflow.log_param("scale_pos_weight", round(spw, 4))
            print(f"  imbalance=weighted -> scale_pos_weight={spw:.4f} (tăng trọng số lớp huỷ)")

        # Sơ đồ pipeline (sklearn estimator TƯƠNG ĐƯƠNG, chỉ để trực quan hoá — model thật vẫn
        # train qua lgb.train() bên dưới để early-stop đúng cách, xem pipeline_diagram.py).
        viz_pipeline = Pipeline([("lightgbm", LGBMClassifier(**run_params, n_estimators=500))])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_lightgbm.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_lightgbm.html"))

        train_set = lgb.Dataset(X_tr, y_tr)
        test_set = lgb.Dataset(X_te, y_te)
        evals_result = {}
        model = lgb.train(
            run_params, train_set,
            num_boost_round=500,
            valid_sets=[train_set, test_set], valid_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(50, verbose=False, first_metric_only=first_metric_only),
                lgb.log_evaluation(100),
                lgb.record_evaluation(evals_result),
            ],
        )
        print(f"  dừng ở vòng {model.best_iteration}")
        mlflow.log_param("best_iteration", model.best_iteration)

        # Log đường cong huấn luyện (AUC/logloss theo từng vòng boosting) — để MLflow vẽ được
        # biểu đồ thật, chứ không chỉ 1 con số cuối như các metric evaluate() ở dưới.
        for split, split_metrics in evals_result.items():
            for metric_name, values in split_metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(f"curve_{split}_{metric_name}", value, step=step)

        print("6 · Đánh giá (train/test giờ CHÍNH LÀ post-dispatch — không cần lọc lại)")
        p_tr = model.predict(X_tr, num_iteration=model.best_iteration)
        p_te = model.predict(X_te, num_iteration=model.best_iteration)
        metrics = {"train": evaluate(y_tr, p_tr, "train"), "test": evaluate(y_te, p_te, "test")}
        log_metrics("train", metrics["train"])
        log_metrics("test", metrics["test"])

        # (Tham khảo) Hiệu năng HỆ THỐNG đầy đủ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
        # model (post-dispatch) — con số này mô phỏng đúng những gì hệ thống thật sẽ trả ra, khác
        # với "test" ở trên (thuần model, chỉ đo trên phần post-dispatch model thực sự học).
        X_te_full = X[is_test_full].copy()
        y_te_full = y[is_test_full]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te_full[c] = pd.Categorical(X_te_full[c], categories=X_tr[c].cat.categories)
        post_te_full = df[is_test_full].is_post_dispatch.astype(int).values == 1
        p_te_full = pd.Series(0.0, index=X_te_full.index)
        p_te_full[post_te_full] = model.predict(X_te_full[post_te_full], num_iteration=model.best_iteration)
        metrics["system_full"] = evaluate(y_te_full, p_te_full.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · SHAP — mức đóng góp từng feature vào từng dự đoán (nguyên tắc hơn gain)")
        # gain (mục 8) đo cây "dùng" feature bao nhiêu để giảm loss trên TRAIN — không nói được
        # feature đó kéo 1 dự đoán CỤ THỂ theo hướng nào. SHAP phân rã mỗi dự đoán thành tổng đóng
        # góp (có dấu) của từng feature so với baseline (giá trị kỳ vọng) — TreeExplainer tính
        # chính xác (không xấp xỉ) cho model dạng cây như LightGBM, chạy trên tập TEST.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)
        if isinstance(shap_values, list):   # 1 số phiên bản shap trả list [class0, class1]
            shap_values = shap_values[1]

        shap_imp = pd.DataFrame({
            "feature": X_te.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_imp_path = os.path.join(ART, "shap_importance.csv")
        shap_imp.to_csv(shap_imp_path, index=False)
        print(shap_imp.to_string(index=False))

        fig = plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_values, X_te, show=False, plot_size=None)
        plt.tight_layout()
        shap_summary_path = os.path.join(ART, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("8 · Lưu artifact")
        model.save_model(os.path.join(ART, "model_v1.txt"), num_iteration=model.best_iteration)
        metrics["features"] = FEATURES
        metrics["best_iteration"] = int(model.best_iteration)
        metrics_path = os.path.join(ART, "metrics.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        imp = pd.DataFrame({"feature": model.feature_name(),
                            "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
        imp_path = os.path.join(ART, "feature_importance.csv")
        imp.to_csv(imp_path, index=False)
        print(imp.to_string(index=False))

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(imp_path)
        mlflow.log_artifact(shap_imp_path)
        mlflow.log_artifact(shap_summary_path)
        mlflow.lightgbm.log_model(model, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
 