"""Ensemble LightGBM + MLP — blend xác suất từ 2 model đã train sẵn (không train lại).

Load model_v1.txt (LightGBM) + mlp_model.pt/mlp_preprocess.pkl (MLP v2 log1p) từ artifacts/,
predict lại trên cùng test set, rồi blend theo 3 cách không cần tune hyperparameter (tránh
peek vào test set để chọn trọng số — không đáng tin cậy với 1 tham số trên 1 test set nhỏ):
  - simple average:   (p_lgb + p_mlp) / 2
  - geometric mean:   sqrt(p_lgb * p_mlp)
  - rank average:     trung bình rank (bền với model có scale xác suất lệch nhau)

    python3 Baseline/ensemble.py
"""
from __future__ import annotations

import os

# LightGBM (libomp Homebrew) + PyTorch trong CÙNG 1 process trên macOS bị deadlock ở lần
# predict/forward đầu tiên của PyTorch (2 runtime OpenMP giẫm chân nhau) — phải set TRƯỚC khi
# import 2 thư viện này, không set sau được.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from evaluation import evaluate
from features import FEATURES, PRE_CATEGORICAL, build_features, load_raw
from mlp_common import TabularMLP, apply_categorical_encoders, transform_numeric
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW, TEST_DATE

torch.set_num_threads(1)
DEVICE = torch.device("cpu")  # inference nhỏ, không cần MPS


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


def rank_average(p_lgb: np.ndarray, p_mlp: np.ndarray) -> np.ndarray:
    r_lgb = rankdata(p_lgb) / len(p_lgb)
    r_mlp = rankdata(p_mlp) / len(p_mlp)
    return (r_lgb + r_mlp) / 2


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ensemble-lgb-mlp-v1"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "ensemble_lgb_mlp"})

        print("1 · Nạp data + build feature + áp rule is_post_dispatch (giống hệt train.py/train_mlp.py)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily)
        is_test_full = df.order_date >= pd.Timestamp(TEST_DATE)
        post_mask = df.is_post_dispatch.astype(int) == 1
        X_tr = X[~is_test_full & post_mask]
        X_te, y_te = X[is_test_full & post_mask], y[is_test_full & post_mask]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)

        print("2 · Load LightGBM (model_v1.txt) và predict")
        booster = lgb.Booster(model_file=os.path.join(ART, "model_v1.txt"))
        p_lgb = booster.predict(X_te)

        print("3 · Load MLP (mlp_model.pt + mlp_preprocess.pkl) và predict")
        prep = joblib.load(os.path.join(ART, "mlp_preprocess.pkl"))
        X_te_num = transform_numeric(X_te, prep["scaler"])
        X_te_cat = apply_categorical_encoders(X_te[prep["categorical_features"]], prep["encoders"])
        mlp = TabularMLP(n_numeric=len(prep["numeric_features"]),
                          cat_cardinalities=prep["cat_cardinalities"],
                          embed_dim=prep["embed_dim"]).to(DEVICE)
        mlp.load_state_dict(torch.load(os.path.join(ART, "mlp_model.pt"), map_location=DEVICE))
        mlp.eval()
        with torch.no_grad():
            p_mlp = torch.sigmoid(mlp(
                torch.tensor(X_te_num, dtype=torch.float32),
                torch.tensor(X_te_cat, dtype=torch.long),
            )).numpy()

        print(f"  p_lgb: AUC riêng lẻ {evaluate(y_te, p_lgb, 'lgb-solo', verbose=False)['roc_auc']:.4f}")
        print(f"  p_mlp: AUC riêng lẻ {evaluate(y_te, p_mlp, 'mlp-solo', verbose=False)['roc_auc']:.4f}")

        print("\n4 · Blend (không tune trọng số trên test — tránh overfit test set nhỏ)")
        blends = {
            "avg": (p_lgb + p_mlp) / 2,
            "geomean": np.sqrt(np.clip(p_lgb, 1e-9, 1) * np.clip(p_mlp, 1e-9, 1)),
            "rank_avg": rank_average(p_lgb, p_mlp),
        }

        metrics = {"solo": {
            "lgb": evaluate(y_te, p_lgb, "lgb-solo"),
            "mlp": evaluate(y_te, p_mlp, "mlp-solo"),
        }, "blends": {}}

        # Log gọn 1 con số/biến thể để so sánh nhanh trên MLflow (roc_auc, brier, ece) — chi
        # tiết đầy đủ (Precision/Recall/confusion matrix/...) từng biến thể vẫn có trong
        # metrics_ensemble.json, không cần biến MLflow thành 90 metric key.
        mlflow.log_metrics({
            "compare_lgb_solo_roc_auc": metrics["solo"]["lgb"]["roc_auc"],
            "compare_mlp_solo_roc_auc": metrics["solo"]["mlp"]["roc_auc"],
        })

        for name, p_blend in blends.items():
            print(f"\n  --- blend: {name} ---")
            m_test = evaluate(y_te, p_blend, f"{name}-test")
            metrics["blends"][name] = {"test": m_test}
            mlflow.log_metrics({
                f"compare_{name}_roc_auc": m_test["roc_auc"],
                f"compare_{name}_brier": m_test["brier"],
                f"compare_{name}_ece": m_test["ece"],
            })

        # rank_avg thắng AUC nhưng output là rank percentile, KHÔNG phải xác suất hợp lệ (Brier vỡ,
        # confusion matrix @0.5 sai) — downstream (matching cost function) cần xác suất thật, nên
        # chỉ chọn "best" trong nhóm blend còn giữ được xác suất đúng nghĩa (avg, geomean).
        VALID_PROB_BLENDS = ["avg", "geomean"]
        best_name = max(VALID_PROB_BLENDS, key=lambda n: metrics["blends"][n]["test"]["roc_auc"])
        print(f"\n5 · Blend tốt nhất (giữ xác suất hợp lệ) trên test: '{best_name}' "
              f"(ROC-AUC {metrics['blends'][best_name]['test']['roc_auc']:.4f} "
              f"so với LightGBM solo {metrics['solo']['lgb']['roc_auc']:.4f})")
        mlflow.log_param("best_blend", best_name)

        # Log THÊM blend tốt nhất dưới tên metric PHẲNG (test_roc_auc, ...) — khớp đúng quy ước
        # của train.py/train_mlp.py/train_hybrid.py, để MLflow gộp chung 1 biểu đồ.
        log_metrics("test", metrics["blends"][best_name]["test"])

        # (Tham khảo) Hiệu năng HỆ THỐNG đầy đủ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
        # blend tốt nhất (post-dispatch) — xem giải thích chi tiết trong train.py.
        X_te_full, y_te_full = X[is_test_full], y[is_test_full]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te_full[c] = pd.Categorical(X_te_full[c], categories=X_tr[c].cat.categories)
        post_te_full = df[is_test_full].is_post_dispatch.astype(int).values == 1
        p_lgb_full = booster.predict(X_te_full)
        X_full_num = transform_numeric(X_te_full, prep["scaler"])
        X_full_cat = apply_categorical_encoders(X_te_full[prep["categorical_features"]], prep["encoders"])
        with torch.no_grad():
            p_mlp_full = torch.sigmoid(mlp(
                torch.tensor(X_full_num, dtype=torch.float32),
                torch.tensor(X_full_cat, dtype=torch.long),
            )).numpy()
        p_blend_full = {"avg": (p_lgb_full + p_mlp_full) / 2,
                         "geomean": np.sqrt(np.clip(p_lgb_full, 1e-9, 1) * np.clip(p_mlp_full, 1e-9, 1))}[best_name]
        p_te_full = np.where(post_te_full, p_blend_full, 0.0)
        metrics["system_full"] = evaluate(y_te_full, p_te_full, "system-full")
        log_metrics("system_full", metrics["system_full"])

        metrics_path = os.path.join(ART, "metrics_ensemble.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
