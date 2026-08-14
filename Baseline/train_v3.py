"""Task 4 (W3) — Model V3: tổng hợp toàn bộ kỹ thuật đã thử nghiệm ở Task 3 thành 1 pipeline
đầy đủ:

  - Model class: XGBoost — THẮNG trong so sánh 4 model class (compare_models.py: ROC-AUC 0,7495,
    nhanh nhất trong 3 GBDT, khác LightGBM baseline theo đúng ưu tiên DOD item 2).
  - Imbalance handling: scale_pos_weight (imbalance.py) — tăng recall lớp huỷ đáng kể
    (0,03 -> ~0,65 ở threshold 0,5) nhưng phá calibration (ECE ~0,006 -> ~0,3).
  - Calibration: chọn TỰ ĐỘNG giữa SplineCalib/Isotonic/Platt (calibration.py) — fit trên
    `calib`, CHỌN theo ECE đo trên `valid` (không đụng `test` ở bước chọn này) — khôi phục lại
    calibration mà không cần biết trước phương pháp nào tốt nhất.
  - Threshold: calibration khôi phục ECE nhưng cũng "kéo" xác suất về gần base rate thật, làm
    recall@0,5 sau calibration tụt lại gần bằng model gốc (đã quan sát ở calibration.py) — nên
    threshold quyết định KHÔNG giữ 0,5 nữa, mà chọn lại (tối đa F1 lớp huỷ) trên `calib` (đã
    calibrate), tránh peek vào `test`.

Kết quả: model V3 vừa xếp hạng tốt (ROC-AUC ~giữ nguyên nhờ calibration đơn điệu), vừa xác
suất ĐÚNG NGHĨA xác suất (ECE thấp), vừa threshold thực sự bắt được nhiều đơn huỷ hơn bản gốc
— khác với chỉ áp riêng lẻ từng kỹ thuật.

    python3 Baseline/train_v3.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xgboost as xgb

from calibration import CALIBRATION_METHODS, apply_calibrator, fit_calibrator
from evaluation import evaluate, reliability_curve
from features import FEATURES, build_features, load_raw
from imbalance import compute_scale_pos_weight
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_xgboost import PARAMS as XGB_PARAMS


def pick_threshold(y_calib, p_calib_calibrated) -> float:
    """Quét threshold trên `calib` (ĐÃ calibrate, KHÔNG phải test) — chọn threshold cho F1 lớp
    huỷ (y=0) cao nhất. F1 cân bằng precision/recall, phù hợp mục tiêu "gắn cờ đơn nguy cơ huỷ
    nhưng không báo động giả tràn lan" hơn recall/precision đơn lẻ.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = evaluate(y_calib, p_calib_calibrated, "threshold-scan", threshold=t, verbose=False)["f1_cancel"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def plot_reliability(curves: dict, path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="calibrate hoàn hảo (y=x)")
    for label, (frac_pos, mean_pred) in curves.items():
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("P(accept) model dự đoán (trung bình mỗi bin)")
    ax.set_ylabel("P(accept) thực tế (trung bình mỗi bin)")
    ax.set_title("Model V3 — Reliability diagram trước/sau calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model-v3-xgboost-weighted-calibrated"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "xgboost",
                          "purpose": "model_v3_full_pipeline"})

        print("1 · Nạp data + chia train/valid/calib/test THEO THỜI GIAN (split.py)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_ca, y_ca = align_categories(X_tr, split.calib[0]), split.calib[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · calib {len(X_ca):,} · test {len(X_te):,}")

        print("2 · Train XGBoost (đề xuất ở compare_models.py, gần như hoà LightGBM V2) + scale_pos_weight")
        base_params = dict(XGB_PARAMS)
        v2_path = os.path.join(ART, "metrics_xgboost_v2.json")
        if os.path.exists(v2_path):
            base_params = {**XGB_PARAMS, **json.load(open(v2_path))["best_params"]}
            print("  dùng XGBoost V2 (Optuna-tuned) làm nền, cộng thêm scale_pos_weight")
        spw = compute_scale_pos_weight(y_tr)
        run_params = {**base_params, "scale_pos_weight": round(spw, 4), "eval_metric": ["auc"]}
        mlflow.log_params(run_params)

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dcalib = xgb.DMatrix(X_ca, enable_categorical=True)
        dtest = xgb.DMatrix(X_te, enable_categorical=True)
        booster = xgb.train(
            run_params, dtrain, num_boost_round=1000,
            evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=False,
        )
        print(f"  dừng ở vòng {booster.best_iteration}")
        mlflow.log_param("best_iteration", booster.best_iteration)

        p_valid_raw = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        p_calib_raw = booster.predict(dcalib, iteration_range=(0, booster.best_iteration + 1))
        p_test_raw = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

        m_before = evaluate(y_te, p_test_raw, "before-calib", verbose=False)
        print(f"  TRƯỚC calib (test, threshold 0.5): ROC-AUC {m_before['roc_auc']:.4f} · "
              f"ECE {m_before['ece']:.4f} · Recall huỷ {m_before['recall_cancel']:.4f}")

        print("3 · Chọn calibration method — fit trên `calib`, ĐO ECE trên `valid` (không đụng test)")
        best_method, best_ece, best_cal = None, np.inf, None
        for method in CALIBRATION_METHODS:
            cal = fit_calibrator(method, p_calib_raw, y_ca)
            p_valid_cal = apply_calibrator(method, cal, p_valid_raw)
            ece_valid = evaluate(y_va, p_valid_cal, f"{method}-valid", verbose=False)["ece"]
            print(f"  {method:9s} -> ECE(valid) = {ece_valid:.4f}")
            if ece_valid < best_ece:
                best_method, best_ece, best_cal = method, ece_valid, cal
        print(f"  -> chọn '{best_method}' (ECE valid thấp nhất = {best_ece:.4f})")
        mlflow.log_param("calibration_method", best_method)

        p_calib_cal = apply_calibrator(best_method, best_cal, p_calib_raw)
        p_test_cal = apply_calibrator(best_method, best_cal, p_test_raw)

        print("4 · Chọn threshold quyết định — quét trên `calib` ĐÃ calibrate (không đụng test)")
        threshold = pick_threshold(y_ca, p_calib_cal)
        print(f"  threshold chọn được = {threshold:.2f} (mặc định thường dùng 0.5)")
        mlflow.log_param("decision_threshold", round(threshold, 2))

        print("5 · Đánh giá cuối cùng trên test — SAU calibration, ĐÚNG threshold đã chọn")
        m_after = evaluate(y_te, p_test_cal, "model-v3-test", threshold=threshold)
        m_after_default_threshold = evaluate(y_te, p_test_cal, "model-v3-test-thr0.5", threshold=0.5, verbose=False)

        print("6 · Reliability diagram trước/sau")
        rc_before = reliability_curve(y_te, p_test_raw)
        rc_after = reliability_curve(y_te, p_test_cal)
        curves = {
            "trước calib (raw, +scale_pos_weight)": (rc_before["fraction_of_positives"], rc_before["mean_predicted_value"]),
            f"sau calib ({best_method})": (rc_after["fraction_of_positives"], rc_after["mean_predicted_value"]),
        }
        plot_path = os.path.join(ART, "model_v3_reliability.png")
        plot_reliability(curves, plot_path)
        mlflow.log_artifact(plot_path)

        for k in ["roc_auc", "pr_auc", "pr_auc_cancel", "log_loss", "brier", "ece",
                  "precision_cancel", "recall_cancel", "f1_cancel", "cancel_flagged_rate"]:
            mlflow.log_metric(f"before_{k}", m_before[k])
            mlflow.log_metric(f"after_{k}", m_after[k])
            mlflow.log_metric(f"after_thr0.5_{k}", m_after_default_threshold[k])
        cm = m_after["confusion_matrix"]
        mlflow.log_metrics({"after_cm_tp": cm["tp"], "after_cm_fp": cm["fp"],
                             "after_cm_fn": cm["fn"], "after_cm_tn": cm["tn"]})

        print("\n7 · Tổng kết — TRƯỚC vs SAU (threshold đã chọn) vs SAU (threshold 0.5 mặc định)")
        compare_keys = ["roc_auc", "pr_auc_cancel", "brier", "ece", "precision_cancel", "recall_cancel", "f1_cancel"]
        header = f"{'':28s}" + "".join(f"{k:>16s}" for k in compare_keys)
        print(header)
        for name, m in [("trước calib (thr 0.5)", m_before),
                         (f"sau calib+threshold={threshold:.2f}", m_after),
                         ("sau calib (thr 0.5)", m_after_default_threshold)]:
            print(f"{name:28s}" + "".join(f"{m[k]:>16.4f}" for k in compare_keys))

        print("8 · Lưu artifact")
        booster.save_model(os.path.join(ART, "model_v3_xgboost.json"))
        import joblib
        joblib.dump(best_cal, os.path.join(ART, "model_v3_calibrator.pkl"))
        out = {
            "model_class": "xgboost", "imbalance": "scale_pos_weight", "scale_pos_weight": spw,
            "calibration_method": best_method, "decision_threshold": threshold,
            "features": FEATURES, "best_iteration": int(booster.best_iteration),
            "before_calibration": m_before, "after_calibration": m_after,
            "after_calibration_threshold_0.5": m_after_default_threshold,
        }
        metrics_path = os.path.join(ART, "metrics_model_v3.json")
        json.dump(out, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.xgboost.log_model(booster, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
