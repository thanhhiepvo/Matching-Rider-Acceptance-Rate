"""Task 3 (W3) — calibration trên tập `calib` RIÊNG (split.py, KHÔNG phải train/valid/test) —
3 kỹ thuật: SplineCalib (ml_insights), Isotonic Regression, Platt scaling (logistic trên
log-odds của xác suất thô).

Vì sao cần: imbalance.py (scale_pos_weight/class_weight/focal loss) cải thiện recall/PR-AUC
nhưng phá vỡ hoàn toàn calibration của xác suất thô — quan sát thực nghiệm: ECE tăng từ ~0,005
lên ~0,3 khi bật `--imbalance weighted/focal` (xem train.py). Calibration ở đây SỬA LẠI xác
suất thô thành xác suất "đúng nghĩa xác suất" (P(accept)=0.8 thì ~80% mẫu đó thật sự accept)
mà KHÔNG đổi thứ hạng — mọi kỹ thuật dưới đây đều là hàm ĐƠN ĐIỆU TĂNG theo p thô nên ROC-AUC
giữ nguyên, chỉ Brier/ECE/reliability curve thay đổi.

Dùng như module (fit_calibrator/apply_calibrator) trong train_v3.py, hoặc chạy trực tiếp để
demo trên LightGBM + scale_pos_weight (biến dạng calibration rõ nhất trong 3 model class đã
train ở W3):

    python3 Baseline/calibration.py
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ml_insights as mli
import mlflow
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate, reliability_curve
from features import build_features, load_raw
from imbalance import compute_scale_pos_weight
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, PARAMS, RAW

CALIBRATION_METHODS = ["spline", "isotonic", "platt"]


def fit_calibrator(method: str, p_calib, y_calib):
    """Fit 1 calibrator trên tập calib — trả về object dùng lại được với apply_calibrator()."""
    p_calib = np.asarray(p_calib, dtype=float)
    y_calib = np.asarray(y_calib, dtype=int)

    if method == "spline":
        cal = mli.SplineCalib(random_state=42)
        cal.fit(p_calib, y_calib)
        return cal
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_calib, y_calib)
        return cal
    if method == "platt":
        # Platt scaling kinh điển: logistic regression trên LOG-ODDS của xác suất thô (1 chiều
        # duy nhất) -> ra xác suất calibrate — fit trên logit(p) chứ không phải p trực tiếp.
        eps = 1e-6
        logit_p = np.log(np.clip(p_calib, eps, 1 - eps) / (1 - np.clip(p_calib, eps, 1 - eps)))
        cal = LogisticRegression()
        cal.fit(logit_p.reshape(-1, 1), y_calib)
        return cal
    raise ValueError(f"Không nhận diện method={method!r}, chọn 1 trong {CALIBRATION_METHODS}")


def apply_calibrator(method: str, calibrator, p) -> np.ndarray:
    """Áp calibrator đã fit lên 1 mảng xác suất thô p bất kỳ (vd test) -> xác suất đã calibrate."""
    p = np.asarray(p, dtype=float)
    if method == "spline":
        return np.asarray(calibrator.predict(p)).ravel()
    if method == "isotonic":
        return calibrator.predict(p)
    if method == "platt":
        eps = 1e-6
        logit_p = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        return calibrator.predict_proba(logit_p.reshape(-1, 1))[:, 1]
    raise ValueError(f"Không nhận diện method={method!r}, chọn 1 trong {CALIBRATION_METHODS}")


def plot_reliability(curves: dict, path: str):
    """curves: {label: (fraction_of_positives, mean_predicted_value)}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="calibrate hoàn hảo (y=x)")
    for label, (frac_pos, mean_pred) in curves.items():
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("P(accept) model dự đoán (trung bình mỗi bin)")
    ax.set_ylabel("P(accept) thực tế (trung bình mỗi bin)")
    ax.set_title("Reliability diagram — trước/sau calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="calibration-lightgbm-weighted-v1"):
        mlflow.set_tags({"phase": "post_dispatch_only", "purpose": "calibration_comparison"})

        print("1 · Nạp data + chia train/valid/calib/test THEO THỜI GIAN (split.py)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_ca, y_ca = align_categories(X_tr, split.calib[0]), split.calib[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · calib {len(X_ca):,} · test {len(X_te):,}")

        print("2 · Train LightGBM + scale_pos_weight (imbalance.py) — biến dạng calibration rõ nhất")
        spw = compute_scale_pos_weight(y_tr)
        run_params = {**PARAMS, "scale_pos_weight": spw}
        mlflow.log_param("scale_pos_weight", round(spw, 4))
        model = lgb.train(
            run_params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
            # first_metric_only: xem giải thích trong train.py — scale_pos_weight làm logloss
            # xấu đi đơn điệu dù AUC vẫn tốt lên, phải tách riêng metric chọn model.
            callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True), lgb.log_evaluation(0)],
        )
        print(f"  dừng ở vòng {model.best_iteration}")

        p_calib_raw = model.predict(X_ca, num_iteration=model.best_iteration)
        p_test_raw = model.predict(X_te, num_iteration=model.best_iteration)

        print("3 · Đánh giá TRƯỚC calibration (xác suất thô)")
        m_before = evaluate(y_te, p_test_raw, "before-calib")
        rc_before = reliability_curve(y_te, p_test_raw)

        print("4 · Fit + áp từng calibrator TRÊN CALIB, đánh giá SAU calibration trên test")
        results = {"before": m_before}
        curves = {"trước calib": (rc_before["fraction_of_positives"], rc_before["mean_predicted_value"])}
        for method in CALIBRATION_METHODS:
            cal = fit_calibrator(method, p_calib_raw, y_ca)
            p_test_cal = apply_calibrator(method, cal, p_test_raw)
            m_after = evaluate(y_te, p_test_cal, f"after-{method}")
            results[method] = m_after
            rc_after = reliability_curve(y_te, p_test_cal)
            curves[f"sau {method}"] = (rc_after["fraction_of_positives"], rc_after["mean_predicted_value"])

            mlflow.log_metrics({
                f"{method}_roc_auc": m_after["roc_auc"], f"{method}_brier": m_after["brier"],
                f"{method}_ece": m_after["ece"], f"{method}_pr_auc_cancel": m_after["pr_auc_cancel"],
            })

        print("5 · So sánh trước/sau (ROC-AUC KHÔNG đổi — mọi calibrator đều đơn điệu tăng)")
        compare_keys = ["roc_auc", "pr_auc_cancel", "log_loss", "brier", "ece"]
        header = f"{'':10s}" + "".join(f"{k:>14s}" for k in compare_keys)
        print(header)
        for name in ["before"] + CALIBRATION_METHODS:
            row = results[name]
            print(f"{name:10s}" + "".join(f"{row[k]:>14.4f}" for k in compare_keys))
        mlflow.log_metrics({f"before_{k}": m_before[k] for k in compare_keys})

        print("6 · Reliability diagram trước/sau")
        plot_path = os.path.join(ART, "reliability_curves.png")
        plot_reliability(curves, plot_path)
        mlflow.log_artifact(plot_path)

        print("7 · Lưu artifact")
        metrics_path = os.path.join(ART, "calibration_comparison.json")
        json.dump(results, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
