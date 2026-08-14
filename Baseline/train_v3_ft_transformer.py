"""Task 4 (W3) — Model V3 trên FT-Transformer: tổng hợp toàn bộ kỹ thuật đã thử nghiệm ở Task 3
thành 1 pipeline đầy đủ, dùng model class ĐỀ XUẤT theo tiêu chí SOTA/ROC-AUC (mục 3-4 báo cáo),
KHÔNG phải theo tiêu chí tốc độ train/predict:

  - Model class: FT-Transformer — ROC-AUC cao nhất trong 6 model class đã thử (0,7475 bản mặc
    định, standalone — xem compare_models.py/train_ft_transformer.py), kiến trúc SOTA cho dữ
    liệu tabular trong nhiều benchmark. KHÔNG chọn theo train/predict time (XGBoost nhanh hơn
    nhiều lần nhưng ROC-AUC thấp hơn) — quyết định chọn theo yêu cầu SOTA/độ chính xác trước.
  - Imbalance handling: FocalLoss (imbalance.py, alpha=base rate lớp huỷ, gamma=2.0) — tương
    đương vai trò scale_pos_weight ở GBDT nhưng phù hợp loss PyTorch hơn (đã hỗ trợ sẵn qua
    `--imbalance focal` ở train_ft_transformer.py).
  - Calibration: chọn TỰ ĐỘNG giữa SplineCalib/Isotonic/Platt (calibration.py) — fit trên
    `calib`, CHỌN theo ECE đo trên `valid` (không đụng `test` ở bước chọn này).
  - Threshold: chọn lại (tối đa F1 lớp huỷ) trên `calib` (đã calibrate), tránh peek vào `test`
    — cùng logic pick_threshold() như train_v3.py (XGBoost).

So sánh với `train_v3.py` (XGBoost V2 + scale_pos_weight + calibration) — file đó vẫn được giữ
làm phương án THAY THẾ nhanh/nhẹ hạ tầng hơn cho production (train/predict nhanh hơn nhiều lần,
không cần GPU) — 2 file phục vụ 2 câu hỏi khác nhau: "model tốt nhất theo độ chính xác" (file
này) vs "model thực dụng nhất để vận hành" (train_v3.py).

    python3 Baseline/train_v3_ft_transformer.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from calibration import CALIBRATION_METHODS, apply_calibrator, fit_calibrator
from evaluation import evaluate, reliability_curve
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_ft_transformer import (
    D_FFN,
    D_TOKEN,
    DEVICE,
    DROPOUT,
    N_HEADS,
    N_LAYERS,
    NUMERIC_FEATURES,
    FTTransformer,
)

EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def pick_threshold(y_calib, p_calib_calibrated) -> float:
    """Quét threshold trên `calib` (ĐÃ calibrate, KHÔNG phải test) — chọn threshold cho F1 lớp
    huỷ (y=0) cao nhất, cùng logic pick_threshold() ở train_v3.py."""
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
    ax.set_title("Model V3 (FT-Transformer) — Reliability diagram trước/sau calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model-v3-ft-transformer-focal-calibrated"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "ft_transformer",
                          "purpose": "model_v3_full_pipeline_sota"})
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "d_token": D_TOKEN, "n_layers": N_LAYERS, "n_heads": N_HEADS, "d_ffn": D_FFN,
            "dropout": DROPOUT, "patience": PATIENCE, "device": str(DEVICE),
        })

        print("1 · Nạp data + chia train/valid/calib/test THEO THỜI GIAN (split.py)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_ca, y_ca = align_categories(X_tr, split.calib[0]), split.calib[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · calib {len(X_ca):,} · test {len(X_te):,}")

        print("2 · Tiền xử lý (log1p + scale numeric, encode categorical — dùng chung mlp_common.py)")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_ca_num, X_te_num = (
            transform_numeric(d, scaler) for d in (X_tr, X_va, X_ca, X_te)
        )
        tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
        X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_ca_cat = apply_categorical_encoders(X_ca[CATEGORICAL_FEATURES], encoders)
        X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
        cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

        def to_tensors(x_num, x_cat, yv=None):
            t = (torch.tensor(x_num, dtype=torch.float32).to(DEVICE),
                 torch.tensor(x_cat, dtype=torch.long).to(DEVICE))
            if yv is None:
                return t
            return t + (torch.tensor(yv.values, dtype=torch.float32).to(DEVICE),)

        X_tr_num_t, X_tr_cat_t, y_tr_t = to_tensors(X_tr_num, X_tr_cat, y_tr)
        X_va_num_t, X_va_cat_t = to_tensors(X_va_num, X_va_cat)
        X_ca_num_t, X_ca_cat_t = to_tensors(X_ca_num, X_ca_cat)
        X_te_num_t, X_te_cat_t = to_tensors(X_te_num, X_te_cat)

        print("3 · Train FT-Transformer + FocalLoss (imbalance.py) — early-stop trên valid")
        model = FTTransformer(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        alpha = compute_focal_alpha(y_tr.values)
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        mlflow.log_param("focal_alpha", round(alpha, 4))

        n_train = len(X_tr)
        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=DEVICE)
            total_loss = 0.0
            for i in range(0, n_train - n_train % BATCH_SIZE, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt.zero_grad()
                logits = model(X_tr_num_t[idx], X_tr_cat_t[idx])
                loss = loss_fn(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            train_loss = total_loss / (n_train - n_train % BATCH_SIZE)

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
            va_auc = roc_auc_score(y_va, p_va)
            va_logloss = log_loss(y_va, p_va)
            print(f"  [epoch {epoch:2d}] train_loss={train_loss:.4f}  valid_auc={va_auc:.4f}  valid_logloss={va_logloss:.4f}")
            mlflow.log_metric("curve_train_loss", train_loss, step=epoch)
            mlflow.log_metric("curve_valid_auc", va_auc, step=epoch)

            if va_auc > best_auc:
                best_auc, best_epoch = va_auc, epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  early stopping ở epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        print(f"  best epoch {best_epoch} · valid AUC {best_auc:.4f}")
        mlflow.log_param("best_epoch", best_epoch)

        model.eval()
        with torch.no_grad():
            p_valid_raw = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
            p_calib_raw = torch.sigmoid(model(X_ca_num_t, X_ca_cat_t)).cpu().numpy()
            p_test_raw = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()

        m_before = evaluate(y_te, p_test_raw, "before-calib", verbose=False)
        print(f"  TRƯỚC calib (test, threshold 0.5): ROC-AUC {m_before['roc_auc']:.4f} · "
              f"ECE {m_before['ece']:.4f} · Recall huỷ {m_before['recall_cancel']:.4f}")

        print("4 · Chọn calibration method — fit trên `calib`, ĐO ECE trên `valid` (không đụng test)")
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

        print("5 · Chọn threshold quyết định — quét trên `calib` ĐÃ calibrate (không đụng test)")
        threshold = pick_threshold(y_ca, p_calib_cal)
        print(f"  threshold chọn được = {threshold:.2f} (mặc định thường dùng 0.5)")
        mlflow.log_param("decision_threshold", round(threshold, 2))

        print("6 · Đánh giá cuối cùng trên test — SAU calibration, ĐÚNG threshold đã chọn")
        m_after = evaluate(y_te, p_test_cal, "model-v3-ft-test", threshold=threshold)
        m_after_default_threshold = evaluate(y_te, p_test_cal, "model-v3-ft-test-thr0.5", threshold=0.5, verbose=False)

        print("7 · Reliability diagram trước/sau")
        rc_before = reliability_curve(y_te, p_test_raw)
        rc_after = reliability_curve(y_te, p_test_cal)
        curves = {
            "trước calib (raw, +focal loss)": (rc_before["fraction_of_positives"], rc_before["mean_predicted_value"]),
            f"sau calib ({best_method})": (rc_after["fraction_of_positives"], rc_after["mean_predicted_value"]),
        }
        plot_path = os.path.join(ART, "model_v3_ft_transformer_reliability.png")
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

        print("\n8 · Tổng kết — TRƯỚC vs SAU (threshold đã chọn) vs SAU (threshold 0.5 mặc định)")
        compare_keys = ["roc_auc", "pr_auc_cancel", "brier", "ece", "precision_cancel", "recall_cancel", "f1_cancel"]
        header = f"{'':28s}" + "".join(f"{k:>16s}" for k in compare_keys)
        print(header)
        for name, m in [("trước calib (thr 0.5)", m_before),
                         (f"sau calib+threshold={threshold:.2f}", m_after),
                         ("sau calib (thr 0.5)", m_after_default_threshold)]:
            print(f"{name:28s}" + "".join(f"{m[k]:>16.4f}" for k in compare_keys))

        print("9 · Lưu artifact")
        model_path = os.path.join(ART, "ft_transformer_model_v3.pt")
        torch.save(model.state_dict(), model_path)
        import joblib
        joblib.dump(best_cal, os.path.join(ART, "model_v3_ft_transformer_calibrator.pkl"))
        out = {
            "model_class": "ft_transformer", "imbalance": "focal_loss", "focal_alpha": alpha,
            "calibration_method": best_method, "decision_threshold": threshold,
            "features": FEATURES, "best_epoch": best_epoch,
            "architecture": f"FT-Transformer(d_token={D_TOKEN}, layers={N_LAYERS}, heads={N_HEADS})",
            "before_calibration": m_before, "after_calibration": m_after,
            "after_calibration_threshold_0.5": m_after_default_threshold,
        }
        metrics_path = os.path.join(ART, "metrics_model_v3_ft_transformer.json")
        json.dump(out, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
