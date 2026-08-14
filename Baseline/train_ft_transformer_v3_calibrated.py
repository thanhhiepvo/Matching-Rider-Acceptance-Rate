"""FT-Transformer V3 (ple_only + FocalLoss) + Calibration — bước tiếp theo sau khi phát hiện
PLE encoding + FocalLoss cải thiện ROC-AUC/Cancel PR-AUC thật (so `train_ft_transformer_v3.py
--ablation ple_only --loss focal`) NHƯNG phá vỡ calibration (ECE 0,006 -> 0,349). Cùng tinh
thần `train_v3_ft_transformer.py` (Model V3): fit calibrator trên `calib` (KHÔNG train/test),
chọn method có ECE thấp nhất đo trên `valid`, chọn lại threshold (F1 lớp huỷ cao nhất) trên
`calib` đã calibrate, rồi mới đánh giá cuối trên `test`.

    python3 Baseline/train_ft_transformer_v3_calibrated.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from calibration import CALIBRATION_METHODS, apply_calibrator, fit_calibrator
from evaluation import evaluate, reliability_curve
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import apply_categorical_encoders, encode_categoricals
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_ft_transformer_v3 import N_BINS, FTTransformerV3, PiecewiseLinearEncoder

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS, PATIENCE = 60, 7

torch.manual_seed(SEED)
np.random.seed(SEED)


def pick_threshold(y_calib, p_calib_calibrated) -> float:
    """Quét threshold trên `calib` (ĐÃ calibrate) — chọn F1 lớp huỷ cao nhất, cùng logic
    train_v3.py/train_v3_ft_transformer.py."""
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
    ax.set_title("FT-Transformer V3 (ple_only + Focal) — Reliability diagram trước/sau calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    v2_path = os.path.join(ART, "metrics_ft_transformer_v2.json")
    bp = json.load(open(v2_path))["best_params"]
    d_token, n_heads, n_layers, d_ffn, dropout = (
        bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
    lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]

    with mlflow.start_run(run_name="ft-transformer-v3-ple_only-focal-calibrated"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "ft_transformer_v3",
                          "ablation": "ple_only", "loss": "focal", "calibrated": "true"})
        mlflow.log_params({
            "d_token": d_token, "n_heads": n_heads, "n_layers": n_layers, "d_ffn": d_ffn,
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        })

        print("1 · Nạp data + chia train/valid/calib/test (post-dispatch, khoá 13/07)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_ca, y_ca = align_categories(X_tr, split.calib[0]), split.calib[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · calib {len(X_ca):,} · test {len(X_te):,}")

        print("2 · Piecewise-linear encode numeric (fit quantile bins trên train) + encode categorical")
        ple = PiecewiseLinearEncoder(N_BINS).fit(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_ca_num, X_te_num = (
            ple.transform(d, NUMERIC_FEATURES) for d in (X_tr, X_va, X_ca, X_te))

        tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
        X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_ca_cat = apply_categorical_encoders(X_ca[CATEGORICAL_FEATURES], encoders)
        X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
        cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

        def to_t(x, dtype):
            return torch.tensor(x, dtype=dtype).to(DEVICE)

        X_tr_num_t, X_va_num_t, X_ca_num_t, X_te_num_t = (
            to_t(a, torch.float32) for a in (X_tr_num, X_va_num, X_ca_num, X_te_num))
        X_tr_cat_t, X_va_cat_t, X_ca_cat_t, X_te_cat_t = (
            to_t(a, torch.long) for a in (X_tr_cat, X_va_cat, X_ca_cat, X_te_cat))
        y_tr_t = to_t(y_tr.values, torch.float32)

        print("3 · Huấn luyện FTTransformerV3 (ple_only) + FocalLoss — early-stop trên valid")
        model = FTTransformerV3(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                                 n_trees=0, num_leaves_cap=0, d_token=d_token, n_layers=n_layers,
                                 n_heads=n_heads, d_ffn=d_ffn, dropout=dropout,
                                 use_ple=True, use_leaf=False).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        alpha = compute_focal_alpha(y_tr.values)
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        mlflow.log_param("focal_alpha", round(alpha, 4))
        print(f"  loss=FocalLoss(alpha={alpha:.4f}, gamma=2.0)")

        n_train = len(X_tr)
        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=DEVICE)
            total_loss = 0.0
            for i in range(0, n_train - n_train % batch_size, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = model(X_tr_num_t[idx], X_tr_cat_t[idx], None)
                loss = loss_fn(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            train_loss = total_loss / (n_train - n_train % batch_size)

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t, None)).cpu().numpy()
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
            p_valid_raw = torch.sigmoid(model(X_va_num_t, X_va_cat_t, None)).cpu().numpy()
            p_calib_raw = torch.sigmoid(model(X_ca_num_t, X_ca_cat_t, None)).cpu().numpy()
            p_test_raw = torch.sigmoid(model(X_te_num_t, X_te_cat_t, None)).cpu().numpy()

        m_before = evaluate(y_te, p_test_raw, "before-calib", verbose=False)
        print(f"\n4 · TRƯỚC calib (test, threshold 0.5): ROC-AUC {m_before['roc_auc']:.4f} · "
              f"Cancel PR-AUC {m_before['pr_auc_cancel']:.4f} · ECE {m_before['ece']:.4f} · "
              f"Recall huỷ {m_before['recall_cancel']:.4f}")

        print("\n5 · Chọn calibration method — fit trên `calib`, đo ECE trên `valid`")
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

        print("\n6 · Chọn threshold quyết định — quét trên `calib` ĐÃ calibrate")
        threshold = pick_threshold(y_ca, p_calib_cal)
        print(f"  threshold chọn được = {threshold:.2f} (mặc định thường dùng 0.5)")
        mlflow.log_param("decision_threshold", round(threshold, 2))

        print("\n7 · Đánh giá cuối cùng trên test")
        m_after = evaluate(y_te, p_test_cal, "after-calib", threshold=threshold)
        m_after_thr05 = evaluate(y_te, p_test_cal, "after-calib-thr0.5", threshold=0.5, verbose=False)

        rc_before = reliability_curve(y_te, p_test_raw)
        rc_after = reliability_curve(y_te, p_test_cal)
        curves = {
            "trước calib (raw, ple+focal)": (rc_before["fraction_of_positives"], rc_before["mean_predicted_value"]),
            f"sau calib ({best_method})": (rc_after["fraction_of_positives"], rc_after["mean_predicted_value"]),
        }
        plot_path = os.path.join(ART, "ft_v3_ple_focal_calibrated_reliability.png")
        plot_reliability(curves, plot_path)
        mlflow.log_artifact(plot_path)

        for k in ["roc_auc", "pr_auc", "pr_auc_cancel", "log_loss", "brier", "ece",
                  "precision_cancel", "recall_cancel", "f1_cancel", "cancel_flagged_rate"]:
            mlflow.log_metric(f"before_{k}", m_before[k])
            mlflow.log_metric(f"after_{k}", m_after[k])
            mlflow.log_metric(f"after_thr0.5_{k}", m_after_thr05[k])

        print("\n8 · Tổng kết — TRƯỚC vs SAU calib+threshold vs SAU calib (thr 0.5)")
        compare_keys = ["roc_auc", "pr_auc_cancel", "brier", "ece", "precision_cancel", "recall_cancel", "f1_cancel"]
        header = f"{'':30s}" + "".join(f"{k:>16s}" for k in compare_keys)
        print(header)
        for name, m in [("trước calib (thr 0.5)", m_before),
                         (f"sau calib+threshold={threshold:.2f}", m_after),
                         ("sau calib (thr 0.5)", m_after_thr05)]:
            print(f"{name:30s}" + "".join(f"{m[k]:>16.4f}" for k in compare_keys))

        metrics = {"before": m_before, "after": m_after, "after_thr0.5": m_after_thr05,
                   "calibration_method": best_method, "threshold": threshold, "best_epoch": best_epoch}
        metrics_path = os.path.join(ART, "metrics_ft_transformer_v3_ple_focal_calibrated.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)

        print(f"\n✓ -> {metrics_path}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
