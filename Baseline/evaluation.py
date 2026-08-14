"""Evaluation Baseline dùng chung — module Python, KHÔNG phụ thuộc model cụ thể nào.

Chỉ nhận (y_true, y_prob) nên dùng được cho bất kỳ model nào (LightGBM, MLP, hay model
của người khác trong team) miễn là output ra xác suất. Đây là bản thống nhất chung —
sửa ở đây thì mọi script train (train.py, train_mlp.py, ...) đều tự động cập nhật theo.

    from evaluation import evaluate
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE: chia [0,1] thành n_bins khoảng đều, mỗi bin so |xác suất dự đoán trung bình -
    tỉ lệ thật| rồi lấy trung bình có trọng số theo số mẫu trong bin. Càng gần 0 càng
    "đáng tin" (P(accept)=0.8 thì thực tế ~80% mẫu đó accept), không liên quan tới AUC
    (AUC đo xếp hạng, ECE đo xác suất có đúng nghĩa xác suất hay không).
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += (mask.sum() / len(y)) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_curve(y, p, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """Dữ liệu cho reliability diagram (biểu đồ tin cậy) — P(accept) THỰC TẾ vs P(accept) MODEL
    DỰ ĐOÁN theo từng bin xác suất. Model calibrate tốt -> các điểm nằm sát đường chéo y=x.
    Bổ sung cho `ece` ở evaluate() — ece chỉ trả 1 số tổng hợp, hàm này trả đủ toạ độ để VẼ
    được đường cong, dùng so sánh trực quan trước/sau calibration (calibration.py).

    strategy: 'uniform' (bin đều theo [0,1], cùng logic với expected_calibration_error() ở
    trên) hoặc 'quantile' (mỗi bin cùng số mẫu — hữu ích khi xác suất dồn cụm ở 1 vùng hẹp,
    tránh bin rỗng).
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    fraction_of_positives, mean_predicted_value = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "n_bins": n_bins,
        "strategy": strategy,
    }


def evaluate(y, p, label: str, threshold: float = 0.5, verbose: bool = True) -> dict:
    """Metric suite: Confusion Matrix, ROC-AUC, PR-AUC, LogLoss, Brier, ECE, và
    Precision/Recall/F1/cancel_flagged_rate cho lớp HUỶ (y=0) — vì mục tiêu nghiệp vụ là
    lọc/gắn cờ đơn có nguy cơ huỷ trước khi vào Matching, không phải lớp accept (y=1, đa số).

    threshold: ngưỡng cắt xác suất -> nhãn 0/1 để tính confusion matrix + Precision/Recall/F1.
    0.5 là mặc định trung lập; với bài toán imbalanced (AR nền ~0.81-0.90) có thể cần chỉnh
    lại sau (vd theo Youden's J hoặc cost matrix của bài toán matching).
    """
    p = np.asarray(p)
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    m = {
        "n": int(len(y)),
        "base_rate": round(float(np.mean(y)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        # pr_auc mặc định coi lớp accept (y=1, đa số ~90%) là positive — số cao (~0.96) chủ yếu
        # phản ánh việc đa số dễ đoán, KHÔNG đánh giá khả năng bắt lớp huỷ. pr_auc_cancel mới là
        # con số đúng cho mục tiêu nghiệp vụ (bắt đơn nguy cơ huỷ) — phải đảo cả nhãn lẫn điểm số
        # (1-y, 1-p) vì average_precision_score(y, p, pos_label=0) KHÔNG tự đảo điểm số, sẽ cho
        # kết quả sai (đã kiểm chứng: cho ra ~0.22 trên 1 ví dụ toy lẽ ra phải là 1.0).
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "pr_auc_cancel": round(float(average_precision_score(1 - np.asarray(y), 1 - p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        # Lớp "huỷ" (y=0): pos_label=0 vì đây là lớp cần bắt được, không phải mặc định
        # sklearn (positive=1=accept).
        "precision_cancel": round(float(precision_score(y, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_cancel": round(float(recall_score(y, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_cancel": round(float(f1_score(y, y_pred, pos_label=0, zero_division=0)), 4),
        # Tỉ lệ đơn bị gắn cờ "nguy cơ huỷ" (y_pred=0) trên tổng — tức tỉ lệ sẽ bị lọc khỏi
        # Matching nếu dùng threshold này trong production.
        "cancel_flagged_rate": round((tn + fn) / len(y), 4),
        "confusion_matrix": {
            "threshold": threshold,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }

    if verbose:
        cm = m["confusion_matrix"]
        print(f"  [{label:9s}] n={m['n']:>7,} · AR nền {m['base_rate']:.4f} · ROC-AUC {m['roc_auc']:.4f}"
              f" · PR-AUC {m['pr_auc']:.4f} (PR-AUC huỷ {m['pr_auc_cancel']:.4f})"
              f" · Brier {m['brier']:.4f} · ECE {m['ece']:.4f}")
        print(f"  {'':11s} confusion@{threshold} — TP={cm['tp']:,} FP={cm['fp']:,} "
              f"FN={cm['fn']:,} TN={cm['tn']:,}")
        print(f"  {'':11s} lớp huỷ — Precision {m['precision_cancel']:.4f} · "
              f"Recall {m['recall_cancel']:.4f} · F1 {m['f1_cancel']:.4f} · "
              f"cancel_flagged_rate {m['cancel_flagged_rate']:.4f}")

    return m
