"""Task 3 (W3) — kỹ thuật xử lý class imbalance. Nhãn lệch: AR nền (accept, y=1) ~0,90-0,91 ở
post-dispatch, tức lớp HUỶ (y=0) — lớp CẦN BẮT theo mục tiêu nghiệp vụ (evaluation.py) — chỉ
chiếm ~9-10%. Mọi hàm dưới đây đều tính trọng số để TĂNG ảnh hưởng của lớp y=0 lên loss, không
phải y=1 (ngược với đa số ví dụ/tutorial imbalance-learning mặc định positive=minority).

Dùng chung 1 chỗ để train.py/train_xgboost.py/train_catboost.py/train_mlp.py áp cùng công
thức, không tự tính riêng lẻ mỗi nơi 1 kiểu.

    from imbalance import compute_scale_pos_weight, compute_class_weights, FocalLoss, compute_focal_alpha
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale_pos_weight(y) -> float:
    """LightGBM/XGBoost: scale_pos_weight nhân vào gradient của lớp POSITIVE (y=1 theo quy ước
    thư viện). Đặt = count(y=0)/count(y=1) (< 1 vì y=1 là lớp ĐA SỐ ở bài toán này) sẽ làm tổng
    trọng số 2 lớp CÂN BẰNG nhau trong loss: count(y=1)*scale_pos_weight = count(y=0) — công
    thức tự động đúng hướng dù lớp thiểu số là 0 hay 1, không cần đảo ngược thủ công.
    """
    y = np.asarray(y)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    return float(n_neg) / float(n_pos)


def compute_class_weights(y) -> dict:
    """CatBoost (`class_weights`)/sklearn-style: trọng số "balanced" chuẩn — mỗi lớp góp trọng
    số TỔNG bằng nhau vào loss. w_c = n_samples / (n_classes * n_c); lớp càng hiếm w càng lớn.
    """
    y = np.asarray(y)
    n = len(y)
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return {0: float(n) / (2 * n0), 1: float(n) / (2 * n1)}


def compute_focal_alpha(y) -> float:
    """Alpha mặc định cho FocalLoss dưới = base rate của lớp y=0 (huỷ, thiểu số) trong dữ liệu
    — lớp càng hiếm, alpha càng nhỏ... nhưng trong FocalLoss bên dưới, trọng số thật sự áp cho
    y=0 là (1-alpha), nên alpha nhỏ => (1-alpha) lớn => lớp y=0 được nhấn mạnh đúng hướng.
    """
    y = np.asarray(y)
    return float((y == 0).mean())


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017), thay cho `nn.BCEWithLogitsLoss()` trong
    train_mlp.py/mlp_common.py khi cần tập trung học nhóm mẫu khó/thiểu số hơn nhóm dễ/đa số.

    alpha: trọng số cho lớp y=1; lớp y=0 nhận (1-alpha). Mặc định lấy alpha =
    `compute_focal_alpha(y_train)` (= base rate y=0) để (1-alpha) — trọng số của lớp huỷ, lớp
    thiểu số — luôn lớn hơn alpha khi y=0 là lớp hiếm, đúng hướng cần.
    gamma: hệ số "focusing" — mẫu dễ (model đã tự tin đúng, p_t gần 1) bị giảm trọng số theo
    (1-p_t)^gamma, buộc gradient tập trung vào mẫu khó/mẫu đang bị đoán sai. gamma=0 tương
    đương weighted BCE thường (không có focusing).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, torch.full_like(targets, self.alpha),
                               torch.full_like(targets, 1 - self.alpha))
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_t * (1 - p_t).clamp(min=1e-6) ** self.gamma * bce
        return loss.mean()
