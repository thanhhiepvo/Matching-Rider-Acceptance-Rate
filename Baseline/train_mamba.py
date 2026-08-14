"""Mamba (selective state-space model, Gu & Dao 2023) — model class thứ 6, quick baseline
kiểu tương tự FT-Transformer (train_ft_transformer.py): CÙNG cách token hoá feature (mỗi
feature 1 token qua FeatureTokenizer), nhưng thay self-attention bằng 1 khối Mamba/SSM để
mix thông tin giữa các token.

⚠️ Lưu ý quan trọng: `mamba-ssm` (package chính thức) cần CUDA kernel, KHÔNG chạy được trên
macOS/MPS — nên file này tự cài đặt lại cơ chế S6 (selective scan) THUẦN PyTorch, đúng công
thức trong paper (không dùng kernel tối ưu). Với chuỗi ngắn (~21 token: 1 CLS + feature) vòng
lặp tuần tự trong `selective_scan()` không phải vấn đề tốc độ — lợi thế thật sự của Mamba
(linear-time trên chuỗi RẤT DÀI) chưa chắc phát huy ở quy mô 21 token này; đây là thử nghiệm
"lắp thử xem sao" (quick baseline), không phải bài toán Mamba được sinh ra để giải.

    python3 Baseline/train_mamba.py
"""
from __future__ import annotations

import argparse
import json
import math
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from pipeline_diagram import (
    ClassificationHeadViz,
    FeatureTokenizerViz,
    MambaEncoderViz,
    save_pipeline_diagram,
)
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_ft_transformer import FeatureTokenizer

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS = 40
BATCH_SIZE = 512
LR = 3e-4        # thấp hơn FT-Transformer (1e-3) — SSM nhạy với LR cao hơn Transformer/MLP
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0  # chống nổ gradient qua nhiều bước selective-scan liên tiếp
PATIENCE = 5
D_TOKEN = 32
N_LAYERS = 3
D_STATE = 16     # chiều không gian trạng thái ẩn (state) của SSM
D_CONV = 4       # kernel size của causal conv1d trước selective scan
EXPAND = 2       # hệ số mở rộng d_inner = EXPAND * D_TOKEN (giống paper gốc)
DROPOUT = 0.15

torch.manual_seed(SEED)
np.random.seed(SEED)


class MambaBlock(nn.Module):
    """Cài lại thuần PyTorch cơ chế S6 (selective state-space) của Mamba — KHÔNG dùng kernel
    CUDA tối ưu của package `mamba-ssm` gốc (không chạy được trên macOS). Từng bước:
      1. Projection đầu vào -> tách 2 nhánh (x để đi qua SSM, z để "cổng" đầu ra).
      2. Causal depthwise conv1d + SiLU trên nhánh x (trộn thông tin cục bộ giữa các token liền kề).
      3. Từ x, sinh ra dt/B/C PHỤ THUỘC INPUT (input-dependent) — đây là điểm "selective" khác
         SSM cổ điển (A/B/C cố định) — cho phép model "chọn" nhớ/quên tuỳ nội dung từng token.
      4. Selective scan tuần tự qua từng vị trí: h_t = exp(dt_t·A)·h_{t-1} + dt_t·B_t·x_t,
         y_t = C_t·h_t + D·x_t.
      5. Cổng đầu ra bằng z (SiLU(z) * y) rồi projection về d_model.
    """

    def __init__(self, d_model: int, d_state: int = D_STATE, d_conv: int = D_CONV, expand: int = EXPAND):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # Init dt_proj.bias sao cho softplus(dt_proj(x)) ~ Uniform[dt_min, dt_max] ở lúc khởi
        # tạo (dt nhỏ, "nhớ lâu") — CHI TIẾT BẮT BUỘC của Mamba gốc, thiếu bước này state dễ
        # nổ/collapse ngay vài epoch đầu (đã gặp thực tế: model collapse về hằng số, AUC=0.5).
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt_init = dt_init.clamp(min=1e-4)
        inv_softplus_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # nghịch đảo softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)
            self.dt_proj.weight.mul_(0.1)  # weight nhỏ để bias (đã tune) quyết định phần lớn dt lúc đầu

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))   # A = -exp(A_log) < 0, đảm bảo state ổn định (không nổ)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)  # (b, l, d_inner) mỗi nhánh

        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :l].transpose(1, 2)  # causal: cắt còn đúng l bước
        x_conv = F.silu(x_conv)

        dt, Bp, Cp = torch.split(self.x_proj(x_conv), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (b, l, d_inner) — luôn dương, "tốc độ cập nhật" state

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(l):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)                 # (b, d_inner, d_state)
            dB = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1)         # (b, d_inner, d_state)
            h = h * dA + dB * x_conv[:, t].unsqueeze(-1)
            ys.append((h * Cp[:, t].unsqueeze(1)).sum(-1))              # (b, d_inner)
        y = torch.stack(ys, dim=1) + x_conv * self.D                    # (b, l, d_inner) + skip (D)

        return self.out_proj(y * F.silu(z))


class MambaTabularLayer(nn.Module):
    """Pre-norm residual block quanh MambaBlock — cùng pattern norm_first=True đã dùng cho
    Transformer encoder layer ở train_ft_transformer.py, chỉ đổi cơ chế mix (SSM thay attention)."""

    def __init__(self, d_model: int, dropout: float = DROPOUT, **mamba_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, **mamba_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaTabular(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_token: int = D_TOKEN,
                 n_layers: int = N_LAYERS, d_state: int = D_STATE, d_conv: int = D_CONV,
                 expand: int = EXPAND, dropout: float = DROPOUT):
        super().__init__()
        # cls_position="last" BẮT BUỘC — Mamba nhân quả (causal), token ở vị trí 0 KHÔNG BAO
        # GIỜ "thấy" được các token phía sau nó, nên CLS phải đứng CUỐI (vị trí duy nhất thấy
        # toàn bộ chuỗi). Đặt "first" như FT-Transformer sẽ khiến CLS hoàn toàn không phụ thuộc
        # input — bug thực tế đã gặp: AUC=0.5000 y hệt cho MỌI input, đã trace tận gốc.
        self.tokenizer = FeatureTokenizer(n_numeric, cat_cardinalities, d_token, cls_position="last")
        self.layers = nn.ModuleList([
            MambaTabularLayer(d_token, dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_token, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.head(tokens[:, -1]).squeeze(1)  # token CUỐI — vị trí duy nhất thấy toàn chuỗi


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
    ap.add_argument("--imbalance", choices=["none", "focal"], default="none")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "mamba-tabular-v1" + (f"-{args.imbalance}" if args.imbalance != "none" else "")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "mamba_tabular", "imbalance": args.imbalance})
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "d_token": D_TOKEN, "n_layers": N_LAYERS, "d_state": D_STATE, "d_conv": D_CONV,
            "expand": EXPAND, "dropout": DROPOUT, "grad_clip": GRAD_CLIP,
            "patience": PATIENCE, "device": str(DEVICE),
            "n_features": len(FEATURES), "imbalance": args.imbalance,
        })

        print("1 · Nạp data + chia train/valid/test THEO THỜI GIAN (split.py, post-dispatch only)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")
        mlflow.log_params({"n_train": len(X_tr), "n_valid": len(X_va), "n_test": len(X_te)})

        print("2 · Tiền xử lý (log1p + scale numeric, encode categorical — dùng chung mlp_common.py)")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
        tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
        X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
        cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

        def to_tensors(x_num, x_cat, y=None):
            t = (torch.tensor(x_num, dtype=torch.float32).to(DEVICE),
                 torch.tensor(x_cat, dtype=torch.long).to(DEVICE))
            if y is None:
                return t
            return t + (torch.tensor(y.values, dtype=torch.float32).to(DEVICE),)

        X_tr_num_t, X_tr_cat_t, y_tr_t = to_tensors(X_tr_num, X_tr_cat, y_tr)
        X_va_num_t, X_va_cat_t = to_tensors(X_va_num, X_va_cat)
        X_te_num_t, X_te_cat_t = to_tensors(X_te_num, X_te_cat)

        # Sơ đồ pipeline (3 khối kiến trúc thật, bọc bằng BaseEstimator "vỏ" chỉ để trực quan
        # hoá — model thật vẫn là MambaTabular/nn.Module bên dưới, xem pipeline_diagram.py).
        # cls_position="last" (không phải "first" như FT-Transformer) vì Mamba nhân quả — xem
        # mục 5 báo cáo.
        viz_pipeline = Pipeline([
            ("feature_tokenizer", FeatureTokenizerViz(
                n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=tuple(cat_cardinalities),
                d_token=D_TOKEN, cls_position="last")),
            ("mamba_encoder", MambaEncoderViz(
                n_layers=N_LAYERS, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND, dropout=DROPOUT)),
            ("classification_head", ClassificationHeadViz(d_token=D_TOKEN)),
        ])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_mamba.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_mamba.html"))

        print("3 · Huấn luyện (early-stop trên valid)")
        model = MambaTabular(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if args.imbalance == "focal":
            alpha = compute_focal_alpha(y_tr.values)
            loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            mlflow.log_param("focal_alpha", round(alpha, 4))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            mlflow.log_metric("curve_valid_logloss", va_logloss, step=epoch)

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

        print("4 · Đánh giá trên test")
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
        metrics = {"test": evaluate(y_te, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("5 · Hệ thống đầy đủ (rule + model) — như train.py")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_tr, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_all_num = transform_numeric(X_te_all, scaler)
        X_all_cat = apply_categorical_encoders(X_te_all[CATEGORICAL_FEATURES], encoders)
        X_all_num_t, X_all_cat_t = to_tensors(X_all_num, X_all_cat)
        with torch.no_grad():
            p_all = torch.sigmoid(model(X_all_num_t, X_all_cat_t)).cpu().numpy()
        p_te_full = np.where(post_te_all, p_all, 0.0)
        metrics["system_full"] = evaluate(y_te_all, p_te_full, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("6 · Lưu artifact")
        model_path = os.path.join(ART, "mamba_tabular_model.pt")
        torch.save(model.state_dict(), model_path)
        metrics["features"] = FEATURES
        metrics["architecture"] = f"MambaTabular(d_token={D_TOKEN}, layers={N_LAYERS}, d_state={D_STATE})"
        metrics_path = os.path.join(ART, "metrics_mamba.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
