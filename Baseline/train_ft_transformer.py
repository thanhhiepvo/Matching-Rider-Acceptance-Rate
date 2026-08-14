"""FT-Transformer (Feature Tokenizer + Transformer, Gorishniy et al. 2021) — model class thứ 5,
trên CÙNG feature set (20) & split (split.py) với LightGBM/XGBoost/CatBoost/MLP.

Ý tưởng khác MLP thuần (mlp_common.TabularMLP): thay vì concat toàn bộ feature thành 1 vector
rồi qua MLP, FT-Transformer biến MỖI FEATURE (numeric lẫn categorical) thành 1 "token" riêng
(numeric: nhân + cộng theo trọng số học được cho từng feature; categorical: embedding như cũ),
rồi cho tự-attention (self-attention) giữa CÁC FEATURE với nhau qua 1 token [CLS] tổng hợp —
mục đích là học được feature nào "chú ý" đến feature nào (interaction), thay vì để MLP tự mò
qua các lớp fully-connected.

Baseline nhanh (quick baseline, KHÔNG phải hướng sequence/lịch sử khách — xem thảo luận trong
hội thoại): dùng ĐÚNG 20 feature phẳng hiện có, không đổi bài toán — nếu không thắng được GBDT
đã tune, đó tự nó là 1 kết quả đáng ghi lại (khớp finding đã biết: MLP thuần cũng thua GBDT
trên bài toán tabular quy mô dữ liệu hiện tại).

    python3 Baseline/train_ft_transformer.py
"""
from __future__ import annotations

import argparse
import json
import os

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    LOG1P_COLS,
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from pipeline_diagram import (
    ClassificationHeadViz,
    FeatureTokenizerViz,
    TransformerEncoderViz,
    save_pipeline_diagram,
)
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
D_TOKEN = 32     # chiều embedding mỗi feature-token
N_LAYERS = 3     # số lớp Transformer encoder
N_HEADS = 4
D_FFN = 64
DROPOUT = 0.15

torch.manual_seed(SEED)
np.random.seed(SEED)


class FeatureTokenizer(nn.Module):
    """Biến mỗi feature thành 1 token d_token chiều — numeric: token_j = x_j * W_j + b_j (mỗi
    feature 1 vector trọng số riêng, KHÔNG share giữa các feature); categorical: embedding
    lookup như thường. Đây là điểm khác biệt cốt lõi với MLP thuần (mlp_common.TabularMLP) —
    numeric feature ở đây có "chỗ đứng" ngang hàng categorical (1 token riêng), không bị ép
    concat phẳng ngay từ đầu.
    """

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_token: int = D_TOKEN,
                 cls_position: str = "first"):
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(n_numeric, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_token) for card in cat_cardinalities  # +1 cho giá trị lạ
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        assert cls_position in ("first", "last")
        self.cls_position = cls_position

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (batch, n_numeric) -> (batch, n_numeric, d_token)
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        feature_tokens = [num_tokens, cat_tokens]
        # "first" (mặc định, dùng cho FT-Transformer/self-attention — thứ tự token không quan
        # trọng vì attention không có tính nhân quả) vs "last" (BẮT BUỘC cho model NHÂN QUẢ như
        # Mamba — token cuối là vị trí DUY NHẤT "thấy" được toàn bộ chuỗi, xem train_mamba.py).
        parts = [cls] + feature_tokens if self.cls_position == "first" else feature_tokens + [cls]
        return torch.cat(parts, dim=1)  # (batch, 1+n_num+n_cat, d_token)


class FTTransformer(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_token: int = D_TOKEN,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS, d_ffn: int = D_FFN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_numeric, cat_cardinalities, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_token, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]  # [CLS] token — tổng hợp toàn bộ self-attention giữa các feature
        return self.head(cls_out).squeeze(1)


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

    run_name = "ft-transformer-v1" + (f"-{args.imbalance}" if args.imbalance != "none" else "")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "ft_transformer", "imbalance": args.imbalance})
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "d_token": D_TOKEN, "n_layers": N_LAYERS, "n_heads": N_HEADS, "d_ffn": D_FFN,
            "dropout": DROPOUT, "patience": PATIENCE, "device": str(DEVICE),
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
        # hoá — model thật vẫn là FTTransformer/nn.Module bên dưới, xem pipeline_diagram.py).
        viz_pipeline = Pipeline([
            ("feature_tokenizer", FeatureTokenizerViz(
                n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=tuple(cat_cardinalities),
                d_token=D_TOKEN, cls_position="first")),
            ("transformer_encoder", TransformerEncoderViz(
                n_layers=N_LAYERS, n_heads=N_HEADS, d_ffn=D_FFN, dropout=DROPOUT)),
            ("classification_head", ClassificationHeadViz(d_token=D_TOKEN)),
        ])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_ft_transformer.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_ft_transformer.html"))

        print("3 · Huấn luyện (early-stop trên valid — KHÔNG phải test, khác train_mlp.py cũ)")
        model = FTTransformer(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities).to(DEVICE)
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
        model_path = os.path.join(ART, "ft_transformer_model.pt")
        torch.save(model.state_dict(), model_path)
        metrics["features"] = FEATURES
        metrics["architecture"] = f"FT-Transformer(d_token={D_TOKEN}, layers={N_LAYERS}, heads={N_HEADS})"
        metrics_path = os.path.join(ART, "metrics_ft_transformer.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
