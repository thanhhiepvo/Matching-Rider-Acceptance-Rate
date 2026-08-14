"""MambaTab (Ahamed & Cheng, 2024 — arXiv:2401.08867, "MambaTab: A Plug-and-Play Model for
Learning Tabular Data") — model class RIÊNG, KHÁC với `train_mamba.py` đã có trong project.

Khác biệt kiến trúc cốt lõi so với `train_mamba.py` (Mamba-tabular tự chế theo kiểu
FT-Transformer, mỗi feature 1 token, chuỗi dài ~21):
  - MambaTab KHÔNG token hoá từng feature. Toàn bộ hàng dữ liệu (mọi feature, numeric lẫn
    categorical đã ordinal-encode) được nén qua 1 feed-forward layer thành 1 EMBEDDING DUY
    NHẤT, rồi coi embedding đó là 1 "token" duy nhất — sequence length = 1 (Batch, Length=1,
    Dimension=d_model). Do đó tính "nhân quả" của Mamba gần như không có ý nghĩa ở đây (chỉ 1
    bước thời gian) — giá trị của khối Mamba trong kiến trúc này là làm 1 "non-linear block"
    (gating + conv + selective-scan suy biến còn 1 bước) xếp chồng nhiều lớp (M blocks, có
    residual) để tăng độ sâu, tương tự tinh thần 1 MLP sâu hơn là "sequence modeling" thật sự.
  - Số Mamba block M mặc định trong paper là 1 (thử tới 100 vẫn ổn định nhờ residual) — ở đây
    dùng M=3 để so sánh công bằng độ sâu với FT-Transformer/Mamba-tabular (N_LAYERS=3).
  - Paper dùng preprocessing riêng (ordinal-encode categorical, min-max [0,1] numeric, impute
    mode) và split ngẫu nhiên 70/10/20. Ở đây ĐỔI sang preprocessing + split THEO THỜI GIAN
    (log1p+standardize numeric qua `mlp_common.py`, `split.py`) để so sánh công bằng với 5
    model class khác trong báo cáo — đây là điểm khác duy nhất so với paper gốc, còn kiến trúc
    (embedding 1 token + Mamba block stack + linear head) giữ đúng.
  - Optimizer/LR theo đúng paper: Adam, lr=1e-4, cosine-annealing (không phải AdamW/1e-3 hay
    3e-4 như 2 model Transformer/Mamba khác trong project).

Tái sử dụng `MambaBlock`/`MambaTabularLayer` (cơ chế S6 thuần PyTorch) đã có sẵn ở
`train_mamba.py` — hoạt động đúng với sequence length=1 (selective scan chỉ chạy 1 bước, vẫn
đúng công thức, không cần sửa).

    python3 Baseline/train_mambatab.py
"""
from __future__ import annotations

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from mlp_common import (
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from pipeline_diagram import ClassificationHeadViz, MambaEncoderViz, save_pipeline_diagram
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_mamba import MambaTabularLayer

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS = 1000        # paper: 1000 epoch + early-stop patience=5 (embedding->1 token nên rẻ hơn hẳn per-epoch)
BATCH_SIZE = 512
LR = 1e-4             # đúng paper — thấp hơn hẳn 2 model token-hoá khác (1e-3 / 3e-4)
WEIGHT_DECAY = 0.0    # paper không dùng weight decay
PATIENCE = 5          # đúng paper
D_MODEL = 32          # "Embedded representation size" — đúng paper
M_BLOCKS = 3           # paper mặc định M=1; dùng 3 để so độ sâu công bằng với FT-Transformer/Mamba (N_LAYERS=3)
D_STATE = 32           # "SSM state expansion factor (N)" — đúng paper (khác D_STATE=16 của train_mamba.py)
D_CONV = 4              # đúng paper
DROPOUT = 0.0           # paper không nêu dropout trong khối Mamba — để 0, khác 0.15 của model kia

torch.manual_seed(SEED)
np.random.seed(SEED)


class MambaTabEmbedding(nn.Module):
    """1 feed-forward layer nén TOÀN BỘ hàng feature (numeric + categorical ordinal-encode)
    thành 1 embedding d_model chiều — LayerNorm rồi ReLU, đúng thứ tự paper mô tả."""

    def __init__(self, n_features: int, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, d_model) — sequence length CỐ ĐỊNH = 1
        return self.act(self.norm(self.proj(x))).unsqueeze(1)


class MambaTab(nn.Module):
    def __init__(self, n_features: int, d_model: int = D_MODEL, m_blocks: int = M_BLOCKS,
                 d_state: int = D_STATE, d_conv: int = D_CONV, dropout: float = DROPOUT):
        super().__init__()
        self.embedding = MambaTabEmbedding(n_features, d_model)
        self.blocks = nn.ModuleList([
            MambaTabularLayer(d_model, dropout, d_state=d_state, d_conv=d_conv, expand=2)
            for _ in range(m_blocks)
        ])
        self.head = nn.Linear(d_model, 1)  # đúng paper — KHÔNG có LayerNorm/ReLU/Dropout trước head như model kia

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)          # (batch, 1, d_model)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(1)  # chỉ 1 vị trí duy nhất (length=1)


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
    ap.add_argument("--m-blocks", type=int, default=M_BLOCKS)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mambatab-v1"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "mambatab",
                          "paper": "arXiv:2401.08867"})
        mlflow.log_params({
            "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "d_model": D_MODEL, "m_blocks": args.m_blocks, "d_state": D_STATE, "d_conv": D_CONV,
            "dropout": DROPOUT, "patience": PATIENCE, "device": str(DEVICE), "n_features": len(FEATURES),
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

        print("2 · Tiền xử lý (log1p + scale numeric; categorical -> ordinal code; NÉN thành 1 vector phẳng)")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
        tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
        # apply_categorical_encoders() trả thẳng mảng ĐÃ stack (Batch, n_cat) theo đúng thứ tự
        # `encoders` (insertion order == CATEGORICAL_FEATURES), KHÔNG phải dict như tr_codes/va_codes.
        te_cat_arr = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
        # Ordinal code -> float, NỐI THẲNG vào vector numeric (không qua embedding table riêng
        # như FeatureTokenizer/FT-Transformer) — đúng "ordinal encoding" của paper MambaTab.
        X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1).astype(np.float32)
        X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1).astype(np.float32)
        X_te_cat = te_cat_arr.astype(np.float32)
        X_tr_flat = np.concatenate([X_tr_num, X_tr_cat], axis=1)
        X_va_flat = np.concatenate([X_va_num, X_va_cat], axis=1)
        X_te_flat = np.concatenate([X_te_num, X_te_cat], axis=1)
        n_features = X_tr_flat.shape[1]

        def to_t(x, yv=None):
            t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            if yv is None:
                return t
            return t, torch.tensor(yv.values, dtype=torch.float32).to(DEVICE)

        X_tr_t, y_tr_t = to_t(X_tr_flat, y_tr)
        X_va_t = to_t(X_va_flat)
        X_te_t = to_t(X_te_flat)

        print("3 · Sơ đồ pipeline (embedding 1-token + Mamba block stack + linear head)")
        viz_pipeline = Pipeline([
            ("mambatab_embedding", MambaEncoderViz(n_layers=1, d_state=D_STATE, d_conv=D_CONV, expand=2, dropout=DROPOUT)),
            ("mambatab_blocks", MambaEncoderViz(n_layers=args.m_blocks, d_state=D_STATE, d_conv=D_CONV, expand=2, dropout=DROPOUT)),
            ("classification_head", ClassificationHeadViz(d_token=D_MODEL, layers="Linear(1) — không LayerNorm/ReLU/Dropout")),
        ])
        save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_mambatab.html"))
        mlflow.log_artifact(os.path.join(ART, "pipeline_mambatab.html"))

        print("4 · Huấn luyện (Adam lr=1e-4, cosine-annealing, early-stop trên valid — đúng paper)")
        model = MambaTab(n_features=n_features, d_model=D_MODEL, m_blocks=args.m_blocks,
                          d_state=D_STATE, d_conv=D_CONV, dropout=DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.BCEWithLogitsLoss()

        n_train = len(X_tr_flat)
        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=DEVICE)
            total_loss = 0.0
            for i in range(0, n_train - n_train % BATCH_SIZE, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt.zero_grad()
                logits = model(X_tr_t[idx])
                loss = loss_fn(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            sched.step()
            train_loss = total_loss / (n_train - n_train % BATCH_SIZE)

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(X_va_t)).cpu().numpy()
            va_auc = roc_auc_score(y_va, p_va)
            va_logloss = log_loss(y_va, p_va)
            if epoch <= 10 or epoch % 10 == 0:
                print(f"  [epoch {epoch:4d}] train_loss={train_loss:.4f}  valid_auc={va_auc:.4f}  "
                      f"valid_logloss={va_logloss:.4f}  lr={sched.get_last_lr()[0]:.2e}")
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

        print("5 · Đánh giá trên test")
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(X_te_t)).cpu().numpy()
        metrics = {"test": evaluate(y_te, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Hệ thống đầy đủ (rule + model) — như train.py")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_tr, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_all_num = transform_numeric(X_te_all, scaler)
        X_all_cat = apply_categorical_encoders(X_te_all[CATEGORICAL_FEATURES], encoders).astype(np.float32)
        X_all_flat = np.concatenate([X_all_num, X_all_cat], axis=1)
        with torch.no_grad():
            p_all = torch.sigmoid(model(to_t(X_all_flat))).cpu().numpy()
        p_te_full = np.where(post_te_all, p_all, 0.0)
        metrics["system_full"] = evaluate(y_te_all, p_te_full, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model_path = os.path.join(ART, "mambatab_model.pt")
        torch.save(model.state_dict(), model_path)
        metrics["features"] = FEATURES
        metrics["architecture"] = (f"MambaTab(d_model={D_MODEL}, m_blocks={args.m_blocks}, "
                                    f"d_state={D_STATE}, seq_len=1)")
        metrics_path = os.path.join(ART, "metrics_mambatab.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
