"""FT-Transformer V3 — 2 thay đổi kiến trúc dựa trên phát hiện ở `reports/W4_dl_error_analysis`
(3 model DL chia sẻ ~97% cùng 1 điểm mù với XGBoost, SHAP hội tụ về cùng feature quan trọng —
gợi ý vấn đề không phải "model chưa tìm ra feature nào", mà là CÁCH biểu diễn feature/tương tác
còn yếu hơn cây quyết định):

1. **Piecewise-linear numeric encoding** (Gorishniy et al. 2022, "On Embeddings for Numerical
   Features in Tabular Deep Learning") — thay vì mỗi numeric feature chỉ 1 phép nhân tuyến tính
   (`FeatureTokenizer.num_weight` ở V1/V2), chia mỗi feature thành N_BINS khoảng theo quantile
   (fit trên train), encode giá trị thành vector "đã đi được bao xa qua từng khoảng" rồi mới
   Linear -> d_token. Cho phép token học được NGƯỠNG phi tuyến giống cây quyết định, thay vì chỉ
   1 đường thẳng duy nhất mỗi feature.
2. **GBDT leaf-embedding token** (kỹ thuật GBDT+NN kinh điển, đã dùng cho MLP ở `train_hybrid.py`
   — tái dùng nguyên `mlp_common.LeafEmbedding`) — train 1 LightGBM V2 (đã tune, `num_leaves=132`)
   làm feature extractor, lấy leaf-index mỗi cây mỗi dòng, embed + mean-pool thành 1 token PHỤ,
   ghép thêm vào chuỗi token trước khi vào Transformer encoder — "mượn" trực tiếp cấu trúc
   tương tác phi tuyến mà cây đã tự học được, thay vì bắt Transformer tự học lại từ đầu.

Giữ NGUYÊN encoder/head + hyperparameter đã tune (`metrics_ft_transformer_v2.json`) để so sánh
công bằng — chỉ đổi CÁCH TOKEN HOÁ đầu vào, không đổi độ sâu/rộng Transformer.

    python3 Baseline/train_ft_transformer_v3.py
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    LeafEmbedding,
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train import PARAMS as LGB_V1_PARAMS

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS, PATIENCE = 60, 7
N_BINS = 48          # số khoảng quantile mỗi numeric feature — theo paper Gorishniy 2022
LEAF_EMBED_DIM = 8   # cùng giá trị với train_hybrid.py

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------------------
# 1 · Piecewise-linear numeric encoding
# ---------------------------------------------------------------------------------------
class PiecewiseLinearEncoder:
    """Fit bin edges theo quantile TRÊN TRAIN cho từng numeric feature (không rò rỉ). Encode 1
    giá trị x thành vector N_BINS chiều — bin đã "đi qua hẳn" = 1, bin đang đứng trong = phân
    số vị trí trong khoảng, bin chưa tới = 0. Numeric feature lệch phải mạnh (total_fee,
    trip_distance_km...) nên bin theo QUANTILE (mật độ đều) thay vì khoảng đều tuyệt đối —
    tránh phần lớn dữ liệu dồn vào 1-2 bin đầu.
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.edges: dict[str, np.ndarray] = {}

    def fit(self, X_tr: pd.DataFrame, numeric_features: list[str]):
        for col in numeric_features:
            vals = X_tr[col].astype(float)
            vals = vals.fillna(vals.median())
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.quantile(vals.values, qs))
            if len(edges) < 2:  # feature gần như hằng số (vd is_weekend đôi khi lệch mạnh)
                edges = np.array([vals.min() - 1.0, vals.max() + 1.0])
            self.edges[col] = edges
        return self

    def transform(self, X: pd.DataFrame, numeric_features: list[str]) -> np.ndarray:
        """Trả về (n, n_features, n_bins) — pad bin thiếu (feature có ít quantile riêng biệt
        hơn n_bins, ví dụ cột gần nhị phân) bằng 0 ở cuối, KHÔNG ảnh hưởng encoding thật vì
        Linear(n_bins, d_token) sau này tự học trọng số ~0 cho phần pad nếu vô nghĩa."""
        n = len(X)
        out = np.zeros((n, len(numeric_features), self.n_bins), dtype=np.float32)
        for j, col in enumerate(numeric_features):
            vals = X[col].astype(float).fillna(X[col].astype(float).median()).values
            edges = self.edges[col]
            n_real_bins = len(edges) - 1
            for k in range(n_real_bins):
                lo, hi = edges[k], edges[k + 1]
                width = hi - lo if hi > lo else 1.0
                frac = np.clip((vals - lo) / width, 0.0, 1.0)
                out[:, j, k] = frac
        return out


class FeatureTokenizerV3(nn.Module):
    """Giống `FeatureTokenizer` (train_ft_transformer.py) nhưng numeric token đi qua
    Linear(n_bins, d_token) trên vector piecewise-linear thay vì Linear(1, d_token) trên giá
    trị thô — categorical token giữ nguyên embedding lookup. Thêm 1 token LEAF (GBDT
    leaf-embedding, mục 2 docstring module) vào cuối chuỗi, trước [CLS]."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], n_trees: int,
                 num_leaves_cap: int, d_token: int, n_bins: int = N_BINS,
                 leaf_embed_dim: int = LEAF_EMBED_DIM, use_ple: bool = True, use_leaf: bool = True):
        super().__init__()
        self.use_ple, self.use_leaf = use_ple, use_leaf
        if use_ple:
            self.num_proj = nn.ModuleList([nn.Linear(n_bins, d_token) for _ in range(n_numeric)])
        else:
            # Ablation: numeric token = num_weight*x + num_bias, GIỐNG HỆT FeatureTokenizer gốc
            # (train_ft_transformer.py) — để cô lập đúng 1 biến thay đổi mỗi lần chạy.
            self.num_weight = nn.Parameter(torch.randn(n_numeric, d_token) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_token))
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_token) for card in cat_cardinalities
        ])
        if use_leaf:
            self.leaf_embedding = LeafEmbedding(n_trees, num_leaves_cap, leaf_embed_dim)
            self.leaf_proj = nn.Linear(leaf_embed_dim, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x_num, x_cat: torch.Tensor, leaf_idx: torch.Tensor | None) -> torch.Tensor:
        if self.use_ple:
            # x_num: (batch, n_numeric, n_bins)
            num_tokens = torch.stack(
                [proj(x_num[:, j, :]) for j, proj in enumerate(self.num_proj)], dim=1
            )
        else:
            # x_num: (batch, n_numeric) — scalar, giống FeatureTokenizer gốc
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        parts = [cls, num_tokens, cat_tokens]
        if self.use_leaf:
            leaf_tok = self.leaf_proj(self.leaf_embedding(leaf_idx)).unsqueeze(1)
            parts.append(leaf_tok)
        return torch.cat(parts, dim=1)


class FTTransformerV3(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], n_trees: int, num_leaves_cap: int,
                 d_token: int, n_layers: int, n_heads: int, d_ffn: int, dropout: float, n_bins: int = N_BINS,
                 use_ple: bool = True, use_leaf: bool = True):
        super().__init__()
        self.tokenizer = FeatureTokenizerV3(n_numeric, cat_cardinalities, n_trees, num_leaves_cap,
                                             d_token, n_bins, use_ple=use_ple, use_leaf=use_leaf)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_token, 1),
        )

    def forward(self, x_num, x_cat, leaf_idx=None):
        tokens = self.tokenizer(x_num, x_cat, leaf_idx)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(1)


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
    ap.add_argument("--ablation", choices=["full", "ple_only", "leaf_only", "none"], default="full",
                     help="full: PLE + leaf-hybrid (mặc định) · ple_only: chỉ đổi numeric encoding, "
                          "GIỮ NGUYÊN tokenizer categorical/CLS gốc, KHÔNG thêm leaf token · "
                          "leaf_only: chỉ thêm leaf token, numeric encoding vẫn Linear gốc · "
                          "none: KHÔNG đổi gì cả, y hệt tokenizer FT-Transformer V2 gốc — dùng để "
                          "cô lập tác dụng của riêng --loss (vd focal) mà không lẫn hiệu ứng PLE/leaf")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce",
                     help="focal: FocalLoss (imbalance.py, alpha=base rate lớp huỷ, gamma=2.0) "
                          "thay BCEWithLogitsLoss — nhấn mạnh lớp huỷ (thiểu số ~9-10%%)")
    args = ap.parse_args()
    use_ple = args.ablation in ("full", "ple_only")
    use_leaf = args.ablation in ("full", "leaf_only")

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    v2_path = os.path.join(ART, "metrics_ft_transformer_v2.json")
    bp = json.load(open(v2_path))["best_params"]
    d_token, n_heads, n_layers, d_ffn, dropout = (
        bp["d_token"], bp["n_heads"], bp["n_layers"], bp["d_ffn"], bp["dropout"])
    lr, weight_decay, batch_size = bp["lr"], bp["weight_decay"], bp["batch_size"]
    print(f"Dùng lại hyperparameter FT-Transformer V2 (Optuna): d_token={d_token} n_heads={n_heads} "
          f"n_layers={n_layers} d_ffn={d_ffn} dropout={dropout:.3f} lr={lr:.2e} batch={batch_size}")

    with mlflow.start_run(run_name=f"ft-transformer-v3-{args.ablation}-{args.loss}"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "ft_transformer_v3",
                          "ablation": args.ablation, "loss": args.loss})
        mlflow.log_params({
            "ablation": args.ablation, "use_ple": use_ple, "use_leaf": use_leaf, "loss": args.loss,
            "epochs": EPOCHS, "patience": PATIENCE, "n_bins": N_BINS, "leaf_embed_dim": LEAF_EMBED_DIM,
            "d_token": d_token, "n_heads": n_heads, "n_layers": n_layers, "d_ffn": d_ffn,
            "dropout": dropout, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        })

        print("1 · Nạp data + chia train/valid/test (post-dispatch, khoá 13/07)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")

        n_trees, num_leaves_cap = 0, 1
        leaf_tr_t = leaf_va_t = leaf_te_t = None
        if use_leaf:
            print("2 · Train LightGBM V2 (đã tune) làm nguồn leaf-embedding — KHÔNG dùng để dự đoán cuối")
            lgb_v2_params = json.load(open(os.path.join(ART, "metrics_lightgbm_v2.json")))["best_params"]
            lgb_params = {**LGB_V1_PARAMS, **lgb_v2_params}
            lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                                   valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
                                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            n_trees = lgb_model.num_trees()
            num_leaves_cap = lgb_params["num_leaves"]
            print(f"  {n_trees} cây · num_leaves cap {num_leaves_cap}")
            leaf_tr = lgb_model.predict(X_tr, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)
            leaf_va = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)
            leaf_te = lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration, pred_leaf=True).astype(np.int64)
            mlflow.log_params({"n_trees": n_trees, "num_leaves_cap": num_leaves_cap})
        else:
            print("2 · use_leaf=False — bỏ qua bước train LightGBM cho leaf-embedding")

        print("3 · Encode numeric" + (" (piecewise-linear, fit quantile bins trên train)" if use_ple
                                       else " (Linear scalar — giống FeatureTokenizer gốc)") +
              " + encode categorical")
        if use_ple:
            ple = PiecewiseLinearEncoder(N_BINS).fit(X_tr, NUMERIC_FEATURES)
            X_tr_num = ple.transform(X_tr, NUMERIC_FEATURES)
            X_va_num = ple.transform(X_va, NUMERIC_FEATURES)
            X_te_num = ple.transform(X_te, NUMERIC_FEATURES)
        else:
            scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
            X_tr_num = transform_numeric(X_tr, scaler)
            X_va_num = transform_numeric(X_va, scaler)
            X_te_num = transform_numeric(X_te, scaler)

        tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
        X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
        X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
        cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

        def to_t(x, dtype):
            return torch.tensor(x, dtype=dtype).to(DEVICE)

        X_tr_num_t, X_va_num_t, X_te_num_t = (to_t(a, torch.float32) for a in (X_tr_num, X_va_num, X_te_num))
        X_tr_cat_t, X_va_cat_t, X_te_cat_t = (to_t(a, torch.long) for a in (X_tr_cat, X_va_cat, X_te_cat))
        if use_leaf:
            leaf_tr_t, leaf_va_t, leaf_te_t = (to_t(a, torch.long) for a in (leaf_tr, leaf_va, leaf_te))
        y_tr_t = to_t(y_tr.values, torch.float32)

        print(f"4 · Huấn luyện FTTransformerV3 (ablation={args.ablation}, early-stop trên valid)")
        model = FTTransformerV3(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                                 n_trees=n_trees, num_leaves_cap=num_leaves_cap, d_token=d_token,
                                 n_layers=n_layers, n_heads=n_heads, d_ffn=d_ffn, dropout=dropout,
                                 use_ple=use_ple, use_leaf=use_leaf).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if args.loss == "focal":
            alpha = compute_focal_alpha(y_tr.values)
            loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            mlflow.log_param("focal_alpha", round(alpha, 4))
            print(f"  loss=FocalLoss(alpha={alpha:.4f}, gamma=2.0)")
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        n_train = len(X_tr)
        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=DEVICE)
            total_loss = 0.0
            for i in range(0, n_train - n_train % batch_size, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = model(X_tr_num_t[idx], X_tr_cat_t[idx], leaf_tr_t[idx] if use_leaf else None)
                loss = loss_fn(logits, y_tr_t[idx])
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            train_loss = total_loss / (n_train - n_train % batch_size)

            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t, leaf_va_t if use_leaf else None)).cpu().numpy()
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

        print("5 · Đánh giá trên test")
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t, leaf_te_t if use_leaf else None)).cpu().numpy()
        metrics = {"test": evaluate(y_te, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Lưu artifact")
        model_path = os.path.join(ART, f"ft_transformer_v3_{args.ablation}_model.pt")
        torch.save(model.state_dict(), model_path)
        metrics["features"] = FEATURES
        metrics["ablation"] = args.ablation
        arch_bits = [f"d_token={d_token}, layers={n_layers}, heads={n_heads}"]
        arch_bits.append(f"numeric=piecewise-linear n_bins={N_BINS}" if use_ple else "numeric=Linear (gốc)")
        arch_bits.append(f"+leaf-embedding token ({n_trees} trees x {num_leaves_cap} leaves)" if use_leaf
                          else "không có leaf token")
        metrics["architecture"] = f"FTTransformerV3({', '.join(arch_bits)})"
        loss_suffix = "" if args.loss == "bce" else f"_{args.loss}"
        metrics_path = os.path.join(ART, f"metrics_ft_transformer_v3_{args.ablation}{loss_suffix}.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ Test ROC-AUC={metrics['test']['roc_auc']:.4f}  Cancel PR-AUC={metrics['test']['pr_auc_cancel']:.4f}")
        print(f"✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
