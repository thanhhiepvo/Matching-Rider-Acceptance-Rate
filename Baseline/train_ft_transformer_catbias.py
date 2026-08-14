"""Thử nghiệm: thêm b_j^(cat) (bias riêng mỗi feature categorical) vào FeatureTokenizer — đúng
công thức Figure 2(a) paper gốc (T_j^(cat) = b_j^(cat) + e_j^T·W_j^(cat)), hiện `train_ft_transformer.py`
KHÔNG có bias này (chỉ lookup thuần). Dùng Focal loss (đã xác nhận là đòn bẩy chính giúp
FT-Transformer, xem reports/W4_dl_error_analysis.ipynb mục 8) + hyperparameter V2 đã tune, để so
sánh CÔNG BẰNG với baseline "V2 gốc + Focal, KHÔNG cat_bias" (ROC-AUC=0.7441, Cancel PR-AUC=0.2508)
— cô lập đúng 1 biến thay đổi: có/không cat_bias.

    python3 Baseline/train_ft_transformer_catbias.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import apply_categorical_encoders, encode_categoricals, fit_numeric_scaler, transform_numeric
from split import align_categories, time_split
from train import ART, RAW
from train_ft_transformer import FeatureTokenizer, FTTransformer

NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS, PATIENCE = 60, 7

torch.manual_seed(SEED)
np.random.seed(SEED)


class FeatureTokenizerCatBias(FeatureTokenizer):
    """Y HỆT FeatureTokenizer gốc, chỉ thêm 1 dòng: cat_bias (1 vector riêng/feature categorical,
    KHÔNG phải riêng/giá trị category — cộng như nhau vào MỌI hàng của bảng embedding feature đó,
    đúng công thức paper: b_j^(cat) cộng sau khi đã chọn hàng theo one-hot)."""

    def __init__(self, n_numeric, cat_cardinalities, d_token, cls_position="first"):
        super().__init__(n_numeric, cat_cardinalities, d_token, cls_position)
        self.cat_bias = nn.Parameter(torch.zeros(len(cat_cardinalities), d_token))

    def forward(self, x_num, x_cat):
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
        ) + self.cat_bias
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        feature_tokens = [num_tokens, cat_tokens]
        parts = [cls] + feature_tokens if self.cls_position == "first" else feature_tokens + [cls]
        return torch.cat(parts, dim=1)


class FTTransformerCatBias(FTTransformer):
    def __init__(self, n_numeric, cat_cardinalities, d_token, n_layers, n_heads, d_ffn, dropout):
        super().__init__(n_numeric, cat_cardinalities, d_token, n_layers, n_heads, d_ffn, dropout)
        self.tokenizer = FeatureTokenizerCatBias(n_numeric, cat_cardinalities, d_token)


def main():
    print("1 · Nạp data + chia train/valid/test (post-dispatch, khoá 13/07)")
    orders, customer_daily = load_raw(RAW)
    df, X, y = build_features(orders, customer_daily, verbose=False)
    split = time_split(df, X, y, post_only=True)
    X_tr, y_tr = split.train
    X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
    X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
    print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")

    scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[CATEGORICAL_FEATURES], X_va[CATEGORICAL_FEATURES])
    X_tr_cat = np.stack([tr_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in CATEGORICAL_FEATURES], axis=1)
    X_te_cat = apply_categorical_encoders(X_te[CATEGORICAL_FEATURES], encoders)
    cat_cardinalities = [len(encoders[c][0]) for c in CATEGORICAL_FEATURES]

    def to_t(x, dtype):
        return torch.tensor(x, dtype=dtype).to(DEVICE)

    X_tr_num_t, X_va_num_t, X_te_num_t = (to_t(a, torch.float32) for a in (X_tr_num, X_va_num, X_te_num))
    X_tr_cat_t, X_va_cat_t, X_te_cat_t = (to_t(a, torch.long) for a in (X_tr_cat, X_va_cat, X_te_cat))
    y_tr_t = to_t(y_tr.values, torch.float32)

    bp = json.load(open(os.path.join(ART, "metrics_ft_transformer_v2.json")))["best_params"]
    print(f"  V2 params: d_token={bp['d_token']} n_heads={bp['n_heads']} n_layers={bp['n_layers']} "
          f"d_ffn={bp['d_ffn']} dropout={bp['dropout']:.3f} lr={bp['lr']:.2e} batch={bp['batch_size']}")

    print("\n2 · Train FTTransformer + cat_bias + FocalLoss")
    model = FTTransformerCatBias(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                                  d_token=bp["d_token"], n_layers=bp["n_layers"], n_heads=bp["n_heads"],
                                  d_ffn=bp["d_ffn"], dropout=bp["dropout"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tổng tham số: {n_params:,} (thêm {model.tokenizer.cat_bias.numel()} so với bản không cat_bias)")

    opt = torch.optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    alpha = compute_focal_alpha(y_tr.values)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    print(f"  loss=FocalLoss(alpha={alpha:.4f}, gamma=2.0)")

    n_train = len(X_tr)
    batch_size = bp["batch_size"]
    best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss = 0.0
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_tr_num_t[idx], X_tr_cat_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / (n_train - n_train % batch_size)

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
        va_auc = roc_auc_score(y_va, p_va)
        va_logloss = log_loss(y_va, p_va)
        print(f"  [epoch {epoch:2d}] train_loss={train_loss:.4f}  valid_auc={va_auc:.4f}  valid_logloss={va_logloss:.4f}")

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

    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
    m = evaluate(y_te, p_te, "test", verbose=False)

    print(f"\n3 · KẾT QUẢ — FT-Transformer V2 + cat_bias + Focal (raw, threshold 0.5)")
    print(f"  ROC-AUC={m['roc_auc']:.4f}  Cancel PR-AUC={m['pr_auc_cancel']:.4f}  ECE={m['ece']:.4f}  "
          f"Recall_huỷ={m['recall_cancel']:.4f}")

    print(f"\n4 · So với baseline đã biết (V2 gốc + Focal, KHÔNG cat_bias):")
    print(f"  V2 gốc + Focal (không cat_bias):  ROC-AUC=0.7441  Cancel PR-AUC=0.2508")
    print(f"  V2 + cat_bias + Focal (lần này):  ROC-AUC={m['roc_auc']:.4f}  Cancel PR-AUC={m['pr_auc_cancel']:.4f}")
    delta = m["pr_auc_cancel"] - 0.2508
    print(f"  Δ Cancel PR-AUC = {delta:+.4f}")

    metrics_path = os.path.join(ART, "metrics_ft_transformer_catbias_focal.json")
    json.dump({"test": m, "best_epoch": best_epoch, "params": bp,
               "n_params_total": n_params, "n_params_cat_bias": model.tokenizer.cat_bias.numel()},
              open(metrics_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n✓ -> {metrics_path}")


if __name__ == "__main__":
    main()
