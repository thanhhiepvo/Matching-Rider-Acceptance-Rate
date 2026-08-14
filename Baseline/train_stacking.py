"""Stacking meta-learner — Logistic Regression trên prediction của 3 base model
(LightGBM, MLP log1p, Hybrid GBDT+NN), thay cho blend trọng số cố định (avg/geomean) ở
ensemble.py. Nguyên tắc hơn: trọng số mỗi model học được từ dữ liệu, không phải đoán tay.

Time-based fold (giữ đúng nguyên tắc "dự đoán tương lai từ quá khứ" xuyên suốt dự án —
KHÔNG random split, KHÔNG đụng test ở bất kỳ bước fit nào):

    inner_train : 06/07 → 11/07   (fit 3 base model)
    meta_val    : 12/07            (base model predict -> làm feature cho meta-learner;
                                     base model cũng dùng slice này để early-stop)
    test        : 13/07            (chỉ đánh giá cuối cùng)

⚠️ Giới hạn phương pháp: meta_val vừa dùng để early-stop base model (MLP/Hybrid) vừa dùng để
fit meta-learner — lý tưởng cần 1 lát cắt thứ 3 riêng, nhưng dữ liệu train chỉ có 7 ngày nên
tách thêm sẽ làm mỗi phần quá ít. Đây là đánh đổi thực dụng thường gặp khi stacking trên
dataset nhỏ, không phải lỗi — nhưng số liệu trên meta_val có thể lạc quan hơn thực tế một
chút vì lý do này. Số liệu đáng tin cậy vẫn là trên `test`.

    python3 Baseline/train_stacking.py
"""
from __future__ import annotations

import os

# LightGBM (libomp Homebrew) + PyTorch cùng process trên macOS deadlock ở lần forward đầu
# của PyTorch nếu không set trước khi import — xem ensemble.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from evaluation import evaluate
from features import FEATURES, PRE_CATEGORICAL, build_features, load_raw
from mlp_common import (
    LeafEmbedding,
    TabularDataset,
    TabularMLP,
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, PARAMS as LGB_PARAMS, RAW, TEST_DATE

torch.set_num_threads(1)
NUMERIC_FEATURES = [f for f in FEATURES if f not in PRE_CATEGORICAL]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
EMBED_DIM = 4
LEAF_EMBED_DIM = 8
META_VAL_DATE = "2026-07-12"  # lát cắt cuối của train, làm "OOF" cho meta-learner

torch.manual_seed(SEED)
np.random.seed(SEED)


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


class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, X_num, X_cat, leaf_idx, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.leaf_idx = torch.tensor(leaf_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_num[i], self.X_cat[i], self.leaf_idx[i], self.y[i]


class HybridNet(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, n_trees, num_leaves_cap,
                 leaf_embed_dim=LEAF_EMBED_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.leaf_embedding = LeafEmbedding(n_trees, num_leaves_cap, leaf_embed_dim)
        self.mlp = TabularMLP(n_numeric, cat_cardinalities, embed_dim=embed_dim, extra_dim=leaf_embed_dim)

    def forward(self, x_num, x_cat, leaf_idx):
        return self.mlp(x_num, x_cat, self.leaf_embedding(leaf_idx))


def train_mlp_like(model, train_ds, eval_fn, epochs=EPOCHS, patience=PATIENCE, label="model"):
    """Vòng lặp train dùng chung cho MLP thuần và Hybrid — eval_fn(model) -> (auc, logloss)."""
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            *xb, yb = [t.to(DEVICE) for t in batch]
            opt.zero_grad()
            loss = loss_fn(model(*xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        auc, logloss = eval_fn(model)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    print(f"  [{label}] best epoch {best_epoch} · meta_val AUC {best_auc:.4f}")
    return model


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="stacking-meta-learner-v1"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "stacking_meta_learner"})
        mlflow.log_params({
            "meta_val_date": META_VAL_DATE, "test_date": TEST_DATE,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "embed_dim": EMBED_DIM,
            "leaf_embed_dim": LEAF_EMBED_DIM, "device": str(DEVICE),
        })

        print("1 · Nạp data + build feature")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily)

        print("2 · Áp rule is_post_dispatch + chia 3 lát theo THỜI GIAN: inner_train / meta_val / test")
        post_mask = df.is_post_dispatch.astype(int) == 1
        is_test_full = df.order_date >= pd.Timestamp(TEST_DATE)
        is_meta_val = (df.order_date >= pd.Timestamp(META_VAL_DATE)) & ~is_test_full
        is_inner_train = ~is_test_full & ~is_meta_val

        X_tr, y_tr = X[is_inner_train & post_mask], y[is_inner_train & post_mask]
        X_val, y_val = X[is_meta_val & post_mask], y[is_meta_val & post_mask]
        X_te, y_te = X[is_test_full & post_mask], y[is_test_full & post_mask]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_val[c] = pd.Categorical(X_val[c], categories=X_tr[c].cat.categories)
                X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
        print(f"  inner_train {len(X_tr):,} · meta_val {len(X_val):,} · test {len(X_te):,}")
        mlflow.log_params({"n_inner_train": len(X_tr), "n_meta_val": len(X_val), "n_test": len(X_te)})

        print("3 · Fit LightGBM trên inner_train (early-stop trên meta_val)")
        train_set = lgb.Dataset(X_tr, y_tr)
        val_set = lgb.Dataset(X_val, y_val)
        booster = lgb.train(
            LGB_PARAMS, train_set, num_boost_round=500,
            valid_sets=[val_set], valid_names=["meta_val"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p_lgb_val = booster.predict(X_val, num_iteration=booster.best_iteration)
        p_lgb_te = booster.predict(X_te, num_iteration=booster.best_iteration)
        print(f"  LightGBM meta_val AUC {roc_auc_score(y_val, p_lgb_val):.4f}")

        print("4 · Tiền xử lý numeric/categorical (log1p + scale, encode) — fit trên inner_train")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_val_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_val, X_te))
        tr_codes, val_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_val[PRE_CATEGORICAL])
        X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
        X_val_cat = np.stack([val_codes[c] for c in PRE_CATEGORICAL], axis=1)
        X_te_cat = apply_categorical_encoders(X_te[PRE_CATEGORICAL], encoders)
        cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

        X_val_num_t = torch.tensor(X_val_num, dtype=torch.float32).to(DEVICE)
        X_val_cat_t = torch.tensor(X_val_cat, dtype=torch.long).to(DEVICE)
        X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
        X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)

        print("5 · Fit MLP (log1p) trên inner_train (early-stop trên meta_val)")
        mlp_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
        mlp = TabularMLP(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                          embed_dim=EMBED_DIM).to(DEVICE)

        def eval_mlp(m):
            with torch.no_grad():
                p = torch.sigmoid(m(X_val_num_t, X_val_cat_t)).cpu().numpy()
            return roc_auc_score(y_val, p), log_loss(y_val, p)

        mlp = train_mlp_like(mlp, mlp_ds, eval_mlp, label="mlp")
        with torch.no_grad():
            p_mlp_val = torch.sigmoid(mlp(X_val_num_t, X_val_cat_t)).cpu().numpy()
            p_mlp_te = torch.sigmoid(mlp(X_te_num_t, X_te_cat_t)).cpu().numpy()

        print("6 · Fit Hybrid GBDT+NN trên inner_train (leaf index từ LightGBM vừa fit ở bước 3)")
        n_trees, num_leaves_cap = booster.num_trees(), LGB_PARAMS["num_leaves"]
        leaf_tr = booster.predict(X_tr, pred_leaf=True).astype(np.int64)
        leaf_val = booster.predict(X_val, pred_leaf=True).astype(np.int64)
        leaf_te = booster.predict(X_te, pred_leaf=True).astype(np.int64)
        leaf_val_t = torch.tensor(leaf_val, dtype=torch.long).to(DEVICE)
        leaf_te_t = torch.tensor(leaf_te, dtype=torch.long).to(DEVICE)

        hybrid_ds = HybridDataset(X_tr_num, X_tr_cat, leaf_tr, y_tr.values)
        hybrid = HybridNet(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                            n_trees=n_trees, num_leaves_cap=num_leaves_cap).to(DEVICE)

        def eval_hybrid(m):
            with torch.no_grad():
                p = torch.sigmoid(m(X_val_num_t, X_val_cat_t, leaf_val_t)).cpu().numpy()
            return roc_auc_score(y_val, p), log_loss(y_val, p)

        hybrid = train_mlp_like(hybrid, hybrid_ds, eval_hybrid, label="hybrid")
        with torch.no_grad():
            p_hybrid_val = torch.sigmoid(hybrid(X_val_num_t, X_val_cat_t, leaf_val_t)).cpu().numpy()
            p_hybrid_te = torch.sigmoid(hybrid(X_te_num_t, X_te_cat_t, leaf_te_t)).cpu().numpy()

        print("7 · Fit meta-learner (Logistic Regression) trên prediction của 3 base model ở meta_val")
        meta_X_val = np.column_stack([p_lgb_val, p_mlp_val, p_hybrid_val])
        meta_X_te = np.column_stack([p_lgb_te, p_mlp_te, p_hybrid_te])
        meta_model = LogisticRegression()
        meta_model.fit(meta_X_val, y_val)
        p_stack_te = meta_model.predict_proba(meta_X_te)[:, 1]

        coef = dict(zip(["lgb", "mlp", "hybrid"], meta_model.coef_[0]))
        print(f"  Trọng số meta-learner: {', '.join(f'{k}={v:.3f}' for k, v in coef.items())}"
              f" · intercept={meta_model.intercept_[0]:.3f}")
        mlflow.log_params({f"meta_coef_{k}": round(float(v), 4) for k, v in coef.items()})
        mlflow.log_param("meta_intercept", round(float(meta_model.intercept_[0]), 4))

        print("8 · Đánh giá trên test — stacking vs 3 base model đứng riêng")
        results = {
            "lgb_solo": evaluate(y_te, p_lgb_te, "lgb-solo"),
            "mlp_solo": evaluate(y_te, p_mlp_te, "mlp-solo"),
            "hybrid_solo": evaluate(y_te, p_hybrid_te, "hybrid-solo"),
            "test": evaluate(y_te, p_stack_te, "stack-test"),
        }
        # Log gọn 1 con số/base-model để so sánh nhanh — chi tiết đầy đủ vẫn có trong
        # metrics_stacking.json, không cần biến MLflow thành hàng chục metric key dư thừa.
        mlflow.log_metrics({f"compare_{name}_roc_auc": results[name]["roc_auc"]
                             for name in ["lgb_solo", "mlp_solo", "hybrid_solo"]})
        log_metrics("test", results["test"])

        # (Tham khảo) Hiệu năng HỆ THỐNG đầy đủ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
        # stacking (post-dispatch) — xem giải thích chi tiết trong train.py.
        X_te_full, y_te_full = X[is_test_full], y[is_test_full]
        for c in X_tr.columns:
            if str(X_tr[c].dtype) == "category":
                X_te_full[c] = pd.Categorical(X_te_full[c], categories=X_tr[c].cat.categories)
        post_te_full = df[is_test_full].is_post_dispatch.astype(int).values == 1
        p_lgb_full = booster.predict(X_te_full, num_iteration=booster.best_iteration)
        X_full_num = transform_numeric(X_te_full, scaler)
        X_full_cat = apply_categorical_encoders(X_te_full[PRE_CATEGORICAL], encoders)
        X_full_num_t = torch.tensor(X_full_num, dtype=torch.float32).to(DEVICE)
        X_full_cat_t = torch.tensor(X_full_cat, dtype=torch.long).to(DEVICE)
        leaf_full = booster.predict(X_te_full, pred_leaf=True).astype(np.int64)
        leaf_full_t = torch.tensor(leaf_full, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            p_mlp_full = torch.sigmoid(mlp(X_full_num_t, X_full_cat_t)).cpu().numpy()
            p_hybrid_full = torch.sigmoid(hybrid(X_full_num_t, X_full_cat_t, leaf_full_t)).cpu().numpy()
        meta_X_full = np.column_stack([p_lgb_full, p_mlp_full, p_hybrid_full])
        p_stack_full = meta_model.predict_proba(meta_X_full)[:, 1]
        p_te_full = np.where(post_te_full, p_stack_full, 0.0)
        results["system_full"] = evaluate(y_te_full, p_te_full, "system-full")
        log_metrics("system_full", results["system_full"])

        print("9 · Lưu artifact")
        metrics_path = os.path.join(ART, "metrics_stacking.json")
        json.dump({**results, "meta_coef": {k: float(v) for k, v in coef.items()},
                   "meta_intercept": float(meta_model.intercept_[0]), "features": FEATURES},
                  open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.sklearn.log_model(meta_model, name="meta_model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
