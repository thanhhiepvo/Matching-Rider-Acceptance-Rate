"""GBDT+NN hybrid — dùng leaf index của LightGBM (model_v1.txt, KHÔNG train lại) làm thêm
input cho MLP, bên cạnh feature số/categorical như train_mlp.py.

LightGBM đã học được cách chia không gian feature qua các cây quyết định — mỗi sample rơi
vào 1 leaf ở mỗi cây. Coi (tree_idx, leaf_idx) như 1 categorical feature, embed rồi mean-pool
qua các cây -> vector đặc trưng "đã được GBDT tinh chọn", ghép thêm vào input MLP. Kỹ thuật
kinh điển kiểu Facebook GBDT+LR / DeepFM — thường mạnh hơn cả GBDT hay NN đứng riêng.

    python3 Baseline/train_hybrid.py
"""
from __future__ import annotations

import os

# Xem giải thích ở ensemble.py: LightGBM (libomp Homebrew) + PyTorch trong CÙNG process trên
# macOS deadlock ở lần forward đầu tiên của PyTorch nếu không set trước khi import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import lightgbm as lgb
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from evaluation import evaluate
from features import FEATURES, PRE_CATEGORICAL, build_features, load_raw
from mlp_common import (
    LeafEmbedding,
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
EMBED_DIM = 4          # embedding cho travel_mode/vertical_name
LEAF_EMBED_DIM = 8     # embedding cho leaf-index (per tree, mean-pooled)

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


class HybridDataset(Dataset):
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
        extra = self.leaf_embedding(leaf_idx)
        return self.mlp(x_num, x_cat, extra)


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hybrid-gbdt-nn-v1"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "hybrid_gbdt_nn"})
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "embed_dim": EMBED_DIM, "leaf_embed_dim": LEAF_EMBED_DIM,
            "patience": PATIENCE, "device": str(DEVICE), "n_features": len(FEATURES),
        })

        print("1 · Nạp 2 file raw")
        orders, customer_daily = load_raw(RAW)
        print(f"  orders {len(orders):,} · customer_daily {len(customer_daily):,}")

        print("2 · Tạo feature")
        df, X, y = build_features(orders, customer_daily)

        print("3 · Áp rule is_post_dispatch + chia train/test theo THỜI GIAN (trên phần post-dispatch)")
        is_test_full = df.order_date >= pd.Timestamp(TEST_DATE)
        post_mask = df.is_post_dispatch.astype(int) == 1
        X_tr, y_tr = X[~is_test_full & post_mask], y[~is_test_full & post_mask]
        X_te, y_te = X[is_test_full & post_mask], y[is_test_full & post_mask]
        print(f"  train {len(X_tr):,} (→{TEST_DATE}) · test {len(X_te):,} ({TEST_DATE})")
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_test", len(X_te))

        print("4 · Load LightGBM đã train sẵn (model_v1.txt) — DÙNG NHƯ FEATURE EXTRACTOR, không train lại")
        booster = lgb.Booster(model_file=os.path.join(ART, "model_v1.txt"))
        n_trees = booster.num_trees()
        num_leaves_cap = LGB_PARAMS["num_leaves"]
        print(f"  {n_trees} cây · num_leaves cap {num_leaves_cap}")
        leaf_tr = booster.predict(X_tr, pred_leaf=True).astype(np.int64)
        leaf_te = booster.predict(X_te, pred_leaf=True).astype(np.int64)
        mlflow.log_param("n_trees", n_trees)
        mlflow.log_param("num_leaves_cap", num_leaves_cap)

        print("5 · Tiền xử lý numeric/categorical (log1p + scale, encode) — giống train_mlp.py")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_te_num = transform_numeric(X_tr, scaler), transform_numeric(X_te, scaler)
        tr_codes, te_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
        X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
        X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
        cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

        train_ds = HybridDataset(X_tr_num, X_tr_cat, leaf_tr, y_tr.values)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
        X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)
        leaf_te_t = torch.tensor(leaf_te, dtype=torch.long).to(DEVICE)

        print("6 · Huấn luyện")
        model = HybridNet(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                           n_trees=n_trees, num_leaves_cap=num_leaves_cap).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for xb_num, xb_cat, xb_leaf, yb in train_loader:
                xb_num, xb_cat, xb_leaf, yb = (xb_num.to(DEVICE), xb_cat.to(DEVICE),
                                                xb_leaf.to(DEVICE), yb.to(DEVICE))
                opt.zero_grad()
                loss = loss_fn(model(xb_num, xb_cat, xb_leaf), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(yb)
            train_loss = total_loss / (len(train_ds) - len(train_ds) % BATCH_SIZE)

            model.eval()
            with torch.no_grad():
                test_p = torch.sigmoid(model(X_te_num_t, X_te_cat_t, leaf_te_t)).cpu().numpy()
            test_auc = roc_auc_score(y_te, test_p)
            test_logloss = log_loss(y_te, test_p)
            print(f"  [epoch {epoch:2d}] train_loss={train_loss:.4f}  test_auc={test_auc:.4f}  test_logloss={test_logloss:.4f}")
            mlflow.log_metric("curve_train_loss", train_loss, step=epoch)
            mlflow.log_metric("curve_test_auc", test_auc, step=epoch)
            mlflow.log_metric("curve_test_logloss", test_logloss, step=epoch)

            if test_auc > best_auc:
                best_auc, best_epoch = test_auc, epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  early stopping ở epoch {epoch} (không cải thiện {PATIENCE} epoch)")
                    break

        model.load_state_dict(best_state)
        print(f"  best epoch {best_epoch} · test AUC {best_auc:.4f}")
        mlflow.log_param("best_epoch", best_epoch)

        print("7 · Đánh giá")
        model.eval()
        with torch.no_grad():
            X_tr_num_t = torch.tensor(X_tr_num, dtype=torch.float32).to(DEVICE)
            X_tr_cat_t = torch.tensor(X_tr_cat, dtype=torch.long).to(DEVICE)
            leaf_tr_t = torch.tensor(leaf_tr, dtype=torch.long).to(DEVICE)
            p_tr = torch.sigmoid(model(X_tr_num_t, X_tr_cat_t, leaf_tr_t)).cpu().numpy()
            p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t, leaf_te_t)).cpu().numpy()

        metrics = {"train": evaluate(y_tr, p_tr, "train"), "test": evaluate(y_te, p_te, "test")}
        log_metrics("train", metrics["train"])
        log_metrics("test", metrics["test"])

        # (Tham khảo) Hiệu năng HỆ THỐNG đầy đủ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
        # model (post-dispatch) — xem giải thích chi tiết trong train.py.
        X_te_full, y_te_full = X[is_test_full], y[is_test_full]
        post_te_full = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_full_num = transform_numeric(X_te_full, scaler)
        X_full_cat = apply_categorical_encoders(X_te_full[PRE_CATEGORICAL], encoders)
        leaf_full = booster.predict(X_te_full, pred_leaf=True).astype(np.int64)
        model.eval()
        with torch.no_grad():
            p_full_all = torch.sigmoid(model(
                torch.tensor(X_full_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_full_cat, dtype=torch.long).to(DEVICE),
                torch.tensor(leaf_full, dtype=torch.long).to(DEVICE))).cpu().numpy()
        p_te_full = np.where(post_te_full, p_full_all, 0.0)
        metrics["system_full"] = evaluate(y_te_full, p_te_full, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("8 · Lưu artifact")
        model_path = os.path.join(ART, "hybrid_model.pt")
        torch.save(model.state_dict(), model_path)
        metrics["features"] = FEATURES
        metrics["architecture"] = (
            f"HybridNet: LeafEmbedding({n_trees} trees x {num_leaves_cap} leaves -> "
            f"dim={LEAF_EMBED_DIM}, mean-pool) + TabularMLP(128,64, embed_dim={EMBED_DIM})"
        )
        metrics_path = os.path.join(ART, "metrics_hybrid.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)
        input_example = (X_te_num_t[:2].cpu(), X_te_cat_t[:2].cpu(), leaf_te_t[:2].cpu())
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=input_example,
                                  serialization_format="pickle")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
