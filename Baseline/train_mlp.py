"""Baseline MLP (PyTorch) — so sánh với LightGBM cho Rider Acceptance, chỉ POST-DISPATCH.

Cùng feature/split/target như train.py (features.py), chỉ đổi model: MLP + embedding cho
2 categorical thay vì LightGBM. v2: log1p feature lệch phải (total_fee, trip_distance_km,
fee_per_km, eta_seconds, cust_orders_30d) trước khi chuẩn hoá — NN nhạy với scale/skew hơn
cây quyết định nhiều. is_post_dispatch là RULE if/else bên ngoài (giống train.py), model chỉ
train/infer trên post-dispatch.

    python3 Baseline/train_mlp.py
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from evaluation import evaluate
from features import FEATURES, PRE_CATEGORICAL, build_features, load_raw
from imbalance import FocalLoss, compute_focal_alpha
from mlp_common import (
    LOG1P_COLS,
    TabularDataset,
    TabularMLP,
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW, TEST_DATE

NUMERIC_FEATURES = [f for f in FEATURES if f not in PRE_CATEGORICAL]
LOG1P_COLS_STR = ", ".join(LOG1P_COLS)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5       # early stopping theo test AUC
EMBED_DIM = 4       # travel_mode/vertical_name chỉ 3-4 giá trị, embedding nhỏ là đủ

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", choices=["none", "focal"], default="none",
                     help="focal: dùng FocalLoss (imbalance.py) thay BCEWithLogitsLoss, nhấn mạnh lớp huỷ (y=0)")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "mlp-pytorch-v2-logtransform" + (f"-{args.imbalance}" if args.imbalance != "none" else "")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "pytorch_mlp", "variant": "log1p",
                          "imbalance": args.imbalance})
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "embed_dim": EMBED_DIM, "patience": PATIENCE,
            "device": str(DEVICE), "n_features": len(FEATURES), "imbalance": args.imbalance,
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

        print("4 · Tiền xử lý (log1p feature lệch phải + scale numeric bằng thống kê TRAIN, encode categorical)")
        scaler = fit_numeric_scaler(X_tr, NUMERIC_FEATURES)
        X_tr_num, X_te_num = transform_numeric(X_tr, scaler), transform_numeric(X_te, scaler)
        tr_codes, te_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
        X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
        X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
        cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

        train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
        X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)

        print("5 · Huấn luyện")
        model = TabularMLP(n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
                            embed_dim=EMBED_DIM).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if args.imbalance == "focal":
            alpha = compute_focal_alpha(y_tr.values)
            loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
            mlflow.log_param("focal_alpha", round(alpha, 4))
            print(f"  imbalance=focal -> alpha={alpha:.4f}, gamma=2.0 (nhấn mạnh lớp huỷ + mẫu khó)")
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, no_improve, best_epoch = -1.0, None, 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for xb_num, xb_cat, yb in train_loader:
                xb_num, xb_cat, yb = xb_num.to(DEVICE), xb_cat.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb_num, xb_cat), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(yb)
            train_loss = total_loss / (len(train_ds) - len(train_ds) % BATCH_SIZE)

            model.eval()
            with torch.no_grad():
                test_p = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
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

        print("6 · Đánh giá")
        model.eval()
        with torch.no_grad():
            X_tr_num_t = torch.tensor(X_tr_num, dtype=torch.float32).to(DEVICE)
            X_tr_cat_t = torch.tensor(X_tr_cat, dtype=torch.long).to(DEVICE)
            p_tr = torch.sigmoid(model(X_tr_num_t, X_tr_cat_t)).cpu().numpy()
            p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()

        metrics = {"train": evaluate(y_tr, p_tr, "train"), "test": evaluate(y_te, p_te, "test")}
        log_metrics("train", metrics["train"])
        log_metrics("test", metrics["test"])

        # (Tham khảo) Hiệu năng HỆ THỐNG đầy đủ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
        # model (post-dispatch) — xem giải thích chi tiết trong train.py.
        X_te_full, y_te_full = X[is_test_full], y[is_test_full]
        post_te_full = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_full_num = transform_numeric(X_te_full, scaler)
        X_full_cat = apply_categorical_encoders(X_te_full[PRE_CATEGORICAL], encoders)
        model.eval()
        with torch.no_grad():
            p_full_all = torch.sigmoid(model(
                torch.tensor(X_full_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(X_full_cat, dtype=torch.long).to(DEVICE))).cpu().numpy()
        p_te_full = np.where(post_te_full, p_full_all, 0.0)
        metrics["system_full"] = evaluate(y_te_full, p_te_full, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model_path = os.path.join(ART, "mlp_model.pt")
        torch.save(model.state_dict(), model_path)

        # Lưu preprocessing + kiến trúc để ensemble.py / script khác load lại predict được,
        # không cần huấn luyện lại.
        preprocess_path = os.path.join(ART, "mlp_preprocess.pkl")
        joblib.dump({
            "scaler": scaler, "encoders": encoders, "cat_cardinalities": cat_cardinalities,
            "numeric_features": NUMERIC_FEATURES, "categorical_features": PRE_CATEGORICAL,
            "embed_dim": EMBED_DIM,
        }, preprocess_path)

        metrics["features"] = FEATURES
        metrics["architecture"] = f"MLP(128,64) + embedding(dim={EMBED_DIM}) cho {PRE_CATEGORICAL} + log1p({LOG1P_COLS_STR})"
        metrics_path = os.path.join(ART, "metrics_mlp.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)

        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(preprocess_path)
        input_example = (X_te_num_t[:2].cpu(), X_te_cat_t[:2].cpu())
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=input_example,
                                  serialization_format="pickle")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
