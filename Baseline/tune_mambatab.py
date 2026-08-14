"""Optuna tuning cho MambaTab (train_mambatab.py) — cùng tinh thần tune_ft_transformer.py:
mỗi trial train ÍT epoch hơn bản gốc, early-stop + chọn trial theo ROC-AUC trên `valid`
(split.py) — KHÔNG đụng `test`/`calib`.

    python3 Baseline/tune_mambatab.py [--n-trials 20] [--epochs-per-trial 60]
"""
from __future__ import annotations

import argparse
import json
import os

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from mlp_common import (
    apply_categorical_encoders,
    encode_categoricals,
    fit_numeric_scaler,
    transform_numeric,
)
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW, ROOT
from train_mambatab import MambaTab

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_DB_PATH = os.path.join(ROOT, "optuna.db")
STUDY_NAME = "mambatab_tuning"
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42

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


def run_epochs(model, opt, sched, loss_fn, X_t, y_t, batch_size, n_epochs,
               eval_fn=None, patience=None):
    """Vòng lặp train dùng chung cho tuning (eval_fn+patience -> early-stop) và final retrain
    (không có eval_fn -> chạy đúng n_epochs, không early-stop). `sched` có thể None (final
    retrain không cần cosine-annealing lại theo epoch budget khác của tuning)."""
    n = len(y_t)
    best_auc, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - n % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()

        if eval_fn is None:
            continue
        model.eval()
        auc = eval_fn(model)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch


def make_objective(X_tr_t, y_tr_t, X_va_t, y_va, n_features: int, epochs: int, patience: int):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "d_model": trial.suggest_categorical("d_model", [16, 32, 48, 64]),
            "m_blocks": trial.suggest_int("m_blocks", 1, 6),
            "d_state": trial.suggest_categorical("d_state", [8, 16, 32, 64]),
            "d_conv": trial.suggest_categorical("d_conv", [2, 4, 8]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }
        model = MambaTab(n_features=n_features, d_model=params["d_model"], m_blocks=params["m_blocks"],
                          d_state=params["d_state"], d_conv=params["d_conv"], dropout=params["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.BCEWithLogitsLoss()

        def eval_fn(m):
            with torch.no_grad():
                p = torch.sigmoid(m(X_va_t)).cpu().numpy()
            return average_precision_score(1 - np.asarray(y_va), 1 - p)  # Cancel PR-AUC, không phải ROC-AUC

        _, best_auc, best_epoch = run_epochs(model, opt, sched, loss_fn, X_tr_t, y_tr_t,
                                              params["batch_size"], epochs, eval_fn, patience)
        trial.set_user_attr("best_epoch", best_epoch)
        return best_auc
    return objective


def build_flat_features(X_tr, X_va, X_te, numeric_features, categorical_features):
    scaler = fit_numeric_scaler(X_tr, numeric_features)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[categorical_features], X_va[categorical_features])
    te_cat_arr = apply_categorical_encoders(X_te[categorical_features], encoders)
    X_tr_cat = np.stack([tr_codes[c] for c in categorical_features], axis=1).astype(np.float32)
    X_va_cat = np.stack([va_codes[c] for c in categorical_features], axis=1).astype(np.float32)
    X_te_cat = te_cat_arr.astype(np.float32)
    X_tr_flat = np.concatenate([X_tr_num, X_tr_cat], axis=1)
    X_va_flat = np.concatenate([X_va_num, X_va_cat], axis=1)
    X_te_flat = np.concatenate([X_te_num, X_te_cat], axis=1)
    return X_tr_flat, X_va_flat, X_te_flat, scaler, encoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs-per-trial", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mambatab-v2-tuned"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "mambatab_optuna"})
        mlflow.log_params({"n_trials": args.n_trials, "epochs_per_trial": args.epochs_per_trial})

        print("1 · Nạp data + chia train/valid/test THEO THỜI GIAN (split.py)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(split.test[0]):,}")

        print("2 · Tiền xử lý (log1p + scale numeric; categorical -> ordinal code; NÉN thành 1 vector phẳng)")
        X_tr_flat, X_va_flat, _, _, _ = build_flat_features(X_tr, X_va, X_va, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        n_features = X_tr_flat.shape[1]
        X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
        X_va_t = torch.tensor(X_va_flat, dtype=torch.float32).to(DEVICE)

        print(f"3 · Optuna search ({args.n_trials} trial × {args.epochs_per_trial} epoch/trial, "
              f"tối ưu Cancel PR-AUC trên valid)")
        storage = f"sqlite:///{OPTUNA_DB_PATH}"
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage,
                                     direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(X_tr_t, y_tr_t, X_va_t, y_va, n_features,
                                       args.epochs_per_trial, args.patience),
                        n_trials=args.n_trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_epoch = best_trial.user_attrs["best_epoch"]
        print(f"  best trial #{best_trial.number}: valid Cancel PR-AUC {best_trial.value:.4f}, best_epoch={best_epoch}")
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_valid_pr_auc_cancel", best_trial.value)
        mlflow.log_param("best_epoch_on_valid", best_epoch)

        trials_path = os.path.join(ART, "optuna_trials_mambatab.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        print(f"4 · Retrain final trên train+valid gộp — ĐÚNG best_epoch={best_epoch}, KHÔNG early-stop lại")
        X_trva, y_trva = split.train_valid
        X_te_final = align_categories(X_trva, split.test[0])
        X_trva_flat, X_te_flat, _, scaler_f, encoders_f = build_flat_features(
            X_trva, X_te_final, X_te_final, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        X_trva_t = torch.tensor(X_trva_flat, dtype=torch.float32).to(DEVICE)
        y_trva_t = torch.tensor(y_trva.values, dtype=torch.float32).to(DEVICE)
        X_te_t = torch.tensor(X_te_flat, dtype=torch.float32).to(DEVICE)

        bp = best_trial.params
        model = MambaTab(n_features=n_features, d_model=bp["d_model"], m_blocks=bp["m_blocks"],
                          d_state=bp["d_state"], d_conv=bp["d_conv"], dropout=bp["dropout"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(best_epoch, 1))
        loss_fn = nn.BCEWithLogitsLoss()
        model, _, _ = run_epochs(model, opt, sched, loss_fn, X_trva_t, y_trva_t,
                                  bp["batch_size"], best_epoch)

        print("5 · Đánh giá trên test")
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(X_te_t)).cpu().numpy()
        metrics = {"test": evaluate(split.test[1], p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Hệ thống đầy đủ (rule + model) — như train_mambatab.py")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_trva, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_all_num = transform_numeric(X_te_all, scaler_f)
        X_all_cat = apply_categorical_encoders(X_te_all[CATEGORICAL_FEATURES], encoders_f).astype(np.float32)
        X_all_flat = np.concatenate([X_all_num, X_all_cat], axis=1)
        with torch.no_grad():
            p_all = torch.sigmoid(model(torch.tensor(X_all_flat, dtype=torch.float32).to(DEVICE))).cpu().numpy()
        p_te_all = np.where(post_te_all, p_all, 0.0)
        metrics["system_full"] = evaluate(y_te_all, p_te_all, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model_path = os.path.join(ART, "mambatab_v2_tuned.pt")
        torch.save(model.state_dict(), model_path)
        metrics["best_params"] = best_trial.params
        metrics["features"] = FEATURES
        metrics_path = os.path.join(ART, "metrics_mambatab_v2.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
