"""Task 3 (W3) — LightGBM V2: Optuna tuning trên lát `valid` (time-based, split.py) + early
stopping, thay cho PARAMS cố định tay ở train.py (LightGBM V1).

Quy trình:
  1. Optuna search (TPE sampler) — mỗi trial train trên `train`, early-stop + đo ROC-AUC trên
     `valid` (KHÔNG đụng `test`/`calib` ở bước này).
  2. Lấy best_params + best_iteration (số vòng boosting của trial tốt nhất) từ bước 1.
  3. Retrain "final" model bằng best_params trên `train+valid` gộp (nhiều dữ liệu hơn), dùng
     ĐÚNG best_iteration đã tìm được — không early-stop lại (tránh phải đụng thêm `calib`/`test`
     chỉ để chọn số vòng).
  4. Đánh giá model final trên `test` — so sánh trực tiếp với LightGBM V1 (train.py).

    python3 Baseline/tune_lightgbm.py [--n-trials 50]
"""
from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd

from evaluation import evaluate
from features import FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW, ROOT

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Lưu vào SQLite riêng (không chung mlflow.db), DÙNG CHUNG 1 file với các script tune_*.py
# khác (mỗi model 1 study_name) — để optuna-dashboard mở 1 nơi xem được cả 4 model, không cần
# 4 dashboard riêng.
OPTUNA_DB_PATH = os.path.join(ROOT, "optuna.db")
STUDY_NAME = "lightgbm_v2_tuning"

FIXED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_threads": 8,
    "seed": 42,
}


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


def make_objective(X_tr, y_tr, X_va, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        model = lgb.train(
            params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        trial.set_user_attr("best_iteration", model.best_iteration)
        p_va = model.predict(X_va, num_iteration=model.best_iteration)
        return evaluate(y_va, p_va, "trial-valid", verbose=False)["pr_auc_cancel"]
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm-v2-tuned"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "lightgbm_v2_optuna"})
        mlflow.log_param("n_trials", args.n_trials)

        print("1 · Nạp data + build feature")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)

        print("2 · Chia train/valid/test THEO THỜI GIAN (post-dispatch only, split.py)")
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,}")

        print(f"3 · Optuna search ({args.n_trials} trial, tối ưu Cancel PR-AUC trên valid)")
        storage = f"sqlite:///{OPTUNA_DB_PATH}"
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        except KeyError:
            pass  # chưa có study trước đó — bỏ qua
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage,
                                     direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(X_tr, y_tr, X_va, y_va), n_trials=args.n_trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_iteration = best_trial.user_attrs["best_iteration"]
        print(f"  best trial #{best_trial.number}: valid Cancel PR-AUC {best_trial.value:.4f}, "
              f"best_iteration={best_iteration}")
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_valid_pr_auc_cancel", best_trial.value)
        mlflow.log_param("best_iteration_on_valid", best_iteration)

        trials_path = os.path.join(ART, "optuna_trials_lightgbm.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        print("4 · Retrain final trên train+valid gộp, dùng ĐÚNG best_iteration (không early-stop lại)")
        X_trva, y_trva = split.train_valid
        X_te_final = align_categories(X_trva, split.test[0])
        y_te_final = split.test[1]
        final_params = {**FIXED_PARAMS, **best_trial.params}
        model = lgb.train(final_params, lgb.Dataset(X_trva, y_trva), num_boost_round=best_iteration)
        mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
        mlflow.log_param("n_train_valid", len(X_trva))

        print("5 · Đánh giá trên test — so với LightGBM V1 (train.py)")
        p_te = model.predict(X_te_final)
        metrics = {"test": evaluate(y_te_final, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Hệ thống đầy đủ (rule + model v2) — như train.py")
        post_mask_full = df.is_post_dispatch.astype(int) == 1
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_trva, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        p_te_all = pd.Series(0.0, index=X_te_all.index)
        p_te_all[post_te_all] = model.predict(X_te_all[post_te_all])
        metrics["system_full"] = evaluate(y_te_all, p_te_all.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model.save_model(os.path.join(ART, "model_v2_tuned.txt"))
        metrics["best_params"] = best_trial.params
        metrics["best_iteration"] = int(best_iteration)
        metrics["features"] = FEATURES
        metrics_path = os.path.join(ART, "metrics_lightgbm_v2.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.lightgbm.log_model(model, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
