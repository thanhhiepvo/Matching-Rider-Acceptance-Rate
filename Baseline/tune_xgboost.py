"""Task 3 (W3) — XGBoost tuning: Optuna trên lát `valid` (time-based, split.py) + early
stopping — cùng phương pháp với tune_lightgbm.py, để so sánh 4 model class ở compare_models.py
công bằng (mọi model đều được tune, không riêng LightGBM).

Quy trình giống hệt tune_lightgbm.py (xem docstring ở đó): search trên train/early-stop+đo
trên valid -> lấy best_params/best_iteration -> retrain trên train+valid -> đánh giá trên test.

    python3 Baseline/tune_xgboost.py [--n-trials 50]
"""
from __future__ import annotations

import argparse
import json
import os

import mlflow
import optuna
import pandas as pd
import xgboost as xgb

from evaluation import evaluate
from features import FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from tune_lightgbm import OPTUNA_DB_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_NAME = "xgboost_tuning"
FIXED_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "seed": 42,
    "nthread": 8,
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


def make_objective(dtrain, dvalid, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        booster = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=False,
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        p_va = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        return evaluate(y_va, p_va, "trial-valid", verbose=False)["pr_auc_cancel"]
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-v2-tuned"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "xgboost_v2_optuna"})
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

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

        print(f"3 · Optuna search ({args.n_trials} trial, tối ưu Cancel PR-AUC trên valid)")
        storage = f"sqlite:///{OPTUNA_DB_PATH}"
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage,
                                     direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(dtrain, dvalid, y_va), n_trials=args.n_trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_iteration = best_trial.user_attrs["best_iteration"]
        print(f"  best trial #{best_trial.number}: valid Cancel PR-AUC {best_trial.value:.4f}, "
              f"best_iteration={best_iteration}")
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_valid_pr_auc_cancel", best_trial.value)
        mlflow.log_param("best_iteration_on_valid", best_iteration)

        trials_path = os.path.join(ART, "optuna_trials_xgboost.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        print("4 · Retrain final trên train+valid gộp, dùng ĐÚNG best_iteration")
        X_trva, y_trva = split.train_valid
        X_te_final = align_categories(X_trva, split.test[0])
        y_te_final = split.test[1]
        final_params = {**FIXED_PARAMS, **best_trial.params}
        dtrva = xgb.DMatrix(X_trva, label=y_trva, enable_categorical=True)
        dtest_final = xgb.DMatrix(X_te_final, enable_categorical=True)
        model = xgb.train(final_params, dtrva, num_boost_round=best_iteration)
        mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
        mlflow.log_param("n_train_valid", len(X_trva))

        print("5 · Đánh giá trên test — so với XGBoost V1 (train_xgboost.py)")
        p_te = model.predict(dtest_final)
        metrics = {"test": evaluate(y_te_final, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Hệ thống đầy đủ (rule + model v2)")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_trva, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        dall_post = xgb.DMatrix(X_te_all[post_te_all], enable_categorical=True)
        p_te_all = pd.Series(0.0, index=X_te_all.index)
        p_te_all[post_te_all] = model.predict(dall_post)
        metrics["system_full"] = evaluate(y_te_all, p_te_all.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model.save_model(os.path.join(ART, "model_xgboost_v2_tuned.json"))
        metrics["best_params"] = best_trial.params
        metrics["best_iteration"] = int(best_iteration)
        metrics["features"] = FEATURES
        metrics_path = os.path.join(ART, "metrics_xgboost_v2.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.xgboost.log_model(model, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
