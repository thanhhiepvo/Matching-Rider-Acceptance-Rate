"""Task 3 (W3) — CatBoost tuning: Optuna trên lát `valid` (time-based, split.py) + early
stopping — cùng phương pháp với tune_lightgbm.py/tune_xgboost.py.

    python3 Baseline/tune_catboost.py [--n-trials 50]
"""
from __future__ import annotations

import argparse
import json
import os

import mlflow
import optuna
import pandas as pd
from catboost import CatBoostClassifier

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, RAW
from train_catboost import to_catboost_categoricals
from tune_lightgbm import OPTUNA_DB_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_NAME = "catboost_tuning"
FIXED_PARAMS = {"eval_metric": "AUC", "random_seed": 42, "thread_count": 8, "verbose": False}


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


def make_objective(X_tr_cb, y_tr, X_va_cb, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **FIXED_PARAMS,
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        model = CatBoostClassifier(**params, cat_features=CATEGORICAL_FEATURES, early_stopping_rounds=50)
        model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        p_va = model.predict_proba(X_va_cb)[:, 1]
        return evaluate(y_va, p_va, "trial-valid", verbose=False)["pr_auc_cancel"]
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="catboost-v2-tuned"):
        mlflow.set_tags({"phase": "post_dispatch_only", "model": "catboost_v2_optuna"})
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

        X_tr_cb, X_va_cb = to_catboost_categoricals(X_tr), to_catboost_categoricals(X_va)

        print(f"3 · Optuna search ({args.n_trials} trial, tối ưu Cancel PR-AUC trên valid)")
        storage = f"sqlite:///{OPTUNA_DB_PATH}"
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(study_name=STUDY_NAME, storage=storage,
                                     direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(X_tr_cb, y_tr, X_va_cb, y_va), n_trials=args.n_trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_iteration = best_trial.user_attrs["best_iteration"]
        print(f"  best trial #{best_trial.number}: valid Cancel PR-AUC {best_trial.value:.4f}, "
              f"best_iteration={best_iteration}")
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_valid_pr_auc_cancel", best_trial.value)
        mlflow.log_param("best_iteration_on_valid", best_iteration)

        trials_path = os.path.join(ART, "optuna_trials_catboost.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        print("4 · Retrain final trên train+valid gộp, dùng ĐÚNG best_iteration")
        X_trva, y_trva = split.train_valid
        X_te_final = align_categories(X_trva, split.test[0])
        y_te_final = split.test[1]
        final_params = {**FIXED_PARAMS, **best_trial.params, "iterations": best_iteration}
        X_trva_cb, X_te_final_cb = to_catboost_categoricals(X_trva), to_catboost_categoricals(X_te_final)
        model = CatBoostClassifier(**final_params, cat_features=CATEGORICAL_FEATURES)
        model.fit(X_trva_cb, y_trva)
        mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
        mlflow.log_param("n_train_valid", len(X_trva))

        print("5 · Đánh giá trên test — so với CatBoost V1 (train_catboost.py)")
        p_te = model.predict_proba(X_te_final_cb)[:, 1]
        metrics = {"test": evaluate(y_te_final, p_te, "test")}
        log_metrics("test", metrics["test"])

        print("6 · Hệ thống đầy đủ (rule + model v2)")
        is_test_full = df.order_date >= pd.Timestamp("2026-07-13")
        X_te_all = align_categories(X_trva, X[is_test_full].copy())
        y_te_all = y[is_test_full]
        post_te_all = df[is_test_full].is_post_dispatch.astype(int).values == 1
        X_te_all_cb = to_catboost_categoricals(X_te_all[post_te_all])
        p_te_all = pd.Series(0.0, index=X_te_all.index)
        p_te_all[post_te_all] = model.predict_proba(X_te_all_cb)[:, 1]
        metrics["system_full"] = evaluate(y_te_all, p_te_all.values, "system-full")
        log_metrics("system_full", metrics["system_full"])

        print("7 · Lưu artifact")
        model.save_model(os.path.join(ART, "model_catboost_v2_tuned.cbm"))
        metrics["best_params"] = best_trial.params
        metrics["best_iteration"] = int(best_iteration)
        metrics["features"] = FEATURES
        metrics_path = os.path.join(ART, "metrics_catboost_v2.json")
        json.dump(metrics, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        mlflow.log_artifact(metrics_path)
        mlflow.catboost.log_model(model, name="model")

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
