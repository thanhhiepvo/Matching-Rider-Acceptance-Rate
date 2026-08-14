"""Task 1 (W3) — so sánh 3 setup xử lý `is_post_dispatch` trên CÙNG 1 test set:

  (a) post-dispatch-only : is_post_dispatch là RULE bên ngoài (không phải feature), model chỉ
                            train/infer trên phần post-dispatch — ĐÚNG architecture production
                            hiện tại (train.py).
  (b) combined-control    : is_post_dispatch LÀ 1 feature, model train trên TOÀN BỘ dữ liệu
                             (pre+post) — giữ làm đối chứng (kiến trúc "cũ", trước khi tách rule
                             ra ngoài).
  (c) rule + model system : hệ thống ĐẦY ĐỦ = rule (pre-dispatch -> huỷ chắc chắn, p=0) +
                             model (a) (post-dispatch) — con số mô phỏng đúng những gì hệ thống
                             thật sẽ trả ra.

(a) và (c) dùng CHUNG 1 model — (c) chỉ là cách đánh giá khác của model (a) trên toàn bộ test
set (pre+post), không train model riêng. (b) là model ĐỘC LẬP, kiến trúc khác hẳn.

Khác 1 điểm so với train.py gốc: early-stopping của cả (a) lẫn (b) dùng lát `valid` riêng
(split.py) thay vì chính test set — đúng tinh thần "time-based validation" mà task 3 (Optuna
tuning) cũng dùng, tránh rò rỉ nhẹ qua early stopping. Điểm số cuối trên `test` vì vậy đáng tin
hơn 1 chút so với train.py's `baseline-v1-post-dispatch-only` run (chênh không đáng kể trên
thực tế, nhưng đúng phương pháp hơn).

    python3 Baseline/compare_setups.py
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import mlflow
import pandas as pd

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, build_features, load_raw
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, PARAMS, RAW


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


def fit_lgb(X_tr, y_tr, X_va, y_va, label: str):
    model = lgb.train(
        PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=500,
        valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  [{label}] dừng ở vòng {model.best_iteration}, train n={len(X_tr):,}")
    return model


def main():
    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="compare-setups-v1"):
        mlflow.set_tags({"phase": "w3_task1", "purpose": "3_setup_comparison"})

        print("1 · Nạp data + build feature")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)

        print("2 · Setup (a): post-dispatch-only (is_post_dispatch = RULE bên ngoài)")
        split_a = time_split(df, X, y, post_only=True)
        Xa_tr, ya_tr = split_a.train
        Xa_va, ya_va = align_categories(Xa_tr, split_a.valid[0]), split_a.valid[1]
        Xa_te, ya_te = align_categories(Xa_tr, split_a.test[0]), split_a.test[1]
        print(f"  train {len(Xa_tr):,} · valid {len(Xa_va):,} · test {len(Xa_te):,}")
        model_a = fit_lgb(Xa_tr, ya_tr, Xa_va, ya_va, "post-dispatch-only")
        p_a_te = model_a.predict(Xa_te, num_iteration=model_a.best_iteration)
        m_a_post = evaluate(ya_te, p_a_te, "a-post-dispatch-only", verbose=False)

        print("3 · Setup (b): combined-control (is_post_dispatch LÀ feature, train trên toàn bộ)")
        X_combined = X.copy()
        X_combined["is_post_dispatch"] = df["is_post_dispatch"]
        split_b = time_split(df, X_combined, y, post_only=False)
        Xb_tr, yb_tr = split_b.train
        Xb_va, yb_va = align_categories(Xb_tr, split_b.valid[0]), split_b.valid[1]
        Xb_te, yb_te = align_categories(Xb_tr, split_b.test[0]), split_b.test[1]
        print(f"  train {len(Xb_tr):,} · valid {len(Xb_va):,} · test {len(Xb_te):,}")
        model_b = fit_lgb(Xb_tr, yb_tr, Xb_va, yb_va, "combined-control")
        p_b_te = model_b.predict(Xb_te, num_iteration=model_b.best_iteration)
        m_b_full = evaluate(yb_te, p_b_te, "b-combined-full", verbose=False)

        post_te_mask = Xb_te["is_post_dispatch"].astype(int).values == 1
        m_b_post = evaluate(yb_te.values[post_te_mask], p_b_te[post_te_mask], "b-combined-post", verbose=False)

        print("4 · Setup (c): rule + model system — dùng model (a), đánh giá trên TOÀN BỘ test set")
        # X_te_full lấy từ split (b) (post_only=False -> có đủ pre+post) nhưng chỉ giữ đúng cột
        # FEATURES (không có is_post_dispatch) để predict bằng model (a).
        Xc_te_full = align_categories(Xa_tr, Xb_te[FEATURES].copy())
        p_c_te = pd.Series(0.0, index=Xc_te_full.index)
        p_c_te[post_te_mask] = model_a.predict(Xc_te_full[post_te_mask], num_iteration=model_a.best_iteration)
        m_c_full = evaluate(yb_te, p_c_te.values, "c-rule-plus-model-full", verbose=False)

        print("5 · Tổng hợp — 3 setup trên CÙNG 1 test set (13/07)")
        rows = []
        metric_keys = ["roc_auc", "pr_auc", "pr_auc_cancel", "log_loss", "brier", "ece",
                        "precision_cancel", "recall_cancel", "f1_cancel", "cancel_flagged_rate"]
        for k in metric_keys:
            rows.append({
                "metric": k,
                "a_post_dispatch_only (post, n={})".format(m_a_post["n"]): m_a_post[k],
                "b_combined_control (full, n={})".format(m_b_full["n"]): m_b_full[k],
                "b_combined_control (post, n={})".format(m_b_post["n"]): m_b_post[k],
                "c_rule_plus_model (full, n={})".format(m_c_full["n"]): m_c_full[k],
            })
        tbl = pd.DataFrame(rows)
        print(tbl.to_string(index=False))

        log_metrics("a_post", m_a_post)
        log_metrics("b_full", m_b_full)
        log_metrics("b_post", m_b_post)
        log_metrics("c_full", m_c_full)

        print("6 · Lưu artifact")
        out = {"a_post_dispatch_only": m_a_post, "b_combined_full": m_b_full,
               "b_combined_post": m_b_post, "c_rule_plus_model_full": m_c_full}
        metrics_path = os.path.join(ART, "compare_setups_metrics.json")
        json.dump(out, open(metrics_path, "w"), indent=2, ensure_ascii=False)
        table_path = os.path.join(ART, "compare_setups_table.csv")
        tbl.to_csv(table_path, index=False)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(table_path)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
