"""Task 3 (W3), DOD item 3 — so sánh model class trên CÙNG feature set & split, kèm thời
gian train/test: LightGBM, XGBoost, CatBoost (V2 tuned nếu đã có kết quả Optuna, else V1),
MLP, FT-Transformer. Train LEAN ngay trong script này (không SHAP/artifact nặng như các script
train_*.py riêng lẻ) để đo timing công bằng, CÙNG 1 lần chạy, CÙNG máy.

Mamba KHÔNG chạy mặc định — train ~30 phút (so với vài giây-vài chục giây các model còn lại,
~600x chậm hơn XGBoost), sẽ phá DOD item 4 ("tái lập bằng 1 lệnh", phục vụ demo W4 cần nhanh)
nếu chạy mỗi lần. Dùng `--include-mamba` để train lại Mamba và đưa vào bảng (script sẽ giữ
nguyên kết quả Mamba của lần chạy trước trong `compare_models_table.csv` nếu không có cờ này,
để bảng so sánh 6 model class trong báo cáo không bị mất dữ liệu).

    python3 Baseline/compare_models.py                  # 5 model, nhanh (mặc định, dùng cho demo/reproduce)
    python3 Baseline/compare_models.py --include-mamba   # 6 model, train lại Mamba (~30 phút)
"""
from __future__ import annotations

import os

# LightGBM (libomp Homebrew) + PyTorch trong CÙNG 1 process trên macOS bị deadlock ở lần
# predict/forward đầu tiên của PyTorch (2 runtime OpenMP giẫm chân nhau) — phải set TRƯỚC khi
# import 2 thư viện này, không set sau được. compare_models.py chạy LightGBM/XGBoost/CatBoost
# RỒI MỚI tới MLP trong CÙNG process, đúng kịch bản gây deadlock — xem ensemble.py/
# train_stacking.py (đã gặp lỗi này trước, đây chỉ áp lại đúng fix).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader

from evaluation import evaluate
from features import CATEGORICAL_FEATURES, FEATURES, PRE_CATEGORICAL, build_features, load_raw
from mlp_common import TabularDataset, TabularMLP, encode_categoricals, fit_numeric_scaler, transform_numeric
from split import align_categories, time_split
from train import ART, EXPERIMENT_NAME, MLFLOW_DB_PATH, PARAMS as LGB_V1_PARAMS, RAW
from train_catboost import PARAMS as CATBOOST_PARAMS, to_catboost_categoricals
from train_ft_transformer import FTTransformer
from train_mamba import MambaTabular
from train_xgboost import PARAMS as XGB_PARAMS
from tune_mlp import HIDDEN_CHOICES
from pipeline_diagram import (
    ClassificationHeadViz,
    FeatureTokenizerViz,
    MambaEncoderViz,
    TransformerEncoderViz,
    save_pipeline_diagram,
)
from sklearn.pipeline import Pipeline

torch.set_num_threads(1)
MLP_NUMERIC = [f for f in FEATURES if f not in PRE_CATEGORICAL]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MLP_EPOCHS, MLP_PATIENCE = 40, 5
MLP_V1_PARAMS = {"embed_dim": 4, "hidden": "medium", "dropout": 0.3, "lr": 1e-3,
                  "weight_decay": 1e-5, "batch_size": 512}


def run_lightgbm(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = dict(LGB_V1_PARAMS)
    v2_path = os.path.join(ART, "metrics_lightgbm_v2.json")
    if os.path.exists(v2_path):
        best_params = json.load(open(v2_path))["best_params"]
        params = {**LGB_V1_PARAMS, **best_params}
        label = "LightGBM V2 (Optuna-tuned)"
    else:
        label = "LightGBM V1 (hand-set params)"

    t0 = time.perf_counter()
    model = lgb.train(
        params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_va, y_va)], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    predict_s = time.perf_counter() - t0
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def run_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = dict(XGB_PARAMS)
    v2_path = os.path.join(ART, "metrics_xgboost_v2.json")
    if os.path.exists(v2_path):
        best_params = json.load(open(v2_path))["best_params"]
        params = {**XGB_PARAMS, **best_params}
        label = "XGBoost V2 (Optuna-tuned)"
    else:
        label = "XGBoost V1 (hand-set params)"

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dtest = xgb.DMatrix(X_te, enable_categorical=True)

    t0 = time.perf_counter()
    booster = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=False,
    )
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_te = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    predict_s = time.perf_counter() - t0
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def run_catboost(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = dict(CATBOOST_PARAMS)
    v2_path = os.path.join(ART, "metrics_catboost_v2.json")
    if os.path.exists(v2_path):
        best_params = json.load(open(v2_path))["best_params"]
        params = {**CATBOOST_PARAMS, **best_params}
        label = "CatBoost V2 (Optuna-tuned)"
    else:
        label = "CatBoost V1 (hand-set params)"

    X_tr_cb, X_va_cb, X_te_cb = (to_catboost_categoricals(d) for d in (X_tr, X_va, X_te))
    model = CatBoostClassifier(**params, cat_features=CATEGORICAL_FEATURES, early_stopping_rounds=50)

    t0 = time.perf_counter()
    model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), use_best_model=True)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_te = model.predict_proba(X_te_cb)[:, 1]
    predict_s = time.perf_counter() - t0
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def run_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = dict(MLP_V1_PARAMS)
    v2_path = os.path.join(ART, "metrics_mlp_v2.json")
    if os.path.exists(v2_path):
        params = json.load(open(v2_path))["best_params"]
        label = "MLP V2 (Optuna-tuned)"
    else:
        label = "MLP V1 (hand-set params)"

    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    _, te_codes, _ = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr.values)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, drop_last=True)
    X_va_num_t = torch.tensor(X_va_num, dtype=torch.float32).to(DEVICE)
    X_va_cat_t = torch.tensor(X_va_cat, dtype=torch.long).to(DEVICE)
    X_te_num_t = torch.tensor(X_te_num, dtype=torch.float32).to(DEVICE)
    X_te_cat_t = torch.tensor(X_te_cat, dtype=torch.long).to(DEVICE)

    model = TabularMLP(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_cardinalities,
                        embed_dim=params["embed_dim"], hidden=HIDDEN_CHOICES[params["hidden"]],
                        dropout=params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for xb_num, xb_cat, yb in loader:
            xb_num, xb_cat, yb = xb_num.to(DEVICE), xb_cat.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb_num, xb_cat), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
        auc = evaluate(y_va, p_va, "mlp-valid", verbose=False)["roc_auc"]
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(X_te_num_t, X_te_cat_t)).cpu().numpy()
    predict_s = time.perf_counter() - t0
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def _prep_token_inputs(X_tr, X_va, X_te):
    """Tiền xử lý dùng chung cho FT-Transformer/Mamba — cùng scaler/encoder, fit trên train."""
    scaler = fit_numeric_scaler(X_tr, MLP_NUMERIC)
    X_tr_num, X_va_num, X_te_num = (transform_numeric(d, scaler) for d in (X_tr, X_va, X_te))
    tr_codes, va_codes, encoders = encode_categoricals(X_tr[PRE_CATEGORICAL], X_va[PRE_CATEGORICAL])
    _, te_codes, _ = encode_categoricals(X_tr[PRE_CATEGORICAL], X_te[PRE_CATEGORICAL])
    X_tr_cat = np.stack([tr_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_va_cat = np.stack([va_codes[c] for c in PRE_CATEGORICAL], axis=1)
    X_te_cat = np.stack([te_codes[c] for c in PRE_CATEGORICAL], axis=1)
    cat_cardinalities = [len(encoders[c][0]) for c in PRE_CATEGORICAL]

    def to_t(x_num, x_cat):
        return (torch.tensor(x_num, dtype=torch.float32).to(DEVICE),
                torch.tensor(x_cat, dtype=torch.long).to(DEVICE))

    return (to_t(X_tr_num, X_tr_cat), to_t(X_va_num, X_va_cat), to_t(X_te_num, X_te_cat), cat_cardinalities)


def _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va, epochs=40, batch_size=512, lr=1e-3,
                        weight_decay=1e-5, patience=5, grad_clip=None):
    """Vòng lặp train dùng chung cho FT-Transformer/Mamba — cùng pattern early-stop trên valid
    như run_mlp() ở trên, chỉ khác model nhận input dạng token thay vì vector phẳng."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    X_tr_num_t, X_tr_cat_t = X_tr_t
    X_va_num_t, X_va_cat_t = X_va_t
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).to(DEVICE)
    n_train = len(y_tr)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train - n_train % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_num_t[idx], X_tr_cat_t[idx]), y_tr_t[idx])
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_num_t, X_va_cat_t)).cpu().numpy()
        auc = evaluate(y_va, p_va, "valid", verbose=False)["roc_auc"]
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


def run_ft_transformer(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """d_token=32/layers=3/heads=4 (mặc định) — THẮNG bản Optuna-tuned (106k tham số) ở thử
    nghiệm riêng (0,7475 vs 0,7329), nên dùng mặc định làm đại diện chính thức ở đây."""
    (X_tr_t, X_va_t, X_te_t, cat_card) = _prep_token_inputs(X_tr, X_va, X_te)
    model = FTTransformer(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_card,
                           d_token=32, n_layers=3, n_heads=4, d_ffn=64, dropout=0.15).to(DEVICE)

    viz_pipeline = Pipeline([
        ("feature_tokenizer", FeatureTokenizerViz(n_numeric=len(MLP_NUMERIC), cat_cardinalities=tuple(cat_card),
                                                    d_token=32, cls_position="first")),
        ("transformer_encoder", TransformerEncoderViz(n_layers=3, n_heads=4, d_ffn=64, dropout=0.15)),
        ("classification_head", ClassificationHeadViz(d_token=32)),
    ])
    save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_ft_transformer.html"))
    mlflow.log_artifact(os.path.join(ART, "pipeline_ft_transformer.html"))

    t0 = time.perf_counter()
    model = _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(*X_te_t)).cpu().numpy()
    predict_s = time.perf_counter() - t0
    label = "FT-Transformer (self-attention)"
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def save_mamba_pipeline_diagram(cat_card):
    """Sơ đồ pipeline Mamba — CHỈ cấu trúc (không cần train), nên gọi UNCONDITIONALLY trong
    main() (không phụ thuộc --include-mamba) để bản 1-lệnh "tái lập nhanh" (DOD 4, không train
    lại Mamba ~30 phút) vẫn luôn có đủ 5 sơ đồ pipeline mới nhất trong artifacts/."""
    viz_pipeline = Pipeline([
        ("feature_tokenizer", FeatureTokenizerViz(n_numeric=len(MLP_NUMERIC), cat_cardinalities=tuple(cat_card),
                                                    d_token=32, cls_position="last")),
        ("mamba_encoder", MambaEncoderViz(n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.15)),
        ("classification_head", ClassificationHeadViz(d_token=32)),
    ])
    save_pipeline_diagram(viz_pipeline, os.path.join(ART, "pipeline_mamba.html"))
    mlflow.log_artifact(os.path.join(ART, "pipeline_mamba.html"))


def run_mamba(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """d_token=32/layers=3/d_state=16 — kiến trúc mặc định của train_mamba.py (CLS ở CUỐI
    chuỗi, bắt buộc cho model nhân quả — xem so sánh kiến trúc trong báo cáo)."""
    (X_tr_t, X_va_t, X_te_t, cat_card) = _prep_token_inputs(X_tr, X_va, X_te)
    model = MambaTabular(n_numeric=len(MLP_NUMERIC), cat_cardinalities=cat_card,
                          d_token=32, n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.15).to(DEVICE)
    save_mamba_pipeline_diagram(cat_card)

    t0 = time.perf_counter()
    model = _train_token_model(model, X_tr_t, y_tr, X_va_t, y_va, lr=3e-4, grad_clip=1.0)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(*X_te_t)).cpu().numpy()
    predict_s = time.perf_counter() - t0
    label = "Mamba (selective state-space)"
    return label, evaluate(y_te, p_te, label, verbose=False), train_s, predict_s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mamba", action="store_true",
                     help="train lại Mamba (~30 phút) và đưa vào bảng — mặc định BỎ QUA để "
                          "pipeline tái lập nhanh (DOD 4); không cần cờ này nếu chỉ muốn xem "
                          "bảng 6 model, kết quả Mamba trước đó vẫn được giữ trong bảng")
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="compare-models-v2-tuned"):
        mlflow.set_tags({"phase": "w3_task3", "purpose": "4_model_class_comparison_fair_tuned"})

        print("1 · Nạp data + chia train/valid/test THEO THỜI GIAN (split.py, post-dispatch only)")
        orders, customer_daily = load_raw(RAW)
        df, X, y = build_features(orders, customer_daily, verbose=False)
        split = time_split(df, X, y, post_only=True)
        X_tr, y_tr = split.train
        X_va, y_va = align_categories(X_tr, split.valid[0]), split.valid[1]
        X_te, y_te = align_categories(X_tr, split.test[0]), split.test[1]
        print(f"  train {len(X_tr):,} · valid {len(X_va):,} · test {len(X_te):,} · {len(FEATURES)} feature")

        if not args.include_mamba:
            # Mamba bị bỏ qua (không train ~30 phút) nhưng sơ đồ pipeline KHÔNG cần train nên
            # vẫn tạo lại — đảm bảo bản 1-lệnh "tái lập nhanh" luôn có đủ 5 sơ đồ mới nhất.
            _, _, _, cat_card_for_diagram = _prep_token_inputs(X_tr, X_va, X_te)
            save_mamba_pipeline_diagram(cat_card_for_diagram)

        runners = [run_lightgbm, run_xgboost, run_catboost, run_mlp, run_ft_transformer]
        if args.include_mamba:
            runners.append(run_mamba)
        rows = []
        for runner in runners:
            print(f"2 · {runner.__name__}")
            label, m, train_s, predict_s = runner(X_tr, y_tr, X_va, y_va, X_te, y_te)
            print(f"  [{label}] train {train_s:.1f}s · predict {predict_s:.2f}s · "
                  f"ROC-AUC {m['roc_auc']:.4f} · PR-AUC huỷ {m['pr_auc_cancel']:.4f} · "
                  f"Brier {m['brier']:.4f} · ECE {m['ece']:.4f}")
            rows.append({
                "model_class": label, "train_seconds": round(train_s, 2), "predict_seconds": round(predict_s, 3),
                "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "pr_auc_cancel": m["pr_auc_cancel"],
                "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
                "precision_cancel": m["precision_cancel"], "recall_cancel": m["recall_cancel"],
                "f1_cancel": m["f1_cancel"],
            })
            mlflow.log_metrics({
                f"{runner.__name__}_roc_auc": m["roc_auc"], f"{runner.__name__}_train_seconds": train_s,
                f"{runner.__name__}_predict_seconds": predict_s,
            })

        table_csv = os.path.join(ART, "compare_models_table.csv")
        table_md = os.path.join(ART, "compare_models_table.md")
        tbl = pd.DataFrame(rows)
        if not args.include_mamba and os.path.exists(table_csv):
            prev = pd.read_csv(table_csv)
            prev_mamba = prev[prev["model_class"].str.contains("Mamba", case=False, na=False)]
            if not prev_mamba.empty:
                tbl = pd.concat([tbl, prev_mamba], ignore_index=True)
                print(f"  (giữ nguyên kết quả Mamba từ lần chạy trước — dùng --include-mamba để train lại)")
        tbl = tbl.sort_values("roc_auc", ascending=False)

        print("\n3 · Bảng so sánh (DOD item 3)")
        print(tbl.to_string(index=False))

        tbl.to_csv(table_csv, index=False)
        open(table_md, "w").write(tbl.to_markdown(index=False))
        mlflow.log_artifact(table_csv)
        mlflow.log_artifact(table_md)

        print(f"\n✓ -> {ART}")
        print(f"✓ MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
