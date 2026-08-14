# %% [markdown]
# # Rider Cancellation Prediction — Baseline Reproduce (Week 1)
#
# Mục tiêu W1: reproduce đúng model production (LightGBM 6 features + SplineCalib),
# đo đầy đủ metric suite, KHÔNG cải tiến gì trong tuần này.
#
# DOD:
# - Baseline reproduce: sai lệch AUC < 0.005 so với model production
# - Metric suite đầy đủ: Confusion Matrix, AUC, PR-AUC, LogLoss
# - Log MLflow run "V0" (frozen benchmark)
#
# TODO (cần điền trước khi chạy thật):
# - [ ] Đường dẫn / query lấy sample data (order, dispatch, log) từ BigQuery/S3
# - [ ] Hyperparameters LightGBM đúng như production (lấy từ notebook mentor / Confluence)
# - [ ] Cách chia train/test đúng như production (time-based? theo ngày nào?)
# - [ ] Xác nhận PHASE ("predispatch" | "postdispatch") và đường dẫn model artifact tương ứng

# %%
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ml_insights cung cấp SplineCalib — dùng để calibrate xác suất đầu ra của LightGBM
import ml_insights as mli

import mlflow

# %% [markdown]
# ## 0. Config

# %%
# "predispatch" -> rider-predispatch-prediciton/sim/
# "postdispatch" -> rider-postdispatch-prediction/sim/
PHASE = "postdispatch"  # TODO: xác nhận pha đang reproduce

MODEL_DIR = {
    "predispatch": Path("rider-predispatch-prediciton/sim"),
    "postdispatch": Path("rider-postdispatch-prediction/sim"),
}[PHASE]

# TODO: trỏ tới sample data thật (order/dispatch/log) khi có access BigQuery/S3
SAMPLE_DATA_PATH = Path("data/sample_orders.parquet")

RANDOM_STATE = 42
AUC_DEVIATION_TOLERANCE = 0.005  # DOD: sai lệch AUC < 0.005 so với production

FEATURES_NUMERIC = [
    "estimate_time_arrival",
    "estimate_distance_arrival",
    "total_fee",
]
FEATURES_CATEGORICAL = [
    "dispatch_mode",
    "driver_contract_type",
    "service_type",
]
FEATURES_ALL = FEATURES_NUMERIC + FEATURES_CATEGORICAL
TARGET = "is_cancel"  # TODO: xác nhận đúng tên cột label trong data thật

mlflow.set_experiment("rider-cancellation-prediction")

# %% [markdown]
# ## 1. Load sample data
#
# Schema mapping (theo docs `rider_cancellation_baseline_model.md`):
#
# | Feature | Cột nguồn |
# |---|---|
# | estimate_time_arrival | estimate_time_arrival |
# | estimate_distance_arrival | estimate_distance_arrival |
# | total_fee | total_fee |
# | dispatch_mode | dispatch_mode / dispatch_types |
# | driver_contract_type | driver_contract_type |
# | service_type | service_type |


# %%
def load_sample_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)

    # Không có data thật -> tạo data giả để pipeline chạy end-to-end được.
    # XOÁ nhánh này khi đã có access data thật.
    print(f"[WARN] {path} không tồn tại — dùng synthetic data để test pipeline.")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 5000
    df = pd.DataFrame(
        {
            "estimate_time_arrival": rng.exponential(300, n).clip(30, 1800),
            "estimate_distance_arrival": rng.exponential(2, n).clip(0.1, 15),
            "total_fee": rng.normal(50000, 15000, n).clip(10000, 200000),
            "dispatch_mode": rng.choice(["auto", "broadcast", "manual"], n),
            "driver_contract_type": rng.choice(["partner", "employee"], n),
            "service_type": rng.choice(["bike", "car"], n),
        }
    )
    cancel_prob = 1 / (1 + np.exp(-(df["estimate_time_arrival"] / 300 - 2)))
    df[TARGET] = rng.binomial(1, cancel_prob.clip(0.05, 0.6))
    return df


df = load_sample_data(SAMPLE_DATA_PATH)
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Feature mapping (categorical -> int)
#
# Dùng đúng mapping JSON production để đảm bảo encode giống hệt lúc train gốc.

# %%
def load_or_fit_mapping(model_dir: Path, mapping_filename: str, series: pd.Series) -> dict:
    mapping_path = model_dir / mapping_filename
    if mapping_path.exists():
        with open(mapping_path) as f:
            return json.load(f)
    print(f"[WARN] {mapping_path} không tồn tại — tự fit mapping từ sample data (không đúng production).")
    return {v: i for i, v in enumerate(sorted(series.dropna().unique()))}


mapping_files = {
    "dispatch_mode": "dispatch_mode_mapping.json",
    "driver_contract_type": "driver_contract_type_mapping.json",
    "service_type": "service_type_type_mapping.json",
}

categorical_mappings = {
    col: load_or_fit_mapping(MODEL_DIR, fname, df[col])
    for col, fname in mapping_files.items()
}

df_enc = df.copy()
for col, mapping in categorical_mappings.items():
    df_enc[col] = df_enc[col].map(mapping)
    unmapped = df_enc[col].isna().sum()
    if unmapped:
        print(f"[WARN] {unmapped} giá trị ở '{col}' không có trong mapping -> NaN")

# %% [markdown]
# ## 3. Train / test split
#
# TODO: production dùng time-based split hay random split? Nếu time-based,
# cần cột timestamp (order_time?) để cắt theo ngày thay vì random_split.

# %%
X = df_enc[FEATURES_ALL]
y = df_enc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)
print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# %% [markdown]
# ## 4. Train LightGBM baseline (6 features)
#
# TODO: thay params bằng đúng hyperparameters production (mentor notebook / Confluence).

# %%
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": RANDOM_STATE,
    "verbosity": -1,
}

train_set = lgb.Dataset(
    X_train, label=y_train, categorical_feature=FEATURES_CATEGORICAL
)
val_set = lgb.Dataset(
    X_val, label=y_val, categorical_feature=FEATURES_CATEGORICAL, reference=train_set
)

model = lgb.train(
    lgb_params,
    train_set,
    num_boost_round=1000,
    valid_sets=[train_set, val_set],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)

raw_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

# %% [markdown]
# ## 5. Calibration (SplineCalib)

# %%
calibrator = mli.SplineCalib()
calibrator.fit(raw_pred_test, y_test.values)
calib_pred_test = calibrator.predict(raw_pred_test)

# %% [markdown]
# ## 6. Metric suite
#
# Confusion Matrix, AUC, PR-AUC, LogLoss — trên xác suất đã calibrate.


# %%
def compute_metric_suite(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


metrics_v0 = compute_metric_suite(y_test, calib_pred_test)
print(json.dumps(metrics_v0, indent=2))

# %% [markdown]
# ## 7. So sánh với model production
#
# Load booster + calibrator production, predict trên cùng test set, so AUC.
# DOD: sai lệch AUC < 0.005.

# %%
prod_model_path = MODEL_DIR / "lightgbm_cr_model.txt"
prod_calib_path = MODEL_DIR / "calib_cr_model.pkl"

if prod_model_path.exists() and prod_calib_path.exists():
    import joblib

    prod_model = lgb.Booster(model_file=str(prod_model_path))
    prod_calibrator = joblib.load(prod_calib_path)

    prod_raw_pred = prod_model.predict(X_test)
    prod_calib_pred = prod_calibrator.predict(prod_raw_pred)
    prod_auc = roc_auc_score(y_test, prod_calib_pred)

    auc_deviation = abs(metrics_v0["auc"] - prod_auc)
    print(f"reproduced AUC = {metrics_v0['auc']:.4f} | production AUC = {prod_auc:.4f} | "
          f"deviation = {auc_deviation:.4f}")
    assert auc_deviation < AUC_DEVIATION_TOLERANCE, (
        f"AUC deviation {auc_deviation:.4f} vượt tolerance {AUC_DEVIATION_TOLERANCE}"
    )
else:
    print(f"[WARN] Không tìm thấy model artifact production tại {MODEL_DIR} — bỏ qua bước so sánh.")
    auc_deviation = None

# %% [markdown]
# ## 8. Log MLflow (run V0 — frozen benchmark)

# %%
with mlflow.start_run(run_name=f"V0-baseline-{PHASE}"):
    mlflow.log_param("phase", PHASE)
    mlflow.log_params(lgb_params)
    mlflow.log_metric("auc", metrics_v0["auc"])
    mlflow.log_metric("pr_auc", metrics_v0["pr_auc"])
    mlflow.log_metric("log_loss", metrics_v0["log_loss"])
    if auc_deviation is not None:
        mlflow.log_metric("auc_deviation_vs_prod", auc_deviation)
    mlflow.log_dict(metrics_v0["confusion_matrix"], "confusion_matrix.json")
    mlflow.lightgbm.log_model(model, "model")

print("Done — run V0 logged to MLflow.")
