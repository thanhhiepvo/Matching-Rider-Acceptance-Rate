"""Biến 3 file raw thành SHORT-LIST 12 feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- SHORT-LIST: 12 feature ------------------------------------------------
NUMERIC_FEATURES = [
    "eta_seconds",         # ETA tài xế tới điểm đón (giây)     — công phải bỏ ra
    "pickup_distance_km",  # quãng đường tới điểm đón           — "chạy không công"
    "trip_distance_km",    # độ dài cuốc                        — công việc chính
    "total_pay",           # tài xế nhận được bao nhiêu         — phần thưởng
    "pay_per_km",          # tiền trên mỗi km                   — dẫn xuất
    "hour_of_day",         # giờ trong ngày (0-23)
    "is_weekend",          # cuối tuần
    "offer_seq",           # lượt chào thứ mấy của cuốc này
    "driver_ar_7d",        # AR của TÀI XẾ 7 ngày trước         — point-in-time
    "driver_offers_7d",    # số lượt chào tài xế nhận được 7 ngày trước
]
CATEGORICAL_FEATURES = [
    "driver_contract_type",  # bike / car / bike_platform / partner …
    "travel_mode",           # bike / car
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "y_accept"

# --- Cột CẤM dùng làm feature ----------------------------------------------------------
# order_key / driver_key : định danh, không phải tín hiệu (và sẽ khiến model học thuộc lòng)
# order_date / order_datetime : dùng để chia train/test và tính lịch sử, không phải feature
# dispatch_mode : đã lọc về một giá trị 'manual', thành hằng số
FORBIDDEN = ["order_key", "driver_key", "order_date", "order_datetime", "dispatch_mode", TARGET]


def driver_history(driver_daily: pd.DataFrame) -> pd.DataFrame:
    """AR lịch sử của tài xế trong cửa sổ [D-7, D-1] — KHÔNG bao gồm ngày D.
    """
    d = driver_daily.sort_values(["driver_key", "stat_date"]).copy()
    cols = ["n_accepted", "n_rejected", "n_ignored"]

    roll = (
        d.set_index("stat_date")
        .groupby("driver_key")[cols]
        .rolling("7D", closed="left")   # ← [D-7, D-1]; bỏ closed='left' là rò rỉ
        .sum()
        .reset_index()
        .rename(columns={c: c + "_7d" for c in cols})
    )

    # Mẫu số đúng của AR = nhận + từ chối + bỏ qua.
    # KHÔNG dùng total_dispatch_job: nó lớn hơn tổng ba cái này (xem DATA_DICTIONARY.md).
    offers = roll.n_accepted_7d + roll.n_rejected_7d + roll.n_ignored_7d
    roll["driver_offers_7d"] = offers
    roll["driver_ar_7d"] = np.where(offers > 0, roll.n_accepted_7d / offers, np.nan)

    return roll[["driver_key", "stat_date", "driver_ar_7d", "driver_offers_7d"]]


def build_features(offers: pd.DataFrame, orders: pd.DataFrame, driver_daily: pd.DataFrame,
                   dispatch_mode: str = "manual", verbose: bool = True):
    """Ráp 3 file raw -> bảng feature. Trả về (df đầy đủ, X, y)."""

    n0 = len(offers)
    df = offers[offers.dispatch_mode == dispatch_mode].copy()
    if verbose:
        print(f"'{dispatch_mode}': {len(df):,}/{n0:,} lượt ghép ({len(df)/n0:.1%})")

    # 2 - Join thuộc tính cuốc — LEFT join, KHÔNG dropna.
    df = df.merge(orders, on="order_key", how="left")

    # 3 - Join lịch sử tài xế theo (driver_key, ngày của lượt chào)
    hist = driver_history(driver_daily)
    df = df.merge(hist, left_on=["driver_key", "order_date"],
                  right_on=["driver_key", "stat_date"], how="left").drop(columns=["stat_date"])

    # 4 - Feature dẫn xuất
    df["hour_of_day"] = df.order_datetime.dt.hour
    df["is_weekend"] = df.order_datetime.dt.dayofweek.isin([5, 6]).astype(int)
    df["pay_per_km"] = df.total_pay / df.trip_distance_km.replace(0, np.nan)

    # 5 - Tách X, y
    y = df[TARGET].astype(int)
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")

    # 6 - Chốt chặn chống rò rỉ — chạy mỗi lần, không phải chỉ một lần lúc viết
    leaked = [c for c in X.columns if c in FORBIDDEN]
    assert not leaked, f"RÒ RỈ: {leaked} không được là feature"
    assert TARGET not in X.columns, "RÒ RỈ: nhãn nằm trong X"

    if verbose:
        print(f"  {X.shape[1]} feature - {len(X):,} dòng - AR {y.mean():.4f}")
    return df, X, y


def load_raw(raw_dir: str):
    return (
        pd.read_parquet(f"{raw_dir}/offers.parquet"),
        pd.read_parquet(f"{raw_dir}/orders.parquet"),
        pd.read_parquet(f"{raw_dir}/driver_daily_stats.parquet"),
    )
 