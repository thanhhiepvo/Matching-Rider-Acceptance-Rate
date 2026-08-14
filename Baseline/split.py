"""Time-based split dùng chung cho toàn bộ pipeline W3 — mọi script train/tune/calibrate đều
import từ đây thay vì tự chia lại, đảm bảo mọi so sánh (3 setup / model class / kỹ thuật
imbalance-calib) đứng trên ĐÚNG 1 định nghĩa train/valid/calib/test.

4 lát cắt THEO THỜI GIAN (không random) — dự đoán tương lai từ quá khứ, không lát nào "nhìn
thấy" ngày sau nó:

    train : 14/06 -> 10/07   (~27 ngày, ~230k đơn post-dispatch — phần lớn dữ liệu)
    valid : 11/07             (Optuna trial selection + early-stopping cho model mới)
    calib : 12/07             (fit calibrator — SplineCalib/Isotonic/Platt)
    test  : 13/07             (chỉ đánh giá cuối cùng, KHÔNG đụng ở bất kỳ bước fit nào)

test giữ đúng TEST_DATE của train.py để mọi model/setup vẫn so sánh trực tiếp được với các
MLflow run đã có. valid/calib là 2 ngày liền kề trước test — khác với train_stacking.py (chỉ
có 1 lát "meta_val" dùng chung 2 việc, vì lúc đó tưởng train chỉ có 7 ngày) — ở đây dữ liệu
thật ra trải dài 29 ngày trước test (~8,5k đơn/ngày), đủ nhiều để tách RIÊNG 2 lát mà không lo
mẫu quá ít.

    from split import time_split, align_categories, TEST_DATE, CALIB_DATE, VALID_DATE
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TEST_DATE = "2026-07-13"
CALIB_DATE = "2026-07-12"
VALID_DATE = "2026-07-11"


@dataclass
class Split:
    train: tuple       # (X, y) — phần lớn dữ liệu, dùng để fit model
    valid: tuple        # (X, y) — early-stopping / Optuna trial selection
    calib: tuple         # (X, y) — fit calibrator (KHÔNG dùng để fit model)
    test: tuple          # (X, y) — chỉ đánh giá cuối cùng
    train_valid: tuple   # (X, y) — train+valid gộp, dùng để retrain model "final" sau khi đã
                          # chọn xong hyperparameter/best_iteration trên valid


def align_categories(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Ép categorical dtype của X_other theo ĐÚNG categories đã thấy ở X_ref (thường là tập
    dùng để fit) — bắt buộc gọi trước khi predict/evaluate trên 1 split khác, nếu không
    LightGBM/model có thể coi category lạ (chưa thấy lúc fit) là giá trị thiếu.
    """
    X_other = X_other.copy()
    for c in X_ref.columns:
        if str(X_ref[c].dtype) == "category":
            X_other[c] = pd.Categorical(X_other[c], categories=X_ref[c].cat.categories)
    return X_other


def time_split(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
    post_only: bool = True,
    test_date: str = TEST_DATE, calib_date: str = CALIB_DATE, valid_date: str = VALID_DATE,
) -> Split:
    """Chia df/X/y thành train/valid/calib/test THEO THỜI GIAN (không random).

    post_only=True (mặc định): lọc thêm is_post_dispatch=1 trước khi chia — khớp architecture
    production hiện tại (rule + model, xem train.py). post_only=False: giữ nguyên toàn bộ dữ
    liệu (combined, is_post_dispatch có thể dùng làm feature) — dùng cho setup "combined-control"
    ở compare_setups.py.

    Categorical dtype KHÔNG được align sẵn ở đây — gọi `align_categories(X_ref, X_other)` sau,
    vì X_ref phụ thuộc bước đang làm (vd Optuna align valid/calib/test theo `train`, còn model
    "final" sau khi tune xong lại align theo `train_valid`).
    """
    mask = (df.is_post_dispatch.astype(int) == 1) if post_only else pd.Series(True, index=df.index)

    is_test = df.order_date >= pd.Timestamp(test_date)
    is_calib = (df.order_date >= pd.Timestamp(calib_date)) & ~is_test
    is_valid = (df.order_date >= pd.Timestamp(valid_date)) & ~is_calib & ~is_test
    is_train = ~is_test & ~is_calib & ~is_valid
    is_train_valid = is_train | is_valid

    def sel(m):
        m = m & mask
        return X[m], y[m]

    return Split(
        train=sel(is_train), valid=sel(is_valid), calib=sel(is_calib), test=sel(is_test),
        train_valid=sel(is_train_valid),
    )


def print_split_summary(split: Split, label: str = ""):
    n_tr, n_va, n_ca, n_te = (len(s[0]) for s in (split.train, split.valid, split.calib, split.test))
    prefix = f"{label} " if label else ""
    print(f"  {prefix}train {n_tr:,} · valid {n_va:,} · calib {n_ca:,} · test {n_te:,}")


if __name__ == "__main__":
    # Sanity check nhanh: chạy `python3 Baseline/split.py` để in kích thước các lát cho cả 2
    # chế độ post_only.
    from features import build_features, load_raw
    import os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orders, customer_daily = load_raw(os.path.join(ROOT, "data", "raw"))
    df, X, y = build_features(orders, customer_daily, verbose=False)

    print("post_only=True (production architecture, rule + model):")
    print_split_summary(time_split(df, X, y, post_only=True))
    print("post_only=False (combined-control):")
    print_split_summary(time_split(df, X, y, post_only=False))
