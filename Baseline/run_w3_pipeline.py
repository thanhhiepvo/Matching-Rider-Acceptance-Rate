"""Task 2 (W3) — "Pipeline train tái lập". DOD item 4: toàn bộ run được log MLflow, tái lập
bằng 1 lệnh (phục vụ Demo tuần 4).

Chạy TUẦN TỰ 10 bước — đủ để chạm mọi DOD item, KHÔNG lặp lại các biến thể đã được cover gián
tiếp (vd không chạy riêng train_xgboost.py/train_catboost.py baseline vì compare_models.py đã
train fresh cả 4 model bên trong, dùng ĐÚNG param đã tune ở bước 2-5; không chạy riêng
train_xgboost.py --imbalance weighted vì train_v3.py đã demo đúng combo model-thắng +
imbalance + calibration):

  1. compare_setups.py   — Task 1: 3 setup xử lý is_post_dispatch trên CÙNG test set
  2. tune_lightgbm.py     — Task 3: LightGBM V2 (Optuna)
  3. tune_xgboost.py      — Task 3: XGBoost V2 (Optuna)
  4. tune_catboost.py     — Task 3: CatBoost V2 (Optuna)
  5. tune_mlp.py          — Task 3: MLP V2 (Optuna) — chậm nhất trong 4 bước tune
  6. compare_models.py    — Task 3 + DOD item 2/3: ≥4 model class ĐÃ TUNE CÔNG BẰNG, kèm thời
                            gian train/test (đọc lại artifact metrics_*_v2.json từ bước 2-5).
                            Mặc định KHÔNG train lại Mamba (~30 phút, xem compare_models.py) —
                            giữ nguyên kết quả Mamba đã có trong compare_models_table.csv để
                            bảng báo cáo vẫn đủ 6 model, còn pipeline tái lập vẫn nhanh (DOD 4).
  7. train.py --imbalance weighted — Task 3: scale_pos_weight, đo tác động PR-AUC/calibration
  8. calibration.py       — Task 3: SplineCalib/Isotonic/Platt trên tập calib riêng
  9. train_v3_ft_transformer.py — Task 4: Model V3 ĐỀ XUẤT CHÍNH, dùng FT-Transformer (chọn
                            theo tiêu chí ROC-AUC/SOTA, KHÔNG theo tốc độ — xem báo cáo mục 3-4)
                            + FocalLoss + calibration + threshold. ~130s.
  10. train_v3.py         — Task 4: Model V3 PHƯƠNG ÁN THAY THẾ, dùng XGBoost V2 (nhanh hơn
                            ~50x, không cần GPU) — vẫn chạy để báo cáo có cả 2 lựa chọn.

Cả 4 bước tune (2-5) lưu Optuna study vào `ROOT/optuna.db` (dùng chung 1 file, study name khác
nhau) — mở `optuna-dashboard sqlite:///optuna.db` để xem optimization history/param importance
cho cả 4 model cùng lúc.

    python3 Baseline/run_w3_pipeline.py [--skip-tuning]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(name: str, cmd: list[str]) -> float:
    print(f"\n{'=' * 70}\n▶ {name}\n  $ {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n✗ THẤT BẠI: {name} (exit code {result.returncode}, {elapsed:.1f}s)")
        sys.exit(result.returncode)
    print(f"✓ {name} xong ({elapsed:.1f}s)")
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgb-trials", type=int, default=50)
    ap.add_argument("--xgb-trials", type=int, default=50)
    ap.add_argument("--cat-trials", type=int, default=25)
    ap.add_argument("--mlp-trials", type=int, default=20)
    ap.add_argument("--mlp-epochs-per-trial", type=int, default=15)
    ap.add_argument("--skip-tuning", action="store_true",
                     help="bỏ qua cả 4 bước tune_*.py (dùng nếu artifacts/metrics_*_v2.json đã có sẵn)")
    args = ap.parse_args()

    py = sys.executable
    steps = [("1/9 · So sánh 3 setup (Task 1)", [py, "compare_setups.py"])]
    if not args.skip_tuning:
        steps += [
            ("2/9 · Optuna tuning LightGBM V2 (Task 3)",
             [py, "tune_lightgbm.py", "--n-trials", str(args.lgb_trials)]),
            ("3/9 · Optuna tuning XGBoost V2 (Task 3)",
             [py, "tune_xgboost.py", "--n-trials", str(args.xgb_trials)]),
            ("4/9 · Optuna tuning CatBoost V2 (Task 3)",
             [py, "tune_catboost.py", "--n-trials", str(args.cat_trials)]),
            ("5/9 · Optuna tuning MLP V2 (Task 3, chậm nhất)",
             [py, "tune_mlp.py", "--n-trials", str(args.mlp_trials),
              "--epochs-per-trial", str(args.mlp_epochs_per_trial)]),
        ]
    else:
        print("2-5/9 · Bỏ qua cả 4 bước tune_*.py (--skip-tuning)")
    steps += [
        ("6/10 · So sánh 4 model class, đã tune công bằng (Task 3 + DOD 2/3)", [py, "compare_models.py"]),
        ("7/10 · LightGBM + imbalance handling (Task 3)", [py, "train.py", "--imbalance", "weighted"]),
        ("8/10 · Calibration trên tập calib riêng (Task 3)", [py, "calibration.py"]),
        ("9/10 · Model V3 (FT-Transformer, đề xuất chính, Task 4)", [py, "train_v3_ft_transformer.py"]),
        ("10/10 · Model V3 (XGBoost, phương án thay thế, Task 4)", [py, "train_v3.py"]),
    ]

    print(f"W3 pipeline — {len(steps)} bước, log MLflow vào {os.path.join(os.path.dirname(HERE), 'mlflow.db')}")
    t_start = time.perf_counter()
    timings = {}
    for name, cmd in steps:
        timings[name] = run_step(name, cmd)

    total = time.perf_counter() - t_start
    print(f"\n{'=' * 70}\n✓ TOÀN BỘ PIPELINE HOÀN TẤT — {total:.1f}s ({total / 60:.1f} phút)\n{'=' * 70}")
    for name, secs in timings.items():
        print(f"  {name:45s} {secs:>7.1f}s")
    print("\nXem kết quả: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Xem Optuna:  optuna-dashboard sqlite:///optuna.db")


if __name__ == "__main__":
    main()
