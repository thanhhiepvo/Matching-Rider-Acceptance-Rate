"""DOD item 5 (Optional, W3) — thử nghiệm feature THỜI TIẾT mới, join từ `data/weather_daily_30d.parquet`.

⚠️ Dataset gốc tự đánh dấu `same_day_full_daily_leakage_risk=True` cho MỌI dòng — đây là
aggregate CẢ NGÀY (mean/max/min trên 24h), fetch RETROACTIVE (historical API, `fetched_at_utc`
= 04/08, sau ngày test 13/07 rất xa). Join THẲNG theo đúng ngày đặt cuốc sẽ RÒ RỈ: 1 đơn đặt
lúc 7h sáng không thể biết trước nhiệt độ cao nhất trong ngày (thường xảy ra ~14h). Xử lý: join
LAG 1 NGÀY (thời tiết hôm QUA làm feature cho đơn hôm nay) — cùng tinh thần `closed='left'` ở
`customer_history()` trong features.py, thời tiết hôm qua CHẮC CHẮN đã biết trước khi đơn hôm
nay được đặt.

    from weather import build_weather_lag1_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert df.same_day_full_daily_leakage_risk.all(), \
        "weather_daily_30d.parquet: kỳ vọng MỌI dòng đều same_day_full_daily_leakage_risk=True"
    return df


def build_weather_lag1_features(orders: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Trả về DataFrame cùng index với `orders`, có các cột thời tiết HÔM QUA (lag 1 ngày) tại
    ô lưới 0,1° gần nhất với điểm đón (`pickup_latitude`/`pickup_longitude`).
    """
    cell_lat = orders.pickup_latitude.round(1)
    cell_lon = orders.pickup_longitude.round(1)
    weather_cell_id = cell_lat.map("{:.1f}".format) + "_" + cell_lon.map("{:.1f}".format)
    lag_date = orders.order_date - pd.Timedelta(days=1)

    key = pd.DataFrame({"weather_cell_id": weather_cell_id, "weather_date": lag_date})
    cols = ["temperature_2m_mean_c", "precipitation_sum_mm", "rain_sum_mm",
            "wind_speed_10m_max_kmh", "cloud_cover_mean_pct", "weather_code_wmo"]
    w = weather[["weather_cell_id", "weather_date"] + cols].copy()
    merged = key.merge(w, on=["weather_cell_id", "weather_date"], how="left")
    merged.index = orders.index

    out = pd.DataFrame(index=orders.index)
    out["temp_mean_c_d1"] = merged["temperature_2m_mean_c"]
    out["rain_sum_mm_d1"] = merged["rain_sum_mm"]
    out["wind_speed_max_kmh_d1"] = merged["wind_speed_10m_max_kmh"]
    out["cloud_cover_pct_d1"] = merged["cloud_cover_mean_pct"]
    out["is_rainy_d1"] = (merged["rain_sum_mm"] > 1.0).astype("float")  # >1mm = "có mưa đáng kể"
    return out


def load_weather_hourly(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_weather_lag1hour_features(orders: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Giống `build_weather_lag1_features()` nhưng LAG 1 GIỜ thay vì 1 NGÀY — tín hiệu SÁT thời
    điểm đặt cuốc hơn nhiều (trong vòng 1 giờ, thay vì tới 24-48h) mà vẫn an toàn không rò rỉ:
    giờ NGAY TRƯỚC giờ đặt cuốc chắc chắn đã "xảy ra xong" (observed), khác với giờ HIỆN TẠI
    (order đặt lúc 14:32 vẫn đang ở giữa khung 14:00-15:00, dữ liệu giờ đó CHƯA đầy đủ tại thời
    điểm đặt) — nguyên tắc y hệt lag-1-ngày, chỉ đổi độ phân giải.
    """
    cell_lat = orders.pickup_latitude.round(1)
    cell_lon = orders.pickup_longitude.round(1)
    weather_cell_id = cell_lat.map("{:.1f}".format) + "_" + cell_lon.map("{:.1f}".format)
    lag_hour = orders.order_datetime.dt.floor("h") - pd.Timedelta(hours=1)

    key = pd.DataFrame({"weather_cell_id": weather_cell_id, "datetime": lag_hour})
    cols = ["temperature_2m_c", "rain_mm", "wind_speed_10m_kmh", "cloud_cover_pct", "weather_code_wmo"]
    w = weather_hourly[["weather_cell_id", "datetime"] + cols].copy()
    merged = key.merge(w, on=["weather_cell_id", "datetime"], how="left")
    merged.index = orders.index

    out = pd.DataFrame(index=orders.index)
    out["temp_c_h1"] = merged["temperature_2m_c"]
    out["rain_mm_h1"] = merged["rain_mm"]
    out["wind_speed_kmh_h1"] = merged["wind_speed_10m_kmh"]
    out["cloud_cover_pct_h1"] = merged["cloud_cover_pct"]
    out["is_rainy_h1"] = (merged["rain_mm"] > 0.1).astype("float")  # >0,1mm/h = "đang mưa"
    return out


def build_weather_lag3hour_features(orders: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Biến thể "mượt" của lag-1-giờ — cộng dồn mưa/TB các chỉ số khác trong 3 GIỜ LIỀN TRƯỚC đó
    (giờ đặt cuốc trở về trước 3 giờ, không tính giờ hiện tại) — giảm nhiễu đơn-giờ trong khi
    vẫn SÁT thời điểm hơn nhiều so với lag-1-ngày.
    """
    cell_lat = orders.pickup_latitude.round(1)
    cell_lon = orders.pickup_longitude.round(1)
    weather_cell_id = cell_lat.map("{:.1f}".format) + "_" + cell_lon.map("{:.1f}".format)
    hour_floor = orders.order_datetime.dt.floor("h")

    w = weather_hourly.set_index(["weather_cell_id", "datetime"]).sort_index()
    key = pd.DataFrame({"weather_cell_id": weather_cell_id, "hour_floor": hour_floor})

    # Tính rolling-3h 1 LẦN cho mỗi cell (không phải mỗi đơn) rồi merge lại theo (cell, lag_hour)
    # — tránh bug thứ tự groupby, và nhanh hơn nhiều so với lặp theo từng đơn.
    roll_rows = []
    for cid, wc in w.groupby(level=0):
        wc = wc.droplevel(0).sort_index()
        full_idx = pd.date_range(wc.index.min(), wc.index.max(), freq="h")
        rc = wc["rain_mm"].reindex(full_idx).fillna(0)
        tc = wc["temperature_2m_c"].reindex(full_idx)
        wsc = wc["wind_speed_10m_kmh"].reindex(full_idx)
        roll = pd.DataFrame({
            "weather_cell_id": cid, "datetime": full_idx,
            "rain_sum_mm_3h": rc.rolling(3, min_periods=1).sum().values,
            "temp_mean_c_3h": tc.rolling(3, min_periods=1).mean().values,
            "wind_max_kmh_3h": wsc.rolling(3, min_periods=1).max().values,
        })
        roll_rows.append(roll)
    roll_df = pd.concat(roll_rows, ignore_index=True)

    lag_hour = key.hour_floor - pd.Timedelta(hours=1)
    merge_key = pd.DataFrame({"weather_cell_id": key.weather_cell_id, "datetime": lag_hour})
    merged = merge_key.merge(roll_df, on=["weather_cell_id", "datetime"], how="left")
    merged.index = orders.index

    out = pd.DataFrame(index=orders.index)
    out["rain_sum_mm_3h"] = merged["rain_sum_mm_3h"]
    out["temp_mean_c_3h"] = merged["temp_mean_c_3h"]
    out["wind_max_kmh_3h"] = merged["wind_max_kmh_3h"]
    out["is_rainy_3h"] = (out["rain_sum_mm_3h"] > 0.1).astype("float")
    return out


if __name__ == "__main__":
    # Sanity check nhanh: coverage của join lag-1 trên orders thật.
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from features import load_raw

    orders, _ = load_raw(os.path.join(ROOT, "data", "raw"))
    weather = load_weather(os.path.join(ROOT, "data", "weather_daily_30d.parquet"))
    feats = build_weather_lag1_features(orders, weather)
    print(f"n orders: {len(orders):,}")
    for c in feats.columns:
        print(f"  {c:24s} null={feats[c].isna().mean():.2%}  "
              f"{'' if feats[c].isna().all() else f'mean={feats[c].mean():.3f}'}")
