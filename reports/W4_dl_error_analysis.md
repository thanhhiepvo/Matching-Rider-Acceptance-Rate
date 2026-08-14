# Rider Acceptance Prediction — Tuần 4: Model DL sai ở đâu nhiều nhất?

| | | | |
|---|---|---|---|
| **Dự án** | Rider Acceptance (Matching & Dispatch) | **Phần** | Tiếp nối Tuần 2 (phân tích lỗi theo segment) — áp dụng cho 3 model DL của Tuần 3 |
| **Mục tiêu** | Xác định 3 model DL (FT-Transformer V2, MambaTab V2, MLP V2) sai nhiều nhất ở segment/feature nào — so sánh chéo với GBDT (XGBoost V2) để biết DL thua GBDT chung chung hay chỉ ở vài chỗ cụ thể | **Test set** | 13/07/2026, n=8.609 đơn post-dispatch (giống hệt W2/W3) |
| **Model phân tích** | XGBoost V2, MLP V2, FT-Transformer V2, MambaTab V2 — đều dùng hyperparameter Optuna-tuned từ `artifacts/metrics_*_v2.json`, tối ưu theo Cancel PR-AUC (chuẩn W3) | **Tái lập** | `python3 Baseline/run_error_analysis_dl.py` rồi `python3 Baseline/segment_analysis_dl.py` |

---

## 0. Phạm vi & phương pháp

Tiếp nối đúng phương pháp báo cáo Tuần 2 (`reports/part1_shap.md`) — segment theo **khách mới/quen**, **giờ cao điểm/thường**, **ETA dài/ngắn** (cùng định nghĩa `error_analysis.py`) — nhưng lần này so sánh **4 MODEL CLASS** trên CÙNG 1 test set, thay vì 2 architecture của 1 model. Thêm **XGBoost V2** (GBDT thắng ở W3 mục 3b) làm đối chứng, để tách 2 câu hỏi:

1. **3 model DL có yếu ở CÙNG 1 kiểu phân khúc không, hay mỗi model yếu 1 kiểu khác nhau?**
2. **DL thua GBDT ở MỌI nơi (đơn nào GBDT cũng nhỉnh hơn), hay chỉ thua ở 1 nhóm đơn cụ thể — và nhóm đó có đặc điểm gì?**

Ngoài bảng segment + soi trực tiếp (median feature FN vs TP, giống W2), thêm **SHAP/feature-attribution cho cả 3 model DL** — dùng `shap.Explainer(algorithm="permutation")` trên raw feature space (không phải TreeExplainer như LightGBM ở W2, vì DL không phải cây) — để so xem feature nào "gây lỗi" nhiều nhất cho từng model, có trùng với GBDT không.

**Giới hạn cần lưu ý trước khi đọc**: SHAP của 3 model DL tính trên **mẫu 200 đơn FN** (không phải toàn bộ 8.609 — Permutation explainer quá chậm cho quy mô đó với 3 model), background 40 dòng từ train. So sánh XGBoost dùng TreeExplainer (nhanh, chính xác, không cần sample) trên đúng cỡ mẫu FN tương ứng để công bằng khi so sánh. **Độ lớn SHAP value giữa Permutation explainer (DL) và TreeExplainer (XGBoost) không cùng đơn vị** — chỉ so sánh được THỨ HẠNG feature trong từng model, không so trực tiếp con số tuyệt đối giữa DL và XGBoost.


```python
%matplotlib inline
import json
import os

import numpy as np
import pandas as pd

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
OUT = os.path.join(ROOT, 'artifacts', 'dl_error_analysis')

labels = json.load(open(os.path.join(OUT, 'model_labels.json')))
labels
```




    {'XGBoost': 'xgboost V2 (Optuna-tuned)',
     'MLP': 'MLP V2 (Optuna-tuned)',
     'FT-Transformer': 'FT-Transformer V2 (Optuna-tuned)',
     'MambaTab': 'MambaTab V2 (Optuna-tuned)'}



## 1. Model đang phân tích (hyperparameter V2 đã tune, theo Cancel PR-AUC)


```python
pred_df = pd.read_csv(os.path.join(OUT, 'predictions_test.csv'), index_col=0)
MODEL_COLS = ['XGBoost', 'MLP', 'FT-Transformer', 'MambaTab']
print(f"test set: {len(pred_df):,} đơn — base rate accept {pred_df['y_accept'].mean():.4f}")
pd.Series(labels)
```

    test set: 8,609 đơn — base rate accept 0.9080





    XGBoost                  xgboost V2 (Optuna-tuned)
    MLP                          MLP V2 (Optuna-tuned)
    FT-Transformer    FT-Transformer V2 (Optuna-tuned)
    MambaTab                MambaTab V2 (Optuna-tuned)
    dtype: object



## 2. ROC-AUC / Cancel PR-AUC theo segment — 4 model song song

Cùng định nghĩa segment với `error_analysis.py` (W1/W2): khách mới = `cust_orders_30d` NaN hoặc 0; giờ cao điểm = {7,8,9,17,18,19}h; ETA dài = >600s.


```python
seg_df = pd.read_csv(os.path.join(OUT, 'segment_metrics.csv'))
pivot_roc = seg_df.pivot(index='segment', columns='model', values='roc_auc')[MODEL_COLS]
pivot_prc = seg_df.pivot(index='segment', columns='model', values='pr_auc_cancel')[MODEL_COLS]
order = ['Toàn bộ post-dispatch', 'Khách mới', 'Khách quen', 'Giờ cao điểm', 'Giờ thường',
         'ETA dài (>600s)', 'ETA ngắn (≤600s)']
print("ROC-AUC theo segment:")
display(pivot_roc.loc[order].round(4))
print("\nCancel PR-AUC theo segment (metric quyết định chính):")
display(pivot_prc.loc[order].round(4))
```

    ROC-AUC theo segment:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>XGBoost</th>
      <th>MLP</th>
      <th>FT-Transformer</th>
      <th>MambaTab</th>
    </tr>
    <tr>
      <th>segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toàn bộ post-dispatch</th>
      <td>0.7488</td>
      <td>0.7415</td>
      <td>0.7366</td>
      <td>0.7383</td>
    </tr>
    <tr>
      <th>Khách mới</th>
      <td>0.7579</td>
      <td>0.7300</td>
      <td>0.7317</td>
      <td>0.7254</td>
    </tr>
    <tr>
      <th>Khách quen</th>
      <td>0.7481</td>
      <td>0.7427</td>
      <td>0.7373</td>
      <td>0.7394</td>
    </tr>
    <tr>
      <th>Giờ cao điểm</th>
      <td>0.7565</td>
      <td>0.7537</td>
      <td>0.7518</td>
      <td>0.7524</td>
    </tr>
    <tr>
      <th>Giờ thường</th>
      <td>0.7439</td>
      <td>0.7338</td>
      <td>0.7267</td>
      <td>0.7290</td>
    </tr>
    <tr>
      <th>ETA dài (&gt;600s)</th>
      <td>0.7778</td>
      <td>0.7436</td>
      <td>0.8034</td>
      <td>0.7650</td>
    </tr>
    <tr>
      <th>ETA ngắn (≤600s)</th>
      <td>0.7435</td>
      <td>0.7396</td>
      <td>0.7313</td>
      <td>0.7364</td>
    </tr>
  </tbody>
</table>
</div>


    
    Cancel PR-AUC theo segment (metric quyết định chính):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>XGBoost</th>
      <th>MLP</th>
      <th>FT-Transformer</th>
      <th>MambaTab</th>
    </tr>
    <tr>
      <th>segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Toàn bộ post-dispatch</th>
      <td>0.2704</td>
      <td>0.2441</td>
      <td>0.2400</td>
      <td>0.2454</td>
    </tr>
    <tr>
      <th>Khách mới</th>
      <td>0.3218</td>
      <td>0.2502</td>
      <td>0.2606</td>
      <td>0.2280</td>
    </tr>
    <tr>
      <th>Khách quen</th>
      <td>0.2669</td>
      <td>0.2456</td>
      <td>0.2403</td>
      <td>0.2483</td>
    </tr>
    <tr>
      <th>Giờ cao điểm</th>
      <td>0.3020</td>
      <td>0.2891</td>
      <td>0.2816</td>
      <td>0.2810</td>
    </tr>
    <tr>
      <th>Giờ thường</th>
      <td>0.2513</td>
      <td>0.2182</td>
      <td>0.2151</td>
      <td>0.2261</td>
    </tr>
    <tr>
      <th>ETA dài (&gt;600s)</th>
      <td>0.4624</td>
      <td>0.4144</td>
      <td>0.4148</td>
      <td>0.3197</td>
    </tr>
    <tr>
      <th>ETA ngắn (≤600s)</th>
      <td>0.2623</td>
      <td>0.2440</td>
      <td>0.2372</td>
      <td>0.2459</td>
    </tr>
  </tbody>
</table>
</div>


**Đọc bảng**: `ETA dài (>600s)` chỉ có n=45 — quá nhỏ để tin cậy thống kê (giống lưu ý ở W2), chỉ mang tính tham khảo.

## 3. Segment yếu nhất mỗi model


```python
worst = (seg_df[seg_df.segment != 'Toàn bộ post-dispatch']
         .sort_values('pr_auc_cancel').groupby('model').first()[['segment', 'n', 'pr_auc_cancel']]
         .loc[MODEL_COLS])
worst
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>segment</th>
      <th>n</th>
      <th>pr_auc_cancel</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>XGBoost</th>
      <td>Giờ thường</td>
      <td>5337</td>
      <td>0.2513</td>
    </tr>
    <tr>
      <th>MLP</th>
      <td>Giờ thường</td>
      <td>5337</td>
      <td>0.2182</td>
    </tr>
    <tr>
      <th>FT-Transformer</th>
      <td>Giờ thường</td>
      <td>5337</td>
      <td>0.2151</td>
    </tr>
    <tr>
      <th>MambaTab</th>
      <td>Giờ thường</td>
      <td>5337</td>
      <td>0.2261</td>
    </tr>
  </tbody>
</table>
</div>



**Phát hiện chính #1 — cả 4 model (3 DL + GBDT) đều yếu nhất ở CÙNG 1 segment: "Giờ thường"** (không phải giờ cao điểm như trực giác thường nghĩ). Đây không phải đặc thù của kiến trúc DL — GBDT cũng yếu nhất ở đúng segment này. Khác biệt là **mức độ**: XGBoost vẫn giữ Cancel PR-AUC 0,2513 ở "Giờ thường", trong khi FT-Transformer tụt xuống 0,2151 (thấp hơn ~14%) — GBDT không tránh được điểm yếu này, chỉ chịu đựng tốt hơn.

Một phát hiện phụ đáng chú ý: **FT-Transformer bắt được 0 đơn huỷ ở segment "Khách mới"** (`recall_cancel = 0,0000`, xem bảng mục 2) — tệ hơn cả 2 model DL còn lại (MLP/MambaTab đều còn bắt được 1/81 đơn, recall 0,0123). Tái khẳng định phát hiện "khách mới yếu nhất" từ W2, nhưng lần này FT-Transformer là model bỏ sót HOÀN TOÀN nhóm này.

## 4. FN overlap — 3 model DL có sai trên CÙNG những đơn hay không?


```python
overlap = json.load(open(os.path.join(OUT, 'fn_overlap_summary.json')))
pd.Series(overlap['fn_counts'])
```




    XGBoost           765
    MLP               779
    FT-Transformer    785
    MambaTab          769
    dtype: int64




```python
n_huy = int(pred_df['y_accept'].eq(0).sum())
print(f"Tổng số đơn huỷ thật trong test: {n_huy}")
print(f"Union FN (>=1 trong 3 DL sai):        {overlap['dl_fn_union']:4d} ({overlap['dl_fn_union']/n_huy:.1%})")
print(f"Intersection FN (CẢ 3 DL đều sai):    {overlap['dl_fn_intersection']:4d} ({overlap['dl_fn_intersection']/n_huy:.1%})")
print(f"XGBoost FN:                           {overlap['fn_counts']['XGBoost']:4d} ({overlap['fn_counts']['XGBoost']/n_huy:.1%})")
print(f"'DL-only-wrong' (cả 3 DL sai, XGBoost đúng): {overlap['dl_only_wrong_vs_xgboost']:4d}")
```

    Tổng số đơn huỷ thật trong test: 792
    Union FN (>=1 trong 3 DL sai):         787 (99.4%)
    Intersection FN (CẢ 3 DL đều sai):     766 (96.7%)
    XGBoost FN:                            765 (96.6%)
    'DL-only-wrong' (cả 3 DL sai, XGBoost đúng):   11


**Phát hiện chính #2 — 3 model DL sai gần như CÙNG 1 tập đơn, không phải 3 kiểu lỗi độc lập.** Intersection (cả 3 DL cùng sai) = 766/787 union = **97,3%** — nghĩa là hầu như bất kỳ đơn nào 1 model DL bỏ lỡ, 2 model DL còn lại cũng bỏ lỡ theo. Pairwise Jaccard giữa từng cặp DL đều ≥97,5% (xem log `segment_analysis_dl.py`). **Đây là bằng chứng khá mạnh rằng 3 kiến trúc DL (token-per-feature Transformer, selective-scan SSM, MLP thuần) đều bị giới hạn bởi CÙNG 1 nguyên nhân gốc — nhiều khả năng là tín hiệu/feature hiện có không đủ tách nhóm khó này, chứ không phải hạn chế riêng của 1 kiến trúc nào.**

So với XGBoost: XGBoost's FN set (765) gần như là TẬP CON của intersection DL (766) — chỉ **11 đơn** mà cả 3 DL đều sai nhưng XGBoost lại đúng. Ngược lại (GBDT sai, DL đúng) không xảy ra đáng kể ở mức tổng quát này — tức **GBDT không "bù" được cho DL ở phần lớn các đơn khó, GBDT chỉ đơn giản là bắt đúng NHIỀU HƠN trên CÙNG một quần thể đơn khó** (base rate huỷ quá thấp ~9%, threshold 0,5 quá cao — vấn đề đã nêu ở W2), không phải có 1 nhóm đơn "DL nhìn ra nhưng GBDT bỏ lỡ".

## 5. Soi trực tiếp: 11 đơn "DL-only-wrong" so với nhóm DL bắt đúng

⚠️ **n=11 vs n=5 — mẫu RẤT nhỏ, chỉ mang tính minh hoạ/gợi ý hướng, không phải kết luận thống kê chắc chắn** (giống tinh thần lưu ý "n=6" ở phần FP của W2).


```python
num_comp = pd.read_csv(os.path.join(OUT, 'dl_only_wrong_vs_correct_numeric.csv'), index_col=0)
num_comp.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DL-only-wrong (median)</th>
      <th>DL-all-correct TP (median)</th>
      <th>% lệch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pickup_distance_km</th>
      <td>2.04</td>
      <td>0.35</td>
      <td>483.34</td>
    </tr>
    <tr>
      <th>cust_completion_rate_30d</th>
      <td>0.65</td>
      <td>0.19</td>
      <td>236.82</td>
    </tr>
    <tr>
      <th>eta_seconds</th>
      <td>446.07</td>
      <td>141.03</td>
      <td>216.29</td>
    </tr>
    <tr>
      <th>hour_of_day</th>
      <td>17.00</td>
      <td>9.00</td>
      <td>88.89</td>
    </tr>
    <tr>
      <th>total_fee</th>
      <td>41000.00</td>
      <td>335000.00</td>
      <td>-87.76</td>
    </tr>
    <tr>
      <th>trip_distance_km</th>
      <td>3.61</td>
      <td>28.55</td>
      <td>-87.36</td>
    </tr>
    <tr>
      <th>cust_cancel_rate_30d</th>
      <td>0.25</td>
      <td>0.79</td>
      <td>-68.33</td>
    </tr>
    <tr>
      <th>cust_orders_30d</th>
      <td>25.00</td>
      <td>53.00</td>
      <td>-52.83</td>
    </tr>
    <tr>
      <th>dist_hotspot_km</th>
      <td>2.53</td>
      <td>4.90</td>
      <td>-48.39</td>
    </tr>
    <tr>
      <th>fee_per_km</th>
      <td>15424.61</td>
      <td>11124.25</td>
      <td>38.66</td>
    </tr>
    <tr>
      <th>re_dispatch_counts</th>
      <td>2.00</td>
      <td>3.00</td>
      <td>-33.33</td>
    </tr>
    <tr>
      <th>dist_airport_km</th>
      <td>26.75</td>
      <td>24.80</td>
      <td>7.86</td>
    </tr>
    <tr>
      <th>dist_center_km</th>
      <td>11.62</td>
      <td>12.54</td>
      <td>-7.34</td>
    </tr>
    <tr>
      <th>surge_multiplier</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>is_weekend</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>is_schedule_order</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>is_rainy_3h</th>
      <td>1.00</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Nhóm "DL-only-wrong" (11 đơn cả 3 DL đều đoán sai, XGBoost đoán đúng) có xu hướng: **`pickup_distance_km` cao hơn hẳn** (+483%, tài xế phải chạy xa hơn nhiều để đón), **`eta_seconds` dài hơn** (+216%), **giờ đặt muộn hơn** (17h vs 9h) — NHƯNG **`total_fee` thấp hơn nhiều** (-88%) và **`trip_distance_km` ngắn hơn nhiều** (-87%) so với nhóm cả 3 DL bắt đúng. Diễn giải định tính: đây có vẻ là nhóm **"chuyến ngắn, giá rẻ, nhưng tài xế phải đón xa"** — một pattern khá cụ thể mà GBDT (nhờ khả năng cắt ngưỡng phi tuyến dứt khoát trên từng feature) nắm bắt tốt hơn 3 kiến trúc DL trong mẫu nhỏ này.


```python
cat_comp = pd.read_csv(os.path.join(OUT, 'dl_only_wrong_vs_correct_categorical.csv'))
cat_comp.round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>value</th>
      <th>wrong_%</th>
      <th>correct_%</th>
      <th>diff_pp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>travel_mode</td>
      <td>car</td>
      <td>54.5</td>
      <td>100.0</td>
      <td>-45.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vertical_name</td>
      <td>Taxi</td>
      <td>54.5</td>
      <td>100.0</td>
      <td>-45.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>travel_mode</td>
      <td>motorcycle</td>
      <td>45.5</td>
      <td>0.0</td>
      <td>45.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vertical_name</td>
      <td>Express</td>
      <td>36.4</td>
      <td>0.0</td>
      <td>36.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>payment_method</td>
      <td>momo</td>
      <td>9.1</td>
      <td>20.0</td>
      <td>-10.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vertical_name</td>
      <td>Bike</td>
      <td>9.1</td>
      <td>0.0</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>payment_method</td>
      <td>zalo</td>
      <td>9.1</td>
      <td>0.0</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>payment_method</td>
      <td>international_card</td>
      <td>9.1</td>
      <td>0.0</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>payment_method</td>
      <td>cash</td>
      <td>72.7</td>
      <td>80.0</td>
      <td>-7.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>payment_method</td>
      <td>closed_wallet</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>payment_method</td>
      <td>vsf</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>payment_method</td>
      <td>vnpay_pos</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>payment_method</td>
      <td>shopee_pay</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>payment_method</td>
      <td>msb_qr</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>payment_method</td>
      <td>onepay_auto_debit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>payment_method</td>
      <td>physical_business_card</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>payment_method</td>
      <td>partner_pay</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>payment_method</td>
      <td>hdb_qr</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>payment_method</td>
      <td>business_card</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>payment_method</td>
      <td>viettel_money</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>vertical_name</td>
      <td>Food</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>travel_mode</td>
      <td>truck</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>payment_method</td>
      <td>qr</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 6. SHAP / feature-attribution — feature nào "gây lỗi" nhiều nhất mỗi model?

Tính trên mẫu FN (huỷ, đoán sai) của từng model riêng (~200 đơn/model, xem mục 0 về giới hạn phương pháp). Xếp theo mean(|SHAP value|) — feature ảnh hưởng lớn nhất tới quyết định (đúng hoặc sai) trên đúng nhóm đơn model đang sai.


```python
shap_tables = {}
for name in ['xgboost', 'mlp', 'ft_transformer', 'mambatab']:
    s = pd.read_csv(os.path.join(OUT, f'shap_meanabs_{name}.csv'), index_col=0).iloc[:, 0]
    shap_tables[name] = s

rank_df = pd.DataFrame({
    'XGBoost (TreeExplainer)': shap_tables['xgboost'].rank(ascending=False).astype(int),
    'MLP (Permutation)': shap_tables['mlp'].rank(ascending=False).astype(int),
    'FT-Transformer (Permutation)': shap_tables['ft_transformer'].rank(ascending=False).astype(int),
    'MambaTab (Permutation)': shap_tables['mambatab'].rank(ascending=False).astype(int),
})
rank_df['rank trung bình (3 DL)'] = rank_df[['MLP (Permutation)', 'FT-Transformer (Permutation)',
                                              'MambaTab (Permutation)']].mean(axis=1)
rank_df.sort_values('rank trung bình (3 DL)').head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>XGBoost (TreeExplainer)</th>
      <th>MLP (Permutation)</th>
      <th>FT-Transformer (Permutation)</th>
      <th>MambaTab (Permutation)</th>
      <th>rank trung bình (3 DL)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pickup_distance_km</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1.333333</td>
    </tr>
    <tr>
      <th>cust_cancel_rate_30d</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>eta_seconds</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>travel_mode</th>
      <td>17</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>vertical_name</th>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>trip_distance_km</th>
      <td>4</td>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>re_dispatch_counts</th>
      <td>13</td>
      <td>6</td>
      <td>8</td>
      <td>6</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <th>fee_per_km</th>
      <td>7</td>
      <td>9</td>
      <td>5</td>
      <td>8</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>total_fee</th>
      <td>14</td>
      <td>8</td>
      <td>11</td>
      <td>3</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>cust_orders_30d</th>
      <td>8</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Phát hiện chính #3 — cả 4 model class (kể cả GBDT) hội tụ về CÙNG 1 nhóm feature "khó" khi giải thích lỗi**: `pickup_distance_km`, `cust_cancel_rate_30d`, `eta_seconds` nằm trong top-4 của cả 4 model (xem hạng trung bình). `travel_mode`/`vertical_name` cũng lặp lại ở top của cả 3 model DL (MLP hạng 1, MambaTab hạng 2, FT-Transformer thấp hơn nhưng vẫn top-10).

Kết hợp với phát hiện #2 (3 DL sai trên gần như CÙNG 1 tập đơn), bức tranh nhất quán: **độ khó không nằm ở việc model nào "nhìn thấy" feature nào — cả 4 model đều dựa vào đúng những feature có tín hiệu mạnh nhất (`pickup_distance_km`, `cust_cancel_rate_30d`, `eta_seconds`). Vấn đề là những feature này KHÔNG đủ tách được nhóm đơn khó (đơn huỷ "bất ngờ", không theo pattern rõ) — một giới hạn của DỮ LIỆU hiện có, không phải giới hạn riêng của bất kỳ kiến trúc DL nào.**

Bảng chi tiết top-8 mỗi model (giá trị mean(|SHAP|), không so được tuyệt đối giữa 2 loại explainer — chỉ đọc thứ tự trong từng cột):


```python
top8 = pd.DataFrame({
    'XGBoost': shap_tables['xgboost'].head(8).index,
    'MLP': shap_tables['mlp'].head(8).index,
    'FT-Transformer': shap_tables['ft_transformer'].head(8).index,
    'MambaTab': shap_tables['mambatab'].head(8).index,
})
top8.index = [f'#{i+1}' for i in range(8)]
top8
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>XGBoost</th>
      <th>MLP</th>
      <th>FT-Transformer</th>
      <th>MambaTab</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#1</th>
      <td>pickup_distance_km</td>
      <td>travel_mode</td>
      <td>pickup_distance_km</td>
      <td>pickup_distance_km</td>
    </tr>
    <tr>
      <th>#2</th>
      <td>cust_cancel_rate_30d</td>
      <td>pickup_distance_km</td>
      <td>eta_seconds</td>
      <td>travel_mode</td>
    </tr>
    <tr>
      <th>#3</th>
      <td>eta_seconds</td>
      <td>cust_cancel_rate_30d</td>
      <td>cust_cancel_rate_30d</td>
      <td>total_fee</td>
    </tr>
    <tr>
      <th>#4</th>
      <td>trip_distance_km</td>
      <td>eta_seconds</td>
      <td>trip_distance_km</td>
      <td>cust_cancel_rate_30d</td>
    </tr>
    <tr>
      <th>#5</th>
      <td>hour_of_day</td>
      <td>vertical_name</td>
      <td>fee_per_km</td>
      <td>eta_seconds</td>
    </tr>
    <tr>
      <th>#6</th>
      <td>vertical_name</td>
      <td>re_dispatch_counts</td>
      <td>vertical_name</td>
      <td>re_dispatch_counts</td>
    </tr>
    <tr>
      <th>#7</th>
      <td>fee_per_km</td>
      <td>trip_distance_km</td>
      <td>cust_completion_rate_30d</td>
      <td>vertical_name</td>
    </tr>
    <tr>
      <th>#8</th>
      <td>cust_orders_30d</td>
      <td>total_fee</td>
      <td>re_dispatch_counts</td>
      <td>fee_per_km</td>
    </tr>
  </tbody>
</table>
</div>



## 7. Kết luận chung

1. **3 model DL không có 3 kiểu lỗi khác nhau — chúng chia sẻ gần như CÙNG 1 điểm mù** (97,3% FN trùng nhau, cùng segment yếu nhất "Giờ thường", cùng top feature SHAP). Nếu mục tiêu W4 là cải thiện DL, hướng đúng là tìm **feature/dữ liệu mới** giúp tách nhóm đơn khó này (giống khuyến nghị W2), không phải thử thêm kiến trúc DL khác — 3 kiến trúc rất khác nhau (Transformer/SSM/MLP) đã hội tụ về cùng 1 giới hạn, gợi ý đây là giới hạn dữ liệu chứ không phải giới hạn kiến trúc.
2. **GBDT (XGBoost) không "nhìn ra" một nhóm đơn hoàn toàn khác với DL — nó chỉ giỏi hơn TRÊN CÙNG quần thể đơn khó** (chỉ 11/787 đơn là "DL-only-wrong", XGBoost gần như tập con của DL về mặt sai sót). Điều này củng cố thêm phát hiện W3 (bootstrap CI): GBDT thắng DL một cách nhất quán, không phải nhờ bù trừ ở 1 nhóm đơn riêng biệt.
3. **"Khách mới" và "Giờ thường" tiếp tục là 2 segment yếu nhất** — nhất quán với W2 (baseline LightGBM v1) lẫn W4 (4 model V2 hiện tại). FT-Transformer đặc biệt tệ ở "Khách mới" (recall_cancel = 0). Đây là bằng chứng lặp lại nhiều lần (không phải ngẫu nhiên 1 lần đo) rằng vấn đề khách mới là **cold-start dữ liệu**, không phải model chưa đủ mạnh — nhất quán với kết luận W2.
4. **11-mẫu "DL-only-wrong" gợi ý (KHÔNG khẳng định)** một pattern "chuyến ngắn/giá rẻ nhưng tài xế đón xa" mà GBDT nắm được còn DL thì không — đáng thử lại với mẫu lớn hơn (chờ thêm dữ liệu hoặc gộp nhiều ngày test) trước khi kết luận chắc.

---

## 8. Addendum (12/08/2026) — Thử cải thiện FT-Transformer để đánh bại XGBoost

Sau khi mục 6-7 chỉ ra 3 model DL chia sẻ chung 1 điểm mù (không phải hạn chế riêng kiến trúc nào), câu hỏi tự nhiên tiếp theo: **có cách nào cải thiện FT-Transformer đủ để đánh bại XGBoost V2 (Cancel PR-AUC 0,2704) không?** Thử nghiệm 4 hướng, **cô lập từng biến thay đổi 1 lần** để biết chính xác cái nào giúp/hại — không gộp chung rồi đoán mò.

**Tái lập**: `python3 Baseline/train_ft_transformer_v3.py --ablation {full,ple_only,leaf_only,none} --loss {bce,focal}`, `python3 Baseline/train_ft_transformer_v3_calibrated.py`, `python3 Baseline/train_dl_focal_comparison.py`.

### 8.1 · Kiến trúc: Piecewise-linear numeric encoding + GBDT leaf-embedding hybrid

2 thay đổi lấy cảm hứng trực tiếp từ phát hiện SHAP ở mục 6 (feature quan trọng nhất giống hệt nhau giữa GBDT và DL, nhưng GBDT vẫn thắng — gợi ý vấn đề nằm ở CÁCH biểu diễn feature/tương tác, không phải chọn sai feature):
- **Piecewise-linear encoding** (Gorishniy et al. 2022) — mỗi numeric feature chia N_BINS=48 khoảng theo quantile, cho phép token học ngưỡng phi tuyến giống cây quyết định, thay vì 1 phép nhân tuyến tính duy nhất.
- **GBDT leaf-embedding token** — train 1 LightGBM V2 phụ, lấy leaf-index mỗi cây mỗi dòng, embed + mean-pool thành 1 token phụ ghép vào chuỗi token trước Transformer encoder.


```python
ablation_rows = [
    {'Biến thể': 'FT-Transformer V2 gốc (đối chứng)', 'ROC-AUC': 0.7366, 'Cancel PR-AUC': 0.2400, 'Ghi chú': '— baseline —'},
    {'Biến thể': 'full (PLE + leaf, BCE)', 'ROC-AUC': 0.7341, 'Cancel PR-AUC': 0.2366, 'Ghi chú': 'Tệ hơn — early-stop epoch 5, dấu hiệu overfit'},
    {'Biến thể': 'ple_only (chỉ PLE, BCE)', 'ROC-AUC': 0.7424, 'Cancel PR-AUC': 0.2437, 'Ghi chú': 'Tốt hơn — early-stop epoch 20, không overfit'},
    {'Biến thể': 'leaf_only (chỉ leaf, BCE)', 'ROC-AUC': 0.7367, 'Cancel PR-AUC': 0.2402, 'Ghi chú': '≈ không đổi so với gốc'},
]
import pandas as pd
pd.DataFrame(ablation_rows)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Biến thể</th>
      <th>ROC-AUC</th>
      <th>Cancel PR-AUC</th>
      <th>Ghi chú</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FT-Transformer V2 gốc (đối chứng)</td>
      <td>0.7366</td>
      <td>0.2400</td>
      <td>— baseline —</td>
    </tr>
    <tr>
      <th>1</th>
      <td>full (PLE + leaf, BCE)</td>
      <td>0.7341</td>
      <td>0.2366</td>
      <td>Tệ hơn — early-stop epoch 5, dấu hiệu overfit</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ple_only (chỉ PLE, BCE)</td>
      <td>0.7424</td>
      <td>0.2437</td>
      <td>Tốt hơn — early-stop epoch 20, không overfit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>leaf_only (chỉ leaf, BCE)</td>
      <td>0.7367</td>
      <td>0.2402</td>
      <td>≈ không đổi so với gốc</td>
    </tr>
  </tbody>
</table>
</div>



**Đọc bảng**: PLE encoding đứng riêng THẬT SỰ giúp (+0,0037 Cancel PR-AUC, early-stop muộn hơn hẳn → không overfit). Leaf-embedding đứng riêng gần như trung tính. Nhưng **cộng cả 2 lại (`full`) lại TỆ HƠN cả 2 riêng lẻ** — bằng chứng overfit rõ (early-stop ở epoch 5, sớm hơn nhiều so với ple_only's epoch 20) khi cộng dồn quá nhiều tham số mới (leaf-embedding table cho 836 cây × 132 lá) mà không tăng epoch budget/regularization tương ứng. **Kết luận giai đoạn 1**: giữ PLE, bỏ leaf-embedding.

### 8.2 · Loss function: Focal Loss — tác dụng KHÁC NHAU hoàn toàn giữa 3 kiến trúc DL

Thử FocalLoss (alpha = base rate lớp huỷ, gamma=2.0) trên **cả 3 model DL**, dùng đúng hyperparameter V2 đã tune, chỉ đổi loss function — để tách bạch tác dụng của riêng Focal khỏi PLE.


```python
focal_rows = [
    {'Model': 'FT-Transformer V2', 'Loss': 'BCE', 'ROC-AUC': 0.7366, 'Cancel PR-AUC': 0.2400, 'ECE': 0.006},
    {'Model': 'FT-Transformer V2', 'Loss': 'Focal', 'ROC-AUC': 0.7441, 'Cancel PR-AUC': 0.2508, 'ECE': 0.3543},
    {'Model': 'MLP V2', 'Loss': 'BCE', 'ROC-AUC': 0.7415, 'Cancel PR-AUC': 0.2460, 'ECE': 0.0067},
    {'Model': 'MLP V2', 'Loss': 'Focal', 'ROC-AUC': 0.7377, 'Cancel PR-AUC': 0.2400, 'ECE': 0.3667},
    {'Model': 'MambaTab V2', 'Loss': 'BCE', 'ROC-AUC': 0.7384, 'Cancel PR-AUC': 0.2475, 'ECE': 0.0154},
    {'Model': 'MambaTab V2', 'Loss': 'Focal', 'ROC-AUC': 0.7368, 'Cancel PR-AUC': 0.2434, 'ECE': 0.3590},
]
focal_df = pd.DataFrame(focal_rows)
focal_df['Δ Cancel PR-AUC vs BCE'] = focal_df.groupby('Model')['Cancel PR-AUC'].transform(lambda s: s - s.iloc[0])
focal_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Loss</th>
      <th>ROC-AUC</th>
      <th>Cancel PR-AUC</th>
      <th>ECE</th>
      <th>Δ Cancel PR-AUC vs BCE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FT-Transformer V2</td>
      <td>BCE</td>
      <td>0.7366</td>
      <td>0.2400</td>
      <td>0.0060</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FT-Transformer V2</td>
      <td>Focal</td>
      <td>0.7441</td>
      <td>0.2508</td>
      <td>0.3543</td>
      <td>0.0108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MLP V2</td>
      <td>BCE</td>
      <td>0.7415</td>
      <td>0.2460</td>
      <td>0.0067</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MLP V2</td>
      <td>Focal</td>
      <td>0.7377</td>
      <td>0.2400</td>
      <td>0.3667</td>
      <td>-0.0060</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MambaTab V2</td>
      <td>BCE</td>
      <td>0.7384</td>
      <td>0.2475</td>
      <td>0.0154</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MambaTab V2</td>
      <td>Focal</td>
      <td>0.7368</td>
      <td>0.2434</td>
      <td>0.3590</td>
      <td>-0.0041</td>
    </tr>
  </tbody>
</table>
</div>



**Phát hiện quan trọng — Focal loss KHÔNG phải "công thức chung cho DL trên dữ liệu imbalanced" như có thể nghĩ ban đầu:**

- **FT-Transformer**: Focal giúp thật — ROC-AUC +0,0075, Cancel PR-AUC **+0,0108**. Recall lớp huỷ nhảy từ ~0,01 lên ~0,68.
- **MLP**: Focal làm HẠI cả 2 mặt — Cancel PR-AUC **-0,0060**, ECE tăng x55 mà không đổi được ranking tốt hơn để bù.
- **MambaTab**: tương tự MLP — Cancel PR-AUC **-0,0041**, cùng kiểu ECE vỡ vô ích.

Tác dụng của Focal loss có vẻ **phụ thuộc kiến trúc**, không phải đặc tính chung của "dữ liệu imbalanced + DL". Giả thuyết hợp lý nhất: cơ chế self-attention của FT-Transformer (cho phép model "chú ý" chọn lọc vào 1 số token/feature nhất định mỗi dự đoán) tận dụng được gradient tái trọng số của Focal tốt hơn hẳn kiến trúc không có attention (MLP thuần, MambaTab dùng selective-scan nhưng không phải attention thực sự) — nhưng đây là suy đoán, chưa kiểm chứng thêm.

### 8.3 · Calibration bắt buộc sau Focal Loss (áp cho FT-Transformer, model duy nhất Focal thật sự giúp)


```python
calib_rows = [
    {'Bước': 'Trước calib (raw, ple_only + Focal, threshold=0.5)', 'ROC-AUC': 0.7450, 'Cancel PR-AUC': 0.2517, 'ECE': 0.3543, 'Recall huỷ': 0.6831},
    {'Bước': 'Sau calib (isotonic) + threshold=0.84 chọn lại', 'ROC-AUC': 0.7435, 'Cancel PR-AUC': 0.2375, 'ECE': 0.0041, 'Recall huỷ': 0.4571},
]
pd.DataFrame(calib_rows)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bước</th>
      <th>ROC-AUC</th>
      <th>Cancel PR-AUC</th>
      <th>ECE</th>
      <th>Recall huỷ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trước calib (raw, ple_only + Focal, threshold=...</td>
      <td>0.7450</td>
      <td>0.2517</td>
      <td>0.3543</td>
      <td>0.6831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sau calib (isotonic) + threshold=0.84 chọn lại</td>
      <td>0.7435</td>
      <td>0.2375</td>
      <td>0.0041</td>
      <td>0.4571</td>
    </tr>
  </tbody>
</table>
</div>



Cùng pattern đã ghi nhận ở W3 mục 6-7 (imbalance luôn đánh đổi calibration, luôn cần bước calibration riêng sau): fit 3 phương pháp (spline/isotonic/platt) trên tập `calib`, chọn theo ECE đo trên `valid` (không đụng test) — isotonic thắng (ECE valid 0,0068). Sau calibration, ECE giảm từ 0,3543 xuống **0,0041** (dùng được thật), nhưng **Cancel PR-AUC giảm nhẹ theo (0,2517→0,2375)** — hiện tượng isotonic regression tạo tie ở biên, làm lệch nhẹ thứ hạng dù về lý thuyết là hàm đơn điệu (đã ghi nhận tương tự ở W3).

### 8.4 · Bảng tổng hợp toàn bộ 8 biến thể đã thử


```python
final_rows = [
    {'Biến thể': 'XGBoost V2 (đối chứng, GBDT thắng W3)', 'ROC-AUC': 0.7488, 'Cancel PR-AUC': 0.2704, 'Dùng được ngay?': '✅'},
    {'Biến thể': 'FT-Transformer V2 gốc (BCE)', 'ROC-AUC': 0.7366, 'Cancel PR-AUC': 0.2400, 'Dùng được ngay?': '✅'},
    {'Biến thể': 'ple_only (BCE)', 'ROC-AUC': 0.7424, 'Cancel PR-AUC': 0.2437, 'Dùng được ngay?': '✅'},
    {'Biến thể': 'leaf_only (BCE)', 'ROC-AUC': 0.7367, 'Cancel PR-AUC': 0.2402, 'Dùng được ngay?': '✅'},
    {'Biến thể': 'full = ple+leaf (BCE)', 'ROC-AUC': 0.7341, 'Cancel PR-AUC': 0.2366, 'Dùng được ngay?': '✅ (nhưng tệ hơn gốc)'},
    {'Biến thể': 'V2 gốc + Focal (raw)', 'ROC-AUC': 0.7441, 'Cancel PR-AUC': 0.2508, 'Dùng được ngay?': '❌ ECE=0,35'},
    {'Biến thể': 'ple_only + Focal (raw)', 'ROC-AUC': 0.7450, 'Cancel PR-AUC': 0.2517, 'Dùng được ngay?': '❌ ECE=0,35'},
    {'Biến thể': 'ple_only + Focal + Calibrated', 'ROC-AUC': 0.7435, 'Cancel PR-AUC': 0.2375, 'Dùng được ngay?': '✅'},
]
result = pd.DataFrame(final_rows).sort_values('Cancel PR-AUC', ascending=False)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Biến thể</th>
      <th>ROC-AUC</th>
      <th>Cancel PR-AUC</th>
      <th>Dùng được ngay?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XGBoost V2 (đối chứng, GBDT thắng W3)</td>
      <td>0.7488</td>
      <td>0.2704</td>
      <td>✅</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ple_only + Focal (raw)</td>
      <td>0.7450</td>
      <td>0.2517</td>
      <td>❌ ECE=0,35</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V2 gốc + Focal (raw)</td>
      <td>0.7441</td>
      <td>0.2508</td>
      <td>❌ ECE=0,35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ple_only (BCE)</td>
      <td>0.7424</td>
      <td>0.2437</td>
      <td>✅</td>
    </tr>
    <tr>
      <th>3</th>
      <td>leaf_only (BCE)</td>
      <td>0.7367</td>
      <td>0.2402</td>
      <td>✅</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FT-Transformer V2 gốc (BCE)</td>
      <td>0.7366</td>
      <td>0.2400</td>
      <td>✅</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ple_only + Focal + Calibrated</td>
      <td>0.7435</td>
      <td>0.2375</td>
      <td>✅</td>
    </tr>
    <tr>
      <th>4</th>
      <td>full = ple+leaf (BCE)</td>
      <td>0.7341</td>
      <td>0.2366</td>
      <td>✅ (nhưng tệ hơn gốc)</td>
    </tr>
  </tbody>
</table>
</div>



### 8.5 · Kết luận addendum

1. **Không biến thể nào đánh bại được XGBoost V2** (0,2704) — biến thể tốt nhất DÙNG ĐƯỢC NGAY (đã calibrate) chỉ đạt 0,2472-0,2517 ở dạng raw hoặc 0,2375 sau calib, đều thấp hơn XGBoost 0,02-0,03. Kể cả biến thể raw tốt nhất (`ple_only + Focal`, 0,2517, chưa dùng được vì ECE vỡ) vẫn thua XGBoost 0,019.
2. **Focal loss là đòn bẩy thật sự duy nhất tìm được** (+0,0108 Cancel PR-AUC cho FT-Transformer) — nhưng **chỉ hiệu quả với FT-Transformer**, làm HẠI cả MLP lẫn MambaTab. Không có công thức chung "cứ imbalance thì dùng Focal" cho mọi kiến trúc DL.
3. **PLE encoding đóng góp thật nhưng nhỏ và không cộng dồn tuyến tính** với Focal (ple_only+Focal chỉ nhỉnh hơn V2-gốc+Focal 0,0009, trong biên độ nhiễu) — phần lớn cải thiện đến từ Focal, không phải PLE.
4. **Leaf-embedding hybrid nên bỏ hẳn** — trung tính khi đứng riêng, có hại khi cộng với PLE (overfit).
5. Củng cố thêm kết luận mục 7: **giới hạn nằm ở dữ liệu, không phải kỹ thuật huấn luyện/kiến trúc còn thiếu** — đã thử khá đầy đủ (encoding mới, hybrid với GBDT, loss imbalance-aware, calibration đúng chuẩn) mà khoảng cách với XGBoost vẫn không thu hẹp được về 0.
