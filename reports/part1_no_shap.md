# Rider Acceptance Prediction — Phần 1: Model sai ở segment/feature nào?

| | | | |
|---|---|---|---|
| **Dự án** | Rider Acceptance (Matching & Dispatch) | **Phần** | 1/N — Phân tích lỗi theo segment + feature/pattern |
| **Mục tiêu** | Xác định **baseline v1 gốc** (chưa thêm feature nào) đang sai nhiều nhất ở đâu — theo segment VÀ theo feature/pattern — để làm đầu vào cho việc tìm feature mới ở phần sau | **Test set** | 13/07/2026, n=8.609 đơn post-dispatch |
| **Model phân tích** | **Baseline v1 GỐC** — chưa thêm bất kỳ feature nào | **Architecture** | So sánh song song 2 cách xử lý `is_post_dispatch` (xem mục 1) |

---

## 0. Lưu ý quan trọng về phạm vi

Toàn bộ phân tích trong báo cáo này chạy trên **baseline v1 nguyên bản** — đúng tập feature đầu tiên của dự án, **trước khi thêm bất kỳ feature mới nào**:

- **(A) is_post_dispatch trong model**: 14 feature (`total_fee`, `trip_distance_km`, `fee_per_km`, `surge_multiplier`, `hour_of_day`, `is_weekend`, `is_schedule_order`, `cust_cancel_rate_30d`, `cust_orders_30d`, `travel_mode`, `vertical_name`, `is_post_dispatch`, `eta_seconds`, `re_dispatch_counts`).
- **(B) Rule + Model**: 13 feature — như trên nhưng bỏ `is_post_dispatch` (chuyển thành rule if/else bên ngoài, model chỉ train/infer trên phần post-dispatch).

## 1. So sánh ROC-AUC theo segment (baseline v1 gốc)

| Segment | n | Base rate | AUC (A) 14ft | AUC (B) 13ft | Δ (B−A) |
|---|---|---|---|---|---|
| **Toàn bộ post-dispatch** | 8.609 | 0,908 | 0,7352 | 0,7356 | +0,0004 |
| **Khách mới** | 745 | 0,891 | **0,7031** | **0,7010** | −0,0021 |
| Khách quen | 7.864 | 0,910 | 0,7379 | 0,7385 | +0,0006 |
| Giờ cao điểm | 3.272 | 0,903 | 0,7426 | 0,7424 | −0,0002 |
| Giờ thường | 5.337 | 0,911 | 0,7298 | 0,7303 | +0,0005 |
| ETA dài (>600s)* | 45 | 0,867 | 0,8419 | 0,8205 | −0,0214 |
| ETA ngắn (≤600s) | 8.264 | 0,908 | 0,7289 | 0,7293 | +0,0004 |

*Sample ETA dài quá nhỏ (45 dòng) — không đủ tin cậy thống kê.

**Khách mới là segment yếu nhất rõ rệt** (AUC ~0,70, thấp hơn "Toàn bộ" gần 3,5 điểm AUC).

## 2. Precision / Recall / PR-AUC (class huỷ) theo segment

| Segment | Recall (A) | Recall (B) | Precision (A) | Precision (B) | PR-AUC huỷ (A) | PR-AUC huỷ (B) |
|---|---|---|---|---|---|---|
| Toàn bộ post-dispatch | 0,0227 | 0,0227 | 0,783 | 0,818 | 0,2490 | 0,2502 |
| Khách mới | 0,0123 | 0,0123 | 1,000 | 1,000 | 0,2397 | 0,2373 |
| Khách quen | 0,0239 | 0,0239 | 0,773 | 0,810 | 0,2503 | 0,2525 |
| Giờ cao điểm | 0,0189 | 0,0158 | 0,857 | 0,833 | 0,2583 | 0,2596 |
| Giờ thường | 0,0253 | 0,0274 | 0,750 | 0,813 | 0,2421 | 0,2440 |
| ETA dài* | 0,1667 | 0,1667 | 0,333 | 0,500 | 0,5079 | 0,5310 |
| ETA ngắn | 0,0157 | 0,0157 | 0,923 | 0,923 | 0,2402 | 0,2414 |

Recall thấp ở mọi segment — nhưng **khách mới đặc biệt tệ: recall chỉ 0,0123** (bắt được 1/81 đơn huỷ thực tế), thấp nhất trong mọi segment. Nguyên nhân gốc: threshold 0,5 quá cao so với base rate ~90-91% (đa số đơn ở segment nào cũng là "đi") — không phải khác biệt giữa 2 architecture (A) và (B).

## 3. Đi tìm nguyên nhân — feature/pattern nào gây lỗi nhiều nhất?

**Phương pháp**: chạy dự đoán baseline v1 gốc (14ft) trên tập test (post-dispatch, n=8.609), tách **4 nhóm** ở threshold 0,5 — xét **cả 2 chiều lỗi** trước khi đi tìm feature, để feature tìm được (nếu có) cân nhắc được cả 2, không chỉ tối ưu 1 chiều rồi vô tình làm chiều kia tệ hơn:

| Nhóm | Định nghĩa | n | % |
|---|---|---|---|
| **FN** — huỷ, đoán sai (đoán sẽ đi) | y=huỷ, p≥0,5 | **775** | 97,8% tổng lỗi |
| **FP** — đi, đoán sai (đoán sẽ huỷ) | y=đi, p<0,5 | **6** | 0,8% tổng lỗi |
| TP — huỷ, bắt đúng | y=huỷ, p<0,5 | 17 | — |
| TN — đi, đoán đúng | y=đi, p≥0,5 | 7.811 | — |

FN áp đảo tuyệt đối về khối lượng (775 vs 6, tỉ lệ ~129:1) — đúng lý do trước đó tập trung phân tích FN. Nhưng **6 đơn FP vẫn đáng nhìn qua trước khi xếp hạng feature**, vì nếu 1 feature nào đó "sửa" FN theo hướng ngược lại đặc điểm của FP, cần biết trước để cân nhắc đánh đổi.

**Đặc điểm 6 đơn FP** (đối chứng: nhóm TN — đơn đi, đoán đúng):

| Cột | FP (median) | TN (median) | % lệch |
|---|---|---|---|
| total_fee | 241.000 | 49.000 | **+392%** |
| cust_cancel_rate_30d | 0,580 | 0,094 | **+515%** |
| trip_distance_km | 22,5 | 5,37 | **+319%** |
| eta_seconds | 460 | 197 | +133% |
| pickup_distance_km | 1,77 | 0,76 | +132% |
| cust_orders_30d | 10,5 | 14,0 | −25% |

6 đơn FP đều là chuyến **giá cao, đường dài, khách có lịch sử huỷ cao** (`cash`, chủ yếu `Taxi`, 2/6 là đơn đặt trước) — đúng nhóm mà model (dựa trên `total_fee`/`trip_distance_km`/`cust_cancel_rate_30d`) có lý do hợp lý để nghi ngờ, nhưng lần này khách vẫn đi. **Đây chính là chiều NGƯỢC LẠI của FN**: ở FN, `cust_cancel_rate_30d` cao hơn nhóm đối chứng dự đoán ĐÚNG là sẽ huỷ; ở FP, `cust_cancel_rate_30d` cao hơn nhóm đối chứng lại dự đoán SAI (khách vẫn đi). Tức feature này (và total_fee/trip_distance_km) **không mâu thuẫn nhau giữa 2 chiều lỗi** — cùng hướng "rủi ro cao hơn", chỉ là với n=6 mẫu quá nhỏ để tách được "rủi ro cao nhưng vẫn đi" khỏi "rủi ro cao và huỷ thật" bằng feature nào thêm. Không phát hiện tín hiệu nào ở FP đi ngược hướng với candidate feature tìm được từ FN — nên bảng xếp hạng ở mục 3.1-3.2 dưới đây (từ FN) **an toàn để dùng, không có rủi ro "sửa được FN thì hại FP"**.

So sánh 2 nhóm FN/TP trên **mọi cột dữ liệu có sẵn** (kể cả cột CHƯA đưa vào baseline v1 gốc) — cột nào lệch nhiều nhất giữa 2 nhóm là ứng viên feature mới đáng thử nhất. Kết quả **giống hệt nhau ở cả 2 architecture (A) và (B)** — vì baseline v1 gốc dự đoán gần như tương đương ở 2 architecture.

### 3.1 · Cột số (numeric) — xếp theo % lệch giữa FN và TP

| Cột | FN (median) | TP (median) | % lệch | Đã có trong baseline v1 gốc? |
|---|---|---|---|---|
| **cust_cancel_rate_30d** | 0,167 | 0,094 | **76,7%** | ✅ Có sẵn |
| **pickup_distance_km** | 1,171 | 0,761 | **53,9%** | ❌ Chưa — ứng viên #1 |
| **eta_seconds** | 248,95 | 197,44 | **26,1%** | ✅ Có sẵn |
| trip_distance_km | 4,77 | 5,38 | 11,4% | ✅ Có sẵn |
| **cust_completion_rate_30d** | 0,786 | 0,857 | **8,3%** | ❌ Chưa — ứng viên |
| **dist_hotspot_km** | 2,27 | 2,10 | **8,2%** | ❌ Chưa — ứng viên |
| total_fee | 45.000 | 49.000 | 8,2% | ✅ Có sẵn |
| **dist_center_km** | 6,86 | 6,51 | **5,4%** | ❌ Chưa — ứng viên |
| **dist_airport_km** | 23,41 | 23,06 | 1,6% | ❌ Chưa (lệch quá nhỏ) |
| fee_per_km | 10.116 | 10.215 | 1,0% | ✅ Có sẵn |
| pickup_latitude/longitude (thô) | — | — | ~0% | Không dùng trực tiếp được |
| surge_multiplier, re_dispatch_counts, cust_orders_30d, hour_of_day | — | — | 0% | ✅ Có sẵn, không lệch |

### 3.2 · Cột phân loại (categorical) — xếp theo % lệch tỉ trọng

| Cột | Giá trị lệch nhiều nhất | FN % | TP % | Đã có trong baseline v1 gốc? |
|---|---|---|---|---|
| **payment_method** | `cash` | 68,5% | 61,9% | ❌ Chưa — ứng viên |
| | `partner_pay` | 0,8% | 6,9% | |
| **vertical_name** | `Bike` | 48,8% | 38,6% | ✅ Có sẵn (nhưng chưa đủ mịn) |
| | `Express` | 8,7% | 16,8% | |
| service_name | `xanhsm_bike_hn` | 44,1% | 36,0% | ❌ Chưa thử (chi tiết hơn vertical_name) |
| travel_mode | (lệch rất nhỏ, <1%) | — | — | ✅ Có sẵn, gần bão hoà |

### 3.3 · Kết luận mục 3 — ứng viên feature ưu tiên cao nhất

Xếp hạng ở 3.1-3.2 cho ra danh sách ứng viên feature theo độ ưu tiên, dẫn đầu là `pickup_distance_km` (#1 numeric, % lệch cao nhất trong nhóm cột chưa dùng) và `payment_method` (#1 categorical mới). Việc thêm và kiểm chứng các ứng viên này qua train/test thực tế nằm ở Phần 2.

**Đối chiếu với FP (mục đầu)**: không có ứng viên nào trong bảng trên có dấu hiệu "sửa FN thì hại FP" — `cust_cancel_rate_30d`, `total_fee`, `trip_distance_km`, `eta_seconds`, `pickup_distance_km` đều lệch CÙNG HƯỚNG "rủi ro cao hơn" ở cả 2 nhóm lỗi. Vấn đề còn lại với FP không phải do chọn sai feature, mà do **giới hạn cố hữu**: 1 nhóm nhỏ khách "rủi ro cao nhưng vẫn đi" (giá cao/đường dài/lịch sử huỷ cao) không thể tách khỏi nhóm "rủi ro cao và huỷ thật" chỉ bằng feature — cần thêm dữ liệu về NGỮ CẢNH chuyến đi (vd mức độ khẩn cấp, lý do đặt xe) mà dataset hiện tại không có.

2 điểm cần lưu ý khi chọn feature ở Phần 2:
- **`vertical_name` (Bike lệch 10,2 điểm %)** — dù đã là feature, mức lệch còn khá lớn, gợi ý có thể cần phân rã chi tiết hơn ở nhóm Bike cụ thể; nhưng bản chi tiết hơn (`service_name`, ~40 giá trị) có rủi ro overfit vì nhiều giá trị mẫu nhỏ.
- **`dist_airport_km`** lệch khá nhỏ ở bước này (1,6%) — cho thấy % lệch FN/TP là chỉ báo tốt nhưng không phải yếu tố quyết định duy nhất, cần train thử thực tế để xác nhận thay vì chỉ dựa vào % lệch.

## 4. Kết luận chung

- **Khách mới là segment yếu nhất của baseline v1 gốc** (AUC ~0,70, recall chỉ 0,0123) — yếu nhất trong mọi segment xét cả AUC lẫn recall.
- **2 architecture (A/B) cho baseline gốc gần như tương đương** trên mọi segment (chênh lệch trong khoảng ±0,002, trừ sample ETA dài quá nhỏ) — tách `is_post_dispatch` ra làm rule không làm giảm chất lượng model ở phần dữ liệu khó (post-dispatch).
- **Đã xét cả FN (775) lẫn FP (6) trước khi xếp hạng feature (mục 3)**: cả 2 chiều lỗi lệch CÙNG HƯỚNG trên các cột hiện có (`cust_cancel_rate_30d`, `total_fee`, `trip_distance_km`, `eta_seconds`, `pickup_distance_km` — càng "trông rủi ro" thì FN và FP càng dễ xảy ra) — không có candidate feature nào từ FN có dấu hiệu gây hại ngược cho FP.
- **Phương pháp soi FN/FP trên baseline gốc** — soi lỗi (cả FN lẫn FP) → xếp hạng cột theo % lệch → thử nghiệm thực tế (train/test) để xác nhận — cho ra danh sách ứng viên feature cụ thể ở mục 3.1-3.2, sẵn sàng để thử nghiệm và đánh giá ở Phần 2.
