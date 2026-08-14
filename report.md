# Rider Acceptance Prediction
### Báo cáo Tuần 2 (phần 1) — Feature Engineering & Phân tích lỗi theo phân khúc

| | | | |
|---|---|---|---|
| **Dự án** | Rider Acceptance (Matching & Dispatch) | **Giai đoạn** | Tuần 2 |
| **Mục tiêu W2 (phần này)** | Feature engineering + chẩn đoán lỗi theo phân khúc + khả năng tái lập kết quả | **Trạng thái** | Hoàn thành — tìm được 6 feature cải thiện thật (`pickup_distance_km`, `payment_method`, `dist_center_km`, `cust_completion_rate_30d`, `dist_airport_km`, `dist_hotspot_km`), fix được sai lệch tái lập kết quả. **Cập nhật thêm (mục 6)**: đổi kiến trúc `is_post_dispatch` từ feature sang rule if/else — model giờ chỉ train/infer trên post-dispatch (19 feature); số liệu mục 1-5 vẫn giữ nguyên làm lịch sử. **Cập nhật thêm (10/08/2026, mục 2 & 6)**: bổ sung Cancel PR-AUC (`pr_auc_cancel`) — tiêu chí quyết định chính từ W3 trở đi — cho 3 thử nghiệm interaction từng bị revert và cho bảng so sánh 5 model; xem ghi chú "Cập nhật Cancel PR-AUC" ở mỗi mục. |

---

## 1. Mục tiêu

Tiếp nối baseline đóng băng ở tuần 1 (ROC-AUC post-dispatch = 0,7267, Brier = 0,0716 — mốc so sánh chuẩn), tuần 2 thực hiện 2 việc theo đúng kế hoạch đã đề ra trong báo cáo tuần 1:

1. **Feature engineering** — mở rộng mô tả hành vi khách hàng (đa cửa sổ lịch sử, recency), vì báo cáo tuần 1 đã chỉ ra đây là hướng tín hiệu có ý nghĩa nhất sau biến phân tầng `is_post_dispatch`.
2. **Chẩn đoán lỗi theo phân khúc** — khách mới, giờ cao điểm, ETA dài.

Bối cảnh: trước khi làm 2 việc trên, dữ liệu training đã được **mở rộng từ 8 ngày lên 30 ngày** (06→13/07 → 14/06→13/07), giúp baseline tự nhiên nhích lên: ROC-AUC post-dispatch 0,7267 → **0,7342**, ROC-AUC tổng 0,8585 → **0,8624** (270.763 dòng train, gấp ~4 lần).

## 2. Feature engineering — 28 thử nghiệm, 6 thành công

Tất cả đều dùng LightGBM (feature gốc + feature thử nghiệm, cộng dồn qua từng thử nghiệm thành công), cùng split train/test theo thời gian như tuần 1.

| Thử nghiệm | Feature thêm | ROC-AUC (tổng) | ROC-AUC (post-dispatch) | Kết luận |
|---|---|---|---|---|
| Gốc (mốc so sánh) | — (chỉ `cust_cancel_rate_30d`) | 0,8624 | 0,7342 | — |
| 1. Đa cửa sổ | `cust_cancel_rate_7d`, `cust_orders_7d`, `cust_cancel_rate_14d`, `cust_orders_14d` | 0,8618 | 0,7330 | Giảm nhẹ |
| 2. Recency | `days_since_last_order`, `days_since_last_cancel` | 0,8611 | 0,7317 | Giảm nhẹ, gain rất thấp (720 và 1.388) |
| 3. Cờ khách mới | `is_new_customer` | 0,8625 | 0,7344 | **Gain = 0** — LightGBM không dùng |
| 4. **`pickup_distance_km`** | Quãng đường tài xế tới điểm đón (đã có sẵn trong `orders.parquet`, chưa từng đưa vào model) | **0,8671** | **0,7433** | ✅ **Cải thiện thật, +0,0091** |
| 5. `service_name` | Tên dịch vụ chi tiết (~40 giá trị, chi tiết hơn `vertical_name`) | 0,8661 | 0,7413 | Giảm nhẹ dù gain cao (29.161) |
| 6. **`payment_method`** | Phương thức thanh toán (16 giá trị: cash/momo/zalo/thẻ...) | **0,8689** | **0,7468** | ✅ **Cải thiện thật, +0,0035** so với mốc sau bước 4 |
| 7. `is_deep_night` | Cờ giờ 0h-4h (phát hiện qua phân tích lỗi, xem mục 3) | 0,8684 | 0,7458 | Giảm nhẹ, gain gần 0 (36) |
| 8. **`dist_center_km`** | Khoảng cách điểm đón tới trung tâm (Hồ Hoàn Kiếm), dẫn xuất từ lat/lon | **0,8699** | **0,7487** | ✅ **Cải thiện thật, +0,0019** so với mốc sau bước 6 |
| 9. `pickup_speed_kmh` | Tốc độ tài xế ước tính = `pickup_distance_km` / `eta_seconds` | 0,8699 | 0,7487 | **Không đổi** dù gain vừa phải (5.250) |
| 10. `pickup_quadrant` | Hướng điểm đón so với trung tâm (NE/NW/SE/SW), dẫn xuất từ dấu lat/lon | 0,8693 | 0,7474 | Giảm nhẹ, gain thấp nhất hệ thống (536) |
| 11. **`cust_completion_rate_30d`** | Tỷ lệ hoàn thành quá khứ của khách = `n_completed_30d`/`n_orders_30d` | **0,8703** | **0,7494** | ✅ **Cải thiện thật, +0,0004** so với mốc sau bước 8 |
| 12. `area_cancel_rate_7d` | Tỷ lệ huỷ quá khứ 7 ngày của khu vực điểm đón (ô lưới ~1km từ lat/lon) | 0,8693 | 0,7475 | Giảm dù gain khá (6.800) — overfit trên ô lưới ít đơn |
| 13. `cust_tenure_days` | Số ngày từ lần đầu xuất hiện của khách trong `customer_daily` | 0,8697 | 0,7483 | Giảm nhẹ, tín hiệu quá yếu (corr EDA 0,031) |
| 14. **`dist_airport_km`** | Khoảng cách điểm đón tới sân bay Nội Bài, dẫn xuất từ lat/lon | **0,8708** | **0,7503** | ✅ **Cải thiện thật, +0,0009** so với mốc sau bước 11 |
| 15. `area_cancel_rate_14d` (thử lại) | Như thử nghiệm 12 nhưng ô lưới thô hơn (~5,5km) + cửa sổ 14 ngày | 0,8696 | 0,7481 | Vẫn giảm — giờ trùng lặp `dist_center_km` (corr 0,609) |
| 16. `fee_percentile_in_vertical` | Percentile `total_fee` so với cùng `vertical_name` (ranh giới fit trên train, áp forward test) | 0,8704 | 0,7497 | Giảm dù gain rất cao (10.023) — không generalize |
| 17. `is_eta_missing` | Cờ eta_seconds bị thiếu (AR 0,928 khi thiếu vs 0,902 khi có) | 0,8708 | 0,7504 | **Gain = 0,000 tuyệt đối** — LightGBM tự xử lý NaN |
| 18. `re_dispatch_counts` dạng categorical | Coi là mã bucket/tier (AR không đơn điệu theo giá trị) thay vì numeric | 0,8703 | 0,7495 | Giảm, gain cũng giảm — numeric đã tối ưu hơn |
| 19. **`dist_hotspot_km`** | Khoảng cách tới tâm cụm điểm đón đông đơn gần nhất (K-means K=15, fit trên train) | **0,8709** | **0,7505** | ✅ **Cải thiện thật, +0,0002** so với mốc sau bước 14 |
| 20. `cancelrate_x_pickupdist` | Tương tác nhân `cust_cancel_rate_30d` × `pickup_distance_km` | 0,8700 | 0,7489 | Giảm dù gain rất cao (33.170) — nhưng PR-AUC huỷ TĂNG (0,2675→0,2702) so mốc tại thời điểm đó¹ |
| 21. `cancelrate_x_distcenter` | Tương tác nhân `cust_cancel_rate_30d` × `dist_center_km` | 0,8696 | 0,7481 | Giảm cả ROC-AUC lẫn PR-AUC huỷ — thất bại rõ ràng |
| 22. `surge_x_pickupdist` | Tương tác nhân `surge_multiplier` × `pickup_distance_km` | 0,8701 | 0,7490 | Giảm dù gain rất cao (49.606, hạng 2 hệ thống) |
| 23. `hotspot_x_redispatch` | Tương tác nhân `dist_hotspot_km` × `re_dispatch_counts` | 0,8702 | 0,7493 | Giảm, PR-AUC huỷ nhích nhẹ (0,2675→0,2682) so mốc tại thời điểm đó¹ |
| 24. `center_hotspot_redispatch` | Tương tác nhân 3 chiều `dist_center_km` × `dist_hotspot_km` × `re_dispatch_counts` | 0,8702 | 0,7493 | Giảm dù cải thiện tương quan EDA cao nhất từng thấy (+0,045) |
| 25. `geo4_redispatch` | Tương tác nhân 4 chiều (`dist_center_km` × `dist_airport_km` × `dist_hotspot_km` × `re_dispatch_counts`) | 0,8708 | 0,7504 | Gần như không đổi (chênh 0,0001) — vô hại nhưng vô dụng |
| 26. `geo5_redispatch` | Tương tác nhân 5 chiều (`trip_distance_km` × `fee_per_km` × `dist_center_km` × `dist_hotspot_km` × `re_dispatch_counts`) | 0,8708 | 0,7503 | Giảm rất nhẹ, PR-AUC huỷ TĂNG (0,2675→0,2702) so mốc tại thời điểm đó¹ |

> ¹ **Cập nhật Cancel PR-AUC (10/08/2026)** — 3 thử nghiệm 20/23/26 là 3 trường hợp DUY NHẤT trong bảng có PR-AUC huỷ tăng dù ROC-AUC giảm lúc đó, nên bị nghi ngờ là revert oan khi W3 đổi tiêu chí quyết định chính sang Cancel PR-AUC. Đã **test lại cả 3** trên baseline production **hiện tại** (20 feature, đã có `is_rainy_3h`, ROC-AUC=0,7519 · Cancel PR-AUC=0,2710 — baseline này mạnh hơn baseline tại thời điểm thử nghiệm gốc), dùng đúng `evaluate()`/`pr_auc_cancel` (không đảo dấu như phép so sánh gốc):
>
> | Thêm feature | ROC-AUC | Cancel PR-AUC | So baseline hiện tại |
> |---|---|---|---|
> | 20. `cancelrate_x_pickupdist` | 0,7501 | 0,2705 | Giảm cả 2 |
> | 23. `hotspot_x_redispatch` | 0,7499 | 0,2673 | Giảm cả 2 |
> | 26. `geo5_redispatch` | 0,7506 | 0,2668 | Giảm cả 2 |
>
> **Kết luận không đổi — cả 3 vẫn nên revert**, kể cả dưới tiêu chí Cancel PR-AUC. "PR-AUC huỷ TĂNG" ghi nhận lúc đó chỉ đúng so với baseline yếu hơn tại thời điểm đó (0,2675); so với baseline hiện tại (đã mạnh hơn nhờ thêm `is_rainy_3h` sau này) cả 3 đều kém hơn trên cả 2 trục. Quyết định revert ban đầu — dù dựa trên tiêu chí ROC-AUC lúc đó — hoá ra vẫn đúng, không phải may mắn nhờ chọn sai metric.
| 27. Batch 4 feature | Gộp cùng lúc: `cust_tenure_days` + `pickup_quadrant` + `area_cancel_rate_7d` + `fee_percentile_in_vertical` | 0,8704 | 0,7495 | Giảm, không vượt baseline — nhưng TỐT HƠN 3/4 khi thử alone |
| 28. Batch 8 toán hạng | Gộp cùng lúc 8 toán hạng tương tác nhân đa dạng (2-4 chiều, chưa thử riêng lẻ) | 0,8704 | 0,7496 | Giảm — không vượt baseline dù 1 toán hạng đạt gain hạng 5 hệ thống |

**Diễn giải:**
- Thử nghiệm 1–2: AUC giảm dần đều qua từng lần thêm feature — không phải nhiễu ngẫu nhiên đơn lẻ mà là xu hướng nhất quán. Nguyên nhân nhiều khả năng: với chỉ 30 ngày dữ liệu đơn hàng, phần lớn khách chưa tích luỹ đủ lịch sử để các cửa sổ ngắn (7d/14d) hay recency tách biệt rõ so với cửa sổ 30 ngày sẵn có — thêm feature chỉ làm model phức tạp hơn mà không thêm thông tin thực chất.
- Thử nghiệm 3: `is_new_customer` (`cust_orders_30d` NaN/0) hoàn toàn **dư thừa** với cây quyết định — LightGBM đã tự suy ra được điều này qua giá trị NaN/0 của `cust_orders_30d`/`cust_cancel_rate_30d` sẵn có, nên thêm 1 cờ tường minh không cho model biết thêm gì mới (gain = 0 xác nhận điều này).
- **Thử nghiệm 4 thành công** vì đây là **loại thông tin hoàn toàn mới** (không phải biến tấu từ `customer_daily` như 3 thử nghiệm trước) — cột đã có sẵn trong `orders.parquet` nhưng chưa từng được đưa vào `FEATURES`. Gain = 91.388, đứng **thứ 2 toàn hệ thống** (chỉ sau `is_post_dispatch`), cao hơn cả `eta_seconds` (42.534).
- Thử nghiệm 5: `service_name` là bản **chi tiết hơn của `vertical_name`** (cùng phân loại dịch vụ, chỉ khác độ mịn) — nhiều giá trị có mẫu quá nhỏ (<50 dòng), dễ overfit trên train và trùng lặp thông tin với `vertical_name` khiến cây phân tán quyết định thay vì tập trung. Gain cao (29.161) nhưng không chuyển thành AUC test tốt hơn — bằng chứng cho thấy gain trên train không đảm bảo generalize.
- **Thử nghiệm 6 thành công** vì cùng lý do với thử nghiệm 4: **nguồn thông tin hoàn toàn mới**, chưa từng khai thác. `payment_method` phản ánh mức độ "chắc đi" của khách — cash có AR thấp nhất (0,799) trong khi thanh toán qua thẻ/đối tác (`partner_pay` 0,981, `business_card` 0,883, `international_card` 0,859) cao hơn hẳn; ví điện tử (momo/zalo/shopee_pay, AR 0,81–0,83) ở mức trung gian — không trùng với các feature khách hàng/giá cước/vận hành hiện có. Gain = 8.103 (hạng 12/16) — thấp hơn nhiều so với `pickup_distance_km` nhưng vẫn đóng góp thật.
- Thử nghiệm 7-8 xuất phát từ **phân tích lỗi trực tiếp** (mổ xẻ các đơn model đoán sai) thay vì thử ngẫu nhiên — xem mục 3 để biết cách phát hiện 2 ứng viên này. `is_deep_night` thất bại vì `hour_of_day` (dạng số liên tục) đã đủ để LightGBM tự tách khung giờ đêm qua các nhát cắt cây — cờ tường minh chỉ trùng lặp, không thêm thông tin (gain ≈ 0, giống hệt bài học ở thử nghiệm 3).
- **Thử nghiệm 8 thành công**: `dist_center_km` phát hiện được nhờ soi kỹ các đơn huỷ mà model đoán sai — nhóm đơn có điểm đón cách trung tâm >20km có AR thấp hẳn (0,82 và 0,71) so với ~0,90-0,91 ở gần trung tâm. Tương quan với `trip_distance_km` chỉ 0,217 và với `pickup_distance_km` chỉ 0,047 — xác nhận đây là tín hiệu **địa lý mới**, không trùng lặp với 2 feature khoảng cách đã có. Gain = 10.207 (hạng 10/17).
- Thử nghiệm 9: `pickup_speed_kmh` là **tổ hợp thuần tuý** của 2 feature đã có sẵn (`eta_seconds`, `pickup_distance_km`), không phải nguồn dữ liệu mới. AUC **giữ nguyên tuyệt đối** (0,8699/0,7487, xác nhận qua 2 lần chạy độc lập) dù gain vừa phải (5.250) — cây tự tổ hợp được thông tin tương đương qua các nhát cắt tuần tự trên 2 cột gốc, chứng minh thêm 1 phép chia không tạo ra tín hiệu mới nếu cả 2 số hạng đều đã là feature.
- Thử nghiệm 10: `pickup_quadrant` là hướng (NE/NW/SE/SW) — EDA cho thấy vẫn có chênh lệch AR ~5 điểm % ngay cả trong nhóm gần trung tâm (<5km), tức không hoàn toàn trùng `dist_center_km`. Nhưng AUC vẫn **giảm nhẹ** (xác nhận qua 2 lần chạy), gain thấp nhất toàn hệ thống (536). Có thể do chỉ chia 4 giá trị theo dấu toán học (không theo phân bố thực tế dữ liệu), tín hiệu quá yếu và quá thô để bù lại chi phí thêm 1 categorical feature cho cây quyết định.
- **Thử nghiệm 11 thành công**: phát hiện qua thấy `n_orders_30d` của khách **KHÔNG LUÔN BẰNG** `n_completed_30d` + `n_cust_cancel_30d` (~7,3% dòng `customer_daily` có chênh lệch) — tức có 1 loại kết quả thứ 3 (có thể tài xế huỷ/không tìm được tài xế) chưa từng khai thác riêng. `cust_completion_rate_30d` không trùng lặp mạnh với `cust_cancel_rate_30d` vì phần dư mang thông tin riêng. Gain = 10.158 (hạng 11/18) — gần bằng `dist_center_km`.
- Thử nghiệm 12: `area_cancel_rate_7d` áp dụng đúng logic point-in-time rolling-window của `cust_cancel_rate_30d` (mục 2) nhưng theo **khu vực điểm đón** (ô lưới ~1km từ lat/lon) thay vì theo khách — ý tưởng hợp lý (mật độ/rủi ro theo khu vực), tương quan EDA thật (-0,077) và không trùng lặp mạnh với feature hiện có. Nhưng AUC vẫn **giảm** dù gain khá (6.800) — đa số ô lưới có rất ít đơn/7 ngày (trung vị ~6 đơn/ô) nên tỷ lệ huỷ theo cửa sổ ngắn dễ nhiễu/overfit trên train, cùng bài học với `service_name`: gain cao trên train không đảm bảo generalize khi mẫu quá thưa.
- Thử nghiệm 13: `cust_tenure_days` đo độ "thâm niên" (khác khái niệm với `cust_orders_30d` là đếm SỐ ĐƠN 30 ngày) — tương quan thấp với `cust_orders_30d` (0,15) nên không hẳn dư thừa, nhưng bản thân tín hiệu quá yếu (EDA corr 0,031) nên AUC vẫn giảm nhẹ dù gain nhỏ (1.407).
- **Thử nghiệm 14 thành công**: `dist_airport_km` — landmark khác với `dist_center_km` (sân bay Nội Bài thay vì Hồ Hoàn Kiếm), gợi ý từ `service_name` có dịch vụ riêng "xanh_airport_limo". Tương quan vừa phải với `dist_center_km` (0,283) — đơn gần sân bay có hành vi khác đơn gần trung tâm thành phố (giờ giấc/áp lực chuyến bay). Gain = 5.869 (hạng 14/19).
- Thử nghiệm 15: thử lại `area_cancel_rate_7d` (thất bại) với ô lưới thô hơn (~5,5km thay vì ~1km) + cửa sổ dài hơn (14 ngày) để sửa đúng vấn đề overfit lần trước (trung vị mẫu/ô tăng từ ~6 lên ~67 đơn). Nhưng vẫn **giảm** — lý do khác lần này: ô lưới thô hơn tương quan quá cao với `dist_center_km` (0,609 so với 0,288 ở ô nhỏ), chỉ còn là bản mã hoá lại "khoảng cách tới trung tâm", không phải tín hiệu mới. 2 lần thất bại liên tiếp → dừng hướng ô lưới địa lý.
- Thử nghiệm 16: `fee_percentile_in_vertical` — để tránh rò rỉ, ranh giới percentile được fit CHỈ trên dữ liệu train rồi áp forward cho test (không tính percentile trên toàn bộ train+test gộp, tránh phân bố giá tương lai lọt vào feature). Gain rất cao (10.023, chỉ sau `dist_center_km` trong nhóm feature thử nghiệm) nhưng AUC vẫn giảm — percentile rank có thể nhạy với đúng phân bố giá của riêng ngày test (chỉ 1 ngày), không ổn định giữa các ngày.
- Thử nghiệm 17: `is_eta_missing` — AR khi `eta_seconds` thiếu (0,928) khác thật so với khi có (0,902), nhưng gain = **0,000 tuyệt đối**, AUC không đổi 1 chút nào. LightGBM đã tự học hướng đi tối ưu cho giá trị NaN ngay khi split trên chính `eta_seconds` — bài học giống `is_new_customer`/`is_deep_night` nhưng dứt khoát hơn (gain=0 chính xác, không chỉ "gần 0").
- Thử nghiệm 18: thử coi `re_dispatch_counts` là **categorical** thay vì numeric — giả thuyết đây là mã bucket/tier rời rạc (chỉ 6 giá trị: 0/2/3/5/10/20) vì AR không đơn điệu theo giá trị (0,803→0,868→0,874→0,753→0,777→1,000). Nhưng AUC vẫn giảm, và gain cũng giảm theo (27.571→12.479 khi đổi sang categorical) — xử lý numeric hoá ra vẫn tách các giá trị hiệu quả hơn (categorical encoding kém regularize hơn khi chỉ có 6 giá trị).
- **Thử nghiệm 19 thành công**: `dist_hotspot_km` dùng K-means (K=15, fit chỉ trên train) để tìm nhiều "điểm nóng" thực tế thay vì neo cố định vào 1 điểm (trung tâm/sân bay). Tương quan vừa phải với `dist_center_km` (0,487) và `dist_airport_km` (0,211) — không trùng lặp hoàn toàn, và các cụm đều đủ mẫu (nhỏ nhất ~2.100 đơn), tránh được vấn đề overfit ô lưới cố định ở thử nghiệm 12/15. Cải thiện nhỏ nhưng thật và ổn định (gain 4.160, hạng 15/20).
- Thử nghiệm 20-26 thử **tương tác nhân** (multiply 2, 3, 4, 5 feature với nhau) — quét hệ thống mọi tổ hợp 2/3/4/5 phần tử trong 13 feature số, mỗi vòng chọn theo tiêu chí "tương quan EDA của tích số MẠNH HƠN thành phần riêng mạnh nhất". Kết quả:
  - **2-way**: `cancelrate_x_pickupdist` (corr -0,163), `cancelrate_x_distcenter` (-0,145), `surge_x_pickupdist` (cải thiện +0,021), `hotspot_x_redispatch` (+0,016).
  - **3-way**: `center_hotspot_redispatch` (+0,045 — điểm cải thiện EDA cao nhất tìm được trong toàn bộ quá trình).
  - **4-way**: `geo4_redispatch` (+0,027) — thấp hơn 3-way.
  - **5-way**: `geo5_redispatch` (+0,017) — thấp hơn cả 4-way.

  Tất cả 7 đều có gain rất cao trên train (33.170, 6.122, 49.606 — hạng 2 toàn hệ thống, 6.489, 7.783, 7.137, 15.656) nhưng **ROC-AUC test-post không cải thiện ở bất kỳ trường hợp nào** — kể cả bộ ba với tương quan EDA ấn tượng nhất. 4/7 thử nghiệm (20, 23, 24, 26) cho PR-AUC lớp huỷ (`pr_auc_cancel` — metric mới bổ sung vào `evaluation.py`, đánh giá đúng khả năng xếp hạng nhóm huỷ, khác `pr_auc` mặc định vốn tính trên lớp accept đa số nên luôn cao ảo ~0,96) **nhích lên hoặc giữ nguyên** dù ROC-AUC giảm — gợi ý hướng tương tác nhân có thể đáng cân nhắc lại nếu ưu tiên nghiệp vụ chuyển hẳn sang recall/PR-AUC nhóm huỷ thay vì ROC-AUC tổng thể, nhưng theo tiêu chí nhất quán xuyên suốt báo cáo (ROC-AUC post-dispatch), cả 7 coi là thất bại và đã revert. **Kết luận sau 7 thử nghiệm (2-way tới 5-way, luôn chọn ứng viên có tương quan EDA tốt nhất tìm được qua quét toàn bộ tổ hợp)**: điểm cải thiện EDA tốt nhất giảm dần khi tăng số chiều (3-way 0,045 > 2-way 0,021 > 4-way 0,027 > 5-way 0,017) — càng nhiều feature nhân với nhau, tín hiệu càng loãng thay vì cộng dồn. Hướng tương tác nhân (2 đến 5 chiều) coi như đã cạn hoàn toàn với LightGBM trên bộ dữ liệu này — cây vốn tự tổ hợp được các feature qua nhiều nhát cắt tuần tự, nên tích số tường minh chỉ tạo thêm 1 cột dễ overfit (gain ảo rất cao trên train) mà không mang lại xếp hạng tốt hơn trên test, bất kể chọn tổ hợp nào hay bao nhiêu chiều. Không khuyến nghị thử 6-way trở lên.
- Thử nghiệm 27 đổi chiến lược: thay vì thêm từng feature một, **gộp cùng lúc 4 feature từng thất bại riêng lẻ** (`cust_tenure_days`, `pickup_quadrant`, `area_cancel_rate_7d`, `fee_percentile_in_vertical`) — chọn có chủ đích, loại trừ các feature đã bị chứng minh dư thừa tuyệt đối (`is_new_customer`/`is_deep_night`/`is_eta_missing`, gain=0 chính xác khi thử alone — gộp các feature "chắc chắn vô dụng" thì không thể sinh ra tín hiệu). Kết quả: AUC vẫn **giảm** (0,8709→0,8704 tổng, 0,7505→0,7495 post, xác nhận 2 lần chạy), không vượt qua baseline sạch. Nhưng có 1 phát hiện thú vị: kết quả gộp (0,7495) lại **tốt hơn 3/4** kết quả khi thử từng cái một (`cust_tenure_days` một mình 0,7483; `pickup_quadrant` một mình 0,7474; `area_cancel_rate_7d` một mình 0,7475) — chỉ thua `fee_percentile_in_vertical` một mình (0,7497). Tức các feature yếu này "trung hoà" bớt nhiễu của nhau khi gộp, nhưng tổng độ phức tạp thêm vào vẫn không đủ bù lại — không có hiệu ứng cộng dồn tích cực. Đã revert cả 4.
- **Thử nghiệm 28**: áp dụng cùng chiến lược "gộp nhiều toán hạng cùng lúc" cho hướng tương tác nhân — thay vì test từng công thức tích số một, thêm **8 toán hạng đa dạng cùng 1 lần** (mỗi toán hạng là 1 feature gốc hoặc tích của 2-4 feature gốc, chọn từ các tổ hợp chưa từng thử riêng lẻ trước đó: `surge_x_cancelrate`, `center_x_redispatch`, `surge_x_eta`, `airport_hotspot_redispatch`, `surge_redispatch_cancelrate`, `fee_center_redispatch`, `center_surge_redispatch_cancelrate`, `surge_pickup_cancelrate_completerate` — tổng 28 feature). Khi đứng cùng nhau, một số toán hạng đạt gain RẤT CAO (`surge_pickup_cancelrate_completerate` = 19.149, hạng 5 toàn hệ thống; `surge_x_cancelrate` = 14.215; `surge_x_eta` = 10.845) — cao hơn hẳn so với lúc test đơn lẻ, cho thấy khi có nhiều toán hạng cạnh tranh, cây "chọn lọc" ra toán hạng phù hợp nhất trong nhóm để dùng nhiều hơn. Nhưng ROC-AUC test-post vẫn **giảm** (0,8709→0,8704 tổng, 0,7505→0,7496 post, xác nhận 2 lần chạy) — không vượt qua baseline. Đã revert cả 8.

**Kết luận sau 9 thử nghiệm tương tác nhân (7 đơn lẻ 2-way đến 5-way + 2 batch gộp 4 và 8 toán hạng)**: dù test từng công thức một hay gộp nhiều công thức cùng lúc, KHÔNG có cấu hình nào vượt qua baseline 20-feature hiện tại. Hướng tương tác nhân (multiply, bất kể số chiều hay số lượng toán hạng gộp cùng lúc) coi như đã cạn hoàn toàn với LightGBM + dữ liệu hiện có. Để đạt mục tiêu 12+ feature mới, hướng khả thi hơn là quay lại tìm **nguồn dữ liệu mới thực sự** (chưa khai thác trong `orders.parquet`/`customer_daily`) thay vì tiếp tục biến đổi toán học trên các feature đã có.

→ **Thử nghiệm 1, 2, 3, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28 đã revert. Thử nghiệm 4 (`pickup_distance_km`), 6 (`payment_method`), 8 (`dist_center_km`), 11 (`cust_completion_rate_30d`), 14 (`dist_airport_km`), 19 (`dist_hotspot_km`) được giữ lại** — baseline production hiện tại có **20 feature** (14 gốc + 6 feature thành công), đã train lại và cập nhật cho cả 5 model (LightGBM, MLP, Hybrid, Ensemble, Stacking) trong MLflow. Bài học chung: feature engineering thành công khi đổi hẳn sang **nguồn dữ liệu/tín hiệu mới thực sự** (cột có sẵn nhưng bị bỏ sót, dẫn xuất mới từ cột/landmark/cụm chưa khai thác, hoặc phần dư dữ liệu bị bỏ sót); thất bại khi chỉ **biến tấu/làm mịn hơn** một nguồn đã bão hoà, **trùng lặp với thông tin cây đã tự suy ra được** (kể cả NaN — LightGBM xử lý nội bộ), **tính trên mẫu/phân bố quá thưa/nhạy cảm 1 ngày** dễ overfit, hoặc **tương tác nhân giữa các feature đã có, dù 2 chiều hay 3 chiều** (cây vốn tự tổ hợp được qua nhiều nhát cắt, tương tác tường minh dễ overfit train dù tương quan EDA cao).

### Kết quả cuối cùng — cả 5 model (20 feature)

| Model | ROC-AUC (tổng) | ROC-AUC (post-dispatch) |
|---|---|---|
| MLP v2 | 0,8666 | 0,7423 |
| LightGBM baseline | 0,8709 | 0,7505 |
| **Hybrid GBDT+NN** | **0,8710** | **0,7508** |
| Ensemble (avg) | 0,8700 | 0,7489 |
| Stacking | 0,8702 | 0,7493 |

Hybrid GBDT+NN dẫn đầu (rất sát LightGBM solo). ROC-AUC post-dispatch **0,7508** — tăng **+0,0241** so với mốc đóng băng tuần 1 (0,7267).

## 3. Phân tích lỗi theo phân khúc + soi trực tiếp các đơn đoán sai (trên tập test post-dispatch, n=8.609)

*Số liệu mục này chạy trên baseline LightGBM production mới nhất — 20 feature (sau khi thêm `dist_hotspot_km`, thử nghiệm cuối cùng thành công).*

| Phân khúc | n | AR nền | ROC-AUC | Brier | ECE | Recall (lớp huỷ) |
|---|---|---|---|---|---|---|
| ETA ngắn (≤600s) | 8.264 | 0,9078 | **0,7445** | 0,0766 | 0,0065 | 0,0223 |
| Giờ thường | 5.337 | 0,9110 | **0,7447** | 0,0745 | 0,0091 | 0,0274 |
| Khách mới | 745 | 0,8913 | 0,7482 | 0,0879 | 0,0211 | 0,0123 |
| **Toàn bộ post-dispatch** | 8.609 | 0,9080 | 0,7505 | 0,0760 | 0,0062 | 0,0290 |
| Khách quen | 7.864 | 0,9096 | 0,7507 | 0,0748 | 0,0051 | 0,0309 |
| Giờ cao điểm | 3.272 | 0,9031 | 0,7590 | 0,0783 | 0,0058 | 0,0315 |
| ETA dài (>600s)* | 45 | 0,8667 | 0,8120 | 0,1065 | 0,1789 | 0,1667 |

*Mẫu ETA dài quá nhỏ (45 dòng) — không đủ tin cậy thống kê, chỉ mang tính tham khảo.

### Phát hiện chính: điểm yếu đã dịch chuyển 2 lần — hiện tại là "ETA ngắn" và "giờ thường"

Hành trình phân khúc yếu nhất qua các vòng feature engineering: 14-feature ban đầu → **khách mới** (AUC 0,7046, yếu nhất hệ thống lúc đó) → 17-19 feature (sau `pickup_distance_km`+`payment_method`) → **giờ thường** (0,7427) → 20-feature hiện tại (sau `dist_center_km`, `dist_airport_km`, `dist_hotspot_km`, `cust_completion_rate_30d`) → **ETA ngắn** (0,7445) và **giờ thường** (0,7447) gần như đồng hạng yếu nhất, "khách mới" đã cải thiện lên gần giữa bảng (0,7482).

Soi trực tiếp các đơn đoán sai trong 2 phân khúc "ETA ngắn"/"giờ thường" (so sánh median các feature hiện có giữa nhóm False Negative và True Positive) cho thấy `eta_seconds` (247 vs 197), `cust_cancel_rate_30d` (0,167 vs 0,094), `pickup_distance_km` (1,162 vs 0,759) đều **lệch đúng hướng rủi ro** ở nhóm đoán sai — tức các feature hiện có đã "biết" đúng hướng, chỉ là tín hiệu tổng hợp chưa đủ mạnh để vượt ngưỡng 0,5 trên nền tỉ lệ chấp nhận cao (~90%). Đây là dấu hiệu cho thấy vấn đề đã chuyển từ "thiếu feature" sang "giới hạn ngưỡng quyết định/model capacity" — không còn là việc feature engineering có thể giải quyết tiếp, xem mục 5.

### Soi trực tiếp các đơn model đoán sai — cách tìm ra `dist_center_km` (thử nghiệm 8, mục 2)

Thay vì tiếp tục thử feature theo trực giác, đã tách rõ 2 loại lỗi ở threshold 0,5 trên tập test post-dispatch (model 16-feature, trước khi thêm `dist_center_km`):

- **False Negative (huỷ nhưng model đoán sẽ đi)**: 768/792 đơn huỷ thực tế (97,0%) — model bắt sót gần như toàn bộ đơn huỷ ở ngưỡng mặc định, và **sai một cách tự tin**: xác suất dự đoán trung bình trên nhóm này là 0,84 (median 0,87).
- **True Negative (huỷ, bắt đúng)**: chỉ 24 đơn, xác suất dự đoán trung bình 0,27 — model phân biệt được rất rõ khi đã bắt đúng, chỉ là số lượng bắt được quá ít.

So sánh các cột **chưa từng đưa vào feature** giữa nhóm False Negative và nhóm True Positive (đơn đi, đoán đúng) để tìm tín hiệu còn sót:

- `pickup_province_name`: 100% là "Thành phố Hà Nội" ở cả 2 nhóm — **không có phương sai, không dùng được làm feature**.
- `service_name`: có lệch nhẹ (`xanhsm_bike_hn` chiếm 44,4% trong nhóm đoán sai vs 35,97% ở nhóm đoán đúng) nhưng đã biết từ thử nghiệm 5 rằng đưa cả `service_name` vào model làm AUC giảm — tín hiệu này không tách ra dùng riêng được mà không kéo theo nhiễu của ~40 giá trị.
- `travel_mode`, `total_fee`, `fee_per_km`, `surge_multiplier`, `re_dispatch_counts`: phân bố gần như giống hệt nhau giữa 2 nhóm — không có tín hiệu mới.
- **`hour_of_day` (giờ 0h-4h)**: nhóm đoán sai có tỉ lệ ở khung giờ đêm cao hơn nhóm đoán đúng (VD giờ 0h: 3,78% vs 0,90%) — kiểm tra AR thực tế theo giờ xác nhận khung 0h-4h có AR thấp hẳn (0,806 vs 0,912 ngoài khung này, n=310). Đây là gợi ý cho thử nghiệm 7 (`is_deep_night`) — nhưng **thất bại** vì `hour_of_day` (dạng số) đã đủ để cây tự tách khung giờ này qua các nhát cắt, xem mục 2.
- **`pickup_latitude`/`pickup_longitude`**: đây là 2 cột duy nhất còn hoàn toàn chưa khai thác. Tính khoảng cách từ điểm đón tới trung tâm Hà Nội (Hồ Hoàn Kiếm) cho thấy nhóm đơn xa trung tâm >20km (n≈9.360/254.591, ~3,7%) có AR thấp hẳn: 0,821 (20-50km) và 0,713 (>50km) so với ~0,90-0,91 ở gần trung tâm — và **không trùng lặp** với `trip_distance_km` (tương quan 0,217) hay `pickup_distance_km` (tương quan 0,047). Đây chính là thử nghiệm 8 (`dist_center_km`) — **thành công**.

**Bài học phương pháp**: soi trực tiếp các đơn đoán sai + đối chiếu với cột dữ liệu thô **chưa từng dùng** hiệu quả hơn thử ngẫu nhiên — cả 2 ứng viên rút ra từ bước này (`is_deep_night`, `dist_center_km`) đều có cơ sở rõ ràng từ dữ liệu thật, và 1/2 thành công (tỉ lệ cao hơn nhiều so với các thử nghiệm biến tấu ở mục 2.1-2.2 trước đó).

### Vì sao feature engineering không sửa được vấn đề khách mới (đến hết mục 2, trước bước 8)

Đã thử thêm `is_new_customer` (mục 2, thử nghiệm 3) như 1 hướng sửa trực tiếp — kết quả: **AUC và Recall của phân khúc khách mới không đổi** (0,7046 → 0,7015, trong khoảng nhiễu; Recall vẫn 0,0000). Đây là bằng chứng thực nghiệm khẳng định: **model không phải "chưa biết" đây là khách mới** — nó đã biết rất rõ qua giá trị NaN sẵn có. Vấn đề không phải model chưa biết đây là khách mới, mà là **cust_cancel_rate_30d bị thiếu hoàn toàn** với nhóm này. Điều bất ngờ: vấn đề cuối cùng lại được giải qua đường vòng — 2 feature ở mức đơn hàng (`pickup_distance_km`, `payment_method`, không liên quan gì đến lịch sử khách) tình cờ mang đủ tín hiệu để bù đắp phần nào chỗ trống đó (mục "Phát hiện chính" ở trên). Đây không phải giải pháp triệt để cho cold-start, nhưng là minh chứng rằng feature engineering ở phạm vi rộng hơn đôi khi giải quyết được vấn đề tưởng chừng chỉ cold-start mới sửa được.

## 4. Khả năng tái lập kết quả (Reproducibility) — 2 sai lệch phát hiện khi đối chiếu với mentor/teammate

Khi so kết quả với mentor và teammate, phát hiện 2 mức sai lệch số liệu khác nhau, đã điều tra và xác định nguyên nhân cho cả 2:

### 4.1 · Sai lệch lớn: 0,7342 (mình) vs 0,7267 (mentor) — khác BỘ DỮ LIỆU

Không phải lỗi — 2 bên đang đo trên 2 input khác nhau:

| Cấu hình | ROC-AUC (post-dispatch) |
|---|---|
| Data 8 ngày (06/07→13/07) + 14 feature gốc | **0,7267** ← khớp chính xác mốc đóng băng tuần 1 |
| Data 30 ngày (14/06→13/07) + 14 feature gốc | **0,7342** ← số của mình sau khi mở rộng data ở đầu tuần 2 |

Đã tái lập chính xác cả 2 con số bằng cách chạy lại đúng cấu hình tương ứng trên data của teammate (đã kiểm tra data 2 bên giống hệt nhau về shape/khoảng ngày). Kết luận: mentor/teammate khi báo cáo 0,7267 vẫn đang dùng bộ data 8 ngày gốc (frozen benchmark tuần 1), chưa cập nhật sang bộ 30 ngày. **Cần thống nhất lại với team đang so sánh trên bộ data nào** để tránh hiểu nhầm là có lỗi.

### 4.2 · Sai lệch nhỏ: 0,7342 (mình, MacBook) vs 0,7333 (teammate, Windows) — 2 nguyên nhân cộng lại

Cùng 1 bộ data, cùng feature, cùng `seed=42`, nhưng vẫn lệch ~0,001. Có 2 nguyên nhân:

1. **LightGBM train đa luồng không đảm bảo bit-exact reproducible** dù đã cố định seed (`num_threads=8`) — thứ tự cộng gradient/histogram giữa các luồng phụ thuộc vào CPU/tải máy tại thời điểm chạy.
2. **Khác hệ điều hành/kiến trúc CPU** (MacBook Apple Silicon/ARM vs Windows x86) — 2 máy dùng 2 bản LightGBM biên dịch khác nhau (Clang+libomp trên Mac vs MSVC+OpenMP trên Windows), có thể cho kết quả làm tròn số học khác nhau ở mức bit, độc lập với vấn đề đa luồng ở trên.

**Đã fix (1), KHÔNG fix được (2)** — thêm 2 tham số vào `PARAMS` trong `train.py`:
```python
"deterministic": True,
"force_row_wise": True,
```
Kiểm chứng: chạy lại `train.py` 2 lần liên tiếp **trên cùng máy** sau khi thêm 2 cờ này → kết quả giống hệt nhau tuyệt đối (0,8671/0,7433 cả 2 lần). Nhưng `deterministic=True` **chỉ đảm bảo reproducible trên cùng 1 máy/OS** — không đảm bảo Mac và Windows ra cùng 1 số bit-exact, vì khác cả kiến trúc CPU lẫn binary, không phải điều LightGBM có thể tự khắc phục. Chênh lệch còn lại (~0,001) giữa 2 máy là mức chấp nhận được do khác phần cứng/OS, không cần truy tiếp. Vì `train_hybrid.py`/`train_stacking.py` import chung `PARAMS` từ `train.py`, fix (1) áp dụng luôn cho toàn bộ pipeline, không cần sửa từng script riêng lẻ.

## 5. Kết luận & khuyến nghị

- **Baseline mới có 20 feature** (14 gốc + `pickup_distance_km`, `payment_method`, `dist_center_km`, `cust_completion_rate_30d`, `dist_airport_km`, `dist_hotspot_km`) — feature engineering không phải lúc nào cũng thất bại, chỉ cần đổi đúng loại nguồn dữ liệu/tín hiệu (cột có sẵn nhưng bị bỏ sót, dẫn xuất chưa khai thác/landmark/cụm mới, hoặc phần dư dữ liệu bị bỏ sót, thay vì biến tấu/làm mịn hơn từ 1 nguồn đã bão hoà).
- **28 thử nghiệm, 6 thành công (~1/5)** — đã quét gần như toàn diện: đa cửa sổ lịch sử khách, recency, 3 hướng địa lý (landmark cố định ×2, ô lưới ×2, K-means cụm thích ứng), categorical encoding thay thế, percentile (fit-on-train an toàn rò rỉ), 9 thử nghiệm tương tác nhân (2 đến 5 chiều, đơn lẻ lẫn gộp batch 4-8 toán hạng cùng lúc). Điểm chung của mọi thất bại: hoặc trùng lặp thông tin cây đã tự suy ra được (kể cả xử lý NaN nội bộ — `is_eta_missing` cho gain=0,000 tuyệt đối), hoặc tính trên mẫu/phân bố quá thưa/nhạy cảm dễ overfit, hoặc là tương tác tường minh mà cây gradient boosting vốn đã tự tổ hợp được qua nhiều nhát cắt tuần tự.
- **Đã kiểm tra: không còn cột dữ liệu nào chưa khai thác trong `orders.parquet`/`customer_daily.parquet`.** Phát hiện thêm 2 bảng `driver_daily_stats.parquet` và `offers.parquet` trong thư mục dự án, nhưng xác nhận **không join được** với `orders.parquet` hiện tại (0% trùng `order_key` dù cùng khoảng ngày) — nhiều khả năng thuộc 1 project khác ("Driver Acceptance Rate", theo tên thư mục `data/raw_driver_ar/`). Cần hỏi mentor/team data nếu muốn dùng 2 bảng này (cần khoá join hợp lệ).
- **Phân khúc yếu nhất đã dịch chuyển 2 lần qua các vòng feature engineering**: 14-feature ban đầu → "khách mới" yếu nhất (AUC 0,7046); 17-19 feature → "giờ thường" yếu nhất; 20-feature hiện tại → **"ETA ngắn" và "giờ thường"** yếu nhất (AUC ~0,744-0,745), "khách mới" đã cải thiện lên gần giữa bảng (0,748). Soi trực tiếp các đơn đoán sai trong 2 phân khúc này (mục cuối) cho thấy **các feature hiện có đã lệch đúng hướng rủi ro** (eta_seconds, cust_cancel_rate_30d, pickup_distance_km cao hơn ở nhóm đoán sai) — tức đây không còn là vấn đề thiếu feature, mà là giới hạn ngưỡng quyết định (threshold 0.5) trên nền tỉ lệ chấp nhận cao (~90%). Hướng cải thiện tiếp theo (nếu cần) nên là tune hyperparameter hoặc điều chỉnh threshold, không phải feature engineering thêm.
- **Phương pháp soi trực tiếp các đơn đoán sai (mục 3) hiệu quả hơn thử feature theo trực giác**: từ việc so sánh nhóm đơn huỷ bị model bỏ sót với nhóm đoán đúng trên các cột dữ liệu thô chưa dùng, tìm ra `dist_center_km` (thành công) và `is_deep_night` (thất bại, nhưng vẫn hữu ích để xác nhận `hour_of_day` đã đủ thông tin).
- **Pipeline giờ đã reproducible bit-exact trên cùng 1 máy** (`deterministic=True`) — chênh lệch nhỏ giữa các máy khác OS/CPU (Mac vs Windows) vẫn có thể còn (~0,001), chấp nhận được. Mọi so sánh số liệu với mentor/teammate từ nay trở đi cần thống nhất rõ đang dùng bộ data nào (8 ngày hay 30 ngày) trước khi đối chiếu.

**Quyết định dừng feature engineering ở đây** — 6/12 feature mục tiêu là kết quả thực chất sau khi đã quét kỹ dữ liệu hiện có; đạt thêm 6 feature nữa nhiều khả năng cần dữ liệu bổ sung thật (không phải biến đổi toán học/thống kê trên dữ liệu sẵn có), nên khuyến nghị trao đổi với mentor về việc lấy thêm nguồn dữ liệu trước khi tiếp tục.

**Mốc mới nhất (Hybrid GBDT+NN, 20 feature, data 30 ngày)**: ROC-AUC 0,7508 (post-dispatch) — tăng **+0,0241** so với mốc đóng băng tuần 1 (0,7267).

---

## 6. Thay đổi kiến trúc: `is_post_dispatch` từ FEATURE thành RULE if/else

*Mục này ghi nhận 1 thay đổi kiến trúc sau khi báo cáo phần trên đã hoàn thành — các số liệu ở mục 1-5 (baseline "combined pre+post", 20 feature) vẫn giữ nguyên làm mốc lịch sử/so sánh, KHÔNG bị thay thế.*

### Vì sao đổi

Xuyên suốt mục 2-3, `is_post_dispatch` luôn là feature áp đảo nhất hệ thống (gain gấp hàng chục lần feature đứng thứ 2, tương quan 0,676 với target — xem mục 3) vì đây gần như là 1 rule tất định: `is_post_dispatch=False` → `y_accept=0` (huỷ) đúng 100% theo định nghĩa dữ liệu, không phải thứ model cần "học". Để model 1 feature tất định làm 1 trong nhiều input khiến:
- Model phải dành 1 phần "ngân sách" cây/tham số để tự tìm ra rule này thay vì tập trung hoàn toàn vào phần khó thật sự (post-dispatch).
- Số liệu "tổng" (combined pre+post) bị thổi phồng, dễ gây hiểu nhầm về chất lượng model thật.

→ Từ nay, `is_post_dispatch` được tách ra làm **rule if/else ở tầng ngoài model**:
- `is_post_dispatch = False` → trả thẳng "huỷ" (không qua model).
- `is_post_dispatch = True` → mới đi qua model. **Model giờ CHỈ train và infer trên phần dữ liệu post-dispatch.**

### Thay đổi kỹ thuật

- `features.py`: bỏ `is_post_dispatch` khỏi `FEATURES` (còn 19 feature, từ 20) — cột vẫn được tính trong `df` để dùng lọc, chỉ không đưa vào `X`.
- Cả 5 script train (`train.py`, `train_mlp.py`, `train_hybrid.py`, `ensemble.py`, `train_stacking.py`): lọc `df.is_post_dispatch == 1` **trước** khi chia train/test theo thời gian, thay vì chia trên toàn bộ rồi lọc lại sau.
- Thêm metric mới **`system_full`** ở cả 5 script — mô phỏng đúng hiệu năng hệ thống thật (rule cho pre-dispatch + model cho post-dispatch) trên tập test gốc (cả pre+post), để vẫn có 1 con số so sánh tương đương với "test" (combined) của kiến trúc cũ.

### So sánh trước/sau (cùng data, cùng feature — chỉ khác cách xử lý `is_post_dispatch`)

| | Kiến trúc cũ (`is_post_dispatch` = feature) | Kiến trúc mới (`is_post_dispatch` = rule) |
|---|---|---|
| Số feature model thấy | 20 | 19 |
| Train | 270.763 dòng (cả pre+post, AR nền 0,8197) | 245.982 dòng (**chỉ post-dispatch**, AR nền 0,9023) |
| "test"/"test-post" (LightGBM) | 0,7505 (đã lọc post-dispatch sau khi train trên cả 2) | **0,7500** (model train từ đầu chỉ trên post-dispatch) |
| "test"/"system_full" (LightGBM, cả pre+post) | 0,8709 | **0,8706** |

Chênh lệch cực nhỏ (~0,0003-0,0005) — xác nhận: bỏ các dòng pre-dispatch ra khỏi train **không làm mất thông tin có ích**, vì is_post_dispatch trước đây đã tách 2 nhóm này gần như hoàn hảo rồi. Đổi kiến trúc chủ yếu là lợi ích về **thiết kế** (model đơn giản hơn, dễ diễn giải hơn, tách rõ phần "rule chắc chắn" khỏi phần "model cần học") chứ không phải để tăng AUC.

### Kết quả cuối cùng — cả 5 model (kiến trúc mới, 19 feature)

| Model | test (post-dispatch, model thuần) | system_full (tham khảo, rule + model) | Cancel PR-AUC (test)¹ |
|---|---|---|---|
| MLP v2 | 0,7417 | 0,8663 | 0,2490 |
| Hybrid GBDT+NN | 0,7466 | 0,8688 | 0,2615 |
| Ensemble (avg) | 0,7486 | 0,8698 | 0,2648 |
| **LightGBM baseline** | 0,7500 | 0,8706 | **0,2685** |
| Stacking | **0,7506** | **0,8709** | 0,2666 |

Stacking vươn lên dẫn đầu theo ROC-AUC (trước đó Hybrid GBDT+NN dẫn đầu ở kiến trúc cũ) — do meta-learner giờ học trực tiếp trên post-dispatch, không còn bị pha loãng bởi phần pre-dispatch dễ đoán trong `meta_val`.

> ¹ **Cập nhật Cancel PR-AUC (10/08/2026)** — lấy trực tiếp từ log MLflow gốc của đúng 5 lần chạy tạo ra bảng này (`evaluate()` đã luôn tính sẵn `pr_auc_cancel`, chỉ chưa được trích vào báo cáo lúc đó), không phải train lại. **Thứ hạng ĐỔI khi xếp theo Cancel PR-AUC**: LightGBM baseline (0,2685) thực ra nhỉnh hơn Stacking (0,2666) — dù Stacking thắng sít sao theo ROC-AUC (0,7506 vs 0,7500, chênh 0,0006). Đây là bằng chứng W2 y hệt phát hiện ở W3 (FT-Transformer từng "thắng" chỉ vì đo bằng ROC-AUC): ROC-AUC bị pha loãng bởi lớp accept đa số (~90%), có thể che mất khác biệt thật ở khả năng xếp hạng lớp huỷ. **Khuyến nghị**: coi LightGBM baseline và Stacking là ngang nhau trong phạm vi nhiễu (0,7500-0,7506 ROC-AUC, 0,2666-0,2685 Cancel PR-AUC đều rất sát), ưu tiên LightGBM baseline nếu cần đơn giản/nhanh hơn Stacking (không cần train meta-learner 2 tầng) — nhất quán với việc XGBoost (GBDT) cũng thắng ở W3.

**Mốc mới nhất (kiến trúc post-dispatch-only): ROC-AUC 0,7506 (Stacking, test, model thuần) / Cancel PR-AUC 0,2685 (LightGBM baseline cao nhất) — 2 model gần như ngang nhau, xem ghi chú Cancel PR-AUC ở trên.**
