"""Biến 2 file raw thành SHORT-LIST feature cho RIDER (customer) ACCEPTANCE.
Bài toán: khách đặt cuốc - sẽ ĐI (accept, y=1) hay TỰ HUỶ (cancel, y=0)?
phân biệt bằng cờ is_post_dispatch.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# DOD item 5 (Optional, W3) — thời tiết, join LAG 1 NGÀY (xem weather.py) để tránh leakage.
# Đường dẫn tự suy ra từ vị trí file này, không phụ thuộc RAW của train.py (weather nằm ở
# data/ trực tiếp, không phải data/raw/) — build_features() tự load nếu file tồn tại, không
# thì bỏ qua êm, không cần sửa signature/caller nào.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
WEATHER_PATH = os.path.join(_ROOT, "data", "weather_daily_30d.parquet")
WEATHER_HOURLY_PATH = os.path.join(_ROOT, "data", "weather_hourly.parquet")

# --- SHORT-LIST feature ---------------------------------------------------------------
# PRE-DISPATCH: biết được ngay khi khách đặt (mọi cuốc đều có)
PRE_FEATURES = [
    "total_fee",             # giá cuốc - cao có làm khách ngại?
    "trip_distance_km",      # độ dài cuốc
    "fee_per_km",            # giá trên mỗi km (dẫn xuất)
    "dist_center_km",        # khoảng cách điểm đón tới trung tâm (Hồ Hoàn Kiếm, dẫn xuất từ lat/lon)
    "dist_airport_km",       # khoảng cách điểm đón tới sân bay Nội Bài — thử nghiệm
    "dist_hotspot_km",       # khoảng cách tới cụm điểm đón đông đơn gần nhất (K-means) — thử nghiệm
    "surge_multiplier",      # hệ số giờ cao điểm
    "hour_of_day",
    "is_weekend",
    "is_schedule_order",     # đặt trước hay đi ngay
    "cust_cancel_rate_30d",  # TỶ LỆ HUỶ QUÁ KHỨ CỦA KHÁCH (point-in-time D-30…D-1)
    "cust_orders_30d",       # khách quen hay khách mới
    "cust_completion_rate_30d",  # tỷ lệ hoàn thành quá khứ — thử nghiệm
    "is_rainy_3h",             # có mưa (>0,1mm) trong 3 GIỜ TRƯỚC đó tại ô lưới điểm đón — thử nghiệm (DOD item 5, hourly)
]
# ĐÃ THỬ: rain_sum_mm_d1 (mưa HÔM QUA, mm, tại ô lưới 0,1° gần điểm đón nhất — join LAG 1 NGÀY
# từ data/weather_daily_30d.parquet để tránh leakage, xem weather.py) — AUC GIẢM (0,7500→0,7486
# post, xác nhận qua 2 lần chạy giống hệt nhau 0,7486). Đã revert.
# ĐÃ THỬ: is_rainy_d1 (nhị phân: mưa HÔM QUA >1mm tại ô lưới điểm đón — cùng nguồn/join với
# rain_sum_mm_d1 ở trên nhưng nhị phân hoá thay vì để nguyên liên tục) — AUC TĂNG (0,7500→0,7510
# post, xác nhận qua 2 lần chạy giống hệt nhau 0,7510). Lưu ý: gain rất nhỏ (0,001, gần như thấp
# nhất hệ thống) dù AUC tăng — nhị phân hoá "giúp" hơn liên tục có thể vì cắt bớt nhiễu đo lường
# (rain_sum_mm_d1 lệch phải rất mạnh, vài giá trị cực đoan).
# ĐÃ THỬ THÊM: temp_mean_c_d1 (nhiệt độ TB HÔM QUA, cộng thêm CÙNG LÚC với is_rainy_d1 đã giữ ở
# trên) — AUC gần như KHÔNG ĐỔI (0,7510→0,7511, +0,0001, trong biên độ nhiễu) — không đáng thêm
# phức tạp. Đã revert.
# --- Nâng độ phân giải thời tiết lên GIỜ (data/weather_hourly.parquet, fetch qua Open-Meteo
# Historical Archive API, xem weather.py) — is_rainy_d1 (ngày) bị THAY THẾ bởi bản giờ tốt hơn: ---
# ĐÃ THỬ: is_rainy_h1 (nhị phân: mưa >0,1mm trong GIỜ NGAY TRƯỚC đó, lag 1 giờ thay vì 1 ngày) —
# AUC GIẢM so với is_rainy_d1 (0,7510→0,7494, xác nhận qua 2 lần chạy giống hệt nhau 0,7494).
# 1 giờ đơn lẻ có vẻ quá "nhiễu" (mưa có thể vừa tạnh vài phút trước) — kém ổn định hơn tín hiệu
# "cả ngày hôm qua" dù xa hơn về thời gian. Đã revert.
# ĐÃ THỬ: is_rainy_3h (nhị phân: mưa >0,1mm CỘNG DỒN trong 3 GIỜ liền trước, rolling sum trước
# khi lag — "mượt" hơn is_rainy_h1 nhưng vẫn sát thời điểm hơn hẳn is_rainy_d1) — AUC TĂNG so
# với is_rainy_d1 (0,7510→0,7519 post, xác nhận qua 2 lần chạy giống hệt nhau 0,7519; +0,0019 so
# baseline gốc không thời tiết 0,7500). Gain 658,9 — CAO HƠN HẲN is_rainy_d1 (0,001), xác nhận
# đây là tín hiệu thật, không phải nhiễu AUC. Giữ lại — THAY is_rainy_d1 bằng is_rainy_3h.
# ĐÃ THỬ: rain_sum_mm_3h (liên tục thay vì nhị phân hoá) — AUC GIẢM (0,7519→0,7501) — cùng bài
# học với rain_sum_mm_d1: liên tục lệch phải mạnh, nhị phân hoá vẫn tốt hơn. Đã revert.
# ĐÃ THỬ THÊM: temp_mean_c_3h (nhiệt độ TB 3 giờ, cộng thêm CÙNG LÚC với is_rainy_3h) — AUC GIẢM
# (0,7519→0,7500) — khác pattern với bản ngày (temp_mean_c_d1 trung tính), ở đây temp gây hại
# thật. Đã revert, chỉ giữ is_rainy_3h.
# ĐÃ THỬ: is_deep_night (giờ 0h-4h, AR giảm rõ 0,91→0,81 trong EDA) — AUC GIẢM (0,8689→0,8684
# tổng, 0,7468→0,7458 post), gain gần 0 (36, thấp nhất hệ thống). LightGBM đã tự tách được
# khung giờ đêm qua các nhát cắt trên hour_of_day (feature liên tục) — thêm cờ tường minh không
# cho biết gì mới, chỉ làm model phức tạp hơn. Đã revert.
# ĐÃ THỬ: dist_center_km (khoảng cách điểm đón tới Hồ Hoàn Kiếm, dẫn xuất từ lat/lon — phát hiện
# qua phân tích lỗi: đơn ở xa trung tâm >20km có AR thấp hẳn 0,82-0,71 so với ~0,90-0,91 ở gần
# trung tâm) — AUC TĂNG (0,8689→0,8699 tổng, 0,7468→0,7487 post), gain 10.207 (hạng 10/17).
# Tương quan yếu với trip_distance_km (0,217) và pickup_distance_km (0,047) — không trùng lặp,
# là tín hiệu địa lý mới (đơn xa trung tâm dễ gặp trục trặc/khách đổi ý hơn). Giữ lại.
# ĐÃ THỬ: thêm is_new_customer vào FEATURES để giúp phân khúc khách mới (xem error_analysis.py:
# AUC 0.70, recall_cancel 0.00 — yếu nhất hệ thống) — gain=0, LightGBM coi hoàn toàn dư thừa vì
# đã tự suy ra qua cust_orders_30d=0. KHÔNG cải thiện được recall khách mới. Bỏ lại, không giữ
# feature không có tác dụng. Vấn đề khách mới là do THIẾU dữ liệu (cold-start), không phải model
# chưa biết đây là khách mới — cần nguồn dữ liệu khác (không có trong dataset hiện tại) mới giải
# quyết được, feature engineering từ dữ liệu sẵn có không đủ.
PRE_CATEGORICAL = ["travel_mode", "vertical_name", "payment_method"]
# ĐÃ THỬ: service_name (~40 giá trị, chi tiết hơn vertical_name) — AUC GIẢM (0,8671→0,8661 tổng,
# 0,7433→0,7413 post) dù gain khá cao (29.161). Nhiều khả năng: nhiều giá trị mẫu quá nhỏ (<50
# dòng) dễ overfit trên train, và trùng lặp thông tin với vertical_name (bản chi tiết hơn của
# cùng 1 nguồn) làm cây phân tán quyết định. Đã revert.
# ĐÃ THỬ: payment_method (16 giá trị: cash/momo/zalo/...) — AUC TĂNG (0,8671→0,8689 tổng,
# 0,7433→0,7468 post), gain vừa phải (8.103, hạng 12/16). Khác cash hay ví điện tử phản ánh mức
# độ "chắc đi" của khách (trả trước qua ví thường ít huỷ hơn cash) — nguồn thông tin mới, không
# trùng với các feature khách hàng/giá cước hiện có. Giữ lại.
# ĐÃ THỬ: pickup_quadrant (NE/NW/SE/SW so với Hồ Hoàn Kiếm, dẫn xuất từ dấu của dlat/dlon — EDA
# cho thấy vẫn có chênh lệch AR ~5 điểm % ngay cả trong nhóm gần trung tâm <5km, tức không hoàn
# toàn trùng dist_center_km) — nhưng AUC GIẢM (0,8699→0,8693 tổng, 0,7487→0,7474 post, xác nhận
# qua 2 lần chạy), gain thấp nhất hệ thống (536). Chỉ 4 giá trị thô (chia theo dấu, không theo
# phân bố thực tế của dữ liệu) có thể làm cây phải hy sinh split cho 1 feature ít thông tin,
# giẫm chân lên các split khác. Đã revert.
# ĐÃ THỬ: cust_completion_rate_30d (= n_completed_30d / n_orders_30d — phát hiện qua thấy
# n_orders_30d KHÔNG LUÔN BẰNG n_completed_30d + n_cust_cancel_30d, tức có 1 loại kết quả thứ 3
# ngoài "hoàn thành"/"khách tự huỷ" chưa từng khai thác riêng, ~7,3% dòng customer_daily có
# chênh lệch này) — AUC TĂNG (0,8699→0,8703 tổng, 0,7487→0,7494 post, xác nhận qua 2 lần chạy),
# gain 10.158 (hạng 11/18) — gần bằng dist_center_km. Không hoàn toàn trùng cust_cancel_rate_30d
# vì phần dư (kết quả thứ 3, có thể là tài xế huỷ/không tìm được tài xế) mang thông tin riêng.
# Giữ lại.
# ĐÃ THỬ: area_cancel_rate_7d (tỷ lệ huỷ quá khứ 7 ngày của KHU VỰC điểm đón — ô lưới ~1km từ
# lat/lon làm tròn 2 chữ số, cùng logic rolling point-in-time với cust_cancel_rate_30d nhưng
# theo khu vực thay vì theo khách. EDA cho tương quan -0,077 với target, không trùng lặp mạnh
# với dist_center_km (0,288) hay cust_cancel_rate_30d (0,065)) — nhưng AUC GIẢM (0,8703→0,8693
# tổng, 0,7494→0,7475 post, xác nhận qua 2 lần chạy) dù gain khá (6.800). Nhiều khả năng: đa số
# ô lưới có rất ít đơn/7 ngày (median ~6 đơn/ô cho toàn bộ dữ liệu, một số ô cực ít) nên tỷ lệ
# huỷ theo cửa sổ ngắn dễ nhiễu/overfit trên train — cùng bài học với service_name (gain cao
# trên train không đảm bảo generalize). Đã revert.
# ĐÃ THỬ: cust_tenure_days (số ngày từ lần đầu xuất hiện của khách trong customer_daily — đo độ
# "thâm niên" khác với cust_orders_30d là đếm SỐ ĐƠN trong 30 ngày) — AUC GIẢM (0,8703→0,8697
# tổng, 0,7494→0,7483 post, xác nhận qua 2 lần chạy) dù gain nhỏ (1.407). Tương quan thấp với
# cust_orders_30d (0,15) nên không hẳn dư thừa, nhưng tín hiệu bản thân quá yếu (EDA corr 0,031)
# để bù chi phí thêm 1 feature. Đã revert.
# ĐÃ THỬ: dist_airport_km (khoảng cách điểm đón tới sân bay Nội Bài, dẫn xuất từ lat/lon — khác
# landmark với dist_center_km, gợi ý từ service_name có "xanh_airport_limo" là dịch vụ riêng) —
# AUC TĂNG (0,8703→0,8708 tổng, 0,7494→0,7503 post, xác nhận qua 2 lần chạy), gain 5.869 (hạng
# 14/19). Tương quan vừa phải với dist_center_km (0,283) — không quá trùng lặp, đơn gần sân bay
# có hành vi khác đơn gần trung tâm thành phố (chuyến đi sân bay thường có giờ giấc/áp lực khác).
# Giữ lại.
# ĐÃ THỬ LẦN 2: area_cancel_rate_14d — thử lại ý tưởng area_cancel_rate_7d (đã fail) với ô lưới
# thô hơn (~5,5km thay vì ~1km) + cửa sổ dài hơn (14 ngày thay vì 7) để mỗi ô có đủ mẫu (trung
# vị ~67 đơn/ô thay vì ~6) — sửa đúng vấn đề overfit lần trước. Nhưng vẫn AUC GIẢM (0,8708→0,8696
# tổng, 0,7503→0,7481 post, xác nhận qua 2 lần chạy). Lý do khác: ô lưới thô hơn giờ tương quan
# quá cao với dist_center_km (0,609, so với 0,288 ở bản ô nhỏ) — chỉ còn là bản mã hoá lại
# "khoảng cách tới trung tâm" theo khu vực, không phải tín hiệu mới. Đã revert (2 lần fail liên
# tiếp — dừng hướng ô lưới địa lý ở đây).
# ĐÃ THỬ: fee_percentile_in_vertical (percentile total_fee so với cùng vertical_name — ranh giới
# percentile CHỈ fit trên TRAIN rồi áp forward cho test, tránh rò rỉ phân bố giá tương lai) —
# AUC GIẢM (0,8708→0,8704 tổng, 0,7503→0,7497 post, xác nhận qua 2 lần chạy) dù gain rất cao
# (10.023, hạng 9/20 — chỉ sau dist_center_km trong nhóm feature "thử nghiệm"). Cùng bài học với
# service_name/area_cancel_rate_7d: gain cao trên train không đảm bảo generalize — percentile
# rank có thể nhạy với đúng phân bố giá của ngày test (chỉ 1 ngày), không ổn định. Đã revert.
# ĐÃ THỬ: is_eta_missing (eta_seconds thiếu ~3,3% đơn post-dispatch, AR khi thiếu 0,928 vs 0,902
# khi có — chênh lệch thật) — nhưng gain = 0,000 TUYỆT ĐỐI, AUC không đổi. LightGBM đã tự xử lý
# NaN nội bộ khi split trên chính eta_seconds (tự học hướng đi tối ưu cho giá trị thiếu) nên cờ
# tường minh hoàn toàn dư thừa — bài học giống is_new_customer/is_deep_night nhưng rõ ràng hơn
# (gain = 0 chứ không chỉ gần 0). Đã revert, không thử is_pickup_distance_missing (cùng cơ chế).
# ĐÃ THỬ: coi re_dispatch_counts là CATEGORICAL thay vì numeric (giả thuyết: đây là mã bucket/tier
# rời rạc — AR không đơn điệu theo giá trị: 0,803→0,868→0,874→0,753→0,777→1,000 cho 6 giá trị
# 0/2/3/5/10/20) — nhưng AUC GIẢM (0,8708→0,8703 tổng, 0,7503→0,7495 post) VÀ gain cũng giảm
# (27.571→12.479) so với xử lý numeric. Xử lý numeric (nhát cắt ngưỡng) hoá ra đã đủ tách các
# giá trị rời rạc hiệu quả hơn categorical (encoding categorical có thể kém regularize hơn với
# chỉ 6 giá trị). Đã revert.
# ĐÃ THỬ: dist_hotspot_km (khoảng cách tới tâm cụm điểm đón đông đơn gần nhất, phân cụm bằng
# K-means K=15 — CHỈ fit trên TRAIN rồi áp forward cho test, tránh rò rỉ. Ý tưởng: thay vì 1 điểm
# neo cố định (trung tâm/sân bay), tìm nhiều "điểm nóng" thực tế theo dữ liệu) — AUC TĂNG
# (0,8708→0,8709 tổng, 0,7503→0,7505 post, xác nhận qua 2 lần chạy), gain 4.160 (hạng 15/20).
# Tương quan vừa phải với dist_center_km (0,487) và dist_airport_km (0,211) — không trùng lặp
# hoàn toàn, các cụm đều đủ mẫu (nhỏ nhất ~2.100 đơn) nên không gặp vấn đề overfit như hướng ô
# lưới cố định trước đó. Giữ lại (cải thiện nhỏ nhưng thật và ổn định).
# ĐÃ THỬ: cancelrate_x_pickupdist (= cust_cancel_rate_30d × pickup_distance_km, tương tác nhân —
# EDA corr -0,163, MẠNH HƠN cả 2 thành phần riêng lẻ cộng lại) — gain rất cao (33.170, hạng 3
# toàn hệ thống) nhưng ROC-AUC test-post GIẢM (0,7505→0,7489, xác nhận 2 lần chạy). Đáng chú ý:
# PR-AUC lớp huỷ lại TĂNG (0,2675→0,2702) — tương tác này cải thiện xếp hạng riêng cho nhóm huỷ
# nhưng làm giảm khả năng xếp hạng tổng thể (ROC-AUC). Theo tiêu chí nhất quán xuyên suốt (ROC-AUC
# post-dispatch), coi là THẤT BẠI — đã revert. Ghi chú lại vì đây là candidate duy nhất có
# trade-off 2 chiều rõ rệt giữa 2 metric, có thể đáng thử lại nếu mục tiêu nghiệp vụ đổi ưu tiên
# sang PR-AUC/recall lớp huỷ thay vì ROC-AUC tổng thể.
# ĐÃ THỬ: cancelrate_x_distcenter (= cust_cancel_rate_30d × dist_center_km — EDA corr -0,145,
# cũng mạnh hơn từng thành phần riêng) — AUC GIẢM cả tổng (0,8709→0,8696) lẫn post (0,7505→0,7481)
# VÀ PR-AUC lớp huỷ cũng giảm (0,2675→0,2640) — thất bại rõ ràng trên mọi mặt, không có trade-off
# như candidate trên. Đã revert. Kết luận chung cho hướng tương tác nhân: gain cao trên train
# (33.170 và 6.122) không đảm bảo generalize — cùng bài học với service_name/fee_percentile.
# ĐÃ THỬ THÊM 3 công thức tương tác nhân nữa (quét hệ thống mọi cặp/bộ ba numeric feature, chọn
# theo "tương quan EDA của tích số MẠNH HƠN từng thành phần riêng"):
#  - surge_x_pickupdist (surge_multiplier × pickup_distance_km, cải thiện EDA +0,021 so với
#    thành phần mạnh nhất) — gain rất cao (49.606, hạng 2 toàn hệ thống) nhưng AUC GIẢM
#    (0,8709→0,8701 tổng, 0,7505→0,7490 post) và PR-AUC huỷ cũng giảm (0,2675→0,2654).
#  - hotspot_x_redispatch (dist_hotspot_km × re_dispatch_counts, cải thiện +0,016) — AUC GIẢM
#    (0,7505→0,7493) dù PR-AUC huỷ nhích nhẹ (0,2675→0,2682).
#  - center_hotspot_redispatch (dist_center_km × dist_hotspot_km × re_dispatch_counts — TAM
#    GIÁC 3 chiều, cải thiện EDA cao nhất từng thấy +0,045) — vẫn AUC GIẢM (0,7505→0,7493),
#    PR-AUC huỷ nhích nhẹ giống trên (0,2682).
# Cả 3 đã revert (xác nhận qua 2 lần chạy mỗi cái).
# ĐÃ THỬ THÊM 4-way và 5-way (quét hệ thống C(13,4) và C(13,5) tổ hợp numeric feature, chọn theo
# cùng tiêu chí "cải thiện tương quan EDA so với thành phần mạnh nhất" — điểm cải thiện cao nhất
# tìm được đã THẤP HƠN bản 3-way tốt nhất (0,027 và 0,017 so với 0,045), dấu hiệu xấu ngay từ đầu):
#  - geo4_redispatch (dist_center_km × dist_airport_km × dist_hotspot_km × re_dispatch_counts,
#    cải thiện EDA +0,027) — AUC gần như KHÔNG ĐỔI (0,7505→0,7504, chênh 0,0001, xác nhận 2 lần
#    chạy) — giống hệt kiểu "vô hại nhưng vô dụng" của pickup_speed_kmh.
#  - geo5_redispatch (trip_distance_km × fee_per_km × dist_center_km × dist_hotspot_km ×
#    re_dispatch_counts, cải thiện EDA +0,017) — AUC giảm rất nhẹ (0,7505→0,7503, xác nhận 2 lần
#    chạy) nhưng PR-AUC lớp huỷ TĂNG (0,2675→0,2702) — cùng kiểu trade-off như cancelrate_x_pickupdist.
# KẾT LUẬN SAU 7 THỬ NGHIỆM TƯƠNG TÁC NHÂN (2-way tới 5-way, luôn chọn ứng viên có tương quan EDA
# tốt nhất tìm được qua quét toàn bộ tổ hợp): không có công thức nào cải thiện ROC-AUC test-post,
# và "điểm cải thiện EDA tốt nhất tìm được" giảm dần khi tăng số chiều (3-way > 2-way > 4-way >
# 5-way) — càng nhiều feature nhân với nhau, tín hiệu càng loãng/nhiễu thay vì cộng dồn. Hướng
# tương tác nhân (multiply, bất kể 2-5 chiều) coi như đã cạn hoàn toàn với LightGBM + dữ liệu
# hiện tại: cây vốn đã tự tổ hợp được các feature qua nhiều nhát cắt tuần tự, nên tích số tường
# minh chỉ tạo thêm 1 cột dễ overfit (gain ảo trên train, luôn xếp hạng rất cao) mà không mang
# lại xếp hạng tốt hơn trên test. KHÔNG khuyến nghị thử thêm 6-way trở lên.
# ĐÃ THỬ THÊM: gộp CÙNG LÚC 4 feature từng thất bại RIÊNG LẺ nhưng KHÔNG bị chứng minh dư thừa
# tuyệt đối (gain>0 khi thử alone, khác với is_new_customer/is_deep_night/is_eta_missing có
# gain=0 chính xác — gộp các feature "chắc chắn vô dụng" thì không thể giúp gì) —
# cust_tenure_days + pickup_quadrant + area_cancel_rate_7d + fee_percentile_in_vertical, thêm
# CÙNG 1 LẦN (24 feature) thay vì từng cái một. Kết quả: AUC vẫn GIẢM (0,8709→0,8704 tổng,
# 0,7505→0,7495 post, xác nhận qua 2 lần chạy) — KHÔNG vượt qua baseline sạch. Điểm thú vị: kết
# quả gộp (0,7495) lại TỐT HƠN 3/4 kết quả khi thử từng cái một (cust_tenure_days một mình 0,7483,
# pickup_quadrant một mình 0,7474, area_cancel_rate_7d một mình 0,7475) — chỉ thua
# fee_percentile_in_vertical một mình (0,7497). Tức các feature yếu này có phần "trung hoà" bớt
# nhiễu của nhau khi gộp, nhưng tổng độ phức tạp thêm vào vẫn không đủ bù lại — không có hiệu ứng
# cộng dồn tích cực nào xảy ra. Đã revert cả 4.
# ĐÃ THỬ: gộp CÙNG LÚC 8 "toán hạng" tương tác nhân đa dạng (mỗi toán hạng = 1 feature gốc hoặc
# tích của 2-4 feature gốc, chọn từ các tổ hợp CHƯA từng thử riêng lẻ để tối đa hoá cơ hội bổ sung
# lẫn nhau, không lặp lại các công thức đã fail ở trên): surge_x_cancelrate, center_x_redispatch,
# surge_x_eta, airport_hotspot_redispatch, surge_redispatch_cancelrate, fee_center_redispatch,
# center_surge_redispatch_cancelrate, surge_pickup_cancelrate_completerate — thêm 1 lần (28
# feature). Một số toán hạng có gain RẤT CAO khi đứng cùng nhau (surge_pickup_cancelrate_
# completerate = 19.149, hạng 5 toàn hệ thống; surge_x_cancelrate = 14.215; surge_x_eta =
# 10.845) — cao hơn nhiều so với lúc test riêng lẻ từng cái, cho thấy khi có nhiều toán hạng
# cùng lúc, cây "chọn" được toán hạng phù hợp nhất trong nhóm để dùng nhiều hơn. NHƯNG ROC-AUC
# test-post vẫn GIẢM (0,8709→0,8704 tổng, 0,7505→0,7496 post, xác nhận qua 2 lần chạy) — không
# vượt qua baseline sạch, dù đỡ hơn so với một số bản 1-đặc-trưng-tại-1-thời-điểm. Đã revert cả 8.
# KẾT LUẬN TỔNG HỢP (9 thử nghiệm tương tác nhân: 7 đơn lẻ + 2 batch): dù thử đơn lẻ (2 đến
# 5-way) hay gộp nhiều toán hạng cùng lúc (4 hoặc 8 toán hạng), KHÔNG có cấu hình nào vượt qua
# baseline 20-feature hiện tại trên ROC-AUC post-dispatch. Hướng tương tác nhân (dù test 1 lúc
# hay nhiều lúc) coi như đã cạn hoàn toàn với LightGBM + dữ liệu hiện có — nguyên nhân gốc không
# đổi: cây tự tổ hợp được thông tin qua nhiều nhát cắt tuần tự trên feature gốc, nên bất kỳ tích
# số tường minh nào cũng chỉ tái tạo lại thông tin cây đã có, đi kèm chi phí overfit (gain ảo
# rất cao trên train). Để đạt mục tiêu 12+ feature mới, hướng khả thi hơn là quay lại tìm NGUỒN
# DỮ LIỆU MỚI THỰC SỰ (chưa khai thác trong orders.parquet/customer_daily) thay vì biến đổi toán
# học trên các feature đã có — xem khuyến nghị ở mục 5 báo cáo.

# POST-DISPATCH: chỉ có nghĩa khi đã gán tài xế; NULL với cuốc huỷ trước khi gán
# is_post_dispatch KHÔNG còn là feature của model — chuyển thành RULE if/else bên ngoài (xem
# train.py): is_post_dispatch=False -> trả thẳng "huỷ" (y=0), không qua model; chỉ is_post_dispatch=True
# mới đi vào model. Model giờ CHỈ train/infer trên phần dữ liệu post-dispatch.
POST_FEATURES = [
    "eta_seconds",           # ETA tài xế tới điểm đón — lâu thì khách dễ huỷ
    "re_dispatch_counts",    # số lần phải tìm lại tài xế
    "pickup_distance_km",    # quãng đường tài xế phải chạy tới điểm đón (km)
]
# ĐÃ THỬ: pickup_speed_kmh (= pickup_distance_km / eta_seconds, tốc độ tài xế ước tính) — AUC
# KHÔNG ĐỔI (0,8699/0,7487, giống hệt không có feature này, xác nhận lại qua 2 lần chạy) dù có
# gain vừa phải (5.250). Đây là tổ hợp thuần tuý của 2 feature đã có sẵn (eta_seconds,
# pickup_distance_km) — cây tự tổ hợp được qua các nhát cắt tuần tự trên 2 cột gốc, không có
# thông tin thực sự mới. Đã revert.

FEATURES = PRE_FEATURES + PRE_CATEGORICAL + POST_FEATURES
CATEGORICAL_FEATURES = PRE_CATEGORICAL
TARGET = "y_accept"

# Phải khớp TEST_DATE trong train.py — chỉ dùng để FIT cụm K-means trên TRAIN, tránh rò rỉ vị trí
# đơn của tập test vào feature (giống tinh thần closed='left' ở customer_history()).
TEST_DATE = "2026-07-13"

# --- Cột CẤM dùng làm feature ---------------------------------------------------------
FORBIDDEN = ["order_key", "customer_key", "order_date", "order_datetime", TARGET]

def customer_history(customer_daily: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ huỷ quá khứ của khách trong cửa sổ [D-30, D-1] — KHÔNG gồm ngày D.
    """
    d = customer_daily.sort_values(["customer_key", "stat_date"]).copy()
    cols = ["n_orders", "n_completed", "n_cust_cancel"]
    roll = (
        d.set_index("stat_date")
        .groupby("customer_key")[cols]
        .rolling("30D", closed="left")   # ← [D-30, D-1]; bỏ closed='left' là rò rỉ
        .sum()
        .reset_index()
        .rename(columns={c: c + "_30d" for c in cols})
    )
    tot = roll.n_orders_30d
    roll["cust_orders_30d"] = tot
    roll["cust_cancel_rate_30d"] = np.where(tot > 0, roll.n_cust_cancel_30d / tot, np.nan)
    roll["cust_completion_rate_30d"] = np.where(tot > 0, roll.n_completed_30d / tot, np.nan)
    return roll[["customer_key", "stat_date", "cust_cancel_rate_30d", "cust_orders_30d",
                 "cust_completion_rate_30d"]]


def dist_hotspot_km(df: pd.DataFrame, n_clusters: int = 15) -> pd.Series:
    """Khoảng cách tới tâm cụm điểm đón (K-means) gần nhất — cụm được FIT CHỈ trên TRAIN (order_date
    < TEST_DATE) để tránh rò rỉ vị trí đơn của tập test vào feature (giống 1 transformer đã fit).
    """
    is_train = df.order_date < pd.Timestamp(TEST_DATE)
    coords_train = df.loc[is_train, ["pickup_latitude", "pickup_longitude"]].values
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(coords_train)
    all_coords = df[["pickup_latitude", "pickup_longitude"]].values
    cluster = km.predict(all_coords)
    return pd.Series(np.linalg.norm(all_coords - km.cluster_centers_[cluster], axis=1) * 111,
                      index=df.index)


def build_features(orders: pd.DataFrame, customer_daily: pd.DataFrame, verbose: bool = True):
    """Ráp 2 file raw -> bảng feature. Trả về (df đầy đủ, X, y)."""
    df = orders.copy()

    # 1 · Join lịch sử khách theo (customer_key, ngày đặt cuốc)
    hist = customer_history(customer_daily)
    df = df.merge(hist, left_on=["customer_key", "order_date"],
                  right_on=["customer_key", "stat_date"], how="left").drop(columns=["stat_date"])

    # 2 · Feature dẫn xuất
    df["hour_of_day"] = df.order_datetime.dt.hour
    df["is_weekend"] = df.order_datetime.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_schedule_order"] = df["is_schedule_order"].astype("float").fillna(0).astype(int)
    df["is_post_dispatch"] = df["is_post_dispatch"].astype(int)
    df["fee_per_km"] = df.total_fee / df.trip_distance_km.replace(0, np.nan)
    CENTER_LAT, CENTER_LON = 21.0285, 105.8542  # Hồ Hoàn Kiếm
    df["dist_center_km"] = np.sqrt((df.pickup_latitude - CENTER_LAT) ** 2
                                    + (df.pickup_longitude - CENTER_LON) ** 2) * 111
    AIRPORT_LAT, AIRPORT_LON = 21.2187, 105.8047  # Sân bay Nội Bài
    df["dist_airport_km"] = np.sqrt((df.pickup_latitude - AIRPORT_LAT) ** 2
                                     + (df.pickup_longitude - AIRPORT_LON) ** 2) * 111
    df["dist_hotspot_km"] = dist_hotspot_km(df)
    df["is_new_customer"] = (df.cust_orders_30d.isna() | (df.cust_orders_30d == 0)).astype(int)

    # 2b · Thời tiết (DOD item 5, optional) — chỉ TÍNH SẴN, không tự vào FEATURES trừ khi
    # thêm tên cột vào PRE_FEATURES/PRE_CATEGORICAL bên dưới sau khi thử nghiệm xác nhận có ích.
    if os.path.exists(WEATHER_PATH):
        from weather import build_weather_lag1_features, load_weather
        weather_df = load_weather(WEATHER_PATH)
        df = pd.concat([df, build_weather_lag1_features(df, weather_df)], axis=1)
    if os.path.exists(WEATHER_HOURLY_PATH):
        from weather import (
            build_weather_lag1hour_features,
            build_weather_lag3hour_features,
            load_weather_hourly,
        )
        weather_hourly_df = load_weather_hourly(WEATHER_HOURLY_PATH)
        df = pd.concat([df, build_weather_lag1hour_features(df, weather_hourly_df)], axis=1)
        df = pd.concat([df, build_weather_lag3hour_features(df, weather_hourly_df)], axis=1)

    # 3 · Tách X, y
    y = df[TARGET].astype(int)
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")

    # 4 · Chốt chặn chống leakage — chạy mỗi lần
    leaked = [c for c in X.columns if c in FORBIDDEN]
    assert not leaked, f"RÒ RỈ: {leaked} không được là feature"

    if verbose:
        print(f"  {X.shape[1]} feature · {len(X):,} dòng · AR {y.mean():.4f}")
    return df, X, y


def load_raw(raw_dir: str):
    return (
        pd.read_parquet(f"{raw_dir}/orders.parquet"),
        pd.read_parquet(f"{raw_dir}/customer_daily.parquet"),
    )