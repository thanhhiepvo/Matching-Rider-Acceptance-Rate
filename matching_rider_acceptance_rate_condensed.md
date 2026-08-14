# ML - MATCHING RIDER ACCEPTANCE RATE

## Báo cáo nghiên cứu tổng quan

## 1. Mục tiêu bài toán

Trong quy trình đặt xe, sau khi khách hàng tạo booking và tài xế nhận chuyến, ứng dụng hiển thị thông tin tài xế, biển số xe, vị trí và thời gian dự kiến đến đón. Tại thời điểm này, khách hàng có thể tiếp tục chờ hoặc hủy chuyến.

Bài toán **Matching Rider Acceptance Rate** sử dụng Machine Learning để dự đoán khả năng khách hàng tiếp tục chuyến sau khi đã có tài xế nhận chuyến. Đây là bài toán dự đoán hành vi của khách hàng, không phải dự đoán tài xế có nhận chuyến hay không.

Luồng nghiệp vụ được tóm tắt như sau:

> Khách đặt chuyến -> hệ thống tìm tài xế -> tài xế nhận chuyến -> khách xem thông tin chuyến -> khách tiếp tục hoặc hủy.

Mô hình được sử dụng tại giai đoạn sau khi tài xế nhận chuyến và trước khi đón khách. Output có thể là xác suất khách tiếp tục chuyến hoặc rủi ro khách hủy chuyến.

## 2. Ý nghĩa của bài toán

Khách hàng hủy sau khi tài xế đã nhận chuyến làm tài xế mất thời gian di chuyển, tài nguyên bị khóa tạm thời và hiệu quả matching giảm. Nền tảng cũng mất cơ hội phục vụ booking khác, trong khi trải nghiệm của cả khách hàng và tài xế đều bị ảnh hưởng.

Nếu phát hiện sớm booking có nguy cơ hủy cao, hệ thống có thể ưu tiên tài xế đến nhanh hơn, cải thiện thông tin hiển thị hoặc hỗ trợ điều chỉnh chiến lược matching. Vì vậy, giá trị của mô hình không chỉ nằm ở performance mà còn ở khả năng hỗ trợ quyết định vận hành.

## 3. Cách tiếp cận ML

Hướng chính là **binary classification**:

- Input: bối cảnh có sẵn tại thời điểm tài xế đã nhận chuyến.
- Output: xác suất khách hàng tiếp tục chuyến.
- Kết quả: `P(customer_continue | trip_context)` hoặc `cancel_risk = 1 - P(customer_continue)`.

LightGBM hiện là baseline với AUC khoảng 0,70. Khi có dữ liệu, có thể so sánh Logistic Regression, LightGBM, XGBoost và CatBoost. Tuy nhiên, ở giai đoạn nghiên cứu hiện tại, mục tiêu quan trọng hơn là hiểu đúng quyết định cần dự đoán, thời điểm dự đoán và cách output được sử dụng.

Một số hướng mở rộng có thể nghiên cứu sau baseline:

- **Ranking:** xếp hạng booking theo nguy cơ hủy để ưu tiên xử lý.
- **Survival analysis:** dự đoán nguy cơ hủy theo thời gian chờ.
- **Predict-then-optimize:** đưa xác suất hủy vào thuật toán matching hoặc dispatch.

Các hướng này chỉ nên triển khai khi bài toán classification cơ bản đã được định nghĩa và đánh giá rõ ràng.

## 4. Hướng nghiên cứu và thực hiện

### Bước 1 - Hiểu nghiệp vụ

Vẽ lại quy trình từ lúc khách đặt chuyến đến khi được đón. Xác định chính xác thời điểm model chạy và phân biệt rider acceptance với driver acceptance.

### Bước 2 - Viết problem statement

Problem statement cần trả lời bốn câu hỏi: mô hình dự đoán hành vi nào, dự đoán tại thời điểm nào, output được sử dụng ra sao và tác động kinh doanh kỳ vọng là gì.

### Bước 3 - Nghiên cứu tài liệu

Tập trung vào passenger cancellation, waiting-time uncertainty, hành vi khách hàng trên marketplace hai phía và cách prediction được kết hợp với dispatch optimization.

### Bước 4 - Chuẩn bị thực nghiệm

Khi có dữ liệu, mới chuyển sang định nghĩa label, kiểm tra chất lượng dữ liệu, xây baseline, so sánh model và đánh giá khả năng cải thiện so với LightGBM AUC 0,70.

## 5. Tài liệu nên đọc

1. **Customer behavioural modelling of order cancellation in coupled ride-sourcing and taxi markets**  
   https://doi.org/10.1016/j.trpro.2019.05.044  
   Nghiên cứu hành vi hủy order và ảnh hưởng của thời gian chờ.

2. **Optimal cancellation penalty for competing ride-sourcing platforms under waiting time uncertainty**  
   https://doi.org/10.1016/j.tre.2023.103107  
   Phân tích quyết định hủy khi thời gian chờ không chắc chắn và khách có lựa chọn thay thế.

3. **Dispatching optimization in ride-sharing platform based on the prediction of passenger's order cancellation**  
   https://doi.org/10.12011/SETP2024-0401  
   Liên hệ trực tiếp giữa dự đoán passenger cancellation và tối ưu dispatch.

4. **A Taxi Order Dispatch Model based on Combinatorial Optimization**  
   https://doi.org/10.1145/3097983.3098138  
   Giúp hiểu matching/dispatch là bài toán tối ưu toàn hệ thống.

5. **The Design of Centralized Matching Systems on Two-Sided Platforms**  
   https://doi.org/10.1287/mksc.2023.0561  
   Cung cấp góc nhìn về matching giữa khách hàng và tài xế trong marketplace hai phía.

6. **LightGBM Documentation**  
   https://lightgbm.readthedocs.io/

**Từ khóa tìm kiếm:** `passenger cancellation prediction ride-hailing`, `waiting time uncertainty ride-hailing`, `rider post-assignment cancellation`, `ride-hailing dispatch optimization`, `predict then optimize ride-hailing`.

## 6. Kết luận

Bài toán cần được hiểu là dự đoán khả năng khách hàng tiếp tục chuyến sau khi tài xế đã nhận. Hướng khởi đầu phù hợp là binary classification, với output là xác suất tiếp tục hoặc rủi ro hủy chuyến.

Trong giai đoạn hiện tại, trọng tâm là thống nhất problem statement, luồng nghiệp vụ và hướng nghiên cứu. Việc lựa chọn feature, định nghĩa label chi tiết và tối ưu model nên được thực hiện sau khi có dữ liệu thực tế.
