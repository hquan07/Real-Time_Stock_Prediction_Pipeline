# Ý tưởng cải tiến Real-Time Stock Prediction Pipeline

## Mục tiêu

Các đề xuất dưới đây giúp dự án tiến từ một dashboard kỹ thuật sang sản phẩm theo dõi và phân tích cổ phiếu có trải nghiệm người dùng tốt hơn. Các hạng mục được chia thành: UI/UX, biểu đồ, mở rộng danh sách mã, và định hướng phát triển.

## 1. Cải tiến UI/UX

### Tìm kiếm và watchlist

- Thêm ô tìm kiếm ticker thay cho danh sách dropdown cố định.
- Cho phép người dùng thêm, ghim, sắp xếp và xóa mã trong watchlist cá nhân.
- Lưu watchlist trong PostgreSQL theo người dùng hoặc local storage khi chưa có authentication.
- Hiển thị giá hiện tại, phần trăm tăng/giảm, volume và trạng thái dự báo ngay trong từng item của watchlist.

### Bố cục dashboard

- Chia dashboard thành các tab rõ ràng: `Overview`, `Technical Analysis`, `Prediction`, `Compare`, `Portfolio`, `Model Health`.
- Tạo trang chi tiết cho mỗi ticker, gồm giá hiện tại, biểu đồ, chỉ báo kỹ thuật, dự báo, thông tin doanh nghiệp và dữ liệu liên quan.
- Dùng layout responsive để hoạt động tốt trên desktop, tablet và điện thoại.
- Thêm dark mode/light mode; duy trì màu sắc thống nhất giữa các biểu đồ.

### Trạng thái dữ liệu và khả năng sử dụng

- Hiển thị thời điểm cập nhật dữ liệu gần nhất, nguồn dữ liệu và trạng thái thị trường mở/đóng.
- Thêm cảnh báo khi dữ liệu cũ, ingestion lỗi hoặc dự báo chưa sẵn sàng.
- Dùng loading skeleton khi đang tải dữ liệu.
- Tạo empty state và error state rõ ràng thay vì để biểu đồ trống khi truy vấn thất bại.
- Thêm tooltip giải thích RSI, MACD, Bollinger Bands, prediction, confidence và metric model.
- Dùng cả màu, mũi tên và phần trăm cho biến động giá để không chỉ phụ thuộc vào màu xanh/đỏ.

## 2. Biểu đồ nên bổ sung

### Biểu đồ phân tích giá và kỹ thuật

| Biểu đồ | Mục đích |
|---|---|
| Giá thực tế và giá dự báo | So sánh prediction với thực tế, thấy ngay model bám xu hướng tới đâu |
| Sai số dự báo theo thời gian | Theo dõi MAE, RMSE, MAPE và phát hiện model suy giảm |
| MACD, signal line và histogram | Đánh giá động lượng và tín hiệu đảo chiều |
| RSI với mức 30/70 | Nhận biết vùng quá bán và quá mua |
| Bollinger Bands | Quan sát volatility và khả năng breakout |
| Moving-average crossover | Theo dõi giao cắt MA20/MA50 hoặc MA50/MA200 |
| Volume profile hoặc volume moving average | Đánh giá lực mua/bán đi kèm biến động giá |
| Returns distribution | Hiểu phân phối lợi nhuận, độ lệch và rủi ro |
| Drawdown | Thấy mức sụt giảm từ đỉnh, hữu ích khi đánh giá rủi ro |

### Biểu đồ so sánh và danh mục

| Biểu đồ | Mục đích |
|---|---|
| Correlation heatmap | So sánh tương quan giữa các mã trong watchlist |
| Normalized return comparison | So sánh hiệu suất nhiều mã từ cùng mốc 100 |
| Sector performance | So sánh hiệu suất theo nhóm ngành |
| Portfolio allocation | Hiển thị tỷ trọng nắm giữ theo mã/ngành |
| Portfolio P&L | Theo dõi lãi/lỗ theo ngày và tổng lũy kế |
| Portfolio vs benchmark | So sánh danh mục với SPY hoặc QQQ |
| Backtest equity curve | Đánh giá chiến lược dựa trên tín hiệu model |

### Timeframe

- Hỗ trợ `1D`, `5D`, `1M`, `3M`, `6M`, `1Y`, `5Y` cho dữ liệu daily.
- Khi có nhà cung cấp dữ liệu intraday phù hợp, bổ sung `15m`, `1h`, `4h`, `1D`.
- Hiển thị rõ timezone và thời điểm dữ liệu được cập nhật.

## 3. Mở rộng danh sách mã cổ phiếu

### Watchlist khởi đầu cho thị trường Mỹ

- Công nghệ/large-cap: `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `META`, `TSLA`, `AMD`, `NFLX`.
- Tài chính: `JPM`, `BAC`, `V`, `MA`.
- Hàng tiêu dùng/bán lẻ: `WMT`, `COST`, `KO`, `PEP`.
- Y tế: `JNJ`, `UNH`, `PFE`.
- Năng lượng: `XOM`, `CVX`.
- Công nghiệp: `CAT`, `BA`.

### ETF benchmark và phân tích thị trường

- Thị trường chung: `SPY`, `QQQ`, `DIA`, `IWM`.
- Theo ngành: `XLK`, `XLE`, `XLF`, `XLV`, `XLY`.
- Những ETF này nên được dùng làm benchmark trong dashboard, portfolio và backtest.

### Mã quốc tế và Việt Nam

- Có thể thêm ADR như `TSM`, `BABA`, `NVO`.
- Với thị trường Việt Nam, cần tích hợp nhà cung cấp dữ liệu phù hợp cho HOSE/HNX/UPCoM thay vì phụ thuộc hoàn toàn vào YFinance.
- Cần đánh giá điều khoản sử dụng, độ trễ, độ đầy đủ dữ liệu lịch sử và giới hạn API của từng data provider trước khi dùng production.

### Thiết kế kỹ thuật để thêm ticker

- Không hard-code ticker trong Python source.
- Tạo bảng `companies` hoặc `watchlist` với các trường: ticker, tên công ty, exchange, sector, active, priority, updated_at.
- Ingestion chỉ lấy những ticker có `active = true`.
- Dashboard đọc ticker từ database hoặc API, có cache và cơ chế tìm kiếm.
- Hỗ trợ bật/tắt ingestion từng mã để kiểm soát rate limit và chi phí data provider.

## 4. Định hướng model và dự báo

### Xác định mục tiêu dự báo

Trước khi phát triển sâu hơn, cần chốt một target cụ thể:

- Giá đóng cửa của ngày kế tiếp.
- Phần trăm lợi nhuận của ngày kế tiếp.
- Xác suất giá tăng/giảm trong phiên tiếp theo.
- Tín hiệu mua/giữ/bán theo một chiến lược xác định.

Nên bắt đầu bằng dự báo `next-day return` hoặc xác suất tăng/giảm; đây thường là mục tiêu phù hợp hơn khi so sánh nhiều mã có mức giá khác nhau.

### Độ tin cậy và đánh giá

- Hiển thị confidence interval hoặc prediction range, không chỉ một giá trị dự báo.
- Hiển thị metric lịch sử theo từng ticker/model: MAE, RMSE, MAPE, directional accuracy.
- Dùng walk-forward validation hoặc time-series split; không dùng random split cho chuỗi thời gian.
- Có baseline đơn giản như “ngày mai bằng hôm nay” để đánh giá model có thực sự tốt hơn baseline không.
- Không fallback sang dữ liệu giả trong production mà không ghi nhận failure rõ ràng.

### Model registry và MLOps

- Lưu model artifact theo ticker và version.
- Lưu cùng model: feature columns, scaler, thời điểm train, training dataset range, metrics và hyperparameters.
- Thêm lịch retrain định kỳ qua Airflow.
- Theo dõi data drift, model drift và cảnh báo khi metric xuống dưới ngưỡng.
- Có thể tích hợp MLflow khi cần experiment tracking/model registry đầy đủ.

## 5. Portfolio, signal và backtesting

### Portfolio tracker

- Cho phép người dùng nhập số lượng, giá vốn và ngày mua.
- Hiển thị market value, realized/unrealized P&L, tỷ trọng theo mã/ngành và hiệu suất theo thời gian.
- So sánh danh mục với benchmark như SPY hoặc QQQ.

### Alert và signal

- Cảnh báo RSI dưới 30 hoặc trên 70.
- Cảnh báo giá vượt Bollinger band.
- Cảnh báo MA crossover hoặc MACD crossover.
- Cảnh báo prediction thay đổi mạnh hoặc confidence thấp.
- Cảnh báo khi dữ liệu ngừng cập nhật hoặc ingestion bị lỗi.

### Backtesting

- Backtest mọi chiến lược mua/bán trước khi hiển thị như một signal đáng tin cậy.
- Bao gồm transaction cost, slippage, thời gian giữ lệnh và giới hạn thanh khoản nếu cần.
- Hiển thị cumulative return, Sharpe ratio, max drawdown, win rate và số giao dịch.
- So sánh với buy-and-hold cùng ticker và benchmark.

## 6. Vận hành và độ tin cậy hệ thống

- Chuẩn hóa toàn bộ config và secret qua `.env`/secret manager.
- Không hard-code database password, port, topic hoặc hostname.
- Dùng một data contract thống nhất giữa Kafka producer và Spark consumer: JSON hoặc Avro.
- Dùng checkpoint Spark persistent volume.
- Thiết kế idempotency/deduplication theo `ticker` và `timestamp`.
- Thêm retry, dead-letter queue và alert cho Kafka message lỗi.
- Dùng migration cho PostgreSQL thay vì phụ thuộc vào database dump.
- Thêm dashboard/app service vào Docker Compose.
- Pin dependency và tách dependency development, application và Airflow.

## 7. Kiểm thử nên bổ sung

- Unit test cho data contract từ fetcher đến Kafka payload.
- Integration test producer → Kafka → Spark → PostgreSQL.
- Test schema migration và database upsert/idempotency.
- Test training/inference với artifact model thật.
- Snapshot/visual test cho các biểu đồ dashboard quan trọng.
- Tách test unit, integration và end-to-end bằng pytest markers.

## Thứ tự triển khai khuyến nghị

### Giai đoạn 1: Nền tảng dữ liệu

1. Chuẩn hóa config, environment variables và secret.
2. Sửa contract fetcher → Kafka và Kafka → Spark.
3. Thống nhất database schema/table đích.
4. Chạy thành công end-to-end cho một ticker.

### Giai đoạn 2: Dashboard hữu ích

1. Watchlist động và tìm kiếm ticker.
2. Trang overview/detail cho ticker.
3. Actual vs predicted, RSI, MACD, Bollinger Bands.
4. Trạng thái dữ liệu, loading và error state.

### Giai đoạn 3: Chất lượng model

1. Chuẩn hóa target, artifact và model version.
2. Thêm baseline, walk-forward validation và metric.
3. Dự báo có confidence range.
4. Retraining và model monitoring.

### Giai đoạn 4: Tính năng nâng cao

1. So sánh nhiều ticker và correlation heatmap.
2. Portfolio tracker.
3. Alerts.
4. Backtesting và benchmark.

## Lưu ý

Hệ thống cần hiển thị disclaimer rõ ràng: dữ liệu và dự báo chỉ phục vụ mục đích nghiên cứu/tham khảo, không phải khuyến nghị đầu tư hoặc tư vấn tài chính cá nhân.
