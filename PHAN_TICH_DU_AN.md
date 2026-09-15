# Phân tích dự án Real-Time Stock Prediction Pipeline

## Tổng quan

Dự án hướng tới hệ thống dự báo giá cổ phiếu theo thời gian thực, với kiến trúc dự kiến:

```text
YFinance → Avro/Kafka → Spark Streaming → PostgreSQL
                                  ↓
                         ML/LSTM inference → Dashboard
Airflow: lịch chạy và điều phối
```

Các thành phần chính:

- `src/data_ingestion`: lấy dữ liệu từ YFinance và gửi Kafka.
- `src/streaming`: Spark Streaming, ETL và ghi dữ liệu.
- `src/database`: kết nối PostgreSQL, bảng dữ liệu thô và dự báo.
- `src/machine_learning`, `src/deep_learning`: feature engineering, Random Forest, LSTM và inference.
- `src/dashboard`: Dash/Plotly dashboard.
- `dags`: Airflow DAG.
- `docker-compose.yml`: PostgreSQL, Airflow, Kafka, Spark và Jupyter.
- `tests`: unit/integration tests cho feature, model, Avro, Kafka và Spark.

## Đánh giá hiện trạng

Mã nguồn được tổ chức khá đầy đủ theo từng tầng của một data/ML pipeline. Tuy nhiên, các thành phần hiện chưa được nối thành một luồng vận hành thống nhất; nhiều phần là khung, mô phỏng hoặc có contract dữ liệu không tương thích. Vì vậy, dự án chưa nên được xem là một pipeline real-time hoàn chỉnh ở trạng thái hiện tại.

## Các vấn đề ưu tiên cao

### 1. Kafka và Spark không cùng định dạng dữ liệu

Kafka producer serialize bản tin thành **Avro binary** bằng `schemaless_writer`. Trong khi đó Spark consumer chuyển giá trị Kafka sang chuỗi rồi parse bằng `from_json`. Avro binary không thể parse bằng JSON, khiến dữ liệu parse ra `null` hoặc pipeline bị lỗi.

- Producer: `src/data_ingestion/kafka_producer/send_data.py`
- Serializer: `src/data_ingestion/kafka_producer/avro_serializer.py`
- Consumer: `src/streaming/transformations/transform_raw_data.py`

Hướng xử lý: hoặc dùng `from_avro` trong Spark với đúng Avro schema, hoặc thay producer sang JSON serialization. Cần chỉ chọn một data contract và áp dụng xuyên suốt.

### 2. Dữ liệu giá bị đưa thành 0 khi build Kafka message

`fetch_price()` trả các key chữ thường:

```python
{"open", "high", "low", "close", "volume"}
```

Nhưng `build_stock_message()` lại đọc các key chữ hoa như `Open`, `High`, `Close`. Khi có dữ liệu, `.get()` không tìm thấy key và dùng giá trị mặc định `0`.

- Nguồn dữ liệu: `src/data_ingestion/fetcher/price.py`
- Ghép Kafka message: `src/data_ingestion/main.py`

Hướng xử lý: dùng thống nhất các key chữ thường trong `build_stock_message()` và thêm unit test kiểm tra giá trị OHLCV không bị thay bằng 0.

### 3. Spark ghi vào bảng không nhất quán với database schema

Spark job mặc định ghi JDBC vào bảng `stock_prices`. Schema chính trong `src/database/init_tables.sql` lại dùng bảng `price_history`; schema streaming riêng định nghĩa bảng `stock_prices_stream`.

- Spark sink: `src/streaming/jobs/spark_consumer_job.py`
- Main schema: `src/database/init_tables.sql`
- Streaming schema: `src/streaming/schemas/postgres/stock_prices_stream.sql`

Hướng xử lý: xác định một bảng đích rõ ràng, thêm migration/DDL vào quy trình khởi tạo DB, đồng thời map cột Spark sang schema đó.

### 4. Airflow DAG chưa chạy pipeline thực tế

DAG hiện tải dữ liệu YFinance trực tiếp, truyền một summary qua XCom, sau đó chỉ in ra màn hình ở bước `save_to_database`. Nó không gọi ingestion module, Kafka, Spark, DB writer, training hoặc inference.

- DAG: `dags/stock_prediction_dag.py`

Hướng xử lý: thay các hàm mô phỏng bằng task gọi các module/script có thật; task phải fail khi write hoặc validation fail thay vì chỉ ghi log.

### 5. Training query không bind số ngày đúng cách

Training query dùng:

```sql
WHERE date >= NOW() - INTERVAL ':days days'
```

`:days` nằm trong SQL string literal, nên SQLAlchemy không bind giá trị tham số như kỳ vọng. Lỗi bị catch và pipeline fallback về dữ liệu giả; kết quả là model có thể không được train từ database production.

- File: `scripts/run_train.py`

Hướng xử lý: dùng expression có parameter thực, ví dụ `NOW() - (:days * INTERVAL '1 day')`.

### 6. Contract artifact của model chưa thống nhất

Training script lưu Random Forest chung ở `src/machine_learning/models/model.pkl`. Dashboard lại tải model theo ticker ở `src/machine_learning/artifacts/rf_model_<TICKER>.pkl`. Inference còn cố load scaler từ artifact riêng, trong khi training flow không bảo đảm tạo artifact/scaler theo cùng chuẩn.

- Training: `scripts/run_train.py`
- Inference: `src/machine_learning/inference/inference.py`
- Dashboard: `src/dashboard/callbacks.py`

Hướng xử lý: xác định một model registry/artifact contract gồm ticker, feature columns, scaler, metrics, version và model file. Training, inference và dashboard phải dùng cùng contract này.

## Vấn đề vận hành và bảo mật

### Secret và cấu hình

- Mật khẩu PostgreSQL đang hard-code ở nhiều module và scripts.
- Port database dùng lẫn `5432` và `5433`.
- Một số module dùng biến `DB_*`, trong khi Spark dùng `POSTGRES_*`.
- `docker compose config` báo nhiều biến môi trường chưa được thiết lập.

Hướng xử lý:

1. Chuyển toàn bộ secret sang biến môi trường.
2. Tạo `.env.example` không chứa credential thật.
3. Xóa secret khỏi lịch sử Git và thay đổi credential đã bị lộ.
4. Tạo một module config duy nhất dùng chung cho app, Spark, scripts và dashboard.

### Docker Compose

Compose hiện khởi chạy PostgreSQL, Airflow, Kafka, Spark và Jupyter nhưng không có service dashboard/app. Vì vậy `docker compose up -d` không tự đưa dashboard hay main application lên.

Dockerfile app cũng có đường dẫn `COPY ../../...`; đường dẫn này không hợp lệ nếu build context là `docker/app`. Dockerfile này hiện chưa được Compose sử dụng.

Hướng xử lý: thêm app/dashboard service với build context là root project hoặc sửa lại các lệnh `COPY` và context tương ứng.

### Dependencies

README nói dự án có XGBoost và PyTorch, nhưng `requirements.txt` chưa khai báo `xgboost` và `torch`. Dashboard dùng `pytz` nhưng dependency này cũng không được khai báo rõ ràng.

Airflow không được pin version trong `requirements.txt`, trong khi Docker image đã pin Airflow 2.9.3. Điều này có thể dẫn tới môi trường local và Docker khác nhau.

Hướng xử lý: tách dependency theo môi trường (`requirements/base.txt`, `requirements/dev.txt`, `requirements/airflow.txt`) hoặc pin các phiên bản tương thích rõ ràng.

## Kiểm thử

Dự án có 16 file test, bao phủ:

- feature engineering và preprocessing;
- train/inference;
- Avro schema và Kafka producer;
- Spark transformation/batch ETL;
- LSTM/Transformer.

Kết quả kiểm tra hiện tại:

```text
pytest tests/unit -q --disable-warnings --maxfail=5
```

Test dừng ở bước collection do:

- thiếu `fastavro`;
- thiếu `pyspark`;
- import `src` không ổn định do package path/test configuration.

Hướng xử lý:

1. Cài dependencies đúng môi trường test.
2. Đảm bảo project root nằm trong `PYTHONPATH` hoặc đóng gói dự án theo chuẩn Python package.
3. Dùng marker để tách test không cần service khỏi test cần DB/Kafka/Spark.
4. Thêm integration test chạy từ producer đến sink với cùng một serialization format.

## Đề xuất lộ trình sửa chữa

### Giai đoạn 1: Làm pipeline dữ liệu chạy được

1. Chuẩn hóa `.env` và module config.
2. Sửa key OHLCV giữa fetcher và producer.
3. Chọn JSON hoặc Avro rồi đồng bộ producer/Spark consumer.
4. Chọn bảng PostgreSQL đích, thêm migration và idempotency strategy.
5. Chạy end-to-end một ticker từ YFinance đến PostgreSQL.

### Giai đoạn 2: Làm ML đáng tin cậy

1. Sửa query load lịch sử giá.
2. Không fallback sang dữ liệu giả trong production mà không báo fail rõ ràng.
3. Chuẩn hóa artifact model/scaler/features/metrics theo ticker và version.
4. Dùng time-series split, theo dõi RMSE/MAE/MAPE trên holdout set.
5. Ghi `target_date` đúng ngày dự báo thay vì cùng ngày prediction.

### Giai đoạn 3: Hoàn thiện vận hành

1. Thay DAG mô phỏng bằng các task gọi pipeline thật.
2. Thêm dashboard/app vào Docker Compose.
3. Hoàn thiện healthcheck cho Kafka, Spark và dashboard.
4. Thiết lập checkpoint Spark có persistent volume.
5. Thêm logging có cấu trúc, alerting và monitoring.

## Kết luận

Project có phạm vi tốt và sẵn những lớp quan trọng của một hệ thống data engineering + ML: ingestion, Kafka, Spark, Postgres, Airflow, model và dashboard. Điểm cần đầu tư trước tiên là tính nhất quán của data contract, config và artifact model. Sau khi xử lý các vấn đề ưu tiên cao, dự án có thể tiến tới một pipeline end-to-end thực sự thay vì các module hoạt động rời rạc.
