# Kế hoạch nâng cấp — Real-Time Stock Prediction Pipeline

## 1. Mục tiêu và phạm vi

Mục tiêu là chuyển dự án từ một MVP phục vụ học tập/demo thành **hệ thống phân tích và phát tín hiệu đầu tư ở mức pilot nội bộ**. Pilot không được đặt lệnh giao dịch thật; mọi tín hiệu phải được backtest, paper-trade và có khả năng truy vết đầy đủ trước khi cân nhắc bất kỳ tích hợp broker nào.

### Các mốc trưởng thành

| Mốc | Kết quả mong muốn | Không bao gồm |
| --- | --- | --- |
| M0 — An toàn & tái lập | Không lộ secret, môi trường chạy tái lập, CI có tín hiệu đáng tin | Thay đổi chiến lược đầu tư |
| M1 — Data platform đúng | Luồng market data có hợp đồng, chất lượng và idempotency | Giao dịch tự động |
| M2 — ML đáng tin | Dataset point-in-time, validation time-series, registry/promotion rõ ràng | Khẳng định lợi nhuận |
| M3 — Signals có thể đánh giá | Tín hiệu, backtest và paper portfolio cùng một quy tắc | Khuyến nghị đầu tư công khai |
| M4 — Pilot vận hành | Dashboard, giám sát, cảnh báo và runbook hoàn chỉnh | Production auto-trading |

### Nguyên tắc bắt buộc

- Không dùng dữ liệu, feature, scaler hoặc model được tạo sau thời điểm dự báo.
- Mọi ghi dữ liệu phải idempotent, có khóa định danh và có thể replay.
- Tách **raw/bronze**, **validated/silver**, **features/gold** và dữ liệu phục vụ dashboard.
- Không để token/mật khẩu trong Git, log hay Docker image.
- Backtest là công cụ đánh giá, không phải bằng chứng về lợi nhuận tương lai.

---

## 2. Roadmap theo pha

| Pha | Thời lượng ước tính | Phụ thuộc | Kết quả/gate kết thúc |
| --- | ---: | --- | --- |
| P-0. Preflight | Trước tuần 1 | — | Secrets được khoanh vùng, backup/baseline/ADR/migration manifest đã được xác nhận |
| 0. Baseline & security | Tuần 1 | — | Secrets đã xoay vòng; CI, compose và inventory có thể tái lập |
| 1. Data contract & storage | Tuần 2–3 | Pha 0 | Một event có schema/version/key; dữ liệu raw và clean truy vết được |
| 2. Streaming & orchestration | Tuần 4–5 | Pha 1 | Consumer chạy liên tục, checkpoint bền vững, replay không tạo bản ghi trùng |
| 3. ML correctness & evaluation | Tuần 6–8 | Pha 1 | Không leakage; walk-forward evaluation, champion/challenger và quality gate |
| 4. Signal, risk & backtest | Tuần 9–10 | Pha 3 | Signal có rule rõ, chi phí giao dịch, benchmark và paper-trading |
| 5. Product & observability | Tuần 11–12 | Pha 2–4 | Dashboard dùng dữ liệu thật, alerting, runbook và UAT pilot |

Thời lượng là ước tính cho một nhóm 1–2 kỹ sư làm bán thời gian. Không bắt đầu pha kế tiếp khi gate của pha trước chưa đạt.

Chi tiết thực thi và các điều kiện còn chặn P0 được ghi tại [preflight README](preflight/README.md).

**Trạng thái 2026-11-26:** phần P0 thuộc phạm vi repository đã hoàn tất và có
bằng chứng tại [P0 implementation status](p0-implementation-status.md). Owner
xác nhận đã thu hồi Finnhub/CloudVault credential cũ; mật khẩu PostgreSQL cũ
đã bị từ chối qua kết nối ứng dụng. Owner chọn giữ lịch sử Git hiện tại sau
khi thu hồi, và nhánh P0 được đưa lên remote để review. Pilot vẫn phải qua
gate của các pha tiếp theo.

### Quy ước branch theo pha

Mỗi pha được triển khai trên một branch riêng, tạo từ `main` hoặc branch của pha trước sau khi gate được phê duyệt. Tên branch dùng mẫu `upr/p<phase>-<muc-tieu-ngan>`; chỉ merge khi toàn bộ tiêu chí nghiệm thu của pha đó đạt. Không dùng branch pha mới để sửa trực tiếp công việc của pha trước.

| Pha | Branch chính | Điểm bắt đầu | Điều kiện merge |
| --- | --- | --- | --- |
| P-0 Preflight | `upg/p0-preflight` *(đã tồn tại, theo tên cũ)* | `main` baseline | Go/no-go preflight được owner xác nhận |
| P0 Baseline & security | `upr/p0-foundation` | Tag/commit preflight đã xác nhận | Secrets, repository hygiene, CI và smoke test đạt gate |
| P1 Data contract & storage | `upr/p1-data-contract` | `main` sau P0 | Contract, migrations, quality gate và replay test đạt |
| P2 Streaming & orchestration | `upr/p2-streaming` | `main` sau P1 | Checkpoint/recovery/idempotency và DAG schedule đạt |
| P3 ML correctness & evaluation | `upr/p3-ml-validation` | `main` sau P1; chỉ ghép P2 khi cần runtime | Walk-forward, baseline, lineage và promotion gate đạt |
| P4 Signal, risk & backtest | `upr/p4-signal-backtest` | `main` sau P3 | Backtest/review/paper-risk gate đạt |
| P5 Product & observability | `upr/p5-pilot-operations` | `main` sau P2–P4 | UAT, monitoring, backup/restore và pilot go/no-go đạt |

Mỗi branch pha phải có: issue/decision record liên kết, checklist phạm vi, test evidence và rollback commit. Khi cần sửa khẩn cấp sau merge, dùng `upr/fix-<muc-tieu>` từ `main`, không mở lại branch pha đã merge.

---

## 3. Pha 0 — Baseline, bảo mật và khả năng tái lập

### 3.0 P0 — Tái cấu trúc repository (triển khai đầu tiên)

Phần này phải hoàn tất **trước** khi sửa logic data/ML. Mục tiêu là tạo ranh giới ownership rõ ràng, không phải đổi tên hàng loạt.

**Hiện trạng cần xử lý:** dashboard có ba hướng song song (Dash Python, FastAPI và React/Vite); logic production bị chia lẫn giữa `src/`, `scripts/`, `dags/`; config/dependency bị trùng; virtualenv, `__pycache__`, logs, notebook checkpoints và model artifacts runtime đang bị tracked.

```text
apps/web/                         # React/Vite dashboard duy nhất
services/api/                     # FastAPI: routers, services, schemas, auth
packages/pipeline/src/stock_pipeline/
  config/ ingestion/ streaming/ data_quality/ features/ ml/ signals/ db/ integrations/ cli/
orchestration/airflow/dags/       # chỉ có DAG, không có business logic
infrastructure/compose/ docker/ migrations/
tests/unit/ integration/ e2e/ contract/
docs/  notebooks/  data/          # data là Git-ignored runtime/local data
```

**Thứ tự thực hiện an toàn:**

1. Lập inventory mọi entrypoint Compose, Airflow và README gọi; gắn owner cho chúng.
2. Chọn React + FastAPI là dashboard chính. Đóng băng Dash, chỉ xóa ở PR riêng sau smoke test React/API.
3. Đưa business logic từ `scripts/` vào `stock_pipeline`; script/DAG chỉ là adapter hoặc CLI mỏng.
4. Hợp nhất hai cây config, loại absolute paths; mọi host/port/path đi qua một settings layer.
5. Chuyển DDL dump sang migrations versioned; tách seed/demo data khỏi schema.
6. Dọn Git index theo PR nhỏ, không xóa local data chưa sao lưu: virtualenv/cache/log/checkpoint/model runtime phải bị ignore; artifact validated đi qua registry/object storage.
7. Hợp nhất Python dependencies thành project/lock file, giữ lock file frontend riêng; cập nhật Compose, CI và README sau mỗi move.

**Tiêu chí nghiệm thu:** root repo không còn runtime files bị tracked; một dashboard runtime và một Python configuration source; không còn absolute project path/hard-coded `localhost` trong application code; Compose config, unit tests, frontend build và API/dashboard smoke test pass sau từng bước.

### 3.1 Secrets và cấu hình

**Hiện trạng cần khắc phục:** `.env` đang được theo dõi bởi Git và mã có fallback credentials.

- Xoay vòng ngay toàn bộ password, token API, token CloudVault và tài khoản quản trị đã từng xuất hiện trong working tree/Git history.
- Gỡ `.env` khỏi index và lịch sử Git theo quy trình được phê duyệt; chỉ giữ `.env.example` với placeholder an toàn.
- Tạo lớp cấu hình duy nhất bằng Pydantic Settings hoặc tương đương; fail-fast nếu thiếu biến bắt buộc ở môi trường staging/production.
- Bỏ mọi fallback password/default admin; dùng Docker secrets hoặc secret manager (Vault, AWS Secrets Manager, GCP Secret Manager) theo môi trường.
- Phân tách `dev`, `test`, `staging`, `pilot`; không tái sử dụng database hay credentials giữa các môi trường.

**Tiêu chí nghiệm thu:** secret scan không tìm thấy credential thật; ứng dụng không khởi động nếu secret bắt buộc thiếu; `.env` mới không xuất hiện trong `git ls-files`.

### 3.2 Chuẩn hóa dependency và repository hygiene

- Chọn một cách đóng gói: `pyproject.toml` + lock file (khuyến nghị) hoặc requirements có constraint file; pin Airflow bằng constraint chính thức theo Python version.
- Tách dependency runtime: ingestion, Spark, Airflow, ML, dashboard, development/test. Không cài `apache-airflow` không pin trong requirements tổng.
- Bổ sung `Makefile`/task runner: `bootstrap`, `lint`, `test-unit`, `test-integration`, `compose-up`, `seed-demo`, `smoke-test`.
- Git-ignore logs, model artifacts sinh ra, cache Spark, checkpoints cục bộ, coverage, dashboard virtualenv và data dump. Không xóa dữ liệu hiện hữu trong pha này nếu chưa được chủ sở hữu phê duyệt.
- Bổ sung pre-commit: Ruff, formatter, type checker, YAML/JSON validation, detect-secrets, dependency audit.

**Tiêu chí nghiệm thu:** clone mới theo Quick Start chạy được bằng một lệnh documented; CI chạy lint + unit test; repository không tăng vì log/artifact runtime.

### 3.3 CI/CD và image hardening

- GitHub Actions/GitLab CI: validate Compose, unit tests, build frontend/backend, secret scan, SCA, image scan.
- Build immutable image có tag Git SHA; không mount source code vào môi trường pilot.
- Chạy container non-root khi image hỗ trợ; thêm healthcheck cho Kafka, Spark, MLflow, API, Redis và frontend.
- Không expose PostgreSQL, Redis, Kafka, MLflow ra host ngoài nhu cầu development; đặt network boundaries.

**Tiêu chí nghiệm thu:** CI đỏ khi test/lint/secret scan fail; smoke test container thành công trên môi trường sạch.

---

## 4. Pha 1 — Data ingestion, contract và database

### 4.1 Xác định nguồn dữ liệu và SLA

- Giữ YFinance chỉ cho historical research/dev vì độ trễ, giới hạn và tính ổn định không phù hợp làm market-data production.
- Chọn provider có license phù hợp cho pilot: xác định asset universe, exchange, độ phân giải (EOD/1 phút/tick), timezone, corporate actions, quotas và chi phí.
- Nếu chưa có provider trả phí, xác định rõ sản phẩm là **EOD/near-real-time research**, không gắn nhãn realtime.
- Định nghĩa data SLA: freshness, completeness, duplicate rate, accepted late arrival, retention và recovery-point/recovery-time objective.

**Tiêu chí nghiệm thu:** tài liệu provider/license/SLA được chấp thuận; dashboard hiển thị timestamp nguồn và độ trễ thực tế.

### 4.2 Event contract và Kafka

- Chuẩn hóa một Avro/Protobuf event `market_price.v1` gồm `event_id`, `symbol`, `event_time_utc`, `ingested_at_utc`, OHLCV, `source`, `schema_version`, quality flags.
- Quy định key Kafka là `symbol`; partition count và retention theo throughput; dùng Schema Registry và compatibility `BACKWARD`.
- Tách topic raw, validated, dead-letter. Không đưa earnings/fundamentals không cùng frequency vào price event; tạo contract riêng.
- Producer cần retry có bounded backoff, delivery callback, metrics và DLQ cho event không hợp lệ.
- Định nghĩa chính sách late/out-of-order events và dedup key `(source, symbol, event_time, interval)`.

**Tiêu chí nghiệm thu:** compatibility test chạy trong CI; event invalid vào DLQ; producer replay cùng file không tăng logical record count.

### 4.3 Storage tiers và schema evolution

- **Bronze:** append-only raw events, payload gốc, source metadata và ingest timestamp.
- **Silver:** event validated/deduplicated/canonicalized; primary/unique key rõ ràng; UTC trong storage.
- **Gold:** bar/features đã versioned, chỉ gồm dữ liệu biết được tại `as_of_time`.
- Tạo Alembic migrations thay cho database dump như cơ chế deploy schema. Tạo index theo `(symbol, event_time)` và partition theo thời gian khi volume yêu cầu.
- Áp dụng upsert/idempotency; ghi raw và processed không dùng `toPandas()` hoặc `to_sql()` không kiểm soát transaction ở đường streaming.
- Lưu data quality results cùng batch/run ID.

**Tiêu chí nghiệm thu:** migration có thể chạy trên database trống; replay 24 giờ dữ liệu không duplicate; có lineage raw → silver → gold cho một symbol/time cụ thể.

### 4.4 Data quality

- Kiểm tra schema, null, range (giá > 0, volume ≥ 0), OHLC consistency, duplicate, market-calendar, freshness và missing bars.
- Thêm Great Expectations/Soda hoặc bộ validator nội bộ có báo cáo persisted.
- Data fail theo ngưỡng phải block feature/training/publish signal, không silently fallback sang dữ liệu giả.

**Tiêu chí nghiệm thu:** bộ test cố ý inject bad events xác nhận chúng bị quarantine; failed quality check ngăn model/signal promotion.

---

## 5. Pha 2 — Streaming, Spark và Airflow

### 5.1 Spark Structured Streaming

- Tạo service/job deployment rõ ràng cho `spark_consumer_job`; Compose chỉ phù hợp dev, pilot dùng Spark standalone/k8s được quản lý với command submit rõ ràng.
- Đặt checkpoint trên persistent volume/object storage; checkpoint path được version theo job/environment, không nằm `/tmp`.
- Dùng watermark/event-time window khi phù hợp; quyết định output semantic (at-least-once + idempotent sink là thực tế cho PostgreSQL).
- Thay Pandas trong `foreachBatch` bằng Spark JDBC/bulk upsert staging table; xử lý từng ticker/window trong Spark hoặc pandas UDF có giới hạn rõ ràng.
- Quan sát consumer lag, processed rows, batch latency, failed batches, DLQ count và checkpoint age.

**Tiêu chí nghiệm thu:** kill/restart consumer không mất/nhân đôi dữ liệu; replay một ngày dữ liệu đạt duplicate rate 0; load test đạt data SLA.

### 5.2 Airflow orchestration

- Phân tách DAG: ingestion/backfill, data quality, feature materialization, training, evaluation/promotion, signal generation/reporting.
- Dùng timezone exchange hoặc UTC được ghi rõ; lịch US market chỉ chạy trading days và sau giờ đóng cửa tương ứng, không chạy signal cuối tuần.
- Không train model mỗi ngày chỉ vì có scheduler: trigger dựa trên new data window, retraining policy và drift/performance threshold.
- Mỗi task nhận `logical_date`, `run_id`, `data_cutoff`; output path/table có run metadata để rerun không overwrite im lặng.
- Cấu hình retries theo loại lỗi, alert khi DAG SLA miss, và task-level logs structured.

**Tiêu chí nghiệm thu:** 3 lần rerun cùng logical date có kết quả idempotent; dry-run/backfill lịch sử tách biệt production data; scheduler đúng market calendar.

### 5.3 Disaster recovery và runbooks

- Document khởi động, dừng, replay topic, restore DB, restore model, rollback model và xử lý DLQ.
- Backup PostgreSQL, registry/model artifacts, MLflow metadata; kiểm thử restore định kỳ.
- Có ownership/on-call rõ cho mỗi service và severity cho incident.

**Tiêu chí nghiệm thu:** tabletop incident và một restore drill hoàn thành trong RTO/RPO đã công bố.

---

## 6. Pha 3 — Feature engineering, ML và MLOps

### 6.1 Sửa correctness của feature/target

- Tất cả `shift`, rolling mean, volatility, lag và target phải `groupby(symbol)` rồi sort theo `event_time`; không để tickers ảnh hưởng lẫn nhau.
- Tạo target bằng future return của **cùng symbol**; chỉ drop NaN sau khi tạo feature/target và trước split.
- Chỉ fit scaler trên train window; transform validation/test/inference bằng scaler của đúng model version.
- Định nghĩa feature contract: names, dtypes, units, lookback, null policy, source, version. Validate contract trước inference.
- Tách feature online (available at inference) và offline (research); parity test hai cách tính trên cùng `as_of_time`.

**Tiêu chí nghiệm thu:** test multi-symbol phát hiện cross-symbol shift; test last target NaN; online/offline feature parity đạt 100% trên fixture.

### 6.2 Evaluation đúng time-series

- Thay split 80/20 đơn bằng rolling/expanding walk-forward validation, có embargo/purge nếu horizon overlap.
- Chọn baseline bắt buộc: zero return, previous close, moving average; model mới chỉ được chấp nhận nếu vượt baseline nhất quán.
- Báo cáo RMSE/MAE trên return và direction accuracy, calibration, per-symbol/per-regime metrics; không chỉ đánh giá trên price level.
- Freeze train/validation/test periods; test cuối chỉ mở một lần khi quyết định release candidate.
- Theo dõi data drift, prediction drift, feature missingness và post-deployment performance khi actual labels đến.

**Tiêu chí nghiệm thu:** report tái lập từ run ID; không có future leakage; model candidate chỉ promoted khi vượt baseline và quality thresholds được product/risk owner phê duyệt.

### 6.3 Model registry và artifact governance

- Giữ local registry/MLflow/CloudVault với vai trò không chồng lấn: MLflow tracking + metrics, registry cho state/promotion, CloudVault cho catalog/durable artifact nếu cần.
- Artifact bundle gồm model, scaler, feature contract, source-data fingerprint, code SHA, environment lock, metrics, checksums và approval record.
- States: `candidate` → `validated` → `paper` → `champion` → `archived`; activation atomic, rollback một bước.
- Không load pickle từ nguồn không tin cậy. Với artifact tải về, tiếp tục checksum verify; cân nhắc format an toàn hơn khi khả thi.

**Tiêu chí nghiệm thu:** bất kỳ prediction nào truy ngược được tới exact model/data/code; failed validation không thể activate model.

### 6.4 Phạm vi model

- Bắt đầu bằng linear/regularized + Random Forest làm baseline; chỉ giữ XGBoost/LSTM nếu walk-forward chứng minh giá trị sau chi phí/complexity.
- Không gọi LSTM là hoàn thiện nếu inference/model serialization/metrics chưa cùng pipeline Random Forest.
- Tối ưu hyperparameter trong nested time-series CV, không trên test holdout.

**Tiêu chí nghiệm thu:** model card cho từng champion, bao gồm intended use, limitations, training period, universe, performance và monitoring threshold.

---

## 7. Pha 4 — Signal engine, backtesting và risk controls

### 7.1 Signal engine

- Tính RSI/MACD/indicator từ chuỗi gold features thật, không dùng default constant. Rule phải versioned và chứa `signal_time`, `data_cutoff`, `model_version`, `feature_version`.
- Tách forecast (predicted return + uncertainty) khỏi decision policy (BUY/SELL/HOLD) để có thể đánh giá độc lập.
- Confidence phải xuất phát từ calibration/uncertainty đã đánh giá, không dùng công thức heuristic.
- Thêm signal validity: stale data, missing features, out-of-distribution, model stale, market closed; trạng thái an toàn là `NO_SIGNAL`, không phải HOLD gây hiểu nhầm.

**Tiêu chí nghiệm thu:** tất cả signal có explanation, version/lineage; rule test cover buy/sell/hold/no-signal và trường hợp data lỗi.

### 7.2 Backtesting engine

- Xây event-driven backtest dùng cùng bar/data cutoff với production; tránh look-ahead, survivorship bias và corporate-action errors.
- Tính commission, bid-ask spread/slippage, delay execution, liquidity/volume cap, position limit, cash, lot sizing, market calendar và delisting policy.
- So sánh với buy-and-hold và baseline signals trên từng symbol/universe/regime.
- Báo cáo CAGR, annualized volatility, Sharpe/Sortino, max drawdown, turnover, hit rate, exposure, tail loss và confidence interval/bootstrap.
- Thực hiện sensitivity analysis cho cost/slippage/latency; không promote strategy nếu chỉ thắng khi giả định chi phí bằng 0.

**Tiêu chí nghiệm thu:** một run backtest tái lập từ data/model/rule versions; report cho nhiều market regimes; review độc lập phê duyệt assumptions.

### 7.3 Paper trading và risk

- Chạy paper portfolio ít nhất 8–12 tuần hoặc qua đủ số phiên/giai đoạn thị trường đã thỏa thuận trước.
- Thiết lập daily loss limit, maximum exposure theo symbol/sector, kill switch, stale-data guard, order-size limits và manual approval.
- Ghi audit log mọi signal, decision, hypothetical order/fill, override và incident.

**Tiêu chí nghiệm thu:** paper result khớp backtest trong tolerance giải thích được; kill switch, stale-data block và manual override được diễn tập.

---

## 8. Pha 5 — Dashboard, API và trải nghiệm người dùng

### 8.1 Backend/API

- Chuẩn hóa REST/WebSocket API versioned, OpenAPI contract, validation, pagination và error format.
- Thêm authentication/authorization phù hợp phạm vi pilot; rate limiting, CORS allowlist, audit logging và Redis cache có TTL/invalidation rõ ràng.
- `/health` phân biệt liveness/readiness/dependency health, không chỉ phản hồi process alive.
- API chỉ trả dữ liệu thật có `as_of_time`, source và freshness; gắn nhãn demo/mock khi chưa có dữ liệu thật.

**Tiêu chí nghiệm thu:** API contract tests, authz tests và load test cơ bản pass; unauthenticated request không đọc portfolio/signal nhạy cảm.

### 8.2 Frontend/dashboard

- Hợp nhất định hướng UI: React/FastAPI là dashboard chính; Dash cũ chỉ giữ nếu có owner/use case rõ ràng, nếu không deprecate có kế hoạch.
- Thay architecture health toggle/mock alerts bằng trạng thái lấy từ API; price alert phải persist và được xử lý server-side.
- Hiển thị: data freshness, provider/source, market status, model/rule version, uncertainty, no-signal reason và disclaimer “not investment advice”.
- Tách dashboard research, signal monitor và paper-portfolio; không trình bày forecast như cam kết lợi nhuận.
- Thêm accessibility, responsive behavior, empty/error/loading state, frontend tests và E2E critical journeys.

**Tiêu chí nghiệm thu:** UI không còn hiển thị số liệu/mock health như production data; E2E dashboard smoke test pass; UAT pilot được sign-off.

---

## 9. Observability, SLO và vận hành pilot

### Chỉ số cần thu thập

| Lớp | Chỉ số |
| --- | --- |
| Ingestion | provider latency/error, rate-limit, events received, event freshness, invalid/DLQ count |
| Kafka/Spark | consumer lag, batch duration, throughput, restart count, checkpoint age, duplicate/replay rate |
| Database | connection pool, query latency, locks, storage growth, backup success |
| ML | training duration, evaluation gate, model age, feature missingness/drift, prediction distribution |
| Signal | signals by type, no-signal reason, stale-data blocks, model/rule version, publish success |
| API/UI | request latency/error, websocket connections, frontend error, data freshness displayed |

### SLO ban đầu cần chốt trước pilot

- Freshness cho mỗi data tier.
- Tỷ lệ event hợp lệ và duplicate rate.
- Tỷ lệ DAG/task thành công và thời gian recovery.
- Prediction/signal availability chỉ khi data/model quality đạt chuẩn.
- RTO/RPO database, artifact registry và checkpoints.

Triển khai Prometheus/OpenTelemetry + Grafana hoặc stack tương đương. Alert phải có owner, severity, suppression rule và link runbook.

**Tiêu chí nghiệm thu:** dashboard vận hành và tối thiểu các alert dữ liệu stale, streaming failure, model gate failure, DB backup failure và API error budget breach hoạt động trên môi trường pilot.

---

## 10. Backlog kỹ thuật ưu tiên

### P0 — Phải hoàn thành trước bất kỳ pilot signal nào

1. **Tái cấu trúc repository theo mục 3.0:** inventory entrypoint, chọn React/FastAPI, tách package/domain, dọn runtime files và hợp nhất config/dependencies. Đây là việc triển khai đầu tiên.
2. Rotate secret, gỡ `.env` khỏi Git/history, xóa hard-coded/default credentials.
3. Sửa feature engineering theo `symbol`, xử lý NaN target và scaler train-only.
4. Thay fallback sample data trong train/inference bằng fail-safe rõ ràng; không cho publish nếu không có validated data.
5. Xác định/triển khai data contract, UTC, market calendar, dedup key và migrations.
6. Khởi chạy Spark consumer độc lập với persistent checkpoint và idempotent sink.
7. Bổ sung walk-forward evaluation + baseline + report artifact.
8. Tắt/đánh dấu mọi signal confidence/RSI giả lập; không public BUY/SELL cho đến khi có backtest.

### P1 — Cần có trước khi mở pilot nội bộ

1. CI, test matrix, image scan, pinned dependencies và smoke test Compose.
2. Data-quality gates/DLQ/replay tooling.
3. Model registry promotion states, model cards, lineage và rollback drill.
4. Backtest gồm costs/slippage; paper portfolio + risk limits.
5. Metrics, dashboards, alerting, backups và runbooks.
6. API auth, frontend real-data contract, persistent alerts/watchlist.

### P2 — Chỉ làm khi có bằng chứng nhu cầu

1. XGBoost/LSTM tuning và feature store chuyên dụng.
2. Kubernetes/managed Kafka/Spark hoặc multi-region DR.
3. Broker integration; phải tách thành dự án mới có legal/compliance/risk review độc lập.
4. Multi-user portfolio, billing hoặc public-facing product features.

---

## 11. Vai trò và governance

| Vai trò | Trách nhiệm chính |
| --- | --- |
| Data engineer | Contract, Kafka/Spark, storage tiers, data quality, replay |
| ML engineer | Features, validation, registry, drift, model card |
| Backend/platform engineer | API, CI/CD, secrets, database, observability, backups |
| Product/risk owner | Universe, signal policy, quality thresholds, backtest assumptions, pilot go/no-go |
| Reviewer độc lập | Kiểm tra leakage, backtest assumptions, security và release gate |

Mọi quyết định promote model/rule, thay data provider, thay quality threshold hoặc chuyển từ paper sang live phải được ghi thành decision record với owner và ngày hiệu lực.

---

## 12. Definition of Done cho pilot

Pilot chỉ được tuyên bố sẵn sàng khi tất cả điều kiện dưới đây đạt:

- Không còn secret thật trong Git/history hiện hành; secret scan và dependency scan pass.
- Môi trường staging/pilot tái lập từ IaC/Compose và image pin theo Git SHA.
- Dữ liệu có contract, quality gates, lineage và replay idempotent đã được kiểm thử.
- Streaming recovery từ checkpoint/backups đã được diễn tập.
- ML không leakage, đã qua walk-forward + untouched holdout; vượt baseline theo ngưỡng được phê duyệt.
- Signal rule, confidence, data freshness và no-signal policy có test và có version.
- Backtest bao gồm realistic costs; paper trading đạt tiêu chí đã chốt trước.
- Dashboard/API dùng dữ liệu thật, có auth phù hợp và minh bạch freshness/model version.
- Observability, alerts, backup/restore và runbook có owner và đã diễn tập.
- Có biên bản go/no-go từ product/risk owner, ghi rõ giới hạn: **research/paper use only, not investment advice, no auto-execution**.

---

## 13. Phụ lục — Đánh giá cấu trúc repository và UI/UX dashboard

Đánh giá dựa trên mã nguồn, Compose, tài liệu và các file đang được Git theo dõi. Không thay thế cho usability test với người dùng thật hoặc kiểm thử dashboard đang chạy.

### 13.1 Kết luận

- **Cấu trúc repository: 30/100.** Domain kỹ thuật đã có, nhưng boundary bị vỡ bởi code trùng, runtime files bị commit và entrypoint cạnh tranh nhau.
- **Nền tảng visual UI: 65/100.** Dark theme, typography, card và biểu đồ nhất quán, phù hợp một dashboard market-monitoring.
- **UI/UX sẵn sàng pilot: 35/100.** Điều hướng/màn hình cơ bản đã có nhưng độ tin cậy dữ liệu, error state, responsive và accessibility chưa đủ cho công cụ hỗ trợ quyết định.

### 13.2 Cấu trúc hiện tại

#### Điểm tốt

- Có các domain cần thiết: ingestion, Kafka, Spark, database, ML/DL, Airflow, integrations, tests và dashboard frontend/backend.
- `tests/unit` và `tests/integration` đã được tách ở mức thư mục.
- Dockerfiles tách cho Spark, Airflow, API và frontend.
- CloudVault integration được cô lập tương đối trong `integrations/`.

#### Vấn đề P0

| Phát hiện | Tác động | Ưu tiên |
| --- | --- | --- |
| 8.809 file virtualenv dashboard, 3.442 file Python cache, 111 file log đang tracked | Repo phình, diff và CI không tái lập | P0 |
| Dash (`src/dashboard`) song song React/FastAPI (`src/dashboard/frontend`, `backend`) | Hai dashboard, tài liệu và runtime mâu thuẫn | P0 |
| `scripts/` chứa business logic, `dags/` gọi script trực tiếp | Khó unit test, versioning, reuse và deploy | P0 |
| Hai cây config (`src/config`, `src/data_ingestion/config`) | Defaults/credentials dễ lệch theo entrypoint | P0 |
| `init_tables.sql` là dump lẫn DDL, không có migrations | Schema deploy/review/rollback không đáng tin | P0 |
| Model/artifact trong source tree và runtime artifact xuất hiện Git status | Lẫn code với data/model; release không tái lập | P0 |
| Nhiều `requirements*.txt` chồng lấn, Airflow thiếu constraint pin đầy đủ | Dependency conflicts giữa Python, Airflow và ML | P0 |
| Absolute local paths và `localhost` trong application code | Không portable giữa local, Docker, staging và pilot | P0 |

Áp dụng cấu trúc đích và thứ tự migrate ở mục **3.0**. Không trộn refactor folder với thay đổi business rule hoặc data model trong cùng một PR.

### 13.3 Đánh giá UI/UX dashboard

#### Điểm mạnh

- Dark theme, accent indigo/violet, card và biểu đồ dùng cùng ngôn ngữ thị giác.
- Overview, Technical, ML Predictions, Portfolio và Alerts phản ánh các câu hỏi chính của người dùng theo dõi thị trường.
- Candlestick, volume, prediction-vs-actual và feature importance là lựa chọn trực quan phù hợp.
- Có disclaimer rằng ML prediction chỉ để tham khảo.

#### Vấn đề UX nghiêm trọng

| Vấn đề | Hệ quả | Ưu tiên |
| --- | --- | --- |
| Landing page mặc định là Architecture | Người dùng market-monitoring bắt đầu ở màn hình kỹ thuật thay vì thông tin hữu ích | P0 |
| Architecture dùng metrics giả và double-click đổi health | Tạo cảm giác sai rằng hệ thống có observability/health thật | P0 |
| Price Alerts chỉ là React local state với 2 alert seed | Refresh làm mất dữ liệu; Pause/Delete không phải hành động thật | P0 |
| API tự tạo confidence interval ±5% | Người dùng có thể hiểu nhầm mức độ bất định của model | P0 |
| API/WebSocket URLs hard-code `localhost` | Lỗi khi deploy qua reverse proxy, staging hoặc dùng thiết bị khác | P0 |
| Error state chủ yếu chỉ `console.error` | Người dùng không biết API lỗi, dữ liệu stale hay chưa có data | P0 |
| Portfolio cố định `default_user` | Không có user context, không phù hợp dữ liệu người dùng thật | P1 |
| Header tabs không có semantic tab/focus/keyboard behavior | Accessibility và điều hướng keyboard yếu | P1 |
| Layout có min-width/6 cột cố định | Rủi ro overflow trên tablet/mobile | P1 |
| Thiếu source, freshness, timezone và model version | Không thể đánh giá dữ liệu thuộc phiên nào hay model nào | P0 |

#### Luồng UX đích cho pilot

1. **Market overview mặc định:** market status, source/freshness, symbol selector, giá gần nhất, độ trễ và biểu đồ. Khi data lỗi/stale, hiển thị trạng thái đó thay vì số 0.
2. **Forecast & signal:** forecast, uncertainty đã hiệu chỉnh, model/rule/data version, `as-of` time và lý do có/không có signal. Không dùng màu xanh/đỏ như một lời khuyên giao dịch.
3. **Research:** indicator, historical performance và feature importance có timeframe/source rõ ràng.
4. **Paper portfolio:** chỉ hiển thị cho người dùng xác thực, có P&L/exposure/cash/timestamp.
5. **Alerts:** persisted qua API, có validation, xác nhận thao tác, notification channel và audit history.
6. **System status:** admin-only, lấy metrics thật từ observability; UI không được phép đổi trạng thái service.

#### Backlog UI/UX

**P0 — làm cùng repository foundation**

1. Đặt Overview làm mặc định; chuyển Architecture thành admin-only hoặc bỏ khỏi pilot.
2. Xóa mock health/metrics và alert seed; hiển thị empty state trung thực cho đến khi backend có dữ liệu thật.
3. Tạo `API_BASE_URL`/`WS_BASE_URL` từ Vite environment và dùng same-origin reverse proxy khi deploy.
4. Chuẩn hóa loading, empty, error, retry và stale-data state cho mọi query; thêm error boundary/toast policy.
5. Hiển thị source, last updated, exchange timezone, market status, model/rule/data version ở Overview/Forecast.
6. Không hiển thị confidence interval nếu backend chưa trả uncertainty đã được đánh giá; dùng `Unavailable` thay vì ước lượng giả.
7. Gắn nhãn research/paper use only, not investment advice; không tạo CTA BUY/SELL.

**P1 — trước UAT pilot**

1. Alert API + persistence + audit log; user context từ authentication.
2. Responsive header, controls, alert list/card và charts tại viewport 320 px, 768 px, 1440 px.
3. Semantic navigation/tab roles, focus-visible, keyboard navigation, labels và contrast audit WCAG AA.
4. Component library/design tokens chung, thay inline style lặp lại; React Testing Library, accessibility checks và Playwright cho flow chính.
5. Đồng bộ ticker/period qua URL để bookmark/share; tránh state xung đột giữa các tab.

#### Tiêu chí nghiệm thu UI pilot

- Không trình bày health, alert, confidence hay portfolio giả như dữ liệu thật.
- Luôn có freshness/source/timezone/model version hoặc lý do thiếu dữ liệu.
- API failure có feedback và retry; không chỉ ghi console.
- Core flows chạy bằng keyboard và ở viewport 320 px, 768 px, 1440 px.
- E2E test cover symbol selection, API error, empty data, forecast unavailable, alert management và portfolio authorization.
