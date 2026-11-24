# Đánh giá cấu trúc repository và UI/UX dashboard

Đánh giá dựa trên mã nguồn, Compose, tài liệu và các file đang được Git theo dõi. Không thay thế cho usability test với người dùng thật hoặc kiểm thử dashboard đang chạy.

## 1. Kết luận

- **Cấu trúc repository: 30/100.** Domain kỹ thuật đã có, nhưng boundary bị vỡ bởi code trùng, runtime files bị commit và entrypoint cạnh tranh nhau.
- **Nền tảng visual UI: 65/100.** Dark theme, typography, card và biểu đồ nhất quán, phù hợp một dashboard market-monitoring.
- **UI/UX sẵn sàng pilot: 35/100.** Điều hướng/màn hình cơ bản đã có nhưng độ tin cậy dữ liệu, error state, responsive và accessibility chưa đủ cho công cụ hỗ trợ quyết định.

## 2. Cấu trúc hiện tại

### Điểm tốt

- Có các domain cần thiết: ingestion, Kafka, Spark, database, ML/DL, Airflow, integrations, tests và dashboard frontend/backend.
- `tests/unit` và `tests/integration` đã được tách ở mức thư mục.
- Dockerfiles tách cho Spark, Airflow, API và frontend.
- CloudVault integration được cô lập tương đối trong `integrations/`.

### Vấn đề P0

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

Áp dụng cấu trúc đích và thứ tự migrate ở mục **3.0** của [upgrade-roadmap.md](upgrade-roadmap.md). Không trộn refactor folder với thay đổi business rule hoặc data model trong cùng một PR.

## 3. Đánh giá UI/UX dashboard

### Điểm mạnh

- Dark theme, accent indigo/violet, card và biểu đồ dùng cùng ngôn ngữ thị giác. [index.css](../src/dashboard/frontend/src/index.css)
- Overview, Technical, ML Predictions, Portfolio và Alerts phản ánh các câu hỏi chính của người dùng theo dõi thị trường.
- Candlestick, volume, prediction-vs-actual và feature importance là lựa chọn trực quan phù hợp.
- Có disclaimer rằng ML prediction chỉ để tham khảo.

### Vấn đề UX nghiêm trọng

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

### Luồng UX đích cho pilot

1. **Market overview mặc định:** market status, source/freshness, symbol selector, giá gần nhất, độ trễ và biểu đồ. Khi data lỗi/stale, hiển thị trạng thái đó thay vì số 0.
2. **Forecast & signal:** forecast, uncertainty đã hiệu chỉnh, model/rule/data version, `as-of` time và lý do có/không có signal. Không dùng màu xanh/đỏ như một lời khuyên giao dịch.
3. **Research:** indicator, historical performance và feature importance có timeframe/source rõ ràng.
4. **Paper portfolio:** chỉ hiển thị cho người dùng xác thực, có P&L/exposure/cash/timestamp.
5. **Alerts:** persisted qua API, có validation, xác nhận thao tác, notification channel và audit history.
6. **System status:** admin-only, lấy metrics thật từ observability; UI không được phép đổi trạng thái service.

### Backlog UI/UX

#### P0 — làm cùng repository foundation

1. Đặt Overview làm mặc định; chuyển Architecture thành admin-only hoặc bỏ khỏi pilot.
2. Xóa mock health/metrics và alert seed; hiển thị empty state trung thực cho đến khi backend có dữ liệu thật.
3. Tạo `API_BASE_URL`/`WS_BASE_URL` từ Vite environment và dùng same-origin reverse proxy khi deploy.
4. Chuẩn hóa loading, empty, error, retry và stale-data state cho mọi query; thêm error boundary/toast policy.
5. Hiển thị source, last updated, exchange timezone, market status, model/rule/data version ở Overview/Forecast.
6. Không hiển thị confidence interval nếu backend chưa trả uncertainty đã được đánh giá; dùng `Unavailable` thay vì ước lượng giả.
7. Gắn nhãn research/paper use only, not investment advice; không tạo CTA BUY/SELL.

#### P1 — trước UAT pilot

1. Alert API + persistence + audit log; user context từ authentication.
2. Responsive header, controls, alert list/card và charts tại viewport 320 px, 768 px, 1440 px.
3. Semantic navigation/tab roles, focus-visible, keyboard navigation, labels và contrast audit WCAG AA.
4. Component library/design tokens chung, thay inline style lặp lại; React Testing Library, accessibility checks và Playwright cho flow chính.
5. Đồng bộ ticker/period qua URL để bookmark/share; tránh state xung đột giữa các tab.

### Tiêu chí nghiệm thu UI pilot

- Không trình bày health, alert, confidence hay portfolio giả như dữ liệu thật.
- Luôn có freshness/source/timezone/model version hoặc lý do thiếu dữ liệu.
- API failure có feedback và retry; không chỉ ghi console.
- Core flows chạy bằng keyboard và ở viewport 320 px, 768 px, 1440 px.
- E2E test cover symbol selection, API error, empty data, forecast unavailable, alert management và portfolio authorization.
