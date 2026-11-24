# Backup inventory — 2026-11-24

## Artifact backup đã tạo

| Nội dung | Trạng thái |
| --- | --- |
| `src/machine_learning/artifacts` | Đã archive |
| `src/deep_learning/artifacts` | Đã archive |
| Archive local | `backups/preflight-20261124-model-artifacts.tar.gz` |
| Kích thước archive | 8.2 MiB |
| Số entry | 32 |
| SHA-256 | `659cb5a547e882e3137bae101ef831dfc895c8be273a5468d9523c0068483c8d` |

Archive nằm trong `backups/` và được Git-ignore để tránh commit binary runtime artifact mới. Archive không bao gồm `.env`, logs hay virtualenv.

## PostgreSQL, Kafka, Redis và MLflow

- `docker compose ps --all` không trả về service nào của project tại thời điểm kiểm tra.
- Docker volume inventory không tìm thấy volume mang tên Compose project hiện tại.
- Do đó không có database/runtime state cục bộ khả dụng để export từ stack này.

Trước P0, owner phải xác nhận một trong hai lựa chọn:

1. Database hiện tại không có state cần giữ; ghi nhận xác nhận trong issue/PR; hoặc
2. Chạy stack đúng environment và thực hiện `pg_dump`, backup Kafka/Redis/MLflow artifacts theo runbook, sau đó lưu checksum và vị trí backup tại đây.

## Khôi phục archive model

1. Kiểm tra SHA-256 của archive khớp giá trị trên.
2. Extract vào thư mục tạm, không ghi đè artifact đang dùng.
3. Kiểm tra `latest.json`, metadata và checksum bundle trước khi activate bất kỳ model nào.
