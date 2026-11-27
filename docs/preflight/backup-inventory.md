# Backup inventory — 2026-11-24

## P1 preflight database baseline — 2026-11-27

The development Compose stack is now running, so the earlier observation that
there was no PostgreSQL state no longer applies. Before P1 schema/storage work,
a PostgreSQL custom-format dump was saved locally at
`backups/p1-preflight-stockdb-20261127.dump` (53,469 bytes, mode `0600`). The
`backups/` directory is Git-ignored; this dump must not be committed or shared
as a repository artifact.

SHA-256: `45a5f90eb119c5de4e455e907e8424c86faf15513bef71e93c675bced14da519`

The dump's archive listing was readable. A restore into a disposable database
on the project PostgreSQL service succeeded, with 2 `schema_migrations` rows
and 0 `stock_prices_stream` rows; the disposable database was then dropped.
The live database remained in place. PostgreSQL table statistics showed zero
rows in the application tables at the time of the snapshot. This backup covers
PostgreSQL only, not Kafka/Redis state or the model artifact archive below.
Keep the dump available until P1 migrations and replay checks finish; a
separate off-host backup/restore policy is required before pilot use.

To verify the local file again, run `sha256sum` and `pg_restore --list` against
the path above. Restore into a disposable database first; do not restore over
the live `stockdb` database.

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
