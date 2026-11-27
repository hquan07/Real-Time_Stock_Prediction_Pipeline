# Baseline evidence — 2026-11-24

## Repository

- Baseline Git tag: `preflight-baseline-20261124`.
- Baseline commit: `b3f030ee8548a954e8794303bdceb6828c4fe723` (`update docs`).
- Preflight branch: `upg/p0-preflight`.
- `.env` và `src/dashboard/backend/venv` hiện bị Git theo dõi; remediation không được thực hiện trước khi credential rotation/backup được xác nhận.

## Compose và Docker

| Check | Kết quả |
| --- | --- |
| `docker compose config --quiet` | Pass; Docker Compose cảnh báo trường `version` đã obsolete |
| `docker compose ps --all` | Không có service nào của project đang chạy |
| Project volumes | Không tìm thấy volume mang tên project để backup |

## Frontend

| Check | Kết quả |
| --- | --- |
| `npm run lint` | Exit 0, còn 6 warnings: unused parameter/imports, synchronous setState trong effects và dependency hook |
| `npm run build` | Pass |
| Production JS bundle | 5.56 MiB (gzip 1.69 MiB); vượt ngưỡng 500 KiB warning của Vite |

## Python/tests

- Python môi trường host không cài dependencies của project; `pytest` không import được.
- Vì vậy không có kết quả unit/integration test có thể dùng làm baseline trong host hiện tại.
- P0 phải tạo lock/bootstrap chính thức trước khi biến kết quả test thành release gate.
