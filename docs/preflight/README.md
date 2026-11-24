# P-0 preflight record

Ngày thực hiện: 2026-11-24. Mục đích của preflight là ghi nhận một baseline và bảo vệ dữ liệu trước P0, không phải thực hiện refactor P0.

## Trạng thái sáu hạng mục

| Hạng mục | Trạng thái | Bằng chứng/kết quả |
| --- | --- | --- |
| 1. Khoanh vùng secrets | **Blocked — cần quyền external** | `.env` và dashboard virtualenv đang bị Git theo dõi; `.gitignore` đã được tăng cường. Xoay vòng CloudVault/Finnhub cần quyền quản trị nhà cung cấp; password local cần thay đổi phối hợp với database/Airflow state. |
| 2. Backup | **Partial** | Đã tạo archive artifact model local; không có service hay volume của Compose project đang chạy để export PostgreSQL. Xem `backup-inventory.md`. |
| 3. Baseline | **Complete with known gaps** | Compose validate pass; Docker project không có service đang chạy; frontend lint/build đã chạy. Xem `baseline-evidence.md`. |
| 4. ADR | **Complete** | [ADR 0001](../adr/0001-canonical-runtime-architecture.md) chọn React/FastAPI và package `stock_pipeline`. |
| 5. Migration manifest | **Complete** | Xem `migration-manifest.md`; không có code/file runtime nào đã được move hay xóa. |
| 6. Git safety point | **Complete locally** | Branch `upg/p0-preflight` và annotated tag `preflight-baseline-20261124` đã được tạo tại commit `b3f030ee8548a954e8794303bdceb6828c4fe723`. |

## Go/no-go để bắt đầu P0

Chỉ bắt đầu P0 khi owner xác nhận:

1. CloudVault/Finnhub và các credential có nguy cơ đã được xoay vòng; `.env` mới không còn trong Git index/history hiện hành.
2. Backup PostgreSQL đã tồn tại hoặc xác nhận database local không có state cần giữ.
3. Người dùng remote repository đã được thông báo trước bất kỳ history rewrite nào.
4. Không còn thay đổi không liên quan trong working tree.

Không rewrite Git history, không xóa tracked virtualenv/log/model artifact và không push branch/tag trong preflight này.
