# Secret remediation runbook

Không ghi giá trị secret vào tài liệu, Git, log, terminal history hoặc issue.

## Credential groups cần xử lý

| Nhóm | Rotation owner | Cách xác nhận |
| --- | --- | --- |
| CloudVault service token | CloudVault project administrator | Tạo service token mới với least privilege; test publish/restore tối thiểu; revoke token cũ |
| Finnhub API key | Finnhub account owner | Tạo/revoke key trong portal; test một request qua local secret store |
| PostgreSQL password | Database owner | Set password mới, cập nhật secret store/connection string, healthcheck và rollback window |
| Airflow admin/database credentials | Airflow/database owner | Cập nhật qua secret store, login/healthcheck, vô hiệu hóa credential cũ |
| Jupyter token | Runtime owner | Tạo token mới hoặc tắt service nếu không thuộc pilot scope |

## Thứ tự bắt buộc

1. Dừng push/deploy và ghi nhận nơi có credential đang dùng.
2. Tạo credential mới trong secret manager hoặc provider portal.
3. Cập nhật runtime environment ngoài Git và test với credential mới.
4. Revoke credential cũ tại provider/database.
5. Kiểm tra log, Docker image, Git working tree và Git history không còn credential.
6. Bỏ `.env` khỏi Git index, sau đó thực hiện history rewrite **chỉ khi tất cả collaborator đã được thông báo**.
7. Force-push/remote cleanup chỉ do repository owner thực hiện; yêu cầu mọi collaborator reclone hoặc reset theo hướng dẫn đã thống nhất.

## Trạng thái preflight

Preflight đã xác nhận `.env` có lịch sử Git và đã thêm ignore guard. Rotation/revocation chưa thể tự động thực hiện vì cần quyền tại CloudVault, Finnhub, PostgreSQL và Airflow runtime.
