# ADR 0001: Chuẩn hóa runtime dashboard và Python package

- **Ngày:** 2026-11-24
- **Trạng thái:** Accepted for P0 migration

## Bối cảnh

Repository hiện có Dash Python, React/Vite và FastAPI cùng phục vụ dashboard. Business logic cũng bị phân tán giữa `src`, `scripts` và Airflow DAG. Điều này làm tài liệu, dependency, deployment và test khó xác định nguồn chuẩn.

## Quyết định

1. React/Vite là dashboard duy nhất cho pilot, đặt tại `apps/web` sau P0 migration.
2. FastAPI là API duy nhất cho dashboard, đặt tại `services/api`.
3. Dash được đóng băng ngay trong P0; chỉ xóa sau khi React/API smoke test và migration manifest hoàn tất.
4. Business logic Python được đóng gói thành `stock_pipeline` trong `packages/pipeline/src`; script và Airflow DAG chỉ là adapter/entrypoint mỏng.
5. One settings layer cung cấp cấu hình cho mọi runtime. Application code không được hard-code host, port, absolute project path hoặc credential fallback.
6. Database schema đi qua versioned migrations; model/runtime artifacts đi qua registry hoặc storage, không ở source tree/Git.

## Hệ quả

- P0 có thể yêu cầu thay đổi import paths, Docker build contexts, Compose mounts, CI và tài liệu.
- Không được xóa Dash hoặc artifact hiện có trong cùng PR với việc di chuyển business logic.
- Các dữ liệu/feature của Pha 1+ vẫn giữ nguyên scope; ADR này chỉ chuẩn hóa runtime và ownership.
