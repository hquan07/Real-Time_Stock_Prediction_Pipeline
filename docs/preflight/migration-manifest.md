# P0 migration manifest

Manifest này là inventory trước refactor. Mỗi dòng chỉ được migrate trong một PR nhỏ cùng test/rollback tương ứng.

| Nguồn hiện tại | Đích/quyết định | Điều kiện trước khi thay đổi |
| --- | --- | --- |
| `src/dashboard/frontend` | `apps/web` | Build React pass, API base URL dùng environment config |
| `src/dashboard/backend` | `services/api` | API smoke test pass; không mang virtualenv sang đích |
| `src/dashboard/app.py`, `callbacks.py`, `layouts.py`, `components/`, `pages/` | Đóng băng Dash; deprecate rồi xóa | React/API đạt smoke test và README được cập nhật |
| `src/data_ingestion`, `src/streaming`, `src/machine_learning`, `src/deep_learning`, `src/database`, `src/utils`, `integrations` | `packages/pipeline/src/stock_pipeline` theo domain | Import contract/unit test có baseline |
| `scripts/*.py` | `stock_pipeline.cli` hoặc command entrypoint | Business logic được chuyển trước, CLI compatibility được test |
| `dags/*.py` | `orchestration/airflow/dags` | DAG dùng package/CLI thay vì import path cũ |
| `src/config`, `src/data_ingestion/config`, config module rải rác | `stock_pipeline.config` | Settings schema và dev/test/pilot config được chấp thuận |
| `src/database/init_tables.sql` | `infrastructure/migrations` | Schema inventory, backup/confirmation database state |
| `docker-compose.yml`, `docker/*` | `infrastructure/compose`, `infrastructure/docker` | Compose config và smoke test pass |
| `requirements*.txt` | `pyproject.toml` + lock/constraints | Python/Airflow target versions được chọn |
| `src/**/artifacts`, `logs`, virtualenv, `__pycache__`, checkpoints | Registry/object storage hoặc Git-ignore | Backup verified; removal được thực hiện trong PR riêng |

## Rollback rule

Không dùng `git reset --hard` để rollback P0. Mỗi PR refactor phải có commit đảo ngược được, giữ compatibility shim cho một release nếu còn caller cũ, và giữ tag preflight làm mốc điều tra.

## Execution record

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| `src/dashboard/frontend` → `apps/web` | Hoàn thành | Frontend lint/build và container smoke pass tại commit `874f7d51` |
| `src/dashboard/backend` → `services/api` | Hoàn thành | `/health` và same-origin `/api/health` pass tại commit `874f7d51` |
| Legacy Dash | Đã đóng băng | Không còn trong navigation/runtime chính; chưa xóa |
| Runtime files khỏi Git index | Hoàn thành | Không còn `.env`, venv, cache, log, checkpoint được track |
| Settings hợp nhất | Đang thực hiện | DB/Kafka/Redis/CORS/schema path đã dùng `src/config/settings.py`; package move còn chờ |
| Pipeline package + CLI/DAG adapters | Chưa bắt đầu | Cần baseline import/unit contract trước khi move |
| SQL migrations | Chưa bắt đầu | Dump hiện tại còn role-bound; cần migration/rollback test |
| Dependency lock/constraints | Chưa hoàn thành | Frontend lock đã sạch; Python/Airflow còn nhiều requirements |
