# Phase 5-Lite Usage

## 1) Health check
```bash
python scripts/health_check.py --skip-network
python scripts/health_check.py --json
```

## 2) Run minimal regression tests
```bash
python -m unittest tests.test_phase5_lite -v
```

## 3) Summarize local metrics
```bash
python scripts/summarize_metrics.py
python scripts/summarize_metrics.py --json
```

## 4) Backup runtime
```bash
python scripts/backup_runtime.py
```

## 5) Cleanup old runtime files
```bash
python scripts/cleanup_runtime.py --keep-days 7
python scripts/cleanup_runtime.py --keep-days 7 --apply
```

## 6) Task lifecycle commands (chat input)
- 查询任务 `<task_id>`
- 取消任务 `<task_id>`
- 重试任务 `<task_id>`
