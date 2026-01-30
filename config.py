from __future__ import annotations

import os
from pathlib import Path

# config.py
# - 尽量不要写死 Windows 绝对路径；路径统一以「项目根目录」为基准构建。

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"环境变量 {name} 需要是整数，但得到：{raw!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 需要是布尔值(1/0/true/false)，但得到：{raw!r}")


# DuckDB（数据集通常不随仓库分发）
# - 通过环境变量配置：SE_PCFG_DUCKDB_PATH 或 DUCKDB_PATH
DUCKDB_PATH = os.environ.get("SE_PCFG_DUCKDB_PATH") or os.environ.get("DUCKDB_PATH")
TABLE_NAME = "breaches"  # 表名
ACCOUNT_COL = "account"
PASSWORD_COL = "password"


def get_duckdb_path() -> str:
    if DUCKDB_PATH:
        return str(Path(DUCKDB_PATH).expanduser())

    # 兼容：如果用户把数据集放到了仓库内（例如 data/combcn2021.duckdb），默认拾取。
    default_in_repo = project_path("data", "combcn2021.duckdb")
    if default_in_repo.exists():
        return str(default_in_repo)

    raise RuntimeError(
        "未配置 DuckDB 路径：请设置环境变量 SE_PCFG_DUCKDB_PATH（或 DUCKDB_PATH）"
    )


# 字典或语料文件路径（相对项目根目录）
ENGLISH_DICT_PATH = str(project_path("data", "english_names.csv"))
CHINESE_PINYIN_PATH = str(project_path("data", "chinese_pinyin.txt"))
CHINESE_NAMES_PATH = str(project_path("data", "chinese_names.csv"))

# 其他可能的配置
BATCH_SIZE = _env_int("SE_PCFG_BATCH_SIZE", 100000)  # 每次处理10万条
TRAIN_DATA_SIZE = _env_int("SE_PCFG_TRAIN_DATA_SIZE", 5000000)  # 本次训练准备处理100万条(示例)
TRAIN_NAME = os.environ.get("SE_PCFG_TRAIN_NAME", "my_training_20250515")
CHECKPOINT_DIR = os.environ.get("SE_PCFG_CHECKPOINT_DIR") or str(project_path("checkpoints"))  # 存放断点/状态文件的位置
WORKERS = _env_int("SE_PCFG_WORKERS", 4)
POOL_ROWS_PER_TASK = _env_int("SE_PCFG_POOL_ROWS_PER_TASK", 200)
USE_POOL_MAP = _env_bool("SE_PCFG_USE_POOL_MAP", False)
IMAP_FLUSH_ROWS = _env_int("SE_PCFG_IMAP_FLUSH_ROWS", 1000)
TOPK_L = _env_int("SE_PCFG_TOPK_L", 5)
EXPAND_TOP_PATHS = _env_int("SE_PCFG_EXPAND_TOP_PATHS", 20)
MERGED_TOP_PATHS = _env_int("SE_PCFG_MERGED_TOP_PATHS", 20)
MP_START_METHOD = os.environ.get("SE_PCFG_MP_START_METHOD", "spawn").strip().lower()
if MP_START_METHOD not in {"spawn", "fork", "forkserver"}:
    raise ValueError(
        f"SE_PCFG_MP_START_METHOD 只支持 spawn/fork/forkserver，但得到：{MP_START_METHOD!r}"
    )

# 数据采样（用于从大表中抽样训练；默认顺序扫描）
SAMPLE_MODE = os.environ.get("SE_PCFG_SAMPLE_MODE", "sequential").strip().lower()
if SAMPLE_MODE not in {"sequential", "random_window"}:
    raise ValueError(
        f"SE_PCFG_SAMPLE_MODE 只支持 sequential/random_window，但得到：{SAMPLE_MODE!r}"
    )
RANDOM_SEED = os.environ.get("SE_PCFG_RANDOM_SEED")  # 可选
if RANDOM_SEED is not None and RANDOM_SEED != "":
    try:
        RANDOM_SEED = int(RANDOM_SEED)
    except ValueError as e:
        raise ValueError(f"SE_PCFG_RANDOM_SEED 需要是整数，但得到：{RANDOM_SEED!r}") from e
else:
    RANDOM_SEED = None

# 是否从已有断点恢复，或重新开始训练
RESUME_TRAINING = _env_bool("SE_PCFG_RESUME_TRAINING", True)  # True: 续训；False: 从头开始新训练
