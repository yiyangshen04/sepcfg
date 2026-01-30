# training_manager.py
# ------------------------------------------------------------
# 并行版（4 核心） · 兼容「多分支解析 + 概率加权计数」 ── bug-fix 版
# ------------------------------------------------------------
import math
import os
import pickle
from typing import List, Tuple

import duckdb
import multiprocessing as mp

from config import (
    BATCH_SIZE, TRAIN_DATA_SIZE, TRAIN_NAME, CHECKPOINT_DIR,
    RESUME_TRAINING, CHINESE_PINYIN_PATH,
    WORKERS, POOL_ROWS_PER_TASK, USE_POOL_MAP,
    IMAP_FLUSH_ROWS,
    TOPK_L, EXPAND_TOP_PATHS, MERGED_TOP_PATHS,
    MP_START_METHOD, get_duckdb_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL,
    SAMPLE_MODE, RANDOM_SEED,
)
from .incremental_trainer import IncrementalTrainer
from .segmenter.postprocessor import postprocess_multibranch
from .segmenter.preprocessor import preprocess_breach
from .segmenter.segment_l_d_s import segment_l_d_s  # ← 只保留这个

# ---------------- 置信度过滤参数 ----------------
POSTERIOR_THRESHOLD = 0.1   # posterior ≥ 10 % 才保留
MAX_SELECTED_PATHS   = 3     # 每条口令最多几条解析
TEMP_ALPHA           = 1.8   # soft-max 温度（>1 => 更尖锐）

# ---------------- 用于子进程的全局变量 ----------------
_GLOBAL_EN_NAME_AC      = None
_GLOBAL_EN_DICT_AC      = None
_GLOBAL_CHINESE_PINYIN  = None
_GLOBAL_EN_AC           = None

# ============================================================
#                     Pool 初始化 / Worker
# ============================================================
def _pool_worker_init(en_name_ac,
                      en_dict_ac,
                      chinese_pinyin_set,
                      ):
    """在每个子进程中缓存大对象，避免重复 pickle."""
    global _GLOBAL_EN_NAME_AC, _GLOBAL_EN_DICT_AC
    global _GLOBAL_CHINESE_PINYIN, _GLOBAL_EN_AC

    _GLOBAL_EN_NAME_AC     = en_name_ac
    _GLOBAL_EN_DICT_AC     = en_dict_ac
    _GLOBAL_CHINESE_PINYIN = chinese_pinyin_set

    # 在 worker 内预构建英文 AC，避免每条口令重复 frozenset()/cache key 构造成本
    from .segmenter.segment_l_d_s import build_en_automaton

    en_name_set = frozenset(en_name_ac.keys())
    en_word_set = frozenset(en_dict_ac.keys())
    _GLOBAL_EN_AC = build_en_automaton(en_name_set, en_word_set)


def _process_row(row):
    """
    子进程真正执行的单行解析逻辑。
    返回：List[(segments, weight)]
    """
    import math  # 子进程内部也需要
    account, password = row

    # ① 预处理粗分段
    rough = preprocess_breach(account, password)

    # ② L-D-S 解析
    pwd_segments = segment_l_d_s(
        rough,
        _GLOBAL_EN_AC,
        _GLOBAL_CHINESE_PINYIN,
        topk_l=TOPK_L
    )

    # ③ 多分支后处理
    merged_paths = postprocess_multibranch(
        pwd_segments,
        _GLOBAL_EN_NAME_AC,
        _GLOBAL_EN_DICT_AC,
        expand_top_paths=EXPAND_TOP_PATHS,
        merged_top_paths=MERGED_TOP_PATHS,
    )

    # ④ soft-max 求 posterior
    max_lp = max(lp for _, lp in merged_paths)
    probs = [math.exp(TEMP_ALPHA * (lp - max_lp)) for _, lp in merged_paths]
    Z = sum(probs) or 1.0

    ranked = sorted(
        ((segs, p / Z) for (segs, _), p in zip(merged_paths, probs)),
        key=lambda x: x[1],
        reverse=True
    )

    kept = []
    for segs, post in ranked:
        if post < POSTERIOR_THRESHOLD:
            break
        kept.append((segs, post))
        if len(kept) >= MAX_SELECTED_PATHS:
            break

    tot = sum(w for _, w in kept) or 1.0
    return [(segs, w / tot) for segs, w in kept]

# ---------------- 通用工具 ----------------
def load_set_from_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        # 允许某些资源文件缺失（例如用户不再使用拼音词表时）
        return set()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def get_training_status_path():
    return os.path.join(CHECKPOINT_DIR, f"{TRAIN_NAME}_status.pkl")


def get_checkpoint_path():
    return os.path.join(CHECKPOINT_DIR, f"{TRAIN_NAME}_counts.pkl")


def load_training_status():
    if not os.path.exists(get_training_status_path()):
        return None
    with open(get_training_status_path(), "rb") as f:
        return pickle.load(f)


def save_training_status(status):
    with open(get_training_status_path(), "wb") as f:
        pickle.dump(status, f)


_CACHED_MAX_ROWID = None


def _get_max_rowid(conn) -> int:
    global _CACHED_MAX_ROWID
    if _CACHED_MAX_ROWID is not None:
        return _CACHED_MAX_ROWID
    _CACHED_MAX_ROWID = int(conn.execute(f"SELECT MAX(rowid) FROM {TABLE_NAME}").fetchone()[0] or 0)
    return _CACHED_MAX_ROWID


def read_batch_from_db_with_mode(
    *,
    offset: int,
    limit: int,
    batch_index: int,
    total_batches: int,
) -> List[Tuple[str, str]]:
    """
    从 DuckDB 读取一批训练数据。

    - SAMPLE_MODE=sequential: 按 rowid 顺序从 offset 起读取（默认）
    - SAMPLE_MODE=random_window: 每个 batch 从表中不同位置随机取一个「连续窗口」(limit 行)
      目的是在不全表排序/全表随机的情况下，尽量覆盖全表区域。
    """
    conn = duckdb.connect(get_duckdb_path(), read_only=True)
    try:
        if SAMPLE_MODE == "random_window":
            import random

            max_rowid = _get_max_rowid(conn)
            if max_rowid <= 0:
                start = 0
            else:
                # 将 rowid 空间分成 total_batches 份，batch 在各自区间内选随机起点，减少重叠
                seg = max(1, max_rowid // max(1, total_batches))
                base = (batch_index % max(1, total_batches)) * seg
                hi = min(max_rowid, base + seg - 1)
                max_start = max(base, hi - max(0, limit))
                rng = random.Random((RANDOM_SEED or 0) + int(batch_index))
                start = rng.randint(base, max_start) if max_start >= base else base

            query = f"""
                SELECT {ACCOUNT_COL}, {PASSWORD_COL}
                FROM {TABLE_NAME}
                WHERE rowid >= {start}
                ORDER BY rowid
                LIMIT {limit}
            """
            return conn.execute(query).fetchall()

        # sequential: DuckDB 上对超大表使用 OFFSET 代价很高；优先用 rowid 范围扫描。
        query_rowid = f"""
            SELECT {ACCOUNT_COL}, {PASSWORD_COL}
            FROM {TABLE_NAME}
            WHERE rowid >= {offset}
            ORDER BY rowid
            LIMIT {limit}
        """
        query_offset = f"""
            SELECT {ACCOUNT_COL}, {PASSWORD_COL}
            FROM {TABLE_NAME}
            LIMIT {limit} OFFSET {offset}
        """
        try:
            return conn.execute(query_rowid).fetchall()
        except Exception:
            return conn.execute(query_offset).fetchall()
    finally:
        conn.close()

# ============================================================
#                       TrainingManager
# ============================================================
class TrainingManager:
    def __init__(self,
                 english_name_ac=None,
                 english_dict_ac=None,
                 *,
                 english_name_automaton=None,
                 english_dict_automaton=None,
                 **kwargs):
        # ── 兼容旧别名 ───────────────────────
        if english_name_ac is None:
            english_name_ac = english_name_automaton
        if english_dict_ac is None:
            english_dict_ac = english_dict_automaton
        if english_name_ac is None or english_dict_ac is None:
            raise ValueError("必须提供英文姓名/词典自动机实例")

        # ── 资源 ─────────────────────────────
        self.en_name_ac = english_name_ac          # Aho-Corasick
        self.en_dict_ac = english_dict_ac

        # 训练/解析阶段的英文词表：直接从 automaton 取 keys()，避免重复读取文件与不一致。
        # 该集合只用于构建用于 L 段扫描的英文 AC。
        self.english_dict = set(self.en_dict_ac.keys())
        self.chinese_pinyin_set  = load_set_from_file(CHINESE_PINYIN_PATH)

        ensure_dir(CHECKPOINT_DIR)
        self.trainer      = IncrementalTrainer()

        # ── 批量参数 ─────────────────────────
        self.batch_size     = BATCH_SIZE
        self.total_size     = TRAIN_DATA_SIZE
        self.total_batches  = math.ceil(self.total_size / self.batch_size)
        self.current_batch  = 0
        if RESUME_TRAINING:
            self._resume_if_possible()
        else:
            for p in (get_training_status_path(), get_checkpoint_path()):
                if os.path.exists(p):
                    os.remove(p)

        # ── 进程池（固定 4 核心） ─────────────
        ctx = mp.get_context(MP_START_METHOD)
        self.pool = ctx.Pool(
            processes=WORKERS,
            initializer=_pool_worker_init,
            initargs=(
                self.en_name_ac,
                self.en_dict_ac,
                self.chinese_pinyin_set,
            )
        )

    # ---------- checkpoint ----------
    def _resume_if_possible(self):
        st = load_training_status()
        if st:
            self.current_batch = st.get("finished_batch_index", 0)
        ckpt = get_checkpoint_path()
        if os.path.exists(ckpt):
            self.trainer.load_state(ckpt)

    def _save_checkpoint(self):
        self.trainer.save_state(get_checkpoint_path())
        save_training_status({"finished_batch_index": self.current_batch})

    # ---------- 训练主循环 ----------
    def train_all_batches(self):
        print(f"Training start: size={self.total_size}, batch={self.batch_size}, "
              f"total_batches={self.total_batches}, resume={RESUME_TRAINING}, "
              f"current={self.current_batch}")

        for b in range(self.current_batch, self.total_batches):
            offset = b * self.batch_size
            limit  = min(self.batch_size, self.total_size - offset)
            if limit <= 0:
                break

            rows = read_batch_from_db_with_mode(
                offset=offset,
                limit=limit,
                batch_index=b,
                total_batches=self.total_batches,
            )

            # --- 并行处理当前 batch ---
            if USE_POOL_MAP:
                list_of_samples_lists = self.pool.map(
                    _process_row,
                    rows,
                    chunksize=POOL_ROWS_PER_TASK,
                )
                weighted_samples = [item for sub in list_of_samples_lists for item in sub]
                self.trainer.update_counts(weighted_samples)
            else:
                buffered = []
                seen = 0
                for samples in self.pool.imap(_process_row, rows, chunksize=POOL_ROWS_PER_TASK):
                    buffered.extend(samples)
                    seen += 1
                    if IMAP_FLUSH_ROWS > 0 and (seen % IMAP_FLUSH_ROWS) == 0:
                        self.trainer.update_counts(buffered)
                        buffered.clear()

                if buffered:
                    self.trainer.update_counts(buffered)

            self.current_batch = b + 1
            self._save_checkpoint()
            print(f"[Batch {b + 1}/{self.total_batches}] done, ckpt saved.")

        print("All batches processed, finalising probabilities…")
        try:
            return self.trainer.finalize_probabilities()
        finally:
            self._close_pool()

    # ---------- 资源回收 ----------
    def _close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        # 避免解释器 shutdown 时的句柄泄漏
        try:
            self._close_pool()
        except Exception:
            pass
