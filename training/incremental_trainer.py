# incremental_trainer.py
# ============================================================
# 现在 update_counts 接收
#     (segments, weight)        ── TrainingManager 新格式
# ============================================================
import multiprocessing
import os
import pickle
from collections import defaultdict
from typing import List, Tuple, Any


# ---------------- 工具：解析 sample ----------------
def _unpack_sample(sample: Tuple[Any, ...]) -> Tuple[List[Tuple[str, str]], float]:
    """
    抽取 (segments, weight)
    期望 sample 只含两个元素：
        segments : List[(surface, tag)]  # 组合后的标注序列
        weight   : float
    """
    if len(sample) != 2:
        raise ValueError(
            f"Expected sample in format (segments, weight), got: {sample}"
        )

    segments, w = sample
    return segments, float(w)


# ---------------- 子进程用函数 ----------------
def process_chunk(chunk):
    """
    处理一个数据块，返回局部计数:
      sp_counts, sp_total, sft_counts, sft_totals     (全是 float)
    """
    sp_counts = defaultdict(float)
    sp_total = 0.0
    sft_counts = defaultdict(lambda: defaultdict(float))
    sft_totals = defaultdict(float)

    for sample in chunk:
        segs, w = _unpack_sample(sample)

        # (1) SP 模板序列
        sft_seq = tuple(lbl for (_, lbl) in segs)
        sp_counts[sft_seq] += w
        sp_total += w

        # (2) SFT → SF
        for sf, sft in segs:
            if sf is not None:
                sft_counts[sft][sf] += w
                sft_totals[sft] += w
    return sp_counts, sp_total, sft_counts, sft_totals


# ============================================================
#                 IncrementalTrainer  (带权版)
# ============================================================
class IncrementalTrainer:
    """
    • update_counts(samples)      现在支持「权重」float
    • update_counts_parallel()    同步升级
    • 内部所有计数改成 float
    """

    def __init__(self):
        self.sp_counts = defaultdict(float)  ### CHANGED ###
        self.sp_total = 0.0  ### CHANGED ###
        self.sft_counts = defaultdict(lambda: defaultdict(float))  ### CHANGED ###
        self.sft_totals = defaultdict(float)  ### CHANGED ###

    # ---------- 串行累计 ----------
    def update_counts(self, samples):
        """
        samples: Iterable[  (segments, weight) |
                            (acc_segs, pwd_segs) |
                            (acc_segs, pwd_segs, weight) ]
        """
        for sample in samples:
            segs, w = _unpack_sample(sample)

            sft_seq = tuple(lbl for (_, lbl) in segs)
            self.sp_counts[sft_seq] += w
            self.sp_total += w

            for sf, sft in segs:
                if sf is not None:
                    self.sft_counts[sft][sf] += w
                    self.sft_totals[sft] += w

    # ---------- 并行累计 ----------
    def update_counts_parallel(self, samples, num_workers=None):
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        chunk_sz = max(1, len(samples) // num_workers)
        chunks = [samples[i: i + chunk_sz] for i in range(0, len(samples), chunk_sz)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, chunks)

        # 合并
        for sp_c, sp_t, sft_c, sft_t in results:
            for k, v in sp_c.items():
                self.sp_counts[k] += v
            self.sp_total += sp_t
            for sft, sf_dict in sft_c.items():
                for sf, v in sf_dict.items():
                    self.sft_counts[sft][sf] += v
            for sft, v in sft_t.items():
                self.sft_totals[sft] += v

    # ---------- 概率计算 ----------
    def finalize_probabilities(self):
        sp_probs = {tpl: cnt / self.sp_total for tpl, cnt in self.sp_counts.items()} \
            if self.sp_total else {}
        sft_probs = {}
        for sft, sf_dict in self.sft_counts.items():
            tot = self.sft_totals[sft]
            sft_probs[sft] = {sf: cnt / tot for sf, cnt in sf_dict.items()} if tot else {}
        return sp_probs, sft_probs

    # ---------- 序列化 ----------
    def save_state(self, path):
        state = {
            "sp_counts": dict(self.sp_counts),
            "sp_total": self.sp_total,
            "sft_counts": {k: dict(v) for k, v in self.sft_counts.items()},
            "sft_totals": dict(self.sft_totals)
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.sp_counts = defaultdict(float, state["sp_counts"])
        self.sp_total = float(state["sp_total"])
        self.sft_counts = defaultdict(lambda: defaultdict(float))
        for sft, sf_dict in state["sft_counts"].items():
            self.sft_counts[sft].update(sf_dict)
        self.sft_totals = defaultdict(float, state["sft_totals"])
