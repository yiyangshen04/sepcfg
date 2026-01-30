#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimised PCFG password generator
author : ChatGPT (May-2025)
"""

from __future__ import annotations
import inspect
import pickle, heapq, math, gc
from dataclasses import dataclass as _dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


def dataclass(*args, **kwargs):
    # Python 3.9 的 dataclass 不支持 slots 参数（3.10+ 才支持）
    if "slots" in kwargs and "slots" not in inspect.signature(_dataclass).parameters:
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)


# ─────────────────────────────── 0-b. 裁剪语义因子概率 ────────────────────────────
def trim_sft_probs(sft_probs: Dict[str, Dict[str, float]],
                   coverage: float | int = 0.95) -> None:
    """
    原地把每个 sft 下面的 sf 列表裁剪到所需覆盖率 / top-k。
      coverage:
        - 若为 0< float ≤1    → 保留达到该局部概率累积的前缀
        - 若为正整数 (k)       → 保留 top-k (按 prob 降序)
    修改 sft_probs 无返回值。
    """
    for sft, sf_dict in list(sft_probs.items()):
        if not sf_dict:          # 空标签直接跳过
            continue

        items = sorted(sf_dict.items(), key=lambda kv: kv[1], reverse=True)

        if isinstance(coverage, float):          # 覆盖率模式
            target = min(max(coverage, 0.0), 1.0)
            cum, kept = 0.0, []
            for sf, p in items:
                cum += p
                kept.append((sf, p))
                if cum >= target:
                    break
        else:                                    # top-k 模式
            kept = items[: int(coverage)]

        # 原地覆写
        sft_probs[sft] = dict(kept)



# ──────────────────────────────── 1. 读 pkl，原地转概率 ──────────────────────────────
def load_pcfg_data(pkl_path: str | Path):
    """
    返回:
      sp_tpl   : list[tuple]      # 所有模板
      sp_logp  : list[float]      # -log(prob) 与 sp_tpl 对齐
      sft_probs: dict{sft: dict{sf: prob}}
    内存优化: 计数→概率后即刻删除旧表
    """
    with open(pkl_path, "rb") as fh:
        state = pickle.load(fh)

    sp_counts: Dict[Tuple[str, ...], int] = state["sp_counts"]
    sp_total: int = state["sp_total"]
    sft_counts: Dict[str, Dict[str, int]] = state["sft_counts"]
    sft_totals: Dict[str, int] = state["sft_totals"]

    # -- sp --
    sp_tpl: List[Tuple[str, ...]] = []
    sp_logp: List[float] = []
    ln = math.log
    for tpl, cnt in sp_counts.items():
        if cnt:
            p = cnt / sp_total
            sp_tpl.append(tpl)
            sp_logp.append(-ln(p))

    # -- sft (原地覆盖) --
    for sft, sf_dict in sft_counts.items():
        total = sft_totals.get(sft, 0)
        if not total:
            sft_counts[sft] = {}
            continue
        inv_total = 1.0 / total
        for sf, cnt in sf_dict.items():
            sf_dict[sf] = cnt * inv_total

    trim_sft_probs(sft_counts, coverage=0.98)


    # 释放计数表
    del state, sp_counts, sft_totals, sp_total
    gc.collect()

    return sp_tpl, sp_logp, sft_counts


# ──────────────────────────────── 2. Grammar/PCFG ────────────────────────────────
class MyPCFG:
    def __init__(self,
                 sp_tpl: List[Tuple[str, ...]],
                 sft_probs: Dict[str, Dict[str, float]],
                 account_labels: Iterable[str] = (
                     "acc_email_domain",
                     "acc_email_domain_com",
                     "acc_pwd_same",
                     "acc_email_name",
                 )):
        self.sp_tpl = sp_tpl
        self.sft_probs = sft_probs
        self.account_labels = set(account_labels)
        # 预排一次最长优先
        self.sorted_labels = sorted(self.account_labels, key=len, reverse=True)

    # ---------- account helpers ----------
    def _is_account_label(self, sft: str) -> bool:
        return any(lbl in sft for lbl in self.account_labels)

    def _convert_sft_with_brackets(self, sft: str) -> str:
        """<prefix>[acc_label]<suffix>"""
        for lbl in self.sorted_labels:
            pos = sft.find(lbl)
            if pos != -1:
                return f"{sft[:pos]}[{lbl}]{sft[pos+len(lbl):]}"
        return f"[{sft}]"  # fallback, 应该不会走到

    # ---------- 展开 ----------
    def children(self, node: "Node"):
        """yield Node 子节点"""
        tpl = self.sp_tpl[node.sp_id]
        idx = node.idx
        if idx >= len(tpl):
            return

        next_sft = tpl[idx]
        parts = node.parts
        base_nlogp = node.neg_logp
        ln = math.log

        if self._is_account_label(next_sft):
            yield Node(base_nlogp,
                       node.sp_id,
                       idx + 1,
                       parts + [self._convert_sft_with_brackets(next_sft)])
        else:
            sf_dict = self.sft_probs.get(next_sft, {})
            for sf, p in sf_dict.items():
                if p == 0.0:   # 保险
                    continue
                yield Node(
                    base_nlogp - ln(p),
                    node.sp_id,
                    idx + 1,
                    parts + [sf]
                )


# ──────────────────────────────── 3. Node & 队列 ────────────────────────────────
@dataclass(order=True, slots=True)
class Node:
    # order 只比较前 3 个字段，parts 不参与比较
    neg_logp: float
    sp_id: int
    idx: int
    parts: List[str] = field(compare=False)


class PcfgQueue:
    """
    最小堆 (neg_logp 越小 => 概率越大 => 越靠前)，
    若 size 超过上限 1.2×MAX 时，用 nsmallest 保留最优 MAX 个节点
    """
    def __init__(self,
                 pcfg: MyPCFG,
                 sp_logp: List[float],
                 min_probability: float = 1e-10,
                 max_queue_size: int = 50000):
        self.pcfg = pcfg
        self.heap: List[Node] = []
        self.max_size = max_queue_size
        self.prune_trigger = int(max_queue_size * 1.2)
        self.threshold_nlogp = -math.log(min_probability)
        self._push_initial(sp_logp)

    # -- public ------------
    def next(self) -> Node | None:
        """
        返回下一个“完整”节点 (idx==len(tpl))；队列空则 None
        """
        while self.heap:
            best = heapq.heappop(self.heap)       # 最高概率
            tpl_len = len(self.pcfg.sp_tpl[best.sp_id])
            if best.idx == tpl_len:
                return best                       # 完整密码
            # 展开子节点
            for child in self.pcfg.children(best):
                if child.neg_logp <= self.threshold_nlogp:
                    self._push(child)
        return None

    # -- internal ----------
    def _push_initial(self, sp_logp: List[float]):
        for sp_id, nlogp in enumerate(sp_logp):
            if nlogp <= self.threshold_nlogp:
                heapq.heappush(self.heap, Node(nlogp, sp_id, 0, []))
        self._prune_if_needed(force=True)

    def _push(self, node: Node):
        heapq.heappush(self.heap, node)
        self._prune_if_needed()

    def _prune_if_needed(self, force: bool = False):
        if not force and len(self.heap) <= self.prune_trigger:
            return
        if len(self.heap) > self.max_size:
            self.heap[:] = heapq.nsmallest(self.max_size, self.heap)
            heapq.heapify(self.heap)


# ──────────────────────────────── 4. 生成器 ───────────────────────────────────────
def generate_password_guesses(pkl_path: str | Path,
                              output_txt: str | Path,
                              num_to_generate: int = 10_000,
                              *,
                              min_probability: float = 1e-10,
                              max_queue_size: int = 50_000):
    # ① 载入
    sp_tpl, sp_logp, sft_probs = load_pcfg_data(pkl_path)
    print(f"[INFO] loaded templates={len(sp_tpl)}   sft={len(sft_probs)}")

    # ② grammar & queue
    pcfg = MyPCFG(sp_tpl, sft_probs)
    pq = PcfgQueue(pcfg,
                   sp_logp,
                   min_probability=min_probability,
                   max_queue_size=max_queue_size)

    # ③ 生成
    out_f = Path(output_txt).expanduser()
    out_f.parent.mkdir(parents=True, exist_ok=True)
    with out_f.open("w", encoding="utf-8") as fh:
        produced = 0
        while produced < num_to_generate:
            node = pq.next()
            if node is None:
                print("[INFO] Queue empty – generation finished early.")
                break
            pwd = "".join(node.parts)
            prob = math.exp(-node.neg_logp)
            fh.write(f"{pwd}\t{prob:.8e}\n")
            produced += 1

    print(f"[INFO] done: {produced} passwords → {out_f}")

def choose_min_prob_from_pkl(pkl_path, coverage_target=0.99):
    """返回满足给定覆盖率的最小模板概率（float）"""
    with open(pkl_path, 'rb') as fh:
        state = pickle.load(fh)
    sp_counts = state['sp_counts']
    sp_total  = state['sp_total']
    # 计算并排序
    probs = sorted((cnt / sp_total for cnt in sp_counts.values()), reverse=True)
    cum = 0.0
    for p in probs:
        cum += p
        if cum >= coverage_target:
            return p
    return 0.0   # 理论上到不了


# ──────────────────────────────── 5. CLI 示例 ───────────────────────────────────
if __name__ == "__main__":
    # === 修改这里：pkl 路径 & 输出路径 ===
    root = Path(__file__).resolve().parents[1]
    tasks = [
        (
            root / "checkpoints" / "my_training_20250412_1M_counts.pkl",
            root / "my_training_20250513_counts.txt",
        )
    ]

    N_OUTPUT = 10_000
    MIN_PROB  = 1e-10
    MAX_QUEUE = 50_000

    for pkl_path, out_txt in tasks:
        print(f"[INFO] >>> {pkl_path} → {out_txt}")
        target_cov = 0.99  # 99 % 模板概率覆盖
        MIN_PROB = choose_min_prob_from_pkl(pkl_path, target_cov)
        print(f"min_probability = {MIN_PROB:.3e}  (covers ≥{target_cov * 100:.1f}% templates)")
        generate_password_guesses(
            pkl_path,
            out_txt,
            num_to_generate=N_OUTPUT,
            min_probability=MIN_PROB,
            max_queue_size=MAX_QUEUE
        )

    print("[INFO] all tasks completed.")
