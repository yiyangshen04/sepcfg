# parallel_pcfg_demo_fixed.py
"""
演示：在單個腳本中用「候選片段 + K‑best Viterbi」並行解析密碼，
并保留前 K 条解析路径（包括可能被模型先验高估但非真实含义的解释），
以免高先验概率的单一路径掩盖其他合理解释。

* 纯 Python 3.8+，无外部依赖。
* 示例词表和概率仅演示，可替换为真实统计数据。
* 支持自定义 K 值，且在生成阶段遍历所有保留解析。

修复内容
----------
1. **K‑best 剪枝方向修正**：原版在堆溢出时弹出的是「最佳解析」，导致概率最高路径被误删。
   现在改为删除 *neg_logp 最大*(即概率最小) 的那一条，真正保留前 K 条最优路径。
2. `Candidate.text` 属性在绑定前调用会 AttributeError；现在在构造时直接保存 `text`，避免潜在坑。
"""

import heapq
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple


# ============================ 数据结构 ============================
@dataclass
class Candidate:
    start: int  # 片段起始索引（含）
    end: int  # 片段结束索引（含）
    label: str  # 语义标签（en_name / en_word / yymmdd ...）
    logp: float  # 对数概率，越大越可能
    _text: str  # 原文（内部用）

    @property
    def text(self) -> str:
        """返回片段原文。"""
        return self._text[self.start: self.end + 1]


# ============================ 示例词表 ============================
EN_NAMES = {"summer", "john", "michael", "jane", "li", "wang"}
EN_WORDS = {
    "summer",
    "winter",
    "password",
    "dragon",
    "letmein",
    "hello",
    "world",
}


# ============================ 候选生成器 ============================

def _bind(text: str, cand: Candidate) -> Candidate:
    cand._text = text
    return cand


def yield_en_name_candidates(text: str):
    lowered = text.lower()
    for m in re.finditer(r"[A-Za-z]+", lowered):
        tok = m.group()
        if tok in EN_NAMES:
            p = 1.0 / len(EN_NAMES)  # 均匀先验
            yield _bind(
                text,
                Candidate(m.start(), m.end() - 1, "en_name", math.log(p), text),
            )


def yield_en_word_candidates(text: str):
    lowered = text.lower()
    for m in re.finditer(r"[A-Za-z]+", lowered):
        tok = m.group()
        if tok in EN_WORDS:
            p = 1.0 / len(EN_WORDS)
            yield _bind(
                text,
                Candidate(m.start(), m.end() - 1, "en_word", math.log(p), text),
            )


def yield_number_candidates(text: str):
    for m in re.finditer(r"\d+", text):
        seq = m.group()
        length = len(seq)
        # 示例先验
        if length == 8:
            label, lp = "yyyymmdd", math.log(0.05)
        elif length == 6:
            label, lp = "yymmdd", math.log(0.05)
        elif len(set(seq)) == 1 and length >= 3:
            label, lp = f"sr{length}", math.log(0.03)
        else:
            label, lp = f"number{length}", math.log(0.01)
        yield _bind(
            text,
            Candidate(m.start(), m.end() - 1, label, lp, text),
        )


# ============================ lattice 构建 ============================

def lattice_from_detectors(text: str, detectors):
    lattice = defaultdict(list)
    for det in detectors:
        for cand in det(text):
            lattice[cand.start].append(cand)
    # fallback 单字符
    for i in range(len(text)):
        lattice[i].append(
            Candidate(i, i, "char", math.log(1e-6), text)
        )
    return lattice


# ============================ K‑best Viterbi ============================

def _drop_worst(heap: List[Tuple[float, list]]):
    """删除小顶堆中 neg_logp 最大(=概率最小) 的元素并保持堆性质。"""
    worst_idx = max(range(len(heap)), key=lambda i: heap[i][0])
    # 把最后一个移到被删位置，再 restore-heap；O(log n)
    heap[worst_idx] = heap[-1]
    heap.pop()
    if worst_idx < len(heap):
        heapq._siftup(heap, worst_idx)
        heapq._siftdown(heap, 0, worst_idx)


def k_best_parse(text: str, lattice, K: int = 5):
    N = len(text)
    dp: List[List[Tuple[float, list]]] = [[] for _ in range(N + 2)]  # 每个位置一个堆
    heapq.heappush(dp[0], (0.0, []))

    for pos in range(N):
        if not dp[pos]:
            continue
        for neg_lp, path in dp[pos]:
            for cand in lattice.get(pos, []):
                new_neg = neg_lp - cand.logp  # 累减对数概率
                new_path = path + [cand]
                heapq.heappush(dp[cand.end + 1], (new_neg, new_path))
                if len(dp[cand.end + 1]) > K:
                    _drop_worst(dp[cand.end + 1])

    # 返回 (neg_logp, path) 列表，按概率从大到小排
    return sorted(dp[N], key=lambda x: x[0])


# ============================ 解析 & 生成接口 ============================

def parse_password(pwd: str, K: int = 5):
    detectors = [
        yield_en_name_candidates,
        yield_en_word_candidates,
        yield_number_candidates,
    ]
    lattice = lattice_from_detectors(pwd, detectors)
    kbest = k_best_parse(pwd, lattice, K)

    results = []
    for neg_lp, path in kbest:
        template = " ".join(f"{pwd[c.start:c.end + 1]}<{c.label}>" for c in path)
        logp = -neg_lp  # 取回 ln P
        results.append((template, logp))
    return results


# ============================ Demo 测试集 ============================

def demo(K: int = 5):
    test_pwds = [
        "summer111111",
        "john20000101",
        "hello2024",
        "111111",
        "dragon99",
        "Summer",
    ]
    for pwd in test_pwds:
        print(f"PASSWORD: {pwd}  (保留 Top-{K} 多条解析)")
        for idx, (tmpl, logp) in enumerate(parse_password(pwd, K), start=1):
            print(f"  {idx}. {tmpl}    logProb={logp:.4f}")
        print()


if __name__ == "__main__":
    demo(K=5)
