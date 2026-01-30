#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_placeholders.py

从训练得到的 PCFG 断点（*_counts.pkl）生成“占位符模板”（template\\tprob）。

注意：模板生成是一个 *k-best* 枚举问题（按概率从高到低输出）。
早期实现会在扩展节点时一次性压入某个 label 的全部 surface-form，
当某些 label 的候选非常多（如 number6/unk_seg）时会导致堆膨胀与性能退化。

当前实现改为对每个 SP 模板做“索引网格”的 Dijkstra/k-best 枚举：
- 每个 label 的 surface-form 按 –ln(p) 排序
- 状态是一个 index tuple（每个维度选第 idx 个 surface-form）
- 每次 pop 只扩展相邻索引（每维 +1），避免一次性生成所有孩子

概率、阈值裁剪(min_prob)、输出排序仍保持一致。
"""

from __future__ import annotations
import pickle, math, heapq, gc, sys, io
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

# ── 全局参数 ──────────────────────────────────────────────────────────────

TPL_CUM_FRAC  = 0.98                # 模板累计概率阈值
SFT_CUM_FRAC  = 0.99                # surface-form 累计概率阈值
LN            = math.log

# ── 通用工具 ───────────────────────────────────────────────────────────────

def normalize(d: Dict[str, float]) -> Dict[str, float]:
    """归一化概率字典；同时过滤掉非正值。"""
    d = {k: v for k, v in d.items() if v > 0}
    s = sum(d.values())
    if abs(s - 1.0) < 1e-9:
        return d
    if s == 0:
        n = len(d) or 1
        return {k: 1 / n for k in d}
    inv = 1.0 / s
    return {k: v * inv for k, v in d.items()}


def prune_by_cumprob(items: Iterable[Tuple[str, float]], threshold: float) -> Dict[str, float]:
    """按累计概率阈值裁剪并重新归一化。"""
    sorted_items = sorted(items, key=lambda kv: kv[1], reverse=True)
    cum = 0.0
    out: Dict[str, float] = {}
    for k, p in sorted_items:
        if cum >= threshold:
            break
        out[k] = p
        cum += p
    return normalize(out)

# ── 载入基础 PCFG ───────────────────────────────────────────────────────────

def load_pcfg_data(
    pkl_path: Path,
    tpl_cum_frac: float = TPL_CUM_FRAC,
    sft_cum_frac: float = SFT_CUM_FRAC,
):
    """从 *.pkl 读取并执行两级累计概率剪枝。"""
    with pkl_path.open("rb") as fh:
        state = pickle.load(fh)

    # —— 1) sp 模板累计概率剪枝 ——
    sp_counts: Dict[Tuple[str, ...], int] = state["sp_counts"]
    sp_total: int = state["sp_total"]

    tpl_probs = [(tpl, cnt / sp_total) for tpl, cnt in sp_counts.items()]
    pruned_tpl_probs = prune_by_cumprob(tpl_probs, tpl_cum_frac)

    sp_tpl: List[Tuple[str, ...]] = []
    sp_neglog: List[float] = []

    for tpl, prob in sorted(pruned_tpl_probs.items(), key=lambda kv: kv[1], reverse=True):
        sp_tpl.append(tpl)
        sp_neglog.append(-LN(prob))

    # —— 2) surface-form 累计概率剪枝 ——
    sft_probs: Dict[str, Dict[str, float]] = {}
    for lab, cd in state["sft_counts"].items():
        tot = state["sft_totals"].get(lab, 0)
        if tot == 0:
            continue
        items = ((sf, cnt / tot) for sf, cnt in cd.items() if cnt > 0)
        sft_probs[lab] = prune_by_cumprob(items, sft_cum_frac)

    del state
    gc.collect()
    return sp_tpl, sp_neglog, sft_probs

# ── 占位符定义 & 注入比例 ───────────────────────────────────────────────────

PLACEHOLDERS = {
    "cn_mobile": "<PHONENUM>",
    "year": "<BIRTH_YEAR>",
    "yymmdd": "<BIRTH_YYMMDD>",
    "yyyymmdd": "<BIRTH_YYYYMMDD>",
    "yymmdd_nopad": "<BIRTH_YYMMDD_NP>",
    "yyyymmdd_nopad": "<BIRTH_YYYYMMDD_NP>",
    "acc_pwd_same": "<ACCOUNT>",
    "acc_email_name": "<EMAIL_NAME>",
    "cn_name_full": "<CN_NAME_FULL>",
    "cn_name_full_special": "<CN_NAME_FULL_SPECIAL>",
    "cn_name_given": "<CN_NAME_GIVEN>",
    "cn_name_last_abbr": "<CN_NAME_LAST_ABBR>",
    "cn_name_abbr": "<CN_NAME_ABBR>",
    "cn_name_first_last": "<CN_NAME_FIRST_LAST>",
    "cn_name_first_last_special": "<CN_NAME_FIRST_LAST_SPECIAL>",
    "cn_name_last_full": "<CN_NAME_LAST_FULL>",
    "yyyymm": "<BIRTH_YYYYMM>",
    "yyyymm_nopad": "<BIRTH_YYYYMM_NP>",
    "mmdd": "<BIRTH_MMDD>",
    "mmdd_nopad": "<BIRTH_MMDD_NP>",
}

INJECT_FRAC = {
    "cn_mobile": 1.0,
    "year": 0.9,
    "yymmdd": 0.98,
    "yyyymmdd": 0.98,
    "yymmdd_nopad": 0.95,
    "yyyymmdd_nopad": 0.95,
    "acc_pwd_same": 1.0,
    "acc_email_name": 1.0,
    "cn_name_full": 0.98,
    "cn_name_first_last": 0.98,
    "cn_name_full_special": 0.9,
    "cn_name_first_last_special": 0.9,
    "cn_name_given": 0.98,
    "cn_name_last_abbr": 0.9,
    "cn_name_last_full": 0.9,
    "cn_name_abbr": 0.98,
    "yyyymm": 0.95,
    "yyyymm_nopad": 0.95,
    "mmdd": 0.9,
    "mmdd_nopad": 0.9,
}

def inject(orig: Dict[str, float], injection: Dict[str, float], keep_orig_frac: float) -> Dict[str, float]:
    """将占位符概率注入到原有 surface-form 中。"""
    keep = max(0.0, min(1.0, keep_orig_frac))
    scale = 1.0 - keep

    newd = {k: v * keep for k, v in orig.items() if v > 0}
    if scale > 1e-12:
        for k, frac in injection.items():
            if frac > 0:
                newd[k] = newd.get(k, 0.0) + frac * scale

    return normalize(newd)

def build_placeholder_sft(base: Dict[str, Dict[str, float]]):
    """在原始 surface-form 概率基础上注入占位符，并预计算 –ln(p)。"""
    sft = {k: dict(v) for k, v in base.items()}

    # 示例：留自定义域名
    sft["acc_email_domain"]      = {"sufe": 0.9, "shufe": 0.1}
    sft["acc_email_domain_com"]  = {"sufe.com": 0.9, "shufe.com": 0.1}

    for lab, ph in PLACEHOLDERS.items():
        orig = sft.get(lab, {})
        frac = INJECT_FRAC.get(lab, 0.0)
        sft[lab] = inject(orig, {ph: 1.0}, keep_orig_frac=1.0 - frac)

    # —— 预计算 –ln(p) —— ④
    sft_ln: Dict[str, List[Tuple[str, float]]] = {}
    for lab, d in sft.items():
        lst = [(sf, -LN(p)) for sf, p in d.items()]
        sft_ln[lab] = lst

    return sft_ln

# ── PCFG 结构；节点=tuple ──────────────────────────────────────────────────
# tuple 结构: (neglog, sp_id, idx, parts_list)

def _kbest_generate_templates(
    *,
    sp_tpl: List[Tuple[str, ...]],
    sp_neglog: List[float],
    sft_ln: Dict[str, List[Tuple[str, float]]],
    top_n: int,
    min_prob: float,
    max_q: int,
    out_path: Path,
) -> int:
    # 1) 先保证每个 label 的候选按 –ln(p) 非减序排列
    for lab, lst in sft_ln.items():
        lst.sort(key=lambda kv: kv[1])

    th = -LN(min_prob)

    # 每个 sp_id 对应的 label 列表（引用 sft_ln 中的 list，避免拷贝）
    sp_lists: List[Optional[List[List[Tuple[str, float]]]]] = [None] * len(sp_tpl)

    # heap item: (neglog_total, sp_id, idx_tuple)
    heap: List[Tuple[float, int, Tuple[int, ...]]] = []
    seen: Dict[int, Set[Tuple[int, ...]]] = {}

    for sp_id, labels in enumerate(sp_tpl):
        lists: List[List[Tuple[str, float]]] = []
        base = sp_neglog[sp_id]
        ok = True
        for lab in labels:
            lst = sft_ln.get(lab)
            if not lst:
                ok = False
                break
            lists.append(lst)
            base += lst[0][1]
        if not ok:
            continue
        if base > th:
            # 当前模板的“最优组合”都过阈值，则其它组合必然更差，直接跳过
            continue

        sp_lists[sp_id] = lists
        idxs = (0,) * len(labels)
        heapq.heappush(heap, (base, sp_id, idxs))
        if len(idxs) > 1:
            seen[sp_id] = {idxs}

    cnt = 0
    with io.open(out_path, "w", buffering=100, encoding="utf-8") as fh:
        while heap and cnt < top_n:
            neglog, sp_id, idxs = heapq.heappop(heap)
            lists = sp_lists[sp_id]
            if lists is None:
                continue

            # 输出该组合
            parts = [lists[i][idx][0] for i, idx in enumerate(idxs)]
            fh.write(f"{''.join(parts)}\t{math.exp(-neglog):.8e}\n")
            cnt += 1

            m = len(idxs)
            if m == 0:
                continue

            # 2) 扩展相邻索引（每维 +1）
            if m == 1:
                i0 = idxs[0]
                j0 = i0 + 1
                if j0 < len(lists[0]):
                    new_neglog = neglog - lists[0][i0][1] + lists[0][j0][1]
                    if new_neglog <= th:
                        heapq.heappush(heap, (new_neglog, sp_id, (j0,)))
            else:
                seen_set = seen.setdefault(sp_id, set())
                for dim in range(m):
                    i = idxs[dim]
                    j = i + 1
                    if j >= len(lists[dim]):
                        continue

                    new_idxs_list = list(idxs)
                    new_idxs_list[dim] = j
                    new_idxs = tuple(new_idxs_list)
                    if new_idxs in seen_set:
                        continue

                    new_neglog = neglog - lists[dim][i][1] + lists[dim][j][1]
                    if new_neglog > th:
                        continue

                    seen_set.add(new_idxs)
                    heapq.heappush(heap, (new_neglog, sp_id, new_idxs))

            # 兼容旧参数：必要时裁剪 heap（避免极端情况下内存占用过大）
            if max_q and len(heap) > max_q:
                heap = heapq.nsmallest(max_q, heap)
                heapq.heapify(heap)

    return cnt

def generate_placeholders(
    pkl_path: str | Path,
    out_path: str | Path,
    top_n: int = 500_000,
    min_prob: float = 1e-8,
    max_q: int = 200_000,
):
    sp_tpl, sp_neglog, base_sft = load_pcfg_data(Path(pkl_path))
    sft_ln = build_placeholder_sft(base_sft)

    cnt = _kbest_generate_templates(
        sp_tpl=sp_tpl,
        sp_neglog=sp_neglog,
        sft_ln=sft_ln,
        top_n=top_n,
        min_prob=min_prob,
        max_q=max_q,
        out_path=Path(out_path),
    )

    print(f"[DONE] placeholders → {out_path} ({cnt} 条)")

# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running via `python generation/password_gen_tools/generate_placeholders.py ...`
    # by ensuring repo root (parents[2]) is on sys.path.
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import argparse

    from config import TRAIN_NAME, project_path

    root = project_path()
    out_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate placeholder templates from a PCFG checkpoint.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=root / "checkpoints" / f"{TRAIN_NAME}_counts.pkl",
        help="Checkpoint pickle path (counts.pkl).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=out_dir / "placeholders1.txt",
        help="Output template file (tab-separated: template\\tprob).",
    )
    parser.add_argument("--top-n", type=int, default=500_000)
    parser.add_argument("--min-prob", type=float, default=1e-8)
    parser.add_argument("--max-q", type=int, default=200_000)
    args = parser.parse_args()

    generate_placeholders(
        pkl_path=args.pkl,
        out_path=args.out,
        top_n=args.top_n,
        min_prob=args.min_prob,
        max_q=args.max_q,
    )
