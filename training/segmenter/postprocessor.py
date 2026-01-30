from __future__ import annotations

# segmenter/postprocessor.py
import heapq
from typing import List, Tuple, Any

# ==================== 旧内容 ====================
# （apply_leet_map, is_english_name, ... 这些函数保持原样）
# ------------------------------------------------
LEET_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '!': 'i', '@': 'a', '$': 's', '9': 'g',
}


def apply_leet_map(raw_str):
    return "".join(LEET_MAP.get(ch, ch) for ch in raw_str.lower())


def is_english_name(candidate, en_name_automaton):
    return candidate.lower() in en_name_automaton


def is_english_word(candidate, en_dict_automaton):
    return candidate.lower() in en_dict_automaton


def detect_label_for_merged_str(merged_raw, en_name_automaton, en_dict_automaton):
    mapped = apply_leet_map(merged_raw)
    if merged_raw.lower() != mapped and len(merged_raw) >= 4:
        if is_english_name(mapped, en_name_automaton) or is_english_word(mapped, en_dict_automaton):
            return "leet"
    if is_english_name(merged_raw, en_name_automaton):
        return "en_name"
    if is_english_word(merged_raw, en_dict_automaton):
        return "en_word"
    return "nn"


def join_segments(segments, i, j):
    return "".join(sf_str for (sf_str, _) in segments[i:j + 1])


def multi_level_merge_with_name_and_dict(segments, en_name_automaton, en_dict_automaton):
    n = len(segments)
    dp = {}

    def dfs(i):
        if i >= n:
            return [], 0
        if i in dp:
            return dp[i]

        sf_str, sf_type = segments[i]
        remain_solution, remain_len = dfs(i + 1)
        best_solution = [(sf_str, sf_type)] + remain_solution
        best_len = 1 + remain_len

        for j in range(i, n):
            merged_raw = join_segments(segments, i, j)
            new_label = detect_label_for_merged_str(merged_raw, en_name_automaton, en_dict_automaton)
            if new_label != "nn":
                sub_solution, sub_len = dfs(j + 1)
                total_len = 1 + sub_len
                if total_len < best_len:
                    best_len = total_len
                    best_solution = [(merged_raw, new_label)] + sub_solution

        dp[i] = (best_solution, best_len)
        return dp[i]

    final_sol, _ = dfs(0)
    return final_sol


# ==================== 新增内容 ====================

def _is_flat_tuple(x: Any) -> bool:
    """判断 segment 是否形如 ('txt','lbl') 的扁平片段"""
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str)


def expand_multibranch_segments(
        segmented: List[Any],
        top_paths: int | None = None,
) -> List[Tuple[List[Tuple[str, str]], float]]:
    """
    把 segment_l_d_s 返回的“混合结构”展开成若干条
       (flat_segments, total_logP)   的确定性路径。

    • 如果同一个 L 段保留 K 条候选，本函数会做笛卡尔积；
      为避免组合爆炸，可传 `top_paths` 仅保留累积 logP 最高的若干条。
    """
    paths: List[Tuple[List[Tuple[str, str]], float]] = [([], 0.0)]
    i, n = 0, len(segmented)

    while i < n:
        entry = segmented[i]
        # -------- 普通扁平片段 --------
        if _is_flat_tuple(entry):
            for p in paths:
                p[0].append(entry)
            i += 1
            continue

        # -------- L 段候选列表 --------
        branch_group = []
        while i < n and not _is_flat_tuple(segmented[i]):
            branch_group.append(segmented[i])  # ([…], logp)
            i += 1

        new_paths = []
        for base_seg, base_lp in paths:
            for branch_seg, branch_lp in branch_group:
                new_paths.append((
                    base_seg + branch_seg,
                    base_lp + branch_lp
                ))

        # 可选：剪枝  top_paths
        if top_paths and len(new_paths) > top_paths:
            # 小顶堆保留 total_logP 最大的 top_paths 条
            heap = [(-lp, seg) for seg, lp in new_paths]
            heapq.heapify(heap)
            new_paths = []
            for _ in range(top_paths):
                if not heap: break
                neg_lp, seg = heapq.heappop(heap)
                new_paths.append((seg, -neg_lp))

        paths = new_paths

    return paths


def postprocess_multibranch(
        segmented: List[Any],
        en_name_automaton,
        en_dict_automaton,
        *,
        expand_top_paths: int | None = 20,
        merged_top_paths: int | None = 10,
):
    """
    入口函数 —— 兼容新版 segment_l_d_s 输出。
    返回  List[ (merged_segments, total_logP) ]  已按 logP ↓ 排序。
    """
    flat_paths = expand_multibranch_segments(segmented, top_paths=expand_top_paths)

    merged_results = []
    for flat_segs, lp in flat_paths:
        merged_segs = multi_level_merge_with_name_and_dict(
            flat_segs, en_name_automaton, en_dict_automaton
        )
        merged_results.append((merged_segs, lp))

    # 按概率降序
    merged_results.sort(key=lambda x: x[1], reverse=True)

    if merged_top_paths:
        merged_results = merged_results[:merged_top_paths]
    # —— 新增：去掉所有 boundary placeholder
    cleaned = []
    for segs, lp in merged_results:
        filtered = [(txt, lbl) for txt, lbl in segs if lbl != "__boundary__"]
        cleaned.append((filtered, lp))
    return cleaned

