# segmenter/cn_name_detection.py
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple, Set, Optional

import ahocorasick


##############################################################################
# 1. 核心匹配引擎
##############################################################################

class NameAutomataManager:
    """
    持有若干 Aho-Corasick 自动机，可对字符串做多标签匹配。
    automata_dict : { label(str) -> ahocorasick.Automaton }
    """

    def __init__(self, automata_dict: Dict[str, ahocorasick.Automaton]):
        self.automata_dict = automata_dict

    # ---------- 工厂方法 ----------
    @classmethod
    def load_automata(cls, pkl_path: str) -> "NameAutomataManager":
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Automata pickle not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            automata_dict = pickle.load(f)
        return cls(automata_dict)

    # ---------- 匹配 ----------
    def find_names(self, text: str) -> List[Tuple[int, int, str]]:
        """
        返回 [(start_idx, end_idx, label), ...]，改为类别内 Longest-First 去重；
        **同一区间可出现多个 label**。
        """
        if not text:
            return []

        lower_text = text.lower()

        # 1) 收集所有原始匹配，按 label 分组
        raw_by_label: Dict[str, List[Tuple[int, int, str, int]]] = {lbl: [] for lbl in self.automata_dict}
        for label, A in self.automata_dict.items():
            for end_pos, (_, pat) in A.iter(lower_text):
                s = end_pos - len(pat) + 1
                raw_by_label[label].append((s, end_pos, label, len(pat)))

        keep_spans: List[Tuple[int, int, str]] = []  # (start, end, label)
        # 2) 对每个类别分别做 Longest-First 去重
        for label, entries in raw_by_label.items():
            if not entries:
                continue
            # 先长后短，再起点升序
            entries.sort(key=lambda x: (-x[3], x[0]))
            spans: List[Tuple[int, int]] = []
            for s, e, lbl, length in entries:
                # 与本类别已保留区间冲突则跳过
                if any(not (e < ss or s > ee) for ss, ee in spans):
                    continue
                spans.append((s, e))
                keep_spans.append((s, e, lbl))

        # 3) 按起点排序所有保留结果
        keep_spans.sort(key=lambda x: x[0])
        return keep_spans


##############################################################################
# 2. 包装类 & Detector 缓存
##############################################################################

class CNNameDetector:
    """薄包装，暴露 .find_names()"""

    def __init__(self, pkl_path: str):
        self.manager = NameAutomataManager.load_automata(pkl_path)

    def find_names(self, text: str):
        return self.manager.find_names(text)


# 按 pickle 路径缓存，支持多份自动机共存
_DETECTOR_CACHE: Dict[str, CNNameDetector] = {}


def build_or_load_detector(pkl_path: str) -> CNNameDetector:
    """
    若缓存中不存在，则加载并缓存；否则复用缓存实例。
    """
    detector = _DETECTOR_CACHE.get(pkl_path)
    if detector is None:
        detector = CNNameDetector(pkl_path)
        _DETECTOR_CACHE[pkl_path] = detector
    return detector


##############################################################################
# 3. 对外辅助函数 – 供预处理 / L-D-S 阶段调用
##############################################################################

def detect_cn_names_before_lds(
        text: str,
        *,
        allowed_labels: Optional[Set[str]] = None,
        detector: Optional[CNNameDetector] = None,
        pkl_path: Optional[str] = None,
) -> Optional[List[Tuple[str, Optional[str]]]]:
    if detector is None:
        if pkl_path is None:
            if not _DETECTOR_CACHE:
                return None
            detector = next(iter(_DETECTOR_CACHE.values()))
        else:
            detector = build_or_load_detector(pkl_path)

    matches = detector.find_names(text)
    if allowed_labels is not None:
        matches = [m for m in matches if m[2] in allowed_labels]

    if not matches:
        return None

    segments: List[Tuple[str, Optional[str]]] = []
    prev_end = -1
    for start, end, label in matches:
        if start > prev_end + 1:
            prefix = text[prev_end + 1: start]
            if prefix:
                segments.append((prefix, None))
        segments.append((text[start: end + 1], label))
        prev_end = end

    if prev_end < len(text) - 1:
        tail = text[prev_end + 1:]
        if tail:
            segments.append((tail, None))

    return segments
