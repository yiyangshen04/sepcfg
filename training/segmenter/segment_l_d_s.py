# segmenter/segment_l_d_s.py
"""
L-D-S 片段语义细分（并行解析 + Top-K）
------------------------------------
* L 段：合并英文姓名/英文词 → 一棵 AC 生成候选 → K-best Viterbi
* D / S 段：沿用旧规则
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Callable, Iterable, Dict, Any
import re

import ahocorasick

# ---------- 自有依赖 ----------
from config import project_path

from .cn_name_detection import build_or_load_detector

# -----------------------------------------------------------

# ---------------- 全局缓存 ----------------
LDS_PKL_PATH = str(project_path("data", "cn_name_automata_lds.pkl"))
_LDS_CN_DETECTOR = build_or_load_detector(LDS_PKL_PATH)  # 中文姓名 AC

# ───────── Path-level penalty ─────────
SPLIT_PEN   = math.log(0.85)   # ≈-0.162
DIV_PEN     = math.log(0.90)   # ≈-0.105
CNMIX_PEN   = math.log(0.50)   # ≈-0.693

LOG_EN2 = math.log(0.50)       # en_word 长度=2
LOG_EN3 = math.log(0.75)       # en_word 长度=3

_UNK_BASE_P   = 0.03     # 长度 = 1 时的基准先验
_UNK_LEN_DEC  = 0.10     # 指数衰减系数；越长越低


_CN_NAME_PRIORS = {
    # 张三丰 ⇒ zhangsanfeng
    "cn_name_full":        0.25,   # 姓拼音 + 名拼音

    # 三丰 ⇒ sanfeng
    "cn_name_given":       0.18,   # 名字拼音

    # 张三丰 ⇒ zhangsf
    "cn_name_last_abbr":   0.1,   # 姓拼音 + 名首字母缩写

    # 张三丰 ⇒ zsf
    "cn_name_abbr":        0.18,   # 姓名首字母缩写

    # 三丰张 ⇒ sanfengzhang
    "cn_name_first_last":  0.15,   # 名拼音 + 姓拼音（先名后姓）

    # 张 ⇒ zhang
    "cn_name_last_full":   0.25,   # 姓拼音
}
_FALLBACK_P = 1e-8  # 兜底，防止 log(0)





# ─────────────────── 统一长度惩罚 ────────────────────
def length_penalty(lbl: str, length: int) -> float:
    """
    返回乘到先验概率上的惩罚系数（0-1].
    规则：
        • cn_name_full         : len < 5           → ×0.50
        • cn_name_given        : len < 3           → ×0.50
                                : len == 3          → ×0.75
        • cn_name_abbr         : len > 5           → ×0.25
                                : len != 3          → ×0.50
        • cn_name_first_last   : len < 5           → ×0.50
        • cn_name_last_full    : len < 2           → ×0.75
        • 其余                 : 无惩罚            → ×1.0
    """
    if lbl == "cn_name_full":
        return 0.5 if length < 5 else 1.0

    if lbl == "cn_name_given":
        if length < 3:
            return 0.5
        if length == 3:
            return 0.75
        return 1.0

    if lbl == "cn_name_abbr":
        if length > 5:
            return 0.25
        if length != 3:
            return 0.5
        return 1.0

    if lbl == "cn_name_first_last":
        return 0.5 if length < 5 else 1.0

    if lbl == "cn_name_last_full":
        if length < 2:  # 仅 1 个字 → 更强惩罚 ×0.20
            return 0.20
        if length < 3:  # 2 个字    → 适中惩罚 ×0.75
            return 0.75
        return 1.0  # ≥3 个字   → 不惩罚

    return 1.0

def _group_of(label: str) -> str:
    if label.startswith("en_"):       return "en"
    if label.startswith("cn_name"):   return "cn"
    if label == "py":                 return "py"
    if label == "unk_seg":            return "unk"
    return "nn"                       # nn / nn_char


def path_penalty(path: List[Candidate]) -> float:
    """
    为整条路径计算额外 log 惩罚（≤0）。
    • 碎片、多类别、中文姓名混搭：沿用旧规则
    • 新增：短 en_word（len==2 / 3）
    """
    n_seg   = len(path)
    labels  = [c.label for c in path]

    # 1) 片段数惩罚
    penalty = (n_seg - 1) * SPLIT_PEN if n_seg > 1 else 0.0

    # 2) 类别散惩罚
    groups = {_group_of(lbl) for lbl in labels}
    if len(groups) > 1:
        penalty += (len(groups) - 1) * DIV_PEN

    # 3) 中文姓名混搭惩罚
    if sum(lbl.startswith("cn_name") for lbl in labels) >= 2 \
       and len({lbl for lbl in labels if lbl.startswith("cn_name")}) >= 2:
        penalty += CNMIX_PEN

    # 4) 短 en_word 惩罚
    for cand in path:
        if cand.label == "en_word":
            seg_len = cand.end - cand.start + 1
            if seg_len == 2:
                penalty += LOG_EN2
            elif seg_len == 3:
                penalty += LOG_EN3

    return penalty
# -----------------------------------------------------------
# 1. 英文姓名 + 英文字典：统一 Aho-Corasick
# -----------------------------------------------------------


# ─────────────────── ① 修改 gen_unknown_seg ────────────────────
def gen_unknown_seg(text: str):
    """
    给整段 L 文本生成一个 unk_seg 候选。

    轻量去重策略：
    ─────────────────────────
    * 仅当 L-段长度 ≥ 2 时才生成 unk_seg
      （单字符由 nn / nn_char 兜底就够了）
    """
    if len(text) == 1:          # <── 新增：跳过单字符
        return

    L  = len(text)
    p  = _UNK_BASE_P * math.exp(-_UNK_LEN_DEC * max(L - 1, 0))
    p  = max(p, 1e-12)          # 防止 log(0)
    yield Candidate(0, L - 1, "unk_seg", math.log(p), text)


@lru_cache(maxsize=1)
def build_en_automaton(
        en_name_set: frozenset[str],
        en_word_set: frozenset[str],
) -> ahocorasick.Automaton:
    """
    把英文姓名、英文词典一次性塞进同一棵 AC。
    value = (label, 原词) 以区分 en_name / en_word。
    使用 frozenset 作 cache key，集合顺序不影响复用。
    """
    A = ahocorasick.Automaton()

    # 统一转小写以避免重复插入
    name_low_set = {w.lower() for w in en_name_set}

    for w in en_name_set:
        A.add_word(w.lower(), ("en_name", w))

    for w in en_word_set:
        wl = w.lower()
        if wl not in name_low_set:                  # 避免重复关键词
            A.add_word(wl, ("en_word", w))

    A.make_automaton()
    return A


def gen_en_cands(text: str, en_ac: ahocorasick.Automaton):
    """
    AC 单次扫描返回所有英文姓名 / 英文字典候选。
    """
    low = text.lower()
    for end, (lbl, w) in en_ac.iter(low):
        s = end - len(w) + 1
        logp = math.log(0.25 if lbl == "en_name" else 0.2)
        yield Candidate(s, end, lbl, logp, text)


# -----------------------------------------------------------
# 2. 并行解析所需数据结构 & 算法
# -----------------------------------------------------------
@dataclass
class Candidate:
    start: int
    end: int          # 右闭区间
    label: str
    logp: float       # ln P
    _src: str

    @property
    def text(self) -> str:
        return self._src[self.start: self.end + 1]


def lattice_from_detectors(
        text: str,
        detectors: List[Callable[[str], Iterable[Candidate]]],
) -> Dict[int, List[Candidate]]:
    """
    构建 lattice，并对未覆盖字符回退插入单字符 nn_char。

    轻量去重策略：
    ─────────────────────────
    * 只有在该位置 **没有任何单字符候选** 时才插入 nn_char，
      从而避免 nn / nn_char 重复。
    """
    lattice: Dict[int, List[Candidate]] = defaultdict(list)

    # 收集所有候选
    for det in detectors:
        for cand in det(text):
            lattice[cand.start].append(cand)

    # fallback：必要时插入 nn_char
    for i in range(len(text)):
        # 若该位置已有单字符（start==end==i）的候选，则跳过
        if not any(c.start == i and c.end == i for c in lattice[i]):
            lattice[i].append(
                Candidate(i, i, "nn_char", math.log(1e-8), text)
            )
    return lattice


def _drop_worst(heap: List[Tuple[float, ...]]):
    """
    堆中保留 neg_logp 最小的前 K 条；丢弃最差者。
    """
    worst_i = max(range(len(heap)), key=lambda i: heap[i][0])
    heap[worst_i] = heap[-1]
    heap.pop()
    if worst_i < len(heap):
        heapq._siftup(heap, worst_i)
        heapq._siftdown(heap, 0, worst_i)


def k_best_parse(
        text: str,
        lattice: Dict[int, List[Candidate]],
        K: int = 5,
) -> List[Tuple[float, List[Candidate]]]:
    """
    Viterbi K-best：返回 [(neg_logp, path)]，已按概率升序。
    """
    N = len(text)
    counter = itertools.count()
    dp: List[List[Tuple[float, int, List[Candidate]]]] = [
        [] for _ in range(N + 1)
    ]
    heapq.heappush(dp[0], (0.0, next(counter), []))

    for pos in range(N):
        if not dp[pos]:
            continue
        for neg_lp, _, path in dp[pos]:
            for cand in lattice.get(pos, []):
                new_neg = neg_lp - cand.logp
                new_path = path + [cand]
                heapq.heappush(
                    dp[cand.end + 1],
                    (new_neg, next(counter), new_path)
                )
                if len(dp[cand.end + 1]) > K:
                    _drop_worst(dp[cand.end + 1])

    return [
        (neg_lp, path)
        for neg_lp, _, path in sorted(dp[N], key=lambda x: x[0])[:K]
    ]


# -----------------------------------------------------------
# 3. 其他候选生成器
# -----------------------------------------------------------
# ─────────────────── 中文姓名候选生成器 ────────────────────
def gen_cn_name_cands(text: str):
    """
    中文姓名候选生成器（含长度惩罚）：
    • 先从 _CN_NAME_PRIORS 查基本先验
    • 再根据 length_penalty() 乘以惩罚系数
    • 保存 natural-log 概率到 Candidate.logp
    """
    for s, e, lbl in _LDS_CN_DETECTOR.find_names(text):
        base_p = _CN_NAME_PRIORS.get(lbl, _FALLBACK_P)
        length  = e - s + 1

        # 动态惩罚
        p = base_p * length_penalty(lbl, length)

        yield Candidate(s, e, lbl, math.log(p), text)

def gen_pinyin_nn_cands(text: str, pinyin_set: set[str]):
    """最长匹配拼音给 0.10，其余字符 nn 给 0.05"""
    if not pinyin_set:
        # 用户不再使用拼音词表时，避免 O(n^2) 的回退扫描
        for i in range(len(text)):
            yield Candidate(i, i, "nn", math.log(0.08), text)
        return

    low = text.lower()
    i, n = 0, len(text)
    while i < n:
        for j in range(n, i, -1):
            if low[i:j] in pinyin_set:
                yield Candidate(i, j - 1, "py", math.log(0.10), text)
                i = j
                break
        else:
            yield Candidate(i, i, "nn", math.log(0.08), text)
            i += 1


# -----------------------------------------------------------
# 4. L-segment 并行解析
# -----------------------------------------------------------
def process_l_segment_parallel(
        seg_text: str,
        en_ac: ahocorasick.Automaton,
        pinyin_set: set[str],
        topk: int = 5,
):
    detectors = [
        lambda t: gen_en_cands(t, en_ac),
        gen_cn_name_cands,
        lambda t: gen_pinyin_nn_cands(t, pinyin_set),
        gen_unknown_seg,
    ]
    lattice = lattice_from_detectors(seg_text, detectors)
    kbest_raw = k_best_parse(seg_text, lattice, K=topk)

    scored = []
    for neg_lp, path in kbest_raw:
        logp = -neg_lp  # 还原 log 概率
        logp += path_penalty(path)  # 加惩罚（≤0）
        scored.append((path, logp))

    # 概率降序排列，格式沿用旧版
    scored.sort(key=lambda x: -x[1])
    return [
        ([(seg_text[c.start:c.end + 1], c.label) for c in path], lp)
        for path, lp in scored
    ]

# -----------------------------------------------------------
# 5. D / S 段（旧规则）
# -----------------------------------------------------------
# ─────────────────── 手机号校验（保持上版逻辑） ────────────────────
_MOBILE_RE = re.compile(
    r"""^1(?:3\d|4[5-9]|5[0-35-9]|6[2567]|7[0-8]|8\d|9[0-35-9])\d{8}$""",
    re.VERBOSE
)
def is_valid_cn_mobile(s: str) -> bool:
    return bool(_MOBILE_RE.match(s))

def _yy_to_year(yy: str) -> int | None:
    """
    将两位年份 yy 映射成 1950-2015 之间的 4 位年份。
    00-15 → 2000-2015
    50-99 → 1950-1999
    其余返回 None 表示不在合法范围。
    """
    v = int(yy)
    if 0 <= v <= 15:          # 2000-2015
        return 2000 + v
    if 50 <= v <= 99:         # 1950-1999
        return 1900 + v
    return None               # 16-49 ⇒ 超出范围




# ─────────────────── 日期校验：固定 6/8 位 ────────────────────────
def is_valid_date_6(s: str) -> bool:          # yy mm dd
    if len(s) != 6 or not s.isdigit():
        return False
    yy, mm, dd = s[:2], s[2:4], s[4:6]

    full_year = _yy_to_year(yy)
    if full_year is None:                     # 16-49 被排除
        return False

    try:
        datetime(full_year, int(mm), int(dd))
        return True
    except ValueError:
        return False


def is_valid_date_8(s: str) -> bool:          # yyyy mm dd
    try:
        dt = datetime.strptime(s, "%Y%m%d")
        return 1950 <= dt.year <= 2015
    except ValueError:
        return False

# ─────────────────── 年-月（yyyy m[m]） ──────────────────────
def is_valid_year_month(s: str) -> bool:
    """
    5-6 位数字串：
        yyyyM   （1-9 月）      例: 19851
        yyyyMM  （01-12 月）    例: 198509
    年份限定 1950-2015，月份 1-12。
    """
    if not (5 <= len(s) <= 6 and s.isdigit()):
        return False

    year  = int(s[:4])
    month = int(s[4:])          # 长度 1 或 2 都支持
    return 1950 <= year <= 2015 and 1 <= month <= 12


# ─────────────────── 月-日（m[m] d[d]） ─────────────────────
def is_valid_mmdd(s: str) -> bool:
    """
    2-4 位数字串，支持前导 0 省略：
        M D      例: 31        (3 月 1 日)
        M DD     例: 731       (7 月 31 日)
        MM D     例: 101       (10 月 1 日)
        MM DD    例: 0731      (07 月 31 日)
    仅校验 1-12 月、1-31 日的真实组合。
    """
    if not (2 <= len(s) <= 4 and s.isdigit()):
        return False

    for m_len in (1, 2):
        d_len = len(s) - m_len
        if d_len < 1 or d_len > 2:
            continue
        month = int(s[:m_len])
        day   = int(s[m_len:])
        try:
            datetime(2000, month, day)        # 年随便取，只为验证月日
            return True
        except ValueError:
            pass
    return False


# ─────────────────── 日期校验：省略 0 版本 ─────────────────────
def _valid_compact_date(year: str, rest: str) -> bool:
    """
    year: 'yy' 或 'yyyy'
    rest: 月日混合字符串（长度 2-4）
    hand-crafted 校验，不再用 strptime 的 %y 规则
    """
    # 解析年份
    if len(year) == 2:                         # yy
        full_year = _yy_to_year(year)
    else:                                      # yyyy
        full_year = int(year) if year.isdigit() else None

    if full_year is None or not (1950 <= full_year <= 2015):
        return False

    # 穷举月 / 日位数组合
    for m_len in (1, 2):
        for d_len in (1, 2):
            if m_len + d_len != len(rest):
                continue
            month = rest[:m_len]
            day   = rest[m_len:]
            if int(month) == 0 or int(day) == 0:
                continue
            try:
                datetime(full_year, int(month), int(day))
                return True
            except ValueError:
                pass
    return False



def is_valid_date_6_compact(s: str) -> bool:   # 4-6 位：yy m d
    return 4 <= len(s) <= 6 and s.isdigit() and \
           _valid_compact_date(s[:2], s[2:])

def is_valid_date_8_compact(s: str) -> bool:   # 6-8 位：yyyy m d
    return 6 <= len(s) <= 8 and s.isdigit() and \
           _valid_compact_date(s[:4], s[4:])


# ─────────────────── 主分派函数 ───────────────────────────────
def process_d_segment(s: str):
    """
    处理 D 段（数字串），对省略 0 的日期分支前后各插入一个 boundary，
    以便后续 expand_multibranch_segments 正确把它们当成单独一组展开。
    """
    L = len(s)

    # 6. 中国大陆手机号（严格 11 位）
    if L == 11 and is_valid_cn_mobile(s):
        return [(s, "cn_mobile")]

    # 1. 年份（4 位，1900-2100）
    if L == 4 and s.isdigit() and 1950 <= int(s) <= 2015:
        return [(s, "year")]

    # 2. yy mm dd —— 固定 6 位
    if L == 6 and s.isdigit() and is_valid_date_6(s):
        return [(s, "yymmdd")]

    # 3. yyyy mm dd —— 固定 8 位
    if L == 8 and s.isdigit() and is_valid_date_8(s):
        return [(s, "yyyymmdd")]

    # 3½. yyyy mm —— 年 + 月（5-6 位，可省 0）
    if 5 <= L <= 6 and is_valid_year_month(s):
        return [
            ("", "__boundary__"),
            ([(s, "yyyymm" if L == 6 else "yyyymm_nopad")], math.log(0.20)),
            ([(s, f"number{L}")],                          math.log(0.12)),
            ("", "__boundary__"),
        ]

    # 3¾. mm dd —— 月 + 日（2-4 位，可全部省 0）
    if 2 <= L <= 4 and is_valid_mmdd(s):
        return [
            ("", "__boundary__"),
            ([(s, "mmdd" if L == 4 else "mmdd_nopad")],    math.log(0.20)),
            ([(s, f"number{L}")],                          math.log(0.12)),
            ("", "__boundary__"),
        ]


    # 4. yy m d —— 省略 0（4-6 位）
    if 4 <= L <= 6 and s.isdigit() and is_valid_date_6_compact(s):
        return [
            ("", "__boundary__"),
            ([(s, "yymmdd_nopad")], math.log(0.25)),
            ([(s, f"number{L}")], math.log(0.2)),
            ("", "__boundary__"),
        ]

    # 5. yyyy m d —— 省略 0（6-8 位）
    if 6 <= L <= 8 and s.isdigit() and is_valid_date_8_compact(s):
        return [
            ("", "__boundary__"),
            ([(s, "yyyymmdd_nopad")], math.log(0.25)),
            ([(s, f"number{L}")], math.log(0.15)),
            ("", "__boundary__"),
        ]

    # 7. 其他数字串
    return [(s, f"number{L}")]

def process_s_segment(s: str):
    return [(s, f"spec{len(s)}")]


# -----------------------------------------------------------
# 6. 顶层接口
# -----------------------------------------------------------
def segment_l_d_s(
        segments: List[Tuple[str, str]],
        en_name_set_or_en_ac,
        en_dict_set_or_pinyin_set,
        pinyin_set: set[str] | None = None,
        topk_l: int = 5,
):
    """
    `segments` 例: [('abc','L'), ('20250101','D'), ...]   # 粗分
    返回混合列表：
        • L 段: [ ([(txt,label)...], logp), … ]   (多分支)
        • D/S 段: (txt, label)

    兼容两种调用方式：
      1) 旧版：segment_l_d_s(segments, en_name_set, en_dict_set, pinyin_set, topk_l=...)
      2) 新版（更快）：segment_l_d_s(segments, en_ac, pinyin_set, topk_l=...)
    """
    if pinyin_set is None:
        en_ac = en_name_set_or_en_ac
        pinyin_set = en_dict_set_or_pinyin_set
    else:
        en_ac = build_en_automaton(
            frozenset(en_name_set_or_en_ac),
            frozenset(en_dict_set_or_pinyin_set),
        )

    new_segs: List[Any] = []
    for txt, typ in segments:
        if typ == "L":
            new_segs.extend(
                process_l_segment_parallel(
                    txt, en_ac, pinyin_set, topk=topk_l
                )
            )
        elif typ == "D":
            new_segs.extend(process_d_segment(txt))
        elif typ == "S":
            new_segs.extend(process_s_segment(txt))
        else:
            new_segs.append((txt, typ))
    return new_segs
