#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
personal_pcfg_generator_fixed.py
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import heapq
import inspect
import math
import pickle
import re
import sys
from dataclasses import dataclass as _dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ────────── 依赖：拼音 ─────────────────────────────────────────────────────────────
try:
    from pypinyin import lazy_pinyin, Style
except ImportError:
    print("❌ 需要先安装 pypinyin：  pip install pypinyin", file=sys.stderr)
    sys.exit(1)


def dataclass(*args, **kwargs):
    # Python 3.9 的 dataclass 不支持 slots 参数（3.10+ 才支持）
    if "slots" in kwargs and "slots" not in inspect.signature(_dataclass).parameters:
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)

# ═══════════════════════ 0. 通用工具 ═══════════════════════════════════════════════

_DIGIT_RE = re.compile(r"\D+")


def safe_eq(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) < eps


def normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if safe_eq(s, 1.0):
        return d
    if s == 0.0:
        n = len(d) or 1
        return {k: 1.0 / n for k in d}
    inv = 1.0 / s
    return {k: v * inv for k, v in d.items()}


def inject(
    orig: Dict[str, float], injection: Dict[str, float], keep_orig_frac: float
) -> Dict[str, float]:
    """
    keep_orig_frac = 1  ➜ 完全保留旧表
    keep_orig_frac = 0  ➜ 全部用 injection
    注：injection 内部应自行归一化 (sum = 1)
    """
    keep_orig_frac = max(0.0, min(1.0, keep_orig_frac))
    new_d: Dict[str, float] = {}

    # ① 旧值按比例缩放
    if keep_orig_frac > 0.0:
        for k, v in orig.items():
            new_d[k] = v * keep_orig_frac

    # ② 注入
    if injection:
        scale = 1.0 - keep_orig_frac
        for k, frac in injection.items():
            new_d[k] = new_d.get(k, 0.0) + frac * scale

    return normalize(new_d)


def name_variants(cn_name: str) -> Dict[str, str]:
    if not cn_name:
        return {}
    surname = cn_name[0]
    given = cn_name[1:]

    sur_py = "".join(lazy_pinyin(surname))
    giv_list = lazy_pinyin(given)
    giv_py = "".join(giv_list)
    giv_initials = "".join(g[0] for g in giv_list)
    return {
        "cn_name_full": f"{sur_py}{giv_py}",
        "cn_name_full_special": f"{sur_py}[SPECIAL]{giv_py}",
        "cn_name_given": giv_py,
        "cn_name_last_abbr": f"{sur_py}{giv_initials}",
        "cn_name_abbr": f"{sur_py[0]}{giv_initials}",
        "cn_name_first_last": f"{giv_py}{sur_py}",
        "cn_name_first_last_special": f"{giv_py}[SPECIAL]{sur_py}",
        "cn_name_last_full": sur_py,
    }


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%m-%d-%Y",
    "%m-%d-%y",
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%y-%m-%d",
    "%y/%m/%d",
]


def parse_birth_date(text: str) -> dt.date | None:
    t = text.strip()
    if not t:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    # 尝试数字压缩格式：20240509 / 240509
    if t.isdigit():
        if len(t) == 8:
            return dt.datetime.strptime(t, "%Y%m%d").date()
        if len(t) == 6:
            return dt.datetime.strptime(t, "%y%m%d").date()
    return None


def date_variants(d: dt.date) -> Dict[str, str]:
    yyyy = f"{d:%Y}"
    yyyymmdd = f"{d:%Y%m%d}"
    yymmdd = f"{d:%y%m%d}"

    mm, dd = d.month, d.day
    yyyymmdd_np = f"{d.year}{mm}{dd}"
    yymmdd_np = f"{d:%y}{mm}{dd}"
    return {
        "year": yyyy,
        "yyyymmdd": yyyymmdd,
        "yymmdd": yymmdd,
        "yyyymmdd_nopad": yyyymmdd_np,
        "yymmdd_nopad": yymmdd_np,
    }


def clean_phone(raw: str) -> str | None:
    digits = _DIGIT_RE.sub("", raw or "")
    return digits if len(digits) == 11 else None


def split_email(raw: str) -> Tuple[str | None, str | None]:
    if "@" not in raw:
        return None, None
    name, domain_full = raw.split("@", 1)
    domain_main = domain_full.rsplit(".", 1)[0]
    return name, domain_main


# ═══════════════════ 1. 读取 PCFG 计数 ════════════════════════════════════════════


def load_pcfg_data(pkl_path: Path):
    with pkl_path.open("rb") as fh:
        state = pickle.load(fh)

    sp_counts = state["sp_counts"]
    sp_total = state["sp_total"]
    sft_counts = state["sft_counts"]
    sft_totals = state["sft_totals"]

    sp_tpl, sp_logp = [], []
    ln = math.log
    for tpl, cnt in sp_counts.items():
        sp_tpl.append(tpl)
        sp_logp.append(-ln(cnt / sp_total))

    sft_probs: Dict[str, Dict[str, float]] = {}
    for k, cnt_dict in sft_counts.items():
        tot = sft_totals.get(k, 0)
        if not tot:
            continue
        inv = 1.0 / tot
        sft_probs[k] = {sf: c * inv for sf, c in cnt_dict.items()}

    del state, sp_counts, sft_counts, sp_total, sft_totals
    gc.collect()
    return sp_tpl, sp_logp, sft_probs


# ═══════════════════ 2. 个性化概率表 ══════════════════════════════════════════════


def build_personal_probs(
    base: Dict[str, Dict[str, float]], row: dict
) -> Dict[str, Dict[str, float]]:
    """
    返回新的 sft_probs；浅拷贝 top-level，仅深拷 label 级别
    """
    sft = {k: v for k, v in base.items()}  # top-level 浅拷

    def ensure_copy(label: str):
        if label in sft:
            sft[label] = dict(sft[label])
        else:
            sft[label] = {}

    # ── 解析字段 ────────────────────────────────────────────────────────────────
    account = str(row.get("account", "")).strip()
    cn_name = str(row.get("教师姓名", "")).strip()
    birth_raw = str(row.get("出生日期", "")).strip()
    phone = clean_phone(str(row.get("联系电话", "")))
    email_raw = str(row.get("电子邮箱", ""))

    birth_dt = parse_birth_date(birth_raw)
    email_name, email_domain = split_email(email_raw)

    name_vars = name_variants(cn_name) if cn_name else {}
    date_vars = date_variants(birth_dt) if birth_dt else {}

    # ── 注入 ────────────────────────────────────────────────────────────────────
    # 1) cn_mobile – 100% 替换
    if phone:
        sft["cn_mobile"] = {phone: 1.0}

    # 2) 年份 & 日期
    if birth_dt:
        # year：50 % CSV
        ensure_copy("year")
        sft["year"] = inject(sft["year"], {date_vars["year"]: 1.0}, 0.5)

        # yymmdd / yyyymmdd：15 % CSV
        for lab in ("yymmdd", "yyyymmdd"):
            ensure_copy(lab)
            sft[lab] = inject(sft[lab], {date_vars[lab]: 1.0}, 0.1)

        # nopad：50 % CSV
        for lab in ("yymmdd_nopad", "yyyymmdd_nopad"):
            ensure_copy(lab)
            sft[lab] = inject(sft[lab], {date_vars[lab]: 1.0}, 0.35)

    # 3) 账号相关
    if account:
        sft["acc_pwd_same"] = {account: 1.0}

    sft["acc_email_domain"] = {"sufe": 0.9, "shufe": 0.1}
    sft["acc_email_domain_com"] = {"sufe.com": 0.9, "shufe.com": 0.1}

    if email_name:
        sft["acc_email_name"] = {email_name: 1.0}

    # 4) 中文姓名 8 个标签
    if name_vars:
        for lab, keep_frac in [
            ("cn_name_full", 0.10),
            ("cn_name_first_last", 0.10),
            ("cn_name_full_special", 0.50),
            ("cn_name_first_last_special", 0.50),
            ("cn_name_given", 0.30),
            ("cn_name_last_abbr", 0.30),
            ("cn_name_last_full", 0.40),
        ]:
            ensure_copy(lab)
            sft[lab] = inject(sft[lab], {name_vars[lab]: 1.0}, keep_frac)

    # 保证所有子 dict 归一化
    for k, d in sft.items():
        sft[k] = normalize(d)

    return sft


# ═══════════════════ 3. Grammar & 生成 ═════════════════════════════════════════════


@dataclass(order=True, slots=True)
class _Node:
    neg_logp: float
    sp_id: int
    idx: int
    parts: List[str] = field(compare=False)


class PCFG:
    def __init__(
        self,
        sp_tpl: List[Tuple[str, ...]],
        sft_probs: Dict[str, Dict[str, float]],
    ):
        self.sp_tpl = sp_tpl
        self.sft = sft_probs
        self.ln = math.log

    def children(self, node: _Node):
        tpl = self.sp_tpl[node.sp_id]
        if node.idx >= len(tpl):
            return
        label = tpl[node.idx]
        sf_dict = self.sft.get(label, {})
        for sf, p in sf_dict.items():
            yield _Node(node.neg_logp - self.ln(p), node.sp_id, node.idx + 1, node.parts + [sf])


class Queue:
    def __init__(
        self,
        pcfg: PCFG,
        sp_logp: List[float],
        min_prob: float,
        max_q: int,
    ):
        self.pcfg = pcfg
        self.heap: List[_Node] = []
        self.max_q = max_q
        self.cut = int(max_q * 1.2)
        self.th = -math.log(min_prob)

        for sp_id, nlogp in enumerate(sp_logp):
            if nlogp <= self.th:
                heapq.heappush(self.heap, _Node(nlogp, sp_id, 0, []))
        self._trim(force=True)

    def _trim(self, force=False):
        if not force and len(self.heap) <= self.cut:
            return
        if len(self.heap) > self.max_q:
            self.heap[:] = heapq.nsmallest(self.max_q, self.heap)
            heapq.heapify(self.heap)

    def pop_complete(self) -> _Node | None:
        while self.heap:
            n = heapq.heappop(self.heap)
            if n.idx == len(self.pcfg.sp_tpl[n.sp_id]):
                return n
            for c in self.pcfg.children(n):
                if c.neg_logp <= self.th:
                    heapq.heappush(self.heap, c)
            self._trim()
        return None


def generate(
    sp_tpl,
    sp_logp,
    sft_personal,
    outfile: Path,
    top_n=10_000,
    min_prob=1e-10,
    max_q=50_000,
):
    pcfg = PCFG(sp_tpl, sft_personal)
    q = Queue(pcfg, sp_logp, min_prob, max_q)

    with outfile.open("w", encoding="utf-8") as fh:
        produced = 0
        while produced < top_n:
            node = q.pop_complete()
            if node is None:
                break
            pwd = "".join(node.parts)
            prob = math.exp(-node.neg_logp)
            fh.write(f"{pwd}\t{prob:.8e}\n")
            produced += 1
    return produced


# ═══════════════════ 4. CLI 示例 ══════════════════════════════════════════════════


def main():
    from config import project_path

    root = project_path()
    PKL_PATH = root / "checkpoints" / "my_training_20250412_1M_counts.pkl"
    CSV_PATH = root / "generation" / "password_gen_tools" / "teacher.csv"
    OUT_DIR = root / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp_tpl, sp_logp, base_sft = load_pcfg_data(PKL_PATH)
    print(f"[INFO] 模板 {len(sp_tpl)}，标签 {len(base_sft)}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    for idx, row in enumerate(rows, 1):
        acc = str(row.get("account", f"row{idx}")).strip() or f"row{idx}"
        out_txt = OUT_DIR / f"{acc}.txt"

        personal_probs = build_personal_probs(base_sft, row)
        cnt = generate(sp_tpl, sp_logp, personal_probs, out_txt)
        print(f"[{idx:>3}/{len(rows)}] {acc}: 生成 {cnt} 条 → {out_txt}")

    print("[INFO] 全部完成")


if __name__ == "__main__":
    main()
