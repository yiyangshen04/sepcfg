#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_placeholders_mod.py  (patched 2025‑05‑19)

对 2025‑05‑17 版本再作两项功能增强：
1. **[SPECIAL] 替换三套符号**：模板产出含 "[SPECIAL]" 的口令时，分别将其替换为
   "@", "_", "."。大小写版本各保留三条；原始含 "[SPECIAL]" 的行不再输出。
2. **<CN_NAME_ABBR> 衍生『名缩写』**：在生成姓首字母+名首字母 (如 xhf / Xhf)
   之外，额外输出仅名缩写 (hf / Hf) 的候选。

其它逻辑保持不变。
"""

import csv
import datetime as dt
import heapq
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

try:
    from pypinyin import lazy_pinyin
except ImportError:  # 若无 pypinyin，保证脚本可跑，但中文占位符全部缺失
    def lazy_pinyin(_: str):
        return []

# ── 常量 ───────────────────────────────────────────────────────────────────

COMPOUND_SURNAMES = {
    "欧阳", "司徒", "上官", "夏侯", "诸葛", "闻人", "东方", "赫连", "皇甫", "尉迟",
    "公孙", "慕容", "申屠", "公冶", "羊舌", "万俟", "南宫", "东郭", "百里", "淳于",
    "端木", "公良", "左丘", "梁丘", "太叔", "叔孙", "西门", "长孙", "呼延", "仲孙",
    "轩辕", "令狐", "钟离", "鲜于", "闾丘", "濮阳", "宇文", "洛阳", "玄穆", "司空",
}

CHN_PLACEHOLDERS: List[str] = [
    "<CN_NAME_FULL>", "<CN_NAME_FULL_SPECIAL>", "<CN_NAME_GIVEN>", "<CN_NAME_LAST_ABBR>",
    "<CN_NAME_ABBR>", "<CN_NAME_FIRST_LAST>", "<CN_NAME_FIRST_LAST_SPECIAL>", "<CN_NAME_LAST_FULL>",
]

SPECIAL_REPLACEMENTS: List[str] = ["@", "_", "."]  # 新增

PH_PATTERN = re.compile(r"<[A-Z_]+?>")


def _normalize_weights(ws: List[float]) -> List[float]:
    if not ws:
        raise ValueError("weights cannot be empty")
    if any(w < 0 for w in ws):
        raise ValueError(f"weights must be non-negative, got: {ws}")
    s = sum(ws)
    if s <= 0:
        raise ValueError(f"weights sum must be > 0, got: {ws}")
    return [w / s for w in ws]


def _patternize_password(s: str) -> str:
    """
    将明文口令映射为可读的“模式串”，避免输出真实内容。
    规则：
      - 数字串 → D{len}
      - 小写字母串 → l{len}
      - 大写字母串 → U{len}
      - 其它字符（常见为符号）原样保留
    """
    if not s:
        return "EMPTY"

    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isdigit():
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            out.append(f"D{j - i}")
            i = j
            continue

        # 注：这里刻意用 islower/isupper，只对 ASCII 拉丁字母的常见情况有意义
        if "a" <= ch <= "z":
            j = i + 1
            while j < n and ("a" <= s[j] <= "z"):
                j += 1
            out.append(f"l{j - i}")
            i = j
            continue

        if "A" <= ch <= "Z":
            j = i + 1
            while j < n and ("A" <= s[j] <= "Z"):
                j += 1
            out.append(f"U{j - i}")
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)

# ── 工具函数 ───────────────────────────────────────────────────────────────

def is_ascii_name(name: str) -> bool:
    return bool(re.search(r"[A-Za-z]", name))


def split_chinese_name(name: str):
    name = name.strip()
    for comp in COMPOUND_SURNAMES:
        if name.startswith(comp):
            return comp, name[len(comp):]
    return name[:1], name[1:]


def capitalize_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def name_vars(name: str):
    """返回两个映射 dict_lower, dict_upper。新增返回 `abbr_given_only`: 名字首字母串。"""

    if not name or is_ascii_name(name):
        return {}, {}

    sur, giv = split_chinese_name(name)
    sp = lazy_pinyin(sur)
    gp = lazy_pinyin(giv)
    if not sp or not gp:
        return {}, {}

    S = "".join(sp)                # 姓全拼
    G = "".join(gp)                # 名全拼
    I = "".join(x[0] for x in gp)   # 名首字母串

    lower = {
        "<CN_NAME_FULL>":               f"{S}{G}",
        "<CN_NAME_FULL_SPECIAL>":       f"{S}[SPECIAL]{G}",
        "<CN_NAME_GIVEN>":              G,
        "<CN_NAME_LAST_ABBR>":          f"{S}{I}",
        "<CN_NAME_ABBR>":               f"{S[0]}{I}",      # xhf
        "<CN_NAME_FIRST_LAST>":         f"{G}{S}",
        "<CN_NAME_FIRST_LAST_SPECIAL>": f"{G}[SPECIAL]{S}",
        "<CN_NAME_LAST_FULL>":          S,
        # 辅助字段（模板中不会直接出现）
        "__ABBR_GIVEN_ONLY__":          I,                  # hf
    }
    upper = {k: capitalize_first(v) if not k.startswith("__") else v for k, v in lower.items()}
    return lower, upper


def parse_date(s: str):
    fmts = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%y/%m/%d", "%Y%m%d", "%y%m%d", "%Y.%m.%d")
    s = s.strip()
    for fmt in fmts:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def date_vars(d: dt.date):
    if not d:
        return {}
    mm, dd = d.month, d.day
    return {
        "<BIRTH_YEAR>":         f"{d:%Y}",
        "<BIRTH_YYYYMM>":       f"{d:%Y}{d:%m}",
        "<BIRTH_YYYYMM_NP>":    f"{d.year}{mm}",
        "<BIRTH_MMDD>":         f"{d:%m}{d:%d}",
        "<BIRTH_MMDD_NP>":      f"{mm}{dd}",
        "<BIRTH_YYYYMMDD>":     f"{d:%Y%m%d}",
        "<BIRTH_YYMMDD>":       f"{d:%y%m%d}",
        "<BIRTH_YYYYMMDD_NP>":  f"{d.year}{mm}{dd}",
        "<BIRTH_YYMMDD_NP>":    f"{d:%y}{mm}{dd}",
    }


def safe_phone(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    if re.fullmatch(r"\d+(\.\d+)?e[+-]?\d+", raw, re.I):
        try:
            raw = str(int(float(raw)))
        except ValueError:
            pass
    return raw

# ── 模板读入 ───────────────────────────────────────────────────────────────

def load_templates(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            pw, p = line.rstrip("\n").split("\t")
            out.append((pw, p))
    return out


def _iter_template_lines(
    *,
    templates_path: Path,
    max_templates: int = 0,
) -> Iterable[Tuple[str, float]]:
    with templates_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            if max_templates and idx > max_templates:
                break
            pw, p = line.rstrip("\n").split("\t")
            yield pw, float(p)


def _identity(s: str) -> str:
    return s


def _precompute_static_mass(
    templates: Iterable[Tuple[str, float]],
    *,
    key_fn: Callable[[str], str],
    ws_special: List[float],
) -> Tuple[Dict[str, float], float, List[Tuple[str, float, bool, bool]]]:
    """
    预处理模板：
    - 静态模板（不含任何 '<'）可对所有账号复用：直接聚合到 key→mass
    - 动态模板（含 '<'）需要每个账号 materialize：
        返回 (tpl, p_tpl, has_cn_placeholder, has_cn_abbr)
    """
    static_prob_by_key: Dict[str, float] = {}
    static_total_mass = 0.0
    dynamic_templates: List[Tuple[str, float, bool, bool]] = []

    for tpl, p_tpl in templates:
        if "<" not in tpl:
            # 注意：静态模板也可能包含 [SPECIAL]，需按权重展开
            if "[SPECIAL]" in tpl:
                for ch, w_special in zip(SPECIAL_REPLACEMENTS, ws_special):
                    if w_special <= 0:
                        continue
                    key = key_fn(tpl.replace("[SPECIAL]", ch))
                    static_prob_by_key[key] = static_prob_by_key.get(key, 0.0) + (p_tpl * w_special)
                    static_total_mass += p_tpl * w_special
            else:
                key = key_fn(tpl)
                static_prob_by_key[key] = static_prob_by_key.get(key, 0.0) + p_tpl
                static_total_mass += p_tpl
            continue

        has_cn = "<CN_NAME_" in tpl
        has_abbr = "<CN_NAME_ABBR>" in tpl
        dynamic_templates.append((tpl, p_tpl, has_cn, has_abbr))

    return static_prob_by_key, static_total_mass, dynamic_templates


def _fill_placeholders_mass(
    *,
    templates_path: Path,
    teacher_csv_path: Path,
    out_dir: Path,
    max_templates: int,
    top_k: int,
    special_weights: Tuple[float, float, float],
    case_weights: Tuple[float, float],
    abbr_weights: Tuple[float, float],
    key_fn: Callable[[str], str],
    mode: str,
    key_name: str,
    warn_plaintext: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if warn_plaintext:
        print(
            "[WARN] plain-mass mode writes plaintext passwords to disk. "
            "Keep output private and do not commit it.",
            file=sys.stderr,
        )

    ws_special = _normalize_weights(list(special_weights))
    ws_case = _normalize_weights(list(case_weights))
    ws_abbr = _normalize_weights(list(abbr_weights))

    static_prob_by_key, static_total_mass, dynamic_templates = _precompute_static_mass(
        _iter_template_lines(templates_path=templates_path, max_templates=max_templates),
        key_fn=key_fn,
        ws_special=ws_special,
    )
    dynamic_non_cn = [(tpl, p) for (tpl, p, has_cn, _has_abbr) in dynamic_templates if not has_cn]
    dynamic_cn = [(tpl, p, has_abbr) for (tpl, p, has_cn, has_abbr) in dynamic_templates if has_cn]

    with teacher_csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        teachers = list(reader)

    for idx, row in enumerate(teachers, 1):
        acct = (row.get("account", "").strip() or f"row{idx}")

        mapping_base: Dict[str, str] = {}
        mapping_base["<PHONENUM>"] = safe_phone(row.get("联系电话", ""))
        mapping_base["<ACCOUNT>"] = row.get("account", "").strip() or None
        email_user = (row.get("电子邮箱", "").split("@", 1)[0] or None)
        mapping_base["<EMAIL_NAME>"] = email_user
        mapping_base.update(date_vars(parse_date(row.get("出生日期", ""))))

        lower_cn, upper_cn = name_vars(row.get("教师姓名", ""))
        has_chinese_name = bool(lower_cn)

        prob_by_key: Dict[str, float] = {}
        get_mass = prob_by_key.get  # type: ignore[assignment]
        materialize_fn = materialize

        mapping_variants_cn = [
            ({**mapping_base, **lower_cn}, ws_case[0]),
            ({**mapping_base, **upper_cn}, ws_case[1]),
        ]

        # 1) 不含中文占位符的动态模板
        for pw_tpl, p_tpl in dynamic_non_cn:
            mv = mapping_base
            pw0 = materialize_fn(pw_tpl, mv)
            if pw0 is None:
                continue

            if "[SPECIAL]" in pw0:
                special_variants = [
                    (pw0.replace("[SPECIAL]", "@"), ws_special[0]),
                    (pw0.replace("[SPECIAL]", "_"), ws_special[1]),
                    (pw0.replace("[SPECIAL]", "."), ws_special[2]),
                ]
            else:
                special_variants = [(pw0, 1.0)]

            for pw1, w_special in special_variants:
                w = w_special
                if w <= 0:
                    continue
                key = key_fn(pw1)
                prob_by_key[key] = get_mass(key, 0.0) + (p_tpl * w)

        # 2) 含中文占位符的动态模板（若缺中文名则整体跳过）
        if has_chinese_name:
            for pw_tpl, p_tpl, has_abbr in dynamic_cn:
                for mv, w_case in mapping_variants_cn:
                    pw0 = materialize_fn(pw_tpl, mv)
                    if pw0 is None:
                        continue

                    if "[SPECIAL]" in pw0:
                        special_variants = [
                            (pw0.replace("[SPECIAL]", "@"), ws_special[0]),
                            (pw0.replace("[SPECIAL]", "_"), ws_special[1]),
                            (pw0.replace("[SPECIAL]", "."), ws_special[2]),
                        ]
                    else:
                        special_variants = [(pw0, 1.0)]

                    abbr_full = mv.get("<CN_NAME_ABBR>") if has_abbr else None
                    abbr_given_only = mv.get("__ABBR_GIVEN_ONLY__") if has_abbr else None

                    for pw1, w_special in special_variants:
                        if has_abbr and abbr_full and abbr_given_only and abbr_full in pw1:
                            abbr_variants = [
                                (pw1, ws_abbr[0]),
                                (pw1.replace(abbr_full, abbr_given_only, 1), ws_abbr[1]),
                            ]
                        else:
                            abbr_variants = [(pw1, 1.0)]

                        for final_pw, w_abbr in abbr_variants:
                            w = w_case * w_special * w_abbr
                            if w <= 0:
                                continue
                            key = key_fn(final_pw)
                            prob_by_key[key] = get_mass(key, 0.0) + (p_tpl * w)

        # 合并静态 + 动态（避免为每个账号重复跑静态模板）
        def _iter_combined_items():
            for key, p_static in static_prob_by_key.items():
                yield key, p_static + prob_by_key.get(key, 0.0)
            for key, p_dyn in prob_by_key.items():
                if key not in static_prob_by_key:
                    yield key, p_dyn

        top_items = heapq.nlargest(top_k, _iter_combined_items(), key=lambda kv: kv[1])
        top_items.sort(key=lambda kv: (-kv[1], kv[0]))  # 稳定输出
        total_mass = static_total_mass + sum(prob_by_key.values())

        out_f = out_dir / f"{acct}.txt"
        with out_f.open("w", encoding="utf-8") as of:
            of.write(f"# mode={mode}\n")
            of.write(f"# templates={templates_path.name}\n")
            of.write(f"# max_templates={max_templates}\n")
            of.write(f"# top_k={top_k}\n")
            of.write(f"# total_mass={total_mass:.8e}\n")
            if warn_plaintext:
                of.write("# WARNING: plaintext passwords below\n")
            of.write(f"# columns: {key_name}\\tprob\n")
            for key, p in top_items:
                of.write(f"{key}\t{p:.8e}\n")

        print(f"[{idx}/{len(teachers)}] → {out_f} (top{top_k}, mass={total_mass:.3e})")


def fill_placeholders(
    *,
    templates_path: Path,
    teacher_csv_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = load_templates(templates_path)

    with teacher_csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        teachers = list(reader)

    for idx, row in enumerate(teachers, 1):
        acct = (row.get("account", "").strip() or f"row{idx}")

        # 基础映射：电话/账号/邮箱/日期
        mapping_base: Dict[str, str] = {}
        mapping_base["<PHONENUM>"] = safe_phone(row.get("联系电话", ""))
        mapping_base["<ACCOUNT>"] = row.get("account", "").strip() or None
        email_user = (row.get("电子邮箱", "").split("@", 1)[0] or None)
        mapping_base["<EMAIL_NAME>"] = email_user
        mapping_base.update(date_vars(parse_date(row.get("出生日期", ""))))

        # 中文姓名映射
        lower_cn, upper_cn = name_vars(row.get("教师姓名", ""))
        has_chinese_name = bool(lower_cn)

        output_lines: List[str] = []

        for pw_tpl, prob in templates:
            contains_cn_ph = "<CN_NAME_" in pw_tpl

            if contains_cn_ph and not has_chinese_name:
                continue

            # 生成映射变体
            variant_list: List[Dict[str, str]] = []
            if contains_cn_ph:
                variant_list.append({**mapping_base, **lower_cn})
                variant_list.append({**mapping_base, **upper_cn})
            else:
                variant_list.append(mapping_base)

            for mv in variant_list:
                pw = materialize(pw_tpl, mv)
                if pw is None:
                    continue

                cand_set: Set[str] = set()

                # —— ① [SPECIAL] →  @  _  . ——
                if "[SPECIAL]" in pw:
                    for ch in SPECIAL_REPLACEMENTS:
                        cand_set.add(pw.replace("[SPECIAL]", ch))
                else:
                    cand_set.add(pw)

                # —— ② 仅名缩写 (hf/Hf) 变体 ——
                if "<CN_NAME_ABBR>" in pw_tpl:
                    abbr_full = mv.get("<CN_NAME_ABBR>")
                    abbr_given_only = mv.get("__ABBR_GIVEN_ONLY__")
                    if abbr_full and abbr_given_only:
                        add_set: Set[str] = set()
                        for cand in cand_set:
                            add_set.add(cand.replace(abbr_full, abbr_given_only, 1))
                        cand_set |= add_set

                for cand in cand_set:
                    output_lines.append(f"{cand}\t{prob}\n")

        out_f = out_dir / f"{acct}.txt"
        with out_f.open("w", encoding="utf-8") as of:
            of.writelines(output_lines)

        print(f"[{idx}/{len(teachers)}] → {out_f} ({len(output_lines)} 条)")


def fill_placeholders_pattern_mass(
    *,
    templates_path: Path,
    teacher_csv_path: Path,
    out_dir: Path,
    max_templates: int = 10_000,
    top_k: int = 100,
    special_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    case_weights: Tuple[float, float] = (1.0, 1.0),
    abbr_weights: Tuple[float, float] = (1.0, 1.0),
) -> None:
    """
    B-style 聚合（截断版）：
    - 模板文件里的 prob 视为模板总质量 p_tpl（不在此处改动）
    - 填充时对派生分支按权重 w_i 拆分（每一类派生权重之和=1）
    - 对相同最终字符串做 P(w)=Σ p_tpl*w_i 的累计
    - 输出为可读的“模式串 + 概率质量”（不落地明文口令）
    """
    _fill_placeholders_mass(
        templates_path=templates_path,
        teacher_csv_path=teacher_csv_path,
        out_dir=out_dir,
        max_templates=max_templates,
        top_k=top_k,
        special_weights=special_weights,
        case_weights=case_weights,
        abbr_weights=abbr_weights,
        key_fn=_patternize_password,
        mode="pattern_mass",
        key_name="pattern",
        warn_plaintext=False,
    )


def fill_placeholders_plain_mass(
    *,
    templates_path: Path,
    teacher_csv_path: Path,
    out_dir: Path,
    max_templates: int = 10_000,
    top_k: int = 100,
    special_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    case_weights: Tuple[float, float] = (1.0, 1.0),
    abbr_weights: Tuple[float, float] = (1.0, 1.0),
) -> None:
    """
    B-style 聚合（明文版，截断版）：
    - 模板文件里的 prob 视为模板总质量 p_tpl（不在此处改动）
    - 填充时对派生分支按权重 w_i 拆分（每一类派生权重之和=1）
    - 对相同最终口令做 P(w)=Σ p_tpl*w_i 的累计，并按质量排序取 top-k

    注意：该模式会将“明文口令”写入 out_dir，请务必妥善保存并避免提交到仓库。
    """
    _fill_placeholders_mass(
        templates_path=templates_path,
        teacher_csv_path=teacher_csv_path,
        out_dir=out_dir,
        max_templates=max_templates,
        top_k=top_k,
        special_weights=special_weights,
        case_weights=case_weights,
        abbr_weights=abbr_weights,
        key_fn=_identity,
        mode="plain_mass",
        key_name="password",
        warn_plaintext=True,
    )


# ── 占位符替换 ──────────────────────────────────────────────────────────────

def materialize(template: str, mapping: Dict[str, str]):
    if "<" not in template:
        return template

    def repl(m):
        ph = m.group()
        val = mapping.get(ph)
        return val if val else ph

    pw = PH_PATTERN.sub(repl, template)
    return None if "<" in pw else pw

# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    import argparse

    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Fill placeholder templates with per-user info.")
    parser.add_argument(
        "--templates",
        type=Path,
        default=base_dir / "placeholders1.txt",
        help="Placeholder template file (tab-separated: template\\tprob).",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=base_dir / "teacher.csv",
        help="Teacher CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "fulllist",
        help="Output directory for per-account candidate lists.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pattern-mass",
        action="store_true",
        dest="pattern_mass",
        help="Aggregate to redacted pattern strings and rank by probability mass (no plaintext).",
    )
    mode.add_argument(
        "--plain-mass",
        action="store_true",
        dest="plain_mass",
        help="Aggregate to plaintext passwords and rank by probability mass (writes plaintext to disk).",
    )
    mode.add_argument(
        "--raw",
        action="store_true",
        help="Write raw per-template candidates (no mass aggregation; may be huge and contains plaintext).",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=10_000,
        help="Only read the first N templates from the template file (mass modes; 0 = no limit).",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Keep top-K outputs per account (mass modes).")
    parser.add_argument(
        "--special-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("W_AT", "W_UNDER", "W_DOT"),
        help="Weights for [SPECIAL] replacements: @  _  . (mass modes).",
    )
    parser.add_argument(
        "--case-weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("W_LOWER", "W_UPPER"),
        help="Weights for chinese-name pinyin case variants: lower / Upper (mass modes).",
    )
    parser.add_argument(
        "--abbr-weights",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("W_FULL", "W_GIVEN_ONLY"),
        help="Weights for CN_NAME_ABBR variants: full-abbr / given-only (mass modes).",
    )
    args = parser.parse_args()

    # 默认行为：给论文分析更友好，且不落地明文
    if not getattr(args, "plain_mass", False) and not args.pattern_mass and not args.raw:
        args.pattern_mass = True

    if getattr(args, "plain_mass", False):
        fill_placeholders_plain_mass(
            templates_path=args.templates,
            teacher_csv_path=args.teacher,
            out_dir=args.out_dir,
            max_templates=args.max_templates,
            top_k=args.top_k,
            special_weights=tuple(args.special_weights),
            case_weights=tuple(args.case_weights),
            abbr_weights=tuple(args.abbr_weights),
        )
    elif args.pattern_mass:
        fill_placeholders_pattern_mass(
            templates_path=args.templates,
            teacher_csv_path=args.teacher,
            out_dir=args.out_dir,
            max_templates=args.max_templates,
            top_k=args.top_k,
            special_weights=tuple(args.special_weights),
            case_weights=tuple(args.case_weights),
            abbr_weights=tuple(args.abbr_weights),
        )
    else:
        fill_placeholders(
            templates_path=args.templates,
            teacher_csv_path=args.teacher,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
