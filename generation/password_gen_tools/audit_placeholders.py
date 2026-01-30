#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_placeholders.py

对占位符模板文件（template\\tprob）做“非明文”的一致性审计：
- 是否按概率非增序排列（允许极小浮点误差）
- 是否有重复模板（同一字符串多次出现；若要“唯一口令概率排序”，需先合并）
- 模板包含占位符 / 纯字面串 的比例

注意：脚本默认只输出 hash/形状信息，不打印明文模板内容。
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path


def _shape(s: str) -> str:
    def cls(ch: str) -> str:
        if ch.isdigit():
            return "D"
        if "a" <= ch <= "z":
            return "l"
        if "A" <= ch <= "Z":
            return "U"
        return "S"

    if not s:
        return "EMPTY"
    out = []
    prev = None
    run = 0
    for ch in s:
        c = cls(ch)
        if prev is None:
            prev = c
            run = 1
        elif c == prev:
            run += 1
        else:
            out.append(f"{prev}{run}")
            prev = c
            run = 1
    out.append(f"{prev}{run}")
    return "".join(out)


def _token_id(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
    return f"<{_shape(s)}>#sha256:{h}"


def iter_templates(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                tpl, p = line.split("\t", 1)
                prob = float(p)
            except Exception as e:
                raise ValueError(f"bad line at {path}:{lineno}: {line[:120]!r}") from e
            yield lineno, tpl, prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit placeholder templates without printing plaintext.")
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path(__file__).resolve().parent / "placeholders1.txt",
        help="Template file (tab-separated: template\\tprob).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-15, help="Allowed prob increase tolerance.")
    parser.add_argument("--top-dups", type=int, default=5, help="Show top N duplicate groups (hashed).")
    args = parser.parse_args()

    n = 0
    n_with_ph = 0
    probs_ok = True
    violations = 0
    prev_prob = math.inf

    dup_counter: Counter[str] = Counter()

    for _, tpl, prob in iter_templates(args.templates):
        n += 1
        dup_counter[tpl] += 1
        if "<" in tpl and ">" in tpl:
            n_with_ph += 1

        if prob > prev_prob + args.epsilon:
            probs_ok = False
            violations += 1
        prev_prob = prob

    uniq = len(dup_counter)
    dup_groups = sum(1 for _tpl, c in dup_counter.items() if c > 1)
    dup_total = sum(c - 1 for c in dup_counter.values() if c > 1)

    print(f"file={args.templates}")
    print(f"lines={n} unique={uniq}")
    print(f"contains_placeholders={n_with_ph} ({(n_with_ph / max(n, 1)):.2%})")
    print(f"sorted_non_increasing={probs_ok} violations={violations} epsilon={args.epsilon}")
    print(f"duplicate_groups={dup_groups} duplicate_extra_lines={dup_total}")

    if args.top_dups > 0 and dup_groups > 0:
        print("\nTop duplicate templates (hashed):")
        for tpl, c in dup_counter.most_common():
            if c <= 1:
                continue
            print(f"  {c}x\t{_token_id(tpl)}")
            args.top_dups -= 1
            if args.top_dups <= 0:
                break


if __name__ == "__main__":
    main()

