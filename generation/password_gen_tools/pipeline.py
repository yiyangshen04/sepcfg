#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py

一键生成“占位符模板”并注入个人信息，输出每个账号的候选口令列表。

默认等价于顺序执行：
1) generate_placeholders.py  → placeholders1.txt
2) fill_placeholders.py      → fulllist/*.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    # generation/password_gen_tools/pipeline.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def main() -> None:
    _ensure_repo_root_on_syspath()

    from config import TRAIN_NAME, project_path
    from generation.password_gen_tools.generate_placeholders import generate_placeholders
    from generation.password_gen_tools.fill_placeholders import (
        fill_placeholders,
        fill_placeholders_pattern_mass,
        fill_placeholders_plain_mass,
    )

    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="End-to-end password generation: generate placeholders then fill them.",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=project_path("checkpoints", f"{TRAIN_NAME}_counts.pkl"),
        help="Checkpoint pickle path (counts.pkl).",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=base_dir / "placeholders1.txt",
        help="Placeholder template output/input file.",
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
    parser.add_argument("--skip-generate", action="store_true", help="Skip template generation step.")
    parser.add_argument("--skip-fill", action="store_true", help="Skip filling step.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pattern-mass",
        action="store_true",
        dest="pattern_mass",
        help="Aggregate to redacted pattern strings and rank by probability mass (fill step; no plaintext).",
    )
    mode.add_argument(
        "--plain-mass",
        action="store_true",
        dest="plain_mass",
        help="Aggregate to plaintext passwords and rank by probability mass (fill step; writes plaintext to disk).",
    )
    mode.add_argument(
        "--raw",
        action="store_true",
        help="Write raw per-template candidates (no mass aggregation; may be huge and contains plaintext).",
    )

    # generate_placeholders 参数（默认保持与 generate_placeholders.py 当前 CLI 一致）
    parser.add_argument("--top-n", type=int, default=500_000)
    parser.add_argument("--min-prob", type=float, default=1e-8)
    parser.add_argument("--max-q", type=int, default=200_000)
    # fill mass 参数
    parser.add_argument(
        "--max-templates",
        type=int,
        default=10_000,
        help="Only read the first N templates (0 = no limit).",
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
    if not args.plain_mass and not args.pattern_mass and not args.raw:
        args.pattern_mass = True

    if not args.skip_generate:
        args.templates.parent.mkdir(parents=True, exist_ok=True)
        generate_placeholders(
            pkl_path=args.pkl,
            out_path=args.templates,
            top_n=args.top_n,
            min_prob=args.min_prob,
            max_q=args.max_q,
        )

    if not args.skip_fill:
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
