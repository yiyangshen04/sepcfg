#!/usr/bin/env python3
"""
把 chinese_names.csv 中的 8 列姓名模式构建成
两份 Aho-Corasick 自动机 pickle：

1) 仅含 *_special 的 2 列      → cn_name_automata_special.pkl
2) 其余 6 列（纯字母形式）     → cn_name_automata_lds.pkl
"""

import csv
import pickle

import ahocorasick

from config import project_path

CSV_PATH = str(project_path("data", "chinese_names.csv"))
OUTPUT_PKL_LDS = str(project_path("data", "cn_name_automata_lds.pkl"))
OUTPUT_PKL_SPECIAL = str(project_path("data", "cn_name_automata_special.pkl"))

# 8 列 → 统一标签
COLUMN_LABEL_MAP = {
    "姓名的全拼": "cn_name_full",
    "姓名的全拼_特殊": "cn_name_full_special",
    "名字的全拼": "cn_name_given",
    "姓全拼+名字缩写": "cn_name_last_abbr",
    "姓名的缩写": "cn_name_abbr",
    "先名后姓": "cn_name_first_last",
    "先名后姓_特殊": "cn_name_first_last_special",
    # ★ 新增列：姓的全拼
    "姓的全拼": "cn_name_last_full",
}

# 哪两列属于 *_special
SPECIAL_COLS = {"姓名的全拼_特殊", "先名后姓_特殊"}
# 其余 6 列自动推导
NORMAL_COLS = set(COLUMN_LABEL_MAP) - SPECIAL_COLS

SPECIAL_REPLACEMENTS = ['@', '_', '.']


def expand_specials(original_str: str) -> set[str]:
    parts = original_str.split("[SPECIAL]")
    count = len(parts) - 1
    if count < 1:
        return {original_str}

    expansions = {parts[0]}
    for i in range(count):
        new_set = set()
        for prefix in expansions:
            for rep in SPECIAL_REPLACEMENTS:
                new_set.add(prefix + rep + parts[i + 1])
        expansions = new_set
    return expansions


def collect_column_values() -> dict[str, set[str]]:
    col_values = {col: set() for col in COLUMN_LABEL_MAP}
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in COLUMN_LABEL_MAP:
                val = row.get(col, "").strip()
                if not val:
                    continue
                for item in expand_specials(val):
                    if item:
                        col_values[col].add(item.lower())
    return col_values


def build_one_automaton(patterns: set[str], label: str) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
    for p in patterns:
        A.add_word(p, (label, p))
    A.make_automaton()
    return A


def build_and_save(col_values: dict[str, set[str]],
                   chosen_cols: set[str],
                   output_path: str):
    automata_dict = {}
    for col in chosen_cols:
        label = COLUMN_LABEL_MAP[col]
        print(f"  构建自动机: {label}  (模式数={len(col_values[col])})")
        automata_dict[label] = build_one_automaton(col_values[col], label)
    print(f"  序列化保存 → {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(automata_dict, f)


def main():
    col_values = collect_column_values()

    print("▼ 生成 *_special 自动机（2 列）")
    build_and_save(col_values, SPECIAL_COLS, OUTPUT_PKL_SPECIAL)

    print("\n▼ 生成普通 6 列自动机（lds 用）")
    build_and_save(col_values, NORMAL_COLS, OUTPUT_PKL_LDS)

    print("\n全部完成!")


if __name__ == "__main__":
    main()
