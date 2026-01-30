# build_english_name_automaton.py

import csv
import os
import pickle

import ahocorasick

from config import project_path


def build_and_save_english_name_automaton(csv_path, pkl_path):
    """
    从 english_names.csv 构建 Aho-Corasick 自动机，并序列化保存为 pkl 文件。

    :param csv_path:  CSV 文件路径, 例如 "./data/english_names.csv"
    :param pkl_path:  要输出的 pkl 文件路径, 例如 "./data/english_name_automaton.pkl"
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"English name CSV not found: {csv_path}")

    A = ahocorasick.Automaton()

    # 读取 CSV，跳过表头，抓取第一列 Name
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_raw = row["Name"].strip()
            name_lower = name_raw.lower()  # 小写，避免大小写差异
            if name_lower not in A:
                A.add_word(name_lower, name_lower)

    # 构建自动机（失配指针）
    A.make_automaton()

    # 将自动机序列化保存到 pkl 文件
    with open(pkl_path, 'wb') as pf:
        pickle.dump(A, pf)

    print(f"[INFO] English name automaton built from '{csv_path}' and saved to '{pkl_path}'.")


def load_english_name_automaton(pkl_path):
    """
    从 pkl 文件加载已构建的英文名自动机 (Aho-Corasick).
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"English name automaton pkl not found: {pkl_path}")

    with open(pkl_path, 'rb') as pf:
        A = pickle.load(pf)

    return A


def match_english_names(text, en_automaton):
    """
    在小写字符串 text 中，用 en_automaton 做多模式匹配，
    并返回不重叠的 (start, end, 'en_name') 列表。

    这里演示“Longest-First”策略，避免重叠时把短词和长词一起返回。
    """
    if not text:
        return []

    text_lower = text.lower()
    raw_matches = []

    # 1) 收集所有原始匹配 (start, end, length)
    for end_pos, name_str in en_automaton.iter(text_lower):
        start_pos = end_pos - len(name_str) + 1
        raw_matches.append((start_pos, end_pos, len(name_str)))

    if not raw_matches:
        return []

    # 2) 按匹配长度降序、start升序排序 (Longest-First)
    raw_matches.sort(key=lambda x: (-x[2], x[0]))

    selected = []
    for (s, e, length) in raw_matches:
        overlap = False
        for (ss, ee, _) in selected:
            # 若与已选区间有重叠，则跳过
            if not (e < ss or s > ee):
                overlap = True
                break
        if not overlap:
            selected.append((s, e, 'en_name'))

    # 3) 最终按起始位置排序
    selected.sort(key=lambda x: x[0])
    return selected


if __name__ == "__main__":
    # 举例：只需在命令行执行一次，构建并持久化自动机
    csv_path_default = str(project_path("data", "english_names.csv"))
    pkl_path_default = str(project_path("data", "english_name_automaton.pkl"))

    print("Building English name automaton from CSV => PKL...")
    build_and_save_english_name_automaton(csv_path_default, pkl_path_default)
    print("[DONE]")
