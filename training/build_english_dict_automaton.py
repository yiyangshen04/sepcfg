import os
import pickle

import ahocorasick

from config import project_path


def build_and_save_english_dict_automaton(dict_path, pkl_path):
    """
    从英文词典文件构建 Aho-Corasick 自动机，并序列化保存为 pkl 文件.

    :param dict_path: 英文词典文件路径，例如 "./data/google-10000-english.txt"
    :param pkl_path: 输出的自动机 pkl 文件路径，例如 "./data/english_dict_automaton.pkl"
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"English dictionary file not found: {dict_path}")

    automaton = ahocorasick.Automaton()
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            # 仅添加长度 >= 2 的单词
            if not word or len(word) < 2:
                continue
            if word not in automaton:
                automaton.add_word(word, word)
    automaton.make_automaton()
    with open(pkl_path, 'wb') as pf:
        pickle.dump(automaton, pf)
    print(f"[INFO] English dictionary automaton built from '{dict_path}' and saved to '{pkl_path}'.")


def load_english_dict_automaton(pkl_path):
    """
    从 pkl 文件加载已经构建好的英文词典自动机.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"English dictionary automaton pkl not found: {pkl_path}")
    with open(pkl_path, 'rb') as pf:
        automaton = pickle.load(pf)
    return automaton


def match_english_words(text, automaton):
    """
    在文本 text 中利用 automaton 查找所有匹配的英文单词，
    返回不重叠的匹配结果，格式为：(start_index, end_index, "en_word")
    """
    if not text:
        return []
    text_lower = text.lower()
    raw_matches = []
    for end_idx, word in automaton.iter(text_lower):
        start_idx = end_idx - len(word) + 1
        raw_matches.append((start_idx, end_idx, len(word)))
    if not raw_matches:
        return []
    # 按匹配长度降序、start 升序：Longest-first 策略
    raw_matches.sort(key=lambda x: (-x[2], x[0]))
    selected = []
    for s, e, l in raw_matches:
        overlap = False
        for ss, ee, _ in selected:
            if not (e < ss or s > ee):
                overlap = True
                break
        if not overlap:
            selected.append((s, e, "en_word"))
    selected.sort(key=lambda x: x[0])
    return selected


if __name__ == "__main__":
    dict_path = str(project_path("data", "merged_words.txt"))
    pkl_path = str(project_path("data", "english_dict_automaton.pkl"))
    build_and_save_english_dict_automaton(dict_path, pkl_path)
