#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import pickle


################################################################################
# 1) 读取并解析 pkl 文件，计算 sp_probs / sft_probs
################################################################################

def load_pcfg_probabilities(pkl_path):
    """
    从指定的 pkl 文件中读取计数信息，并计算得到:
      - sp_probs: dict, { sp_tuple: probability }
      - sft_probs: dict, { sft: { sf: probability } }
    """
    with open(pkl_path, 'rb') as f:
        state = pickle.load(f)

    sp_counts = state['sp_counts']  # dict, key = tuple of sft, value = count
    sp_total = state['sp_total']  # int
    sft_counts = state['sft_counts']  # dict, key = sft, value = dict(sf -> count)
    sft_totals = state['sft_totals']  # dict, key = sft, value = int

    # 计算 sp_probs
    sp_probs = {}
    for sp_tuple, cnt in sp_counts.items():
        sp_probs[sp_tuple] = cnt / sp_total if sp_total > 0 else 0

    # 计算 sft_probs
    sft_probs = {}
    for sft, sf_dict in sft_counts.items():
        total = sft_totals[sft]
        sft_probs[sft] = {}
        for sf, cnt in sf_dict.items():
            sft_probs[sft][sf] = cnt / total if total > 0 else 0

    return sp_probs, sft_probs


################################################################################
# 2) 定义一个简易的 PCFG 类，实现 initalize_base_structures / find_children
################################################################################

class MyPCFG:
    """
    将 sp_probs + sft_probs 封装成一个可被优先队列(PcfgQueue)使用的Grammar。
    """

    def __init__(self, sp_probs, sft_probs):
        self.sp_probs = sp_probs
        self.sft_probs = sft_probs

    def initalize_base_structures(self):
        """
        返回一个迭代器，每个元素是一个 "parse_item" (dict)，用于初始化优先队列。
        每个 parse_item 对应一个 sp_tuple (模板)，还没有展开任何 sf。
        """
        for sp_tuple, prob in self.sp_probs.items():
            yield {
                "sp_tuple": sp_tuple,
                "index": 0,
                "current_str": "",
                "prob": prob
            }

    def find_children(self, parse_item):
        """
        对 parse_item 进行 "下一步展开"：
          - 若尚未展开完 sp_tuple，则从 sft_probs 中列举所有可能sf，并更新概率
          - 若已经是终态(展开完所有 sft)，则不yield子节点
        """
        sp_tuple = parse_item["sp_tuple"]
        idx = parse_item["index"]
        base_str = parse_item["current_str"]
        base_prob = parse_item["prob"]

        if idx >= len(sp_tuple):
            return  # 没有子节点，parse_item 已经完整

        next_sft = sp_tuple[idx]
        sf_dict = self.sft_probs.get(next_sft, {})
        for sf, sf_prob in sf_dict.items():
            child_prob = base_prob * sf_prob
            child_str = base_str + sf
            yield {
                "sp_tuple": sp_tuple,
                "index": idx + 1,
                "current_str": child_str,
                "prob": child_prob
            }


################################################################################
# 3) 改进后的优先队列结构，支持“概率下限 + 最大队列大小”剪枝
################################################################################

class QueueItem:
    """
    小包装类，让 heapq 处理为"大顶堆"效果
    """

    def __init__(self, pt_item):
        self.pt_item = pt_item

    def __lt__(self, other):
        # 概率大的优先
        return self.pt_item['prob'] > other.pt_item['prob']


class PcfgQueue:
    """
    使用 heapq 实现的优先队列，并支持：
      - min_probability: 若子项概率低于此阈值，直接丢弃
      - max_queue_size: 超过时弹出最低概率的节点
    """

    def __init__(self, pcfg, min_probability=1e-10, max_queue_size=50000):
        self.pcfg = pcfg
        self.p_queue = []
        self.min_probability = min_probability
        self.max_queue_size = max_queue_size

        # 初始化：将 grammar 的 base items 全部push进队列
        for base_item in self.pcfg.initalize_base_structures():
            # 如果 base_item.prob 也可能很小，可做一次判断
            if base_item['prob'] >= self.min_probability:
                heapq.heappush(self.p_queue, QueueItem(base_item))

        # 若初始就过大，可以进行一次修剪
        self._prune_queue_if_needed()

    def next(self):
        """
        每次 pop 出概率最高的 parse_item，若它还可展开，就将子节点 push 回队列。
        当 parse_item 已展开完所有 sft => 返回它；若队列空则返回 None。
        """
        while self.p_queue:
            top = heapq.heappop(self.p_queue)
            pt_item = top.pt_item

            # 判断是否已展开完
            if pt_item["index"] >= len(pt_item["sp_tuple"]):
                # 已是完整密码 => 直接返回
                return pt_item

            # 否则展开子节点
            for child in self.pcfg.find_children(pt_item):
                # 剪枝1：概率阈值
                if child['prob'] < self.min_probability:
                    continue
                # 入队
                heapq.heappush(self.p_queue, QueueItem(child))

            # 剪枝2：若队列超限，移除最低概率的节点
            self._prune_queue_if_needed()

        # 队列空了
        return None

    def _prune_queue_if_needed(self):
        """
        如果当前队列大小超过 max_queue_size，
        就弹出最低概率的节点，直到回到安全范围。
        """
        while len(self.p_queue) > self.max_queue_size:
            # heapq 是小顶堆，但我们通过重载 __lt__ 让概率大的排前
            # => 队列最末端是最低概率
            # 但 heapq 并不支持直接弹“末端”元素；这里的方法是：
            #   先把 items 全部弹到临时数组，再仅保留前max_queue_size个
            # 不过这样效率不是最优，但写起来简单直观。
            temp_items = []
            while self.p_queue:
                temp_items.append(heapq.heappop(self.p_queue))
            # temp_items 已经是从最大到最小(QueueItem.__lt__反转了逻辑)，
            # 其实需要我们再 sort 一下以保证顺序；或者保留 top K
            # 这里演示简单做法:
            temp_items.sort(key=lambda x: x.pt_item['prob'], reverse=True)
            # 保留前 max_queue_size
            temp_items = temp_items[:self.max_queue_size]
            # 放回队列
            for it in temp_items:
                heapq.heappush(self.p_queue, it)


################################################################################
# 4) 生成指定数量的密码，写入文件
################################################################################

def generate_password_guesses(
        pkl_path="checkpoints/my_training_20250515_counts.pkl",
        output_txt="generated_passwords.txt",
        num_to_generate=100,
        min_probability=1e-10,  # 剪枝阈值
        max_queue_size=50000  # 优先队列最大大小
):
    """
    读取 pkl 文件，计算 sp_probs/sft_probs，
    用优先队列从高到低生成指定数量的密码（带剪枝），写到 output_txt。
    """
    # (1) 加载并计算概率
    sp_probs, sft_probs = load_pcfg_probabilities(pkl_path)
    print(f"[INFO] Loaded sp_probs({len(sp_probs)}) & sft_probs({len(sft_probs)}) from {pkl_path}")

    # (2) 构建PCFG, 初始化优先队列(带剪枝)
    pcfg = MyPCFG(sp_probs, sft_probs)
    pq = PcfgQueue(
        pcfg,
        min_probability=min_probability,
        max_queue_size=max_queue_size
    )

    # (3) 从队列中依次获取完整密码
    with open(output_txt, "w", encoding="utf-8") as f_out:
        count = 0
        while count < num_to_generate:
            parse_item = pq.next()
            if parse_item is None:
                print("[INFO] No more parse items in queue, generation ended.")
                break
            pwd = parse_item["current_str"]
            prob = parse_item["prob"]
            count += 1
            # 写入文本: password \t probability
            f_out.write(f"{pwd}\t{prob:.8e}\n")

    print(f"[INFO] Generated {count} passwords into {output_txt}.")


################################################################################
# 5) 主入口：示例
################################################################################
if __name__ == "__main__":
    PKL_PATH = "checkpoints/my_training_20250515_counts.pkl"
    OUTPUT_TXT = "generated_passwords_pruned.txt"
    N = 2000  # 想要的生成数量

    # 你可以视情况调整 min_probability / max_queue_size
    generate_password_guesses(
        pkl_path=PKL_PATH,
        output_txt=OUTPUT_TXT,
        num_to_generate=N,
        min_probability=1e-10,
        max_queue_size=50000
    )
