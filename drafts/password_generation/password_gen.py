#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import heapq
import re

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

    sp_counts = state['sp_counts']   # dict, key = tuple of sft, value = count
    sp_total  = state['sp_total']    # int
    sft_counts = state['sft_counts'] # dict, key = sft, value = dict(sf -> count)
    sft_totals = state['sft_totals'] # dict, key = sft, value = int

    # 计算 sp_probs
    sp_probs = {}
    for sp_tuple, cnt in sp_counts.items():
        sp_probs[sp_tuple] = cnt / sp_total if sp_total > 0 else 0

    # 计算 sft_probs
    sft_probs = {}
    for sft, sf_dict in sft_counts.items():
        total = sft_totals.get(sft, 0)
        sft_probs[sft] = {}
        for sf, cnt in sf_dict.items():
            sft_probs[sft][sf] = cnt / total if total > 0 else 0

    return sp_probs, sft_probs


################################################################################
# 2) 定义一个简易的 PCFG 类，实现 initalize_base_structures / find_children
################################################################################

class MyPCFG:
    """
    将 sp_probs + sft_probs 封装成一个可被优先队列 (PcfgQueue) 使用的 Grammar。
    """
    def __init__(self, sp_probs, sft_probs):
        self.sp_probs = sp_probs
        self.sft_probs = sft_probs

        # 账号相关标签的白名单（凡是 sft 中包含以下任何一个关键字，都当作账号相关）
        self.account_labels = {
            "acc_email_domain",
            "acc_email_domain_com",  # ← 新增
            "acc_pwd_same",
            "acc_email_name",
        }

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
          - 若尚未展开完 sp_tuple，则根据下一个 sft 查找对应的候选 sf。
            * 如果 sft 是账号相关 (acc_*), 则只产出一个孩子节点，概率视为 1.0，
              并把对应的特殊标记加到 current_str 中。
            * 否则，就遍历 sft_probs[sft] 中的所有 sf, 并乘以各自概率。
          - 若已经是终态(展开完所有 sft)，则不 yield 子节点。
        """
        sp_tuple = parse_item["sp_tuple"]
        idx = parse_item["index"]
        base_str = parse_item["current_str"]
        base_prob = parse_item["prob"]

        if idx >= len(sp_tuple):
            return  # 没有子节点，parse_item 已经完整

        next_sft = sp_tuple[idx]
        # 检查是否属于账号相关的标签
        if self._is_account_label(next_sft):
            # 账号相关 => 只生成一个子节点，概率=base_prob (×1.0)
            child_str = base_str + self._convert_sft_with_brackets(next_sft)
            yield {
                "sp_tuple": sp_tuple,
                "index": idx + 1,
                "current_str": child_str,
                "prob": base_prob
            }
        else:
            # 常规情况：遍历 sft_probs 中的所有 sf
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

    def _is_account_label(self, sft):
        """
        判断一个 sft 是否为“账号相关”标签:
          - 只要 sft 中包含 self.account_labels 的任意一个关键字, 就认为是账号相关。
        """
        for acc_label in self.account_labels:
            if acc_label in sft:
                return True
        return False

    def _convert_sft_with_brackets(self, sft: str) -> str:
        """
        将包含账号标签的 sft 统一转换成:
          <prefix>[<acc_label>]<suffix>
        - prefix   : 标签**左侧**的任何字符（允许为空）
        - suffix   : 标签**右侧**的任何字符（允许为空）
        这样即可同时兼容 “前缀+标签”、“标签+后缀” 以及 “前缀+标签+后缀” 三种形态。
        """
        # ② 遍历所有已登记的账号标签，优先匹配**最长**的（避免子串干扰）
        for acc_label in sorted(self.account_labels, key=len, reverse=True):
            pos = sft.find(acc_label)
            if pos != -1:
                prefix = sft[:pos]                       # 标签左边
                suffix = sft[pos + len(acc_label):]      # 标签右边
                return f"{prefix}[{acc_label}]{suffix}"

        # 理论上不会走到这里；保险起见做兜底处理
        return f"[{sft}]"


################################################################################
# 3) 改进后的优先队列结构，支持“概率下限 + 最大队列大小”剪枝
################################################################################

class QueueItem:
    """
    小包装类，让 heapq 处理为"大顶堆"效果。
    由于默认 heapq 是最小堆，这里通过 __lt__ 反转比较，让概率大的排在前面。
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
      - prune_interval: 控制每多少次展开后做一次大规模剪枝
    """
    def __init__(self, pcfg, min_probability=1e-10, max_queue_size=50000, prune_interval=1000):
        self.pcfg = pcfg
        self.p_queue = []
        self.min_probability = min_probability
        self.max_queue_size = max_queue_size

        # prune_interval: 可以根据实际情况调整，减少频繁剪枝的开销
        self.prune_interval = prune_interval
        self.expand_count = 0  # 统计累计展开次数

        # 初始化：将 grammar 的 base items 全部 push 进队列
        for base_item in self.pcfg.initalize_base_structures():
            if base_item['prob'] >= self.min_probability:
                heapq.heappush(self.p_queue, QueueItem(base_item))

        # 若初始就过大，可以进行一次修剪
        self._prune_queue_if_needed(force=True)

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

            # 每次展开计数 +1
            self.expand_count += 1
            # 若累计展开次数到了阈值，就做一次剪枝
            if (self.expand_count % self.prune_interval) == 0:
                self._prune_queue_if_needed(force=True)

        # 队列空了
        return None

    def _prune_queue_if_needed(self, force=False):
        """
        如果当前队列大小超过 max_queue_size，就弹出概率最低的节点，直到回到安全范围。
        若 force=True 则强制执行剪枝；否则只在超过限制时才进行。
        """
        if (not force) and (len(self.p_queue) <= self.max_queue_size):
            return

        while len(self.p_queue) > self.max_queue_size:
            # 取出所有元素
            temp_items = []
            while self.p_queue:
                temp_items.append(heapq.heappop(self.p_queue))
            # 只保留概率前 max_queue_size 大的项目
            temp_items = heapq.nlargest(self.max_queue_size, temp_items, key=lambda x: x.pt_item['prob'])
            # 放回队列
            for it in temp_items:
                heapq.heappush(self.p_queue, it)


################################################################################
# 4) 生成指定数量的密码，写入文件
################################################################################

def generate_password_guesses(
    pkl_path,
    output_txt,
    num_to_generate = 100,
    min_probability = 1e-10,    # 剪枝阈值
    max_queue_size = 50000,     # 优先队列最大大小
    prune_interval = 1000       # 每多少次展开后做一次大剪枝
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
        max_queue_size=max_queue_size,
        prune_interval=prune_interval
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
# 5) 主入口：顺序执行示例（从 60M 到 10M）
################################################################################
if __name__ == "__main__":
    from config import TRAIN_NAME, project_path

    root = project_path()
    tasks = [
        (
            root / "checkpoints" / f"{TRAIN_NAME}_counts.pkl",
            root / f"{TRAIN_NAME}_counts.txt",
        ),
    ]


    # 要生成多少条密码，以及剪枝参数（可根据需要自行调整）
    N = 10000
    MIN_PROB = 1e-10
    MAX_QUEUE_SIZE = 50000
    PRUNE_INTERVAL = 1000

    # 顺序处理：从 tasks[0] ~ tasks[-1] 依次执行
    for pkl_path, output_txt in tasks:
        print(f"[INFO] Start processing {pkl_path} -> {output_txt} ...")
        generate_password_guesses(
            pkl_path=pkl_path,
            output_txt=output_txt,
            num_to_generate=N,
            min_probability=MIN_PROB,
            max_queue_size=MAX_QUEUE_SIZE,
            prune_interval=PRUNE_INTERVAL
        )

    print("[INFO] All tasks completed in sequence.")
