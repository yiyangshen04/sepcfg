#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import inspect
import pickle
import re
from dataclasses import dataclass as _dataclass, field
from pathlib import Path

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

    sp_counts   = state['sp_counts']     # dict, key = tuple of sft, value = count
    sp_total    = state['sp_total']      # int
    sft_counts  = state['sft_counts']    # dict, key = sft, value = dict(sf -> count)
    sft_totals  = state['sft_totals']    # dict, key = sft, value = int

    # 计算 sp_probs
    sp_probs = {sp: cnt / sp_total if sp_total else 0.0
                for sp, cnt in sp_counts.items()}

    # 计算 sft_probs
    sft_probs = {}
    for sft, sf_dict in sft_counts.items():
        total = sft_totals.get(sft, 0)
        sft_probs[sft] = {sf: cnt / total if total else 0.0
                          for sf, cnt in sf_dict.items()}
    return sp_probs, sft_probs


################################################################################
# 2) PCFG 定义（新增正则结果缓存）
################################################################################

def dataclass(*args, **kwargs):
    # Python 3.9 的 dataclass 不支持 slots 参数（3.10+ 才支持）
    if "slots" in kwargs and "slots" not in inspect.signature(_dataclass).parameters:
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)


@dataclass(slots=True)
class ParseItem:
    sp_tuple: tuple
    index:   int
    parts:   list = field(default_factory=list)   # 字符片段累积
    prob:    float = 1.0


class MyPCFG:
    """
    将 sp_probs + sft_probs 封装成可供优先队列使用的 Grammar。
    """
    _ACC_REGEX = re.compile(
        r"^(.*?)+("
        r"acc_email_domain_com|"
        r"acc_email_name|acc_email_domain|acc_pwd_same"
        r")$"
    )

    def __init__(self, sp_probs, sft_probs):
        self.sp_probs   = sp_probs
        self.sft_probs  = sft_probs

        self.account_labels = {
            "acc_email_domain_com",
            "acc_email_domain",
            "acc_email_name",
            "acc_pwd_same",
        }        # —— 新增 —— 结果缓存，避免反复正则匹配
        self._bracket_cache = {}

    # ---------- 公共接口 ---------- #

    def initalize_base_structures(self):
        for sp_tuple, prob in self.sp_probs.items():
            yield ParseItem(
                sp_tuple=sp_tuple,
                index=0,
                parts=[],
                prob=prob
            )

    def find_children(self, parse_item: ParseItem):
        sp_tuple   = parse_item.sp_tuple
        idx        = parse_item.index
        base_parts = parse_item.parts
        base_prob  = parse_item.prob

        if idx >= len(sp_tuple):
            return  # 终态

        next_sft = sp_tuple[idx]

        # 账号相关标签：只生成一个孩子
        if self._is_account_label(next_sft):
            child_parts = base_parts + [self._convert_sft_with_brackets(next_sft)]
            yield ParseItem(sp_tuple, idx + 1, child_parts, base_prob)
        else:
            sf_dict = self.sft_probs.get(next_sft, {})
            for sf, sf_prob in sf_dict.items():
                child_parts = base_parts + [sf]
                yield ParseItem(sp_tuple, idx + 1, child_parts, base_prob * sf_prob)

    # ---------- 私有工具 ---------- #

    def _is_account_label(self, sft: str) -> bool:
        return any(lbl in sft for lbl in self.account_labels)

    def _convert_sft_with_brackets(self, sft: str) -> str:
        """
        将类似 "123+acc_email_name" → "123[acc_email_name]"；结果带缓存。
        """
        if sft in self._bracket_cache:
            return self._bracket_cache[sft]

        m = self._ACC_REGEX.match(sft)
        if m:
            prefix, acc_label = m.groups()
            res = f"{prefix}[{acc_label}]"
        else:
            for acc_label in self.account_labels:
                if acc_label in sft:
                    prefix = sft.split(acc_label, 1)[0]
                    res = f"{prefix}[{acc_label}]"
                    break
            else:   # 理论不会走到
                res = f"[{sft}]"

        self._bracket_cache[sft] = res
        return res


################################################################################
# 3) 优先队列结构：使用 heappush + 溢出时 heappop（max-size 强限容）
################################################################################

class QueueItem:
    """薄包装，使 heapq 成‘大顶堆’效果（概率大 → 优先级高）"""
    __slots__ = ("pt_item",)
    def __init__(self, pt_item: ParseItem):
        self.pt_item = pt_item
    def __lt__(self, other):
        return self.pt_item.prob > other.pt_item.prob    # 翻转比较符号


class PcfgQueue:
    def __init__(self, pcfg: MyPCFG,
                 min_probability=1e-10,
                 max_queue_size=50_000,
                 prune_interval=1_000):
        self.pcfg            = pcfg
        self.min_probability = min_probability
        self.max_queue_size  = max_queue_size
        self.prune_interval  = prune_interval
        self.expand_cnt      = 0

        # 一次性 heapify
        base_items  = (QueueItem(it) for it in pcfg.initalize_base_structures()
                       if it.prob >= min_probability)
        self.p_queue = list(base_items)
        heapq.heapify(self.p_queue)

    # ---------------- 主循环 ---------------- #

    def next(self):
        while self.p_queue:
            top = heapq.heappop(self.p_queue).pt_item      # 最优节点

            if top.index >= len(top.sp_tuple):             # 完成态
                return top

            # 展开
            for child in self.pcfg.find_children(top):
                if child.prob < self.min_probability:
                    continue
                heapq.heappush(self.p_queue, QueueItem(child))

            # ——— 批量修剪：每 prune_interval 次展开后统一删除多余节点 ———
            self.expand_cnt += 1
            if (self.expand_cnt % self.prune_interval) == 0:
                self._prune_queue()

        return None

    # ---------------- 只保留概率最高的 max_queue_size 个 ---------------- #

    def _prune_queue(self):
        if len(self.p_queue) <= self.max_queue_size:
            return
        # nlargest 用真实概率排序，保留前 k 高
        keep = heapq.nlargest(self.max_queue_size,
                              self.p_queue,
                              key=lambda x: x.pt_item.prob)
        self.p_queue[:] = keep   # 原地替换
        heapq.heapify(self.p_queue)


################################################################################
# 4) 生成指定数量的密码，写入文件
################################################################################

def generate_password_guesses(
    pkl_path,
    output_txt,
    num_to_generate = 100,
    min_probability = 1e-10,
    max_queue_size  = 50_000,
):
    # (1) 加载概率
    sp_probs, sft_probs = load_pcfg_probabilities(pkl_path)
    print(f"[INFO] Loaded sp_probs({len(sp_probs)}) & sft_probs({len(sft_probs)})")

    # (2) 构建 PCFG 与队列
    pcfg = MyPCFG(sp_probs, sft_probs)
    pq   = PcfgQueue(pcfg, min_probability, max_queue_size)

    # (3) 生成
    with open(output_txt, "w", encoding="utf-8") as f_out:
        count = 0
        while count < num_to_generate:
            item = pq.next()
            if item is None:
                print("[INFO] Queue exhausted.")
                break
            pwd  = ''.join(item.parts)      # 最后一次 join
            prob = item.prob
            f_out.write(f"{pwd}\t{prob:.8e}\n")
            count += 1

    print(f"[INFO] Generated {count} passwords → {output_txt}")


################################################################################
# 5) CLI 示例
################################################################################
if __name__ == "__main__":
    from config import TRAIN_NAME, project_path

    root = project_path()
    TASKS = [
        (
            root / "checkpoints" / f"{TRAIN_NAME}_counts.pkl",
            root / f"{TRAIN_NAME}_counts_fast.txt",
        )
    ]

    N               = 10_000
    MIN_PROB        = 1e-10
    MAX_QUEUE_SIZE  = 50_000

    for pkl_path, output_txt in TASKS:
        print(f"[INFO] Processing {pkl_path} …")
        generate_password_guesses(
            pkl_path=pkl_path,
            output_txt=output_txt,
            num_to_generate=N,
            min_probability=MIN_PROB,
            max_queue_size=MAX_QUEUE_SIZE,
        )

    print("[INFO] All tasks completed.")
