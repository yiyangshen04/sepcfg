# pcfg_trainer.py

from collections import defaultdict


def build_sp_sft_sequences(account_segments, password_segments):
    """
    将账号和密码的 (SF, SFT) 序列合并形成一个完整序列。
    也可在此加上分隔符SFT，例如 (None, "acc_pwd_sep")，以免混淆。
    """
    # 示例：在账号和密码之间插入一个特殊标记
    combined = account_segments + [(None, "acc_pwd_sep")] + password_segments
    return combined


def train_semantic_pcfg(records_segments):
    """
    records_segments: [ (account_segments, password_segments), ... ]
      其中account_segments/password_segments都已经是 [(sf, sft), ...]的形式

    返回:
      sp_probs: dict, { tuple_of_sft: prob }
      sft_probs: dict, { sft: { sf: prob } }
    """
    sp_counts = defaultdict(int)
    sp_total = 0
    sft_counts = defaultdict(lambda: defaultdict(int))
    sft_totals = defaultdict(int)

    for (acc_segs, pwd_segs) in records_segments:
        combined = build_sp_sft_sequences(acc_segs, pwd_segs)
        sft_sequence = tuple(sft for (_, sft) in combined)

        # 统计SP出现
        sp_counts[sft_sequence] += 1
        sp_total += 1

        # 统计SFT->SF
        for (sf, sft) in combined:
            if sf is not None:
                sft_counts[sft][sf] += 1
                sft_totals[sft] += 1

    # 计算概率
    sp_probs = {}
    for sp, cnt in sp_counts.items():
        sp_probs[sp] = cnt / sp_total

    sft_probs = {}
    for sft, sf_dict in sft_counts.items():
        total_sft = sft_totals[sft]
        sft_probs[sft] = {}
        for sf, cnt in sf_dict.items():
            sft_probs[sft][sf] = cnt / total_sft

    return sp_probs, sft_probs
