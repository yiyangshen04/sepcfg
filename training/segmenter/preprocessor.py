from __future__ import annotations

# segmenter/preprocessor.py
import re

# ---------------- 中文姓名（仅 *_special 两类） ---------------- #
from config import project_path

from .cn_name_detection import (
    detect_cn_names_before_lds,
    build_or_load_detector,
)
from .keyboard_walk import detect_keyboard_walk

_SPECIAL_PKL_PATH = str(project_path("data", "cn_name_automata_special.pkl"))
_SPECIAL_LABELS = {
    "cn_name_full_special",
    "cn_name_first_last_special",
}

# 全局缓存，避免重复反序列化
_SPECIAL_DETECTOR = build_or_load_detector(_SPECIAL_PKL_PATH)
# ------------------------------------------------------------- #

_EMAIL_RE = re.compile(r"^[\w.%+\-]+@[\w.\-]+\.\w+$")

# --------- 额外允许的 3 位键盘模式 --------- #
_KB3_PATTERNS = {"qaz", "zxc", "asd", "qwe"}
_KB3_RE = re.compile(r"(qaz|zxc|asd|qwe|wsx|xsw|zaq)", re.IGNORECASE)



def is_email_str(text):
    return bool(_EMAIL_RE.match(text))


def extract_email_user_part(email_str):
    return email_str.split('@', 1)[0]


def extract_email_domain_part(email_str):
    """
    从 email 中提取“主域名”：
      - 对 a.b.c.d，返回倒数第三段（b.c.d → b）
      - 对 a.b，返回 a
      - 否则返回整个 domain_str
    """
    domain_str = email_str.split('@', 1)[1].lower()
    parts = domain_str.split('.')
    if len(parts) >= 3:
        return parts[-3]
    elif len(parts) == 2:
        return parts[0]
    else:
        return domain_str


def _same_token_category(a: str, b: str) -> bool:
    """
    判断 a、b 两段文本是否属于同一种“基本字符类别”：
      · 全为数字
      · 全为字母
      · 全为非字母数字（特殊字符）
    只要同类就允许在 detect_alpha_runs 的合并阶段连在一起。
    """
    return (
        (a.isdigit() and b.isdigit()) or
        (a.isalpha() and b.isalpha()) or
        (not a.isalnum() and not b.isalnum())
    )

def detect_kb3_special(text: str):
    """
    检测 text 中的 'qaz' / 'zxc' / 'asd' / 'qwe'（忽略大小写）。
    若命中则返回 [(seg_text, seg_label_or_None), ...]，
    其中命中的子串统一打 'kb3'，未命中的子串打 None，
    并保持原顺序以便后续递归处理。
    未命中返回 None。
    """
    if text.isdigit():
        return None  # 纯数字不用检测

    matches = list(_KB3_RE.finditer(text))
    if not matches:
        return None

    segs = []
    idx = 0
    for m in matches:
        if m.start() > idx:                       # 前缀
            segs.append((text[idx:m.start()], None))
        segs.append((m.group(), "kb3"))           # 命中片段
        idx = m.end()
    if idx < len(text):                           # 尾巴
        segs.append((text[idx:], None))

    return segs


# ================= 连续字母检测 ================= #
def detect_alpha_runs(text: str, min_alpha_run: int = 3):
    """
    将字符串切成若干段，标记其中按字母表顺序连续 ≥ min_alpha_run 的子串，
    但 **仅在该子串左右两端都不是字母** 时才打 alphaN 标签。

    返回 [(seg_text, label_or_None), ...]。
    """
    n = len(text)
    i = 0
    out: list[tuple[str, str | None]] = []

    # ────────── 主扫描：识别升序字母子串 ──────────
    while i < n:
        ch = text[i]

        # 非字母 → 直接输出
        if not ch.isalpha():
            out.append((ch, None))
            i += 1
            continue

        # 检测严格升序序列
        j = i
        while (
            j + 1 < n
            and text[j + 1].isalpha()
            and (ord(text[j + 1].lower()) - ord(text[j].lower()) == 1)
        ):
            j += 1

        run_len = j - i + 1
        run_txt = text[i : j + 1]

        # 左右字符判定：只要一端还是字母就不打标签
        left_char  = text[i - 1] if i > 0 else None
        right_char = text[j + 1] if j + 1 < n else None
        left_is_alpha  = left_char is not None and left_char.isalpha()
        right_is_alpha = right_char is not None and right_char.isalpha()

        if run_len >= min_alpha_run and not left_is_alpha and not right_is_alpha:
            out.append((run_txt, f"alpha{run_len}"))
        else:
            out.append((run_txt, None))

        i = j + 1

    # ────────── 合并相邻无标签片段 ──────────
    merged: list[tuple[str, str | None]] = []
    for seg_text, seg_label in out:
        if (
            seg_label is None
            and merged
            and merged[-1][1] is None
            and _same_token_category(seg_text, merged[-1][0])
        ):
            # 同类别（数字 / 字母 / 特殊字符） → 直接拼接
            merged[-1] = (merged[-1][0] + seg_text, None)
        else:
            merged.append((seg_text, seg_label))

    return merged

def label_substring_in_text(original_text, substring, label):
    segments = []
    start_idx = 0

    lower_text = original_text.lower()
    lower_sub = substring.lower()
    sub_len = len(substring)

    while True:
        idx = lower_text.find(lower_sub, start_idx)
        if idx == -1:
            if start_idx < len(original_text):
                rest = original_text[start_idx:]
                segments.extend(preprocess_string(rest))
            break

        if idx > start_idx:
            prefix_str = original_text[start_idx:idx]
            segments.extend(preprocess_string(prefix_str))

        matched_substring = original_text[idx: idx + sub_len]
        segments.append((matched_substring, label))

        start_idx = idx + sub_len

    return segments




# ------------- 其余辅助函数（保持不变，省略行内注释） -------------
def detect_special_patterns(text: str):
    """
    依次检测：
      1) 重复子串 (srX)
      2) 键盘路径 (kbX) —— 纯数字段不当作键盘路径
      3) 字母表升序序列 (alphaN)
    若都不匹配则返回 None，让调用方继续走中文名检测或 L/D/S 拆分。
    返回值格式：[(seg_text, seg_label_or_None), ...]  ── 至少有一个 seg_label 非 None
    """

    # ---------- 1) 重复子串检测 ----------
    # ---------------- 连续重复子串检测（新版） ---------------- #



    def _sr_label_for_block(block: str, repeats: int) -> str | None:
        # 1) 纯数字：重复 ≥3 次
        if block.isdigit() and repeats >= 3:
            return f"sr{repeats}"

        # 2) 纯字母且单元长度 = 1：重复 ≥3 次
        if block.isalpha() and len(block) == 1 and repeats >= 3:
            return f"sr{repeats}"

        # 3) 包含字母或混合（非纯数字），单元长度 ≥2，重复 ≥2 次
        if repeats >= 2 and len(block) >= 2 and not block.isdigit():
            return f"sr{repeats}"

        return None

    def detect_sr_segments(text: str):
        """
        连续重复子串检测
        额外约束：在纯数字串中，只有当 *整段数字* 都由某个子单元重复构成，
        才打 srX 标签；否则整段数字留给后续拆分。
        """
        n = len(text)
        if n < 2:
            return None

        i, prefix_start = 0, 0
        out = []

        while i < n:
            matched = False
            # 尝试不同 unit_len（最短 1，最长 (n-i)//2）
            for unit_len in range(1, (n - i) // 2 + 1):
                unit = text[i:i + unit_len]
                # 仅处理纯字母数字单元
                if not re.match(r'^[a-zA-Z0-9]+$', unit):
                    continue

                # 统计连续重复次数
                repeats, j = 1, i + unit_len
                while j + unit_len <= n and text[j:j + unit_len].lower() == unit.lower():
                    j += unit_len
                    repeats += 1

                # ────── 纯数字额外校验：必须整段覆盖 ──────
                if unit.isdigit():
                    left_ok = (i == 0 or not text[i - 1].isdigit())
                    right_ok = (j == n or not text[j].isdigit())
                    if not (left_ok and right_ok):
                        # 只是在数字串中截到一块，不算有效 sr
                        continue

                # —— 新增：非数字前后不能接字母 ——
                if not unit.isdigit():
                    left_char = text[i - 1] if i > 0 else None
                    right_char = text[j] if j < n else None
                    # 前后任何一边是字母，就放弃 srX 检测
                    if (left_char and left_char.isalpha()) or \
                            (right_char and right_char.isalpha()):
                        continue

                # 满足三条规则之一才算 sr
                if repeats >= 2 and (lbl := _sr_label_for_block(unit, repeats)):
                    # ① flush 前缀
                    if i > prefix_start:
                        out.append((text[prefix_start:i], None))
                    # ② push 重复块
                    out.append((text[i:j], lbl))
                    # ③ 移动指针
                    prefix_start = j
                    i = j
                    matched = True
                    break  # unit_len loop

            if not matched:
                i += 1  # 继续向右扫描

        # flush 尾巴
        if prefix_start < n:
            out.append((text[prefix_start:], None))

        # 合并连续 None 段
        merged = []
        for t, l in out:
            if l is None and merged and merged[-1][1] is None:
                merged[-1] = (merged[-1][0] + t, None)
            else:
                merged.append((t, l))

        # 如果没有任何 srX，则返回 None
        return None if all(l is None for _, l in merged) else merged



    sr_segments = detect_sr_segments(text)
    if sr_segments is not None:
        return sr_segments



    # ---------- 2) 键盘路径检测 ----------
    if text.isdigit():
        kb_sections = []  # 纯数字段直接跳过
    else:
        kb_sections = detect_keyboard_walk(text, min_keyboard_run=4)

    # 2-A. 整段正好是一条键盘路径
    if (
        len(kb_sections) == 1
        and kb_sections[0][1]  # 有标签
        and kb_sections[0][0] == text
    ):
        kb_len = kb_sections[0][1][1:]  # K5 → 5
        return [(text, f"kb{kb_len}")]


    # 2-B. 混合字符串里包含键盘路径
    any_kb = any(lbl for _t, lbl in kb_sections if lbl)
    if any_kb:
        refined = []
        for seg_text, seg_label in kb_sections:

            if seg_label and seg_label.startswith("K") and not seg_text.isdigit():
                refined.append((seg_text, f"kb{seg_label[1:]}"))
            else:
                refined.append((seg_text, None))

        # ---- 合并 & 回退策略 ----
        merged = []
        for txt, lbl in refined:
            if (
                lbl is None
                and merged
                and merged[-1][1] is None
            ):
                prev_txt = merged[-1][0]
                # 同为纯数字
                if txt.isdigit() and prev_txt.isdigit():
                    merged[-1] = (prev_txt + txt, None)
                    continue
                # 同为纯字母
                if txt.isalpha() and prev_txt.isalpha():
                    merged[-1] = (prev_txt + txt, None)
                    continue
            merged.append((txt, lbl))

        # 如果最终全部 label 都是 None，说明实际上没有有效 kb 匹配
        if all(lbl is None for _t, lbl in merged):
            return None  # 交回调用者

        return merged
    # ---------- 3) kb3 固定模式检测 --------------- #
    # 只有在上面没找到任何 kb4+ 时才会走到这里，所以不会截胡
    kb3_sections = detect_kb3_special(text)
    if kb3_sections is not None:
        return kb3_sections



    # ---------- 4) 什么都没检测到 ----------
    return None


def split_into_LDS(text):
    segments = []
    buf = ""
    current_type = None

    def flush_buffer(b, t):
        if b:
            segments.append((b, t))

    for ch in text:
        if ch.isalpha():
            t = "L"
        elif ch.isdigit():
            t = "D"
        else:
            t = "S"

        if current_type is None:
            current_type = t
            buf = ch
        else:
            if t == current_type:
                buf += ch
            else:
                flush_buffer(buf, current_type)
                buf = ch
                current_type = t

    flush_buffer(buf, current_type)
    return segments


# ---------------------------------------------------------------


def preprocess_string(text: str):
    """
    统一预处理流程
    ----------------
    1) 先跑 detect_special_patterns(text)                      → srX / kbX
       · **若至少有一个 seg_label ≠ None**：
           ─ 有标签的片段直接保留；
           ─ 无标签的片段递归调用本函数；
           ─ 立即 return 组合结果。
       · **若全部 seg_label 都是 None** 或直接返回 None：
           ─ 说明未命中 sr/kbX（或纯数字 kb 被降级为空）。继续后续步骤。

    2) 检测连续升序字母串 detect_alpha_runs(text, ≥3)           → alphaN
       · 若存在任何 alphaN 标签，则直接返回 detect_alpha_runs 的结果。

    3) 仅对 *_special 两类中文姓名做匹配                       → cn_name_*
       · 若整段命中 ≥2 处中文名，则 fallback 到 split_into_LDS；
       · 否则，命中的中文名直接打标签，其余再用 split_into_LDS 拆分。

    4) 全部都没命中 → 最后 fallback 到 split_into_LDS(text)
    """
    # ---------- 1) srX / kbX 检测 ----------
    sp_result = detect_special_patterns(text)
    if sp_result is not None and any(lbl for _t, lbl in sp_result if lbl):
        final_segments: list[tuple[str, str | None]] = []
        for seg_text, seg_label in sp_result:
            if seg_label is None:
                # 无标签片段再递归跑一遍
                final_segments.extend(preprocess_string(seg_text))
            else:
                final_segments.append((seg_text, seg_label))
        return final_segments

    # ---------- 2) 字母表升序序列 ----------
    alpha_sections = detect_alpha_runs(text, min_alpha_run=3)
    if any(lbl for _t, lbl in alpha_sections if lbl):
        final_sections: list[tuple[str, str | None]] = []
        for seg_text, seg_label in alpha_sections:
            if seg_label is None:
                # 让无标签子串继续拆 → 得到 'D' / 'L' / 'S'
                final_sections.extend(split_into_LDS(seg_text))
            else:
                final_sections.append((seg_text, seg_label))
        return final_sections

    # ---------- 3) 中文姓名（*_special） ----------
    name_segments = detect_cn_names_before_lds(
        text,
        allowed_labels=_SPECIAL_LABELS,
        detector=_SPECIAL_DETECTOR,
    )
    if name_segments:
        final_segments: list[tuple[str, str | None]] = []
        cn_cnt = sum(
            1
            for _t, _lbl in name_segments
            if _lbl and _lbl.startswith("cn_name_")
        )
        if cn_cnt >= 2:
            # 命中 ≥2 处中文名 → 认为误判，走 L/D/S 粗拆
            final_segments.extend(split_into_LDS(text))
        else:
            for seg_text, seg_label in name_segments:
                if seg_label and seg_label.startswith("cn_name_"):
                    final_segments.append((seg_text, seg_label))
                else:
                    final_segments.extend(split_into_LDS(seg_text))
        return final_segments

    # ---------- 4) Fallback: L/D/S ----------
    return split_into_LDS(text)



# ------------------------- 账户-密码入口 ------------------------- #
def preprocess_breach(account: str, password: str):
    acc_lower = account.lower()
    pwd_lower = password.lower()

    # ① 完全相同（忽略大小写）
    if acc_lower == pwd_lower:
        return [(password, "acc_pwd_same")]

    # ②   account 作为子串出现在 password 中（忽略大小写，非 email 版）
    #     —— 注意：如果帐号本身就是 email，仍走后面的 email 分支
    if (not is_email_str(account)) and (acc_lower in pwd_lower):
        # 把帐号子串打上 acc_pwd_same，其余片段继续拆分
        segments = label_substring_in_text(password, account, "acc_pwd_same")
        output = []
        for seg_text, seg_label in segments:
            if seg_label is None:
                output.extend(preprocess_string(seg_text))
            else:
                output.append((seg_text, seg_label))
        return output

    if is_email_str(account):
        user_part = extract_email_user_part(account)
        domain_part = extract_email_domain_part(account)
        full_domain = account.split('@', 1)[1]

        segments = []
        if full_domain and full_domain.lower() in password.lower():
            segments = label_substring_in_text(
                password, full_domain, "acc_email_domain_com"
            )
        else:
            segments = [(password, None)]

        final = []
        for seg_text, seg_label in segments:
            if seg_label is None:
                if user_part and user_part.lower() in seg_text.lower():
                    sub = label_substring_in_text(seg_text, user_part, "acc_email_name")
                else:
                    sub = [(seg_text, None)]
                for t, l in sub:
                    if l is None and domain_part and domain_part.lower() in t.lower():
                        final.extend(
                            label_substring_in_text(t, domain_part, "acc_email_domain")
                        )
                    else:
                        final.append((t, l))
            else:
                final.append((seg_text, seg_label))

        output = []
        for t, l in final:
            if l is None:
                output.extend(preprocess_string(t))
            else:
                output.append((t, l))
        return output

    else:
        return preprocess_string(password)
