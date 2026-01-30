# segmenter/keyboard_walk.py

"""
Module: keyboard_walk
---------------------
Provides functionality to detect dynamic keyboard walks (e.g. 1qaz, qwerty, xsw2)
on a US QWERTY layout. The core logic is adapted from the user-provided code snippet.
"""


def _get_us_keyboard():
    """
    Returns a dictionary of key mappings for a US QWERTY keyboard.
    """
    keyboard_mapping = {
        'name': "qwerty",
        'row1': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
        's_row1': ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+'],

        'row2': ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
        's_row2': ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],

        'row3': ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
        's_row3': ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],

        'row4': ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
        's_row4': ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    }
    return keyboard_mapping


def find_keyboard_row_column(char, keyboards):
    """
    Finds the keyboard row/column of a character across multiple keyboard definitions.
    Here we only have US QWERTY, but it can be extended.

    Returns: dict
      { "keyboard_name": {'row': int, 'pos': int}, ... }
    """
    pos_list = {}
    for board in keyboards:
        # row1 / s_row1
        if char in board['row1']:
            pos_list[board['name']] = {'row': 1, 'pos': board['row1'].index(char)}
        elif char in board['s_row1']:
            pos_list[board['name']] = {'row': 1, 'pos': board['s_row1'].index(char)}

        # row2 / s_row2
        elif char in board['row2']:
            pos_list[board['name']] = {'row': 2, 'pos': board['row2'].index(char)}
        elif char in board['s_row2']:
            pos_list[board['name']] = {'row': 2, 'pos': board['s_row2'].index(char)}

        # row3 / s_row3
        elif char in board['row3']:
            pos_list[board['name']] = {'row': 3, 'pos': board['row3'].index(char)}
        elif char in board['s_row3']:
            pos_list[board['name']] = {'row': 3, 'pos': board['s_row3'].index(char)}

        # row4 / s_row4
        elif char in board['row4']:
            pos_list[board['name']] = {'row': 4, 'pos': board['row4'].index(char)}
        elif char in board['s_row4']:
            pos_list[board['name']] = {'row': 4, 'pos': board['s_row4'].index(char)}

    return pos_list


def is_next_on_keyboard(past, current):
    """
    Checks if 'current' character is adjacent to 'past' character on the same keyboard layout.
    past/current are dicts from find_keyboard_row_column().
    Returns dict of all matching keyboard runs: { 'keyboard_name': {...}, ... }
    """
    current_runs = {}

    if (past is None) or (current is None):
        return current_runs

    # Loop through the past keyboards that were detected
    for past_name, past_data in past.items():

        # Quick fail if both characters were not found in the same dictionary
        if past_name not in current:
            continue

        cur_data = current[past_name]

        # Exclude repeated exactly same char (pos)
        if (cur_data['row'] == past_data['row']) and (cur_data['pos'] == past_data['pos']):
            continue

        # Same row, position +/- 1
        if cur_data['row'] == past_data['row']:
            if abs(cur_data['pos'] - past_data['pos']) == 1:
                current_runs[past_name] = {
                    'past_row': past_data['row'],
                    'past_pos': past_data['pos'],
                    'cur_row': cur_data['row'],
                    'cur_pos': cur_data['pos']
                }
        # One row down
        elif cur_data['row'] == past_data['row'] + 1:
            # positions can match or be offset by 1
            if (cur_data['pos'] == past_data['pos']) or (cur_data['pos'] == past_data['pos'] - 1):
                current_runs[past_name] = {
                    'past_row': past_data['row'],
                    'past_pos': past_data['pos'],
                    'cur_row': cur_data['row'],
                    'cur_pos': cur_data['pos']
                }
        # One row up
        elif cur_data['row'] == past_data['row'] - 1:
            if (cur_data['pos'] == past_data['pos']) or (cur_data['pos'] == past_data['pos'] + 1):
                current_runs[past_name] = {
                    'past_row': past_data['row'],
                    'past_pos': past_data['pos'],
                    'cur_row': cur_data['row'],
                    'cur_pos': cur_data['pos']
                }

    return current_runs


def is_horizontal_run_in_one_row(combo):
    """
    检测 combo 是否在键盘的同一行上连续（横向相邻）。
    例如: '1234', 'asdf', 'dfgh', 'poiu' 等都满足：
      - 所有字符都在同一 row
      - 相邻字符在 pos 上差值为 +1 或 -1（可正向可反向）
    """
    keyboards = [_get_us_keyboard()]
    row_pos_list = []

    for c in combo:
        pos_list = find_keyboard_row_column(c, keyboards)
        # 若在当前键盘上找不到任何位置，说明无法匹配
        if not pos_list or 'qwerty' not in pos_list:
            return False
        row_pos_list.append(pos_list['qwerty'])

    base_row = row_pos_list[0]['row']
    for i in range(1, len(row_pos_list)):
        # 必须保证行相同
        if row_pos_list[i]['row'] != base_row:
            return False
        # 必须保证 pos 连续 (正向或反向)
        if abs(row_pos_list[i]['pos'] - row_pos_list[i - 1]['pos']) != 1:
            return False

    return True


def interesting_keyboard(combo):
    """
    A simple heuristic to decide if a found combo is 'interesting' enough
    to be considered a keyboard walk.
    """
    if len(combo) < 1:
        return False

    # --------------------
    # 新增: 若长度 >= 4 并且满足横向连续，则直接认定为键盘模式
    # --------------------
    if len(combo) >= 4 and is_horizontal_run_in_one_row(combo):
        return True

    # 你原有的过滤/黑名单逻辑（false_positive_words 等）可保留
    false_positive_words = [
        "drew", "kiki", "fred", "were", "pop",
        # 其他略...
    ]
    test_str = ''.join(combo).lower()
    for fp in false_positive_words:
        if fp in test_str:
            return False

    # 复杂度要求：同时出现字母/数字/特殊字符中至少两种
    alpha = 0
    digit = 0
    special = 0
    for c in combo:
        if c.isalpha():
            alpha = 1
        elif c.isdigit():
            digit = 1
        else:
            special = 1

    if (alpha + digit + special) >= 2:
        return True

    return False


def detect_keyboard_walk(password, min_keyboard_run=4):
    """
    Iteratively detect keyboard walks in `password` of length >= min_keyboard_run.
    Returns a list of segments: [(sub_str, "K4"), (sub_str, None), ...].
      - "K4"/"K5" etc. indicates the length of the walk.
      - Plain segments are labeled None.
    """
    keyboards = [_get_us_keyboard()]
    sections = []
    last_cut_idx = 0  # 上次切分位置

    idx = 0
    while idx < len(password):
        # 尝试在 idx 处启动一个走位探索
        cur_run = [password[idx]]
        past_pos = find_keyboard_row_column(password[idx], keyboards)
        j = idx + 1

        # 向后探索相邻走位
        while j < len(password):
            pos = find_keyboard_row_column(password[j], keyboards)
            if is_next_on_keyboard(past_pos, pos):
                cur_run.append(password[j])
                past_pos = pos
                j += 1
            else:
                break

        # 判断这段 run 是否足够长且有趣
        if len(cur_run) >= min_keyboard_run and interesting_keyboard(cur_run):
            # 先切分普通段
            if idx > last_cut_idx:
                straight = password[last_cut_idx:idx]
                sections.append((straight, None))
            # 再切分走位段
            run_str = ''.join(cur_run)
            sections.append((run_str, f"K{len(cur_run)}"))
            # 更新指针到 run 之后
            last_cut_idx = j
            idx = j
        else:
            # 如果不是走位，就跳过一个字符（当作普通段的一部分）
            idx += 1

    # 处理末尾残余的普通段
    if last_cut_idx < len(password):
        tail = password[last_cut_idx:]
        sections.append((tail, None))

    return sections
