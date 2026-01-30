import ast
import json
from pathlib import Path

from config import project_path


def main():
    src = project_path("checkpoints", "sp_counts.json")
    data = json.loads(Path(src).read_text(encoding="utf-8"))

    # 统计 key 的平均长度（即元组中元素的数量）
    lengths = []
    for key in data.keys():
        tuple_key = ast.literal_eval(key)  # 将字符串转换为元组
        lengths.append(len(tuple_key))

    average_length = sum(lengths) / (len(lengths) or 1)
    print(f"平均长度为: {average_length:.2f}")


if __name__ == "__main__":
    main()
