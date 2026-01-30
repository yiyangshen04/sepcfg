import json
from pathlib import Path

from config import project_path


def main():
    src = project_path("checkpoints", "sp_counts.json")
    out = project_path("checkpoints", "sp_counts_sorted.json")

    data = json.loads(Path(src).read_text(encoding="utf-8"))

    # 按值从高到低排序
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # 打印排序后的结果
    for k, v in sorted_data.items():
        print(f"{k}: {v}")

    # 可选：保存为新文件
    Path(out).write_text(
        json.dumps(sorted_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
