import pickle
from pathlib import Path

import pandas as pd

from config import project_path


def main():
    pkl_path = project_path("checkpoints", "my_training_20250515_counts.pkl")
    output_dir = project_path("exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with Path(pkl_path).open("rb") as f:
        data = pickle.load(f)

    # 处理各部分数据并排序
    df_sp_counts = pd.DataFrame([
        {"structure": sp, "count": count} for sp, count in data["sp_counts"].items()
    ]).sort_values(by="count", ascending=False)

    df_sft_counts = pd.DataFrame([
        {"sft": sft, "sf": sf, "count": count}
        for sft, subdict in data["sft_counts"].items()
        for sf, count in subdict.items()
    ]).sort_values(by="count", ascending=False)

    df_sft_totals = pd.DataFrame([
        {"sft": sft, "total": total} for sft, total in data["sft_totals"].items()
    ]).sort_values(by="total", ascending=False)

    df_sp_total = pd.DataFrame([{"sp_total": data["sp_total"]}])  # 只有一行，无需排序

    # 保存为 CSV 文件
    df_sp_counts.to_csv(output_dir / "sp_counts.csv", index=False)
    df_sft_counts.to_csv(output_dir / "sft_counts.csv", index=False)
    df_sft_totals.to_csv(output_dir / "sft_totals.csv", index=False)
    df_sp_total.to_csv(output_dir / "sp_total.csv", index=False)

    print("✅ 所有 CSV 文件已保存到目录：", output_dir)


if __name__ == "__main__":
    main()
