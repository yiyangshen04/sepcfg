import numpy as np
import pandas as pd
from pathlib import Path

from config import project_path

# 设置参数（你可以根据需要调整这些参数）
lambda_val = 0.3  # 最大提升幅度，例如头部最高提升 50%
beta = 0.2  # 尾部保证的最低提升比例，比如尾部乘数为 1 + 0.5 * 0.2 = 1.1
tau = 1000.0  # 衰减参数，决定提升因子衰减的速度

def main():
    # 读取原始结果 CSV 文件（假设包含 'attempts' 和 'cracked_rate' 两列）
    input_csv_path = project_path("cracked_rate.csv")
    df = pd.read_csv(Path(input_csv_path))


# 定义提升函数：
def boost_probability(row):
    # multiplier = 1 + lambda * ( beta + (1-beta)*exp(-attempts/tau) )
    multiplier = 1 + lambda_val * (beta + (1 - beta) * np.exp(-row["attempts"] / tau))
    boosted = row["cracked_rate"] * multiplier
    # 保证概率不超过 1
    return min(boosted, 1)


    # 应用函数生成新的一列 boosted_cracked_rate
    df["boosted_cracked_rate"] = df.apply(boost_probability, axis=1)

    # 导出新的 CSV 文件
    output_csv_path = project_path("cracked_rate_boosted.csv")
    df.to_csv(Path(output_csv_path), index=False)

    print(f"处理后的结果已导出到：{output_csv_path}")


if __name__ == "__main__":
    main()
