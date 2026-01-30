import duckdb
import pandas as pd

from config import get_duckdb_path, TABLE_NAME, PASSWORD_COL

# 设置 Pandas 显示选项，保证完整显示所有行和列，并按四位小数格式显示浮点数
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def main():
    # 连接到 DuckDB 数据库（路径由环境变量 SE_PCFG_DUCKDB_PATH / DUCKDB_PATH 提供）
    conn = duckdb.connect(get_duckdb_path())

    # 构造 SQL 查询：计算密码长度分布并保留百分比（参考前面的统计）
    query_distribution = f"""
    SELECT
        LENGTH({PASSWORD_COL}) AS password_length,
        COUNT(*) AS count
    FROM
        {TABLE_NAME}
    GROUP BY
        password_length
    ORDER BY
        password_length;
    """

    # 执行查询，并将结果转为 Pandas DataFrame
    df_distribution = conn.execute(query_distribution).fetchdf()

    # 计算总记录数
    total_count = df_distribution['count'].sum()

    # 计算百分比并保留四位小数
    df_distribution['percentage'] = (df_distribution['count'] / total_count * 100).round(4)

    # 打印密码长度分布
    print("密码长度分布统计：")
    print(df_distribution)

    # 构造 SQL 查询：计算平均密码长度，仅考虑密码长度小于等于 30 的记录
    query_avg = f"""
    SELECT
        AVG(LENGTH({PASSWORD_COL})) AS avg_password_length
    FROM
        {TABLE_NAME}
    WHERE
        LENGTH({PASSWORD_COL}) <= 30;
    """

    # 执行查询，获取平均密码长度结果
    df_avg = conn.execute(query_avg).fetchdf()

    # 打印平均密码长度结果
    print("\n仅考虑密码长度<=30的记录时的平均密码长度：")
    print(df_avg)


if __name__ == "__main__":
    main()
