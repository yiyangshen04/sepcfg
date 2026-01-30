import duckdb
import pandas as pd

from config import get_duckdb_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL

# 设置 Pandas 显示选项，保证完整显示所有行和列，并按四位小数格式显示浮点数
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def main():
    # 连接到 DuckDB 数据库（路径由环境变量 SE_PCFG_DUCKDB_PATH / DUCKDB_PATH 提供）
    conn = duckdb.connect(get_duckdb_path())

    # 构造 SQL 查询：统计每个密码长度的数量
    query = f"""
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
    df = conn.execute(query).fetchdf()

    # 计算总记录数
    total_count = df['count'].sum()

    # 计算百分比并保留四位小数
    df['percentage'] = (df['count'] / total_count * 100).round(4)

    # 打印结果，完整显示 DataFrame
    print(df)


if __name__ == "__main__":
    main()
