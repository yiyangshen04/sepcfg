import duckdb

from config import get_duckdb_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL


def main():
    # 连接到 DuckDB 数据库
    conn = duckdb.connect(database=get_duckdb_path())

    # 仅查询账户和密码均非空的数据
    query = f"""
        SELECT {ACCOUNT_COL}, {PASSWORD_COL}
        FROM {TABLE_NAME}
        WHERE {ACCOUNT_COL} IS NOT NULL AND {PASSWORD_COL} IS NOT NULL
    """
    df = conn.execute(query).fetchdf()

    # 定义邮箱地址匹配的正则表达式（与 is_email_str 内部相同）
    email_pattern = r"^[\w.%+\-]+@[\w.\-]+\.\w+$"

    # 利用 vectorized 方式筛选出合法的邮箱记录（忽略空值情况）
    valid_email_mask = df[ACCOUNT_COL].str.match(email_pattern, na=False)
    df_valid = df[valid_email_mask].copy()

    total_email_records = df_valid.shape[0]

    # 向量化操作：提取邮箱用户名部分（即 "@" 前的部分）
    df_valid['username'] = df_valid[ACCOUNT_COL].str.split('@').str[0]

    # 向量化比较：判断密码是否与邮箱用户名完全一致（忽略大小写）
    exact_match_mask = df_valid[PASSWORD_COL].str.lower() == df_valid['username'].str.lower()
    exact_match_records = exact_match_mask.sum()

    percentage = (exact_match_records / total_email_records * 100) if total_email_records > 0 else 0

    print(f"总邮箱记录数: {total_email_records}")
    print(f"完全匹配（密码 == 邮箱用户名）的记录数: {exact_match_records}")
    print(f"百分比: {percentage:.2f}%")


if __name__ == "__main__":
    main()
