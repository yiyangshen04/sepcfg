# data_retriever.py

import duckdb

from config import get_duckdb_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL


def fetch_breaches_data(limit=None):
    """
    从DuckDB数据库中读取账号和密码。
    limit: 可选，从数据库中只取前n条测试。
    返回: [(account, password), (account, password), ...]
    """
    conn = duckdb.connect(database=get_duckdb_path(), read_only=True)
    query = f"SELECT {ACCOUNT_COL}, {PASSWORD_COL} FROM {TABLE_NAME}"
    if limit is not None:
        query += f" LIMIT {limit}"
    results = conn.execute(query).fetchall()
    conn.close()
    return results
