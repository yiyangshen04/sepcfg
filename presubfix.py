#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_policy_affixes.py

基于“策略型”正则表（单数字、两位数字、年份、单符号、符号+数字、数字+符号）
检测每条密码的前缀/后缀（1–4 字符），并输出排名前几的结果到 CSV：
  - suffixes.csv （后缀 & 频次）
  - prefixes.csv （前缀 & 频次）
"""

import duckdb

# ========== 配置区域 ==========
from config import get_duckdb_path, TABLE_NAME, PASSWORD_COL

# 正则模式：策略型前/后缀（POSIX 风格）
AFFIX_PATTERN = r'^([0-9]{1,2}|(19|20)[0-9]{2}|[!@#$%^&*]|[!@#$%^&*][0-9]|[0-9][!@#$%^&*])$'
# 核心最小长度：去掉前/后缀后，核心部分必须 ≥ 4 字符且含至少一个字母
MIN_CORE_LEN = 4


# ==============================

def main():
    # 连接 DuckDB（只读）
    con = duckdb.connect(database=get_duckdb_path(), read_only=True)

    # --- 后缀提取与频率统计 ---
    suffix_sql = f"""
    WITH lengths AS (
      SELECT 1 AS L UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
    ),
    cand AS (
      SELECT
        substring({PASSWORD_COL}, -L, L) AS suffix,
        {PASSWORD_COL} AS pwd,
        L
      FROM {TABLE_NAME}, lengths
    )
    SELECT
      suffix,
      COUNT(*) AS freq
    FROM cand
    WHERE
      length(pwd) - L >= {MIN_CORE_LEN}
      AND suffix ~ '{AFFIX_PATTERN}'
      AND substring(pwd, 1, length(pwd) - L) ~ '.*[A-Za-z].*'
    GROUP BY suffix
    ORDER BY freq DESC
    """
    suffix_df = con.execute(suffix_sql).df()
    suffix_df.to_csv("suffixes.csv", index=False, encoding="utf-8-sig")

    # --- 前缀提取与频率统计 ---
    prefix_sql = f"""
    WITH lengths AS (
      SELECT 1 AS L UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
    ),
    cand AS (
      SELECT
        substring({PASSWORD_COL}, 1, L) AS prefix,
        {PASSWORD_COL} AS pwd,
        L
      FROM {TABLE_NAME}, lengths
    )
    SELECT
      prefix,
      COUNT(*) AS freq
    FROM cand
    WHERE
      length(pwd) - L >= {MIN_CORE_LEN}
      AND prefix ~ '{AFFIX_PATTERN}'
      AND substring(pwd, L + 1) ~ '.*[A-Za-z].*'
    GROUP BY prefix
    ORDER BY freq DESC
    """
    prefix_df = con.execute(prefix_sql).df()
    prefix_df.to_csv("prefixes.csv", index=False, encoding="utf-8-sig")

    # 打印前 10 条供快速检查
    print("=== Top 10 后缀 ===")
    print(suffix_df.head(10).to_string(index=False))
    print("\n=== Top 10 前缀 ===")
    print(prefix_df.head(10).to_string(index=False))
    print("\n已生成文件：suffixes.csv, prefixes.csv")


if __name__ == "__main__":
    main()
