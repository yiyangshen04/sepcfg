#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pcfg_targeted_eval_L2_sample_parallel_fixed.py

• 从 breaches 表中随机抽 100k 条账号-密码
• 基于抽样集合完成 targeted‐PCFG 覆盖率评估（并行版）
• 修正了静态模板去重和 None→int rank 填充
"""

import re
import duckdb
import pandas as pd
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count

from config import get_duckdb_path, project_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL

# ========== 全局配置 ==========
TEMPLATE_FILE = str(project_path("my_training_20250513_counts.txt"))
OUTPUT_CSV = str(project_path("cracked_rate_sample100k_new0513.csv"))

MAX_ATTEMPTS  = 30_000
STEP          = 100
SAMPLE_SIZE   = 100_0000
# ========================================== #

# ========== 占位符渲染函数（保持不变） ==========
ACC_RE = re.compile(r'\[(acc_email_name|acc_email_domain|acc_email_domain_com)\]')

def _extract_email_parts(acc: str):
    if '@' not in acc:
        return acc, '', ''
    user, full = acc.split('@', 1)
    parts = full.lower().split('.')
    main = parts[-3] if len(parts) >= 3 else (parts[0] if len(parts) == 2 else full.lower())
    return user, main, full.lower()

def render(template: str, account: str) -> str:
    user, dom_main, dom_full = _extract_email_parts(account)
    def repl(m):
        tag = m.group(1)
        return {
            'acc_email_name':       user,
            'acc_email_domain':     dom_main,
            'acc_email_domain_com': dom_full
        }[tag]
    return ACC_RE.sub(repl, template)

# ========== 子进程处理函数 ==========
def process_chunk(df_chunk: pd.DataFrame, static_map: dict, dyn_list: list) -> pd.DataFrame:
    out_rows = []
    for account, pwd in df_chunk.itertuples(index=False):
        best = None
        # 静态模板查哈希
        if pwd in static_map:
            best = static_map[pwd]
        # 动态模板
        for templ, rn in dyn_list:
            if pwd == render(templ, account):
                if best is None or rn < best:
                    best = rn
        out_rows.append((account, best))
    return pd.DataFrame(out_rows, columns=['account', 'rank'])

def main():
    # —— 1. 抽样 ——
    print("[INFO] Sampling 100k records from DuckDB…")
    conn = duckdb.connect(database=get_duckdb_path(), read_only=True)
    sample_df = conn.execute(f"""
        SELECT {ACCOUNT_COL} AS account, {PASSWORD_COL} AS password
        FROM {TABLE_NAME}
        ORDER BY RANDOM()
        LIMIT {SAMPLE_SIZE}
    """).fetchdf()
    conn.close()

    # —— 2. 加载模板 ——
    print("[INFO] Loading template counts…")
    tpl_df = pd.read_csv(
        TEMPLATE_FILE,
        sep='\t',
        header=None,
        names=['template', 'prob'],
        dtype={'template': str, 'prob': float}
    ).dropna(subset=['prob'])
    tpl_df = tpl_df.sort_values('prob', ascending=False).reset_index(drop=True).iloc[:MAX_ATTEMPTS]
    tpl_df['rn'] = tpl_df.index + 1

    # 静态 vs 动态模板
    static_tpl = tpl_df[~tpl_df['template'].str.contains(r'\[acc_')].copy()
    # 去重：同一个明文只保留概率最高的一条
    static_tpl = static_tpl.drop_duplicates('template', keep='first')
    static_map = dict(zip(static_tpl['template'], static_tpl['rn']))

    dyn_tpl = tpl_df[tpl_df['template'].str.contains(r'\[acc_')].copy()
    dyn_list = list(zip(dyn_tpl['template'], dyn_tpl['rn']))

    # —— 3. 并行计算 best_rank ——
    n_workers = cpu_count()
    print(f"[INFO] Running evaluation on {n_workers} parallel workers…")
    chunk_size = math.ceil(len(sample_df) / n_workers)
    chunks = [
        sample_df.iloc[i*chunk_size:(i+1)*chunk_size]
        for i in range(n_workers)
    ]

    with Pool(n_workers) as pool:
        results = pool.starmap(
            process_chunk,
            [(chunk, static_map, dyn_list) for chunk in chunks]
        )
    best_rank_df = pd.concat(results, ignore_index=True)

    # —— 4. 填充 None 并转 int ——
    best_rank_df['rank'] = (
        best_rank_df['rank']
        .fillna(MAX_ATTEMPTS + 1)  # 未命中视为超过上限
        .astype(int)
    )

    # —— 5. 统计覆盖率 ——
    print("[INFO] Computing cracked-rate curve…")
    attempts = list(range(STEP, MAX_ATTEMPTS + 1, STEP))
    rows = []
    total = SAMPLE_SIZE
    for a in attempts:
        cracked = best_rank_df['rank'].le(a).sum()
        rows.append((a, round(cracked / total, 8)))
    result_df = pd.DataFrame(rows, columns=['attempts', 'cracked_rate'])

    # —— 6. 保存 ——
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] cracked-rate curve saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
