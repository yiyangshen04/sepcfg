import concurrent.futures
import math
import os
from pathlib import Path

import duckdb
import pandas as pd

from config import get_duckdb_path, project_path, TABLE_NAME, ACCOUNT_COL, PASSWORD_COL

##############################
# 1) 帮助函数：替换占位符
##############################

def extract_email_user_part(email_str):
    """提取邮箱的用户名前缀, 如 alice@gmail.com -> alice"""
    return email_str.split('@', 1)[0]


def extract_email_domain_part(email_str):
    """提取邮箱域名主体, 如 alice@gmail.com -> gmail.com (或 'gmail')"""
    return email_str.split('@', 1)[1]


def replace_placeholders(candidate_str, account):
    """
    将带 [acc_xxx] 占位符的候选密码替换成真实账号信息。
    假设:
     - [acc_pwd_same]    -> 直接替换为 account
     - [acc_email_name]  -> 账号是邮箱时替换为邮箱前缀
     - [acc_email_domain]-> 替换为邮箱域名(可自行决定只保留主体或含后缀)
    若不是邮箱，则简单使用 account 替换即可。
    """
    if '@' in account:
        user_part = extract_email_user_part(account)
        domain_part = extract_email_domain_part(account)
    else:
        user_part = account
        domain_part = account

    replaced = candidate_str
    replaced = replaced.replace("[acc_pwd_same]", account)
    replaced = replaced.replace("[acc_email_name]", user_part)
    replaced = replaced.replace("[acc_email_domain]", domain_part)
    return replaced


##############################
# 2) 读取PCFG生成的密码文件
##############################

def load_generated_passwords(txt_path, max_k):
    """
    从txt文件中读取形如:
      [acc_email_name]\t3.48118600e-02
      123456           \t2.51652687e-02
    等数据, 返回list[(candidate_str, prob), ...]
    假设文件已经按概率从大到小排序.
    只保留前 max_k 条, 以避免浪费时间.
    """
    candidates = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_k:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                pwd_str, prob_str = parts
                try:
                    prob_val = float(prob_str)
                except ValueError:
                    prob_val = 0.0
                candidates.append((pwd_str, prob_val))
    return candidates


##############################
# 3) 子任务：对一批用户扫描(单线程)
##############################

def scan_users_chunk(users_chunk, all_candidates):
    """
    对一批用户 (account, real_pwd) 完整扫描 all_candidates[0..len(all_candidates)],
    找到各用户'最早匹配'的索引. 若未匹配则返回 None.
    """
    results = []
    for (acct, real_pwd) in users_chunk:
        found_idx = None
        # 如果字符串替换和对比很大量，这里也可以考虑进一步优化
        for i, (cand_str, _) in enumerate(all_candidates):
            replaced_str = replace_placeholders(cand_str, acct)
            if replaced_str == real_pwd:
                found_idx = i
                break
        results.append(found_idx)
    return results


##############################
# 4) 评估单个模型的函数
##############################
def evaluate_model_for_topk(user_records, model_file, max_k_list, chunk_size_factor=0.1):
    """
    针对某个PCFG模型(给定生成文件), 在 user_records(账号+密码) 数据上
    计算不同Top-K时的成功率.

    由于外面会针对多个模型并行，这里就不要再启进程池了，直接单进程处理。
    如果数据量特别大且确实需要在单模型内部也并行，可改用ThreadPoolExecutor。
    """
    max_k_all = max(max_k_list)
    print(f"[INFO] 从 {model_file} 读取候选, 只保留前 {max_k_all} 条...")
    all_candidates = load_generated_passwords(model_file, max_k_all)
    real_max_len = len(all_candidates)
    print(f"[INFO] 实际用于匹配的候选数量: {real_max_len}")

    total_users = len(user_records)
    min_chunk_size = 100  # 每个块最小为100个用户
    chunk_size = max(min_chunk_size, math.floor(total_users * chunk_size_factor))

    all_earliest_indices = []
    # 单线程分块扫描
    for i in range(0, total_users, chunk_size):
        chunk = user_records[i:i + chunk_size]
        chunk_result = scan_users_chunk(chunk, all_candidates)
        all_earliest_indices.extend(chunk_result)

    if len(all_earliest_indices) != total_users:
        print("[WARN] 匹配结果数量与用户数不一致, 请检查逻辑!")

    # 计算各个Top-K的成功率
    results_dict = {}
    for K in max_k_list:
        actual_k = min(K, real_max_len)
        hits = sum(1 for idx in all_earliest_indices if (idx is not None and idx < actual_k))
        rate = hits / total_users if total_users > 0 else 0
        results_dict[K] = rate
    return results_dict


##############################
# 5) 并行主入口：计算多个模型的评估结果
##############################

def evaluate_multiple_models(user_records, model_files, max_k_list):
    """
    针对多个PCFG模型进行并行评估。如果有6个模型，就最多开6个进程，
    每个进程只负责评估一个模型，避免嵌套开进程池。
    """
    results = []

    # 这里的 max_workers 可以根据模型数和 CPU 核心数综合决定
    # 若只想按模型数开进程，则可用:
    max_workers = min(len(model_files), os.cpu_count())

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {}
        for label, model_file in model_files.items():
            future = executor.submit(
                evaluate_model_for_topk,
                user_records,
                model_file,
                max_k_list
            )
            future_to_label[future] = label

        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            result_dict = future.result()
            for k_val, succ_rate in result_dict.items():
                results.append({
                    "model_label": label,
                    "Top_K": k_val,
                    "success_rate": succ_rate
                })
    return results


##############################
# 6) 主入口
##############################
if __name__ == "__main__":
    data_dir = project_path("data")
    MODEL_FILES = {
        "1M": str(Path(data_dir) / "generated_passwords_pruned_1M.txt"),
        "2M": str(Path(data_dir) / "generated_passwords_pruned_2M.txt"),
        "3M": str(Path(data_dir) / "generated_passwords_pruned_3M.txt"),
        "4M": str(Path(data_dir) / "generated_passwords_pruned_4M.txt"),
        "5M": str(Path(data_dir) / "generated_passwords_pruned_5M.txt"),
        "6M": str(Path(data_dir) / "generated_passwords_pruned_6M.txt"),
        "7M": str(Path(data_dir) / "generated_passwords_pruned_7M.txt"),
        "8M": str(Path(data_dir) / "generated_passwords_pruned_8M.txt"),
        "9M": str(Path(data_dir) / "generated_passwords_pruned_9M.txt"),
        "10M": str(Path(data_dir) / "generated_passwords_pruned_10M.txt"),
    }

    # 过滤掉仓库内不存在的模型文件，避免直接报错
    MODEL_FILES = {k: v for k, v in MODEL_FILES.items() if Path(v).exists()}
    if not MODEL_FILES:
        print(f"[ERROR] 未找到任何模型文件（期望在目录：{data_dir}）")
        raise SystemExit(2)

    max_k_list = list(range(1000, 10001, 1000))  # 要评估的Top-K列表

    # ============ 读取用户泄露数据 =============
    print("[INFO] 从 DUCKDB 加载用户记录...")
    con = duckdb.connect(get_duckdb_path())
    query = f"SELECT {ACCOUNT_COL}, {PASSWORD_COL} FROM {TABLE_NAME} LIMIT 100000"
    df = con.execute(query).fetchdf()
    con.close()
    print(f"[INFO] 数据加载完成, 一共 {len(df)} 条.")
    user_records = list(zip(df[ACCOUNT_COL], df[PASSWORD_COL]))

    # ============ 并行计算多个模型评估结果 =============
    print(f"[INFO] 开始评估多个模型...")
    final_results = evaluate_multiple_models(
        user_records=user_records,
        model_files=MODEL_FILES,
        max_k_list=max_k_list
    )

    # ============ 保存到CSV =============
    results_df = pd.DataFrame(final_results)
    results_df.sort_values(by=["model_label", "Top_K"], inplace=True)
    out_csv_path = "eval_cracking_results_all_models.csv"
    results_df.to_csv(out_csv_path, index=False, float_format="%.6f")
    print(f"[INFO] 评估结果已保存至: {os.path.abspath(out_csv_path)}")
