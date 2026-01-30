本项目为口令（密码）研究相关代码，核心流程为基于泄露数据训练语义/结构化 PCFG，并生成候选口令用于评估。

## 运行环境
- Python 3.9+
- 依赖（按脚本需要安装）：`duckdb`、`pandas`、`pyahocorasick`、`pypinyin`

## 安装依赖（推荐 venv）
由于系统 Python 可能没有写权限，建议在项目内创建虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install duckdb pandas pyahocorasick pypinyin
```

> `pypinyin` 仅用于 `fill_placeholders.py` 的中文姓名占位符；不装也能跑，但中文占位符会被跳过。

## 路径与跨平台说明
历史版本中存在大量 Windows 绝对路径（如 `C:\\Users\\...` / `D:\\...`）。现在已统一改为：
- 仓库内文件：以项目根目录为基准拼接（`config.PROJECT_ROOT` / `config.project_path()`）
- 外部 DuckDB 数据库：通过环境变量提供路径

需要先配置 DuckDB 路径（数据集一般不随仓库分发）：

```bash
export SE_PCFG_DUCKDB_PATH="/path/to/your_dataset.duckdb"
```

## 入口示例
训练主流程（会读取 DuckDB）：

```bash
.venv/bin/python -m training.main
```

## 生成候选口令（基于训练出的 checkpoint）
生成脚本已整理到 `generation/password_gen_tools/`（最终版本为“先生成占位符模板，再注入个人信息”）：

```bash
# 选择要使用的 checkpoint（对应 checkpoints/<TRAIN_NAME>_counts.pkl）
export SE_PCFG_TRAIN_NAME="my_training_20250515"

# 一键跑完两步（生成模板 + 注入）
.venv/bin/python generation/password_gen_tools/pipeline.py

# 1) 由 checkpoint 生成占位符模板（输出 generation/password_gen_tools/placeholders1.txt）
.venv/bin/python generation/password_gen_tools/generate_placeholders.py

# 2) 读取 teacher.csv，将占位符填充为每个账号的候选口令列表（默认脱敏聚合输出；输出 generation/password_gen_tools/fulllist/*.txt）
.venv/bin/python generation/password_gen_tools/fill_placeholders.py

# 分析用（脱敏输出）：聚合并按概率质量排序（输出“模式串”，不落地明文）
.venv/bin/python generation/password_gen_tools/pipeline.py --pattern-mass --top-k 100 --max-templates 10000

# 分析用（明文输出）：聚合并按概率质量排序（会落地明文，请勿提交到仓库）
.venv/bin/python generation/password_gen_tools/pipeline.py --plain-mass --top-k 100 --max-templates 10000

# 如需旧版“逐模板逐变体输出”（可能非常大且包含明文），使用：
.venv/bin/python generation/password_gen_tools/pipeline.py --raw
```
