from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

# === 配置 ===
SRC = Path(r"data/Chinese-common-password-list.txt")   # 源文件
KEEP = 1_000_000                                       # 要保留的前 N 行

# === 主逻辑 ===
with SRC.open("r", encoding="utf-8", errors="ignore") as fr, \
     NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:

    for i, line in enumerate(fr):
        if i >= KEEP:          # 超过 N 行就停止复制
            break
        tmp.write(line)

# 用生成的临时文件覆盖原文件
shutil.move(tmp.name, SRC)
print(f"Done! 仅保留前 {KEEP:,} 行。")
