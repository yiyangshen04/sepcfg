# main.py
from __future__ import annotations

import hashlib
import os

from config import TRAIN_NAME, project_path

from .build_english_dict_automaton import load_english_dict_automaton
from .build_english_name_automaton import load_english_name_automaton
from .segmenter.cn_name_detection import build_or_load_detector
from .training_manager import TrainingManager


def initialize_automata(name_pkl_path: str, dict_pkl_path: str):
    """
    加载并返回英文名和英文词典自动机实例。
    """
    en_name_ac = load_english_name_automaton(name_pkl_path)
    en_dict_ac = load_english_dict_automaton(dict_pkl_path)
    return en_name_ac, en_dict_ac


def main():
    # 初始化并获取英文名 & 英文词典自动机实例
    en_name_ac, en_dict_ac = initialize_automata(
        str(project_path("data", "english_name_automaton.pkl")),
        str(project_path("data", "english_dict_automaton.pkl")),
    )

    # 3. 将自动机注入 TrainingManager
    manager = TrainingManager(
        english_name_automaton=en_name_ac,
        english_dict_automaton=en_dict_ac
    )

    # 4. 执行训练与概率计算
    sp_probs, sft_probs = manager.train_all_batches()

    # 5. 打印结果
    print("=== Final SP Probabilities (Top 10) ===")
    for sp, prob in sorted(sp_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(sp, f"{prob:.4f}")

    print("\n=== Final SFT->SF Probabilities (Sample) ===")
    show_raw = os.environ.get("SE_PCFG_SHOW_RAW_SFT", "").strip().lower() in {"1", "true", "yes", "y", "on"}

    def _shape(s: str) -> str:
        def cls(ch: str) -> str:
            if ch.isdigit():
                return "D"
            if "a" <= ch <= "z":
                return "l"
            if "A" <= ch <= "Z":
                return "U"
            return "S"

        if not s:
            return "EMPTY"
        out = []
        prev = None
        run = 0
        for ch in s:
            c = cls(ch)
            if prev is None:
                prev = c
                run = 1
            elif c == prev:
                run += 1
            else:
                out.append(f"{prev}{run}")
                prev = c
                run = 1
        out.append(f"{prev}{run}")
        return "".join(out)

    def _mask_sf(sf: str) -> str:
        h = hashlib.sha256(sf.encode("utf-8")).hexdigest()[:10]
        return f"<{_shape(sf)}>#sha256:{h}"

    for sft in list(sft_probs)[:5]:
        print(f"[{sft}] top 5:")
        for sf, p in sorted(sft_probs[sft].items(), key=lambda x: x[1], reverse=True)[:5]:
            shown = sf if show_raw else _mask_sf(sf)
            print(f"  {shown}: {p:.4f}")
        print()

    export_report = os.environ.get("SE_PCFG_EXPORT_REPORT", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if export_report:
        out_dir = project_path("exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{TRAIN_NAME}__summary.txt"

        lines = []
        lines.append(f"train_name={TRAIN_NAME}\n")
        lines.append("top_sp_templates=100\n")
        for sp, prob in sorted(sp_probs.items(), key=lambda x: x[1], reverse=True)[:100]:
            lines.append(f"SP\t{sp}\t{prob:.8e}\n")

        # For each SFT label: show top-20 surface forms as masked tokens (shape+hash)
        lines.append("\nmasked_sft_top20_per_label\n")
        for sft in sorted(sft_probs.keys()):
            lines.append(f"SFT\t{sft}\n")
            for sf, p in sorted(sft_probs[sft].items(), key=lambda x: x[1], reverse=True)[:20]:
                lines.append(f"  SF\t{_mask_sf(sf)}\t{p:.8e}\n")
        out_path.write_text("".join(lines), encoding="utf-8")
        print(f"[REPORT] wrote {out_path}")


if __name__ == "__main__":
    main()
