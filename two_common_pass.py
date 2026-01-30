#!/usr/bin/env python3
"""filter_passwords.py

Read Chinese-common-password-list.txt and write two filtered files:
    1. passwords longer than 8 chars containing at least one letter and one digit
    2. passwords longer than 8 chars containing at least one lowercase letter, one uppercase letter, and one digit

Usage:
    python filter_passwords.py Chinese-common-password-list.txt
    # optional custom output paths
    python filter_passwords.py Chinese-common-password-list.txt --out1 alphanum.txt --out2 upperlowernum.txt
"""

from __future__ import annotations
import re
from pathlib import Path
import argparse

# Pre‑compiled regex patterns
PAT_LETTER_DIGIT = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{9,}$")
PAT_UPPER_LOWER_DIGIT = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{9,}$")


def filter_passwords(input_path: str | Path, out1_path: str | Path, out2_path: str | Path) -> None:
    """Filter passwords into two files according to the given rules."""
    input_path = Path(input_path)
    out1_path = Path(out1_path)
    out2_path = Path(out2_path)

    # Ensure output directories exist
    out1_path.parent.mkdir(parents=True, exist_ok=True)
    out2_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
            out1_path.open("w", encoding="utf-8") as fout1, \
            out2_path.open("w", encoding="utf-8") as fout2:
        for line in fin:
            pwd = line.rstrip("\n")
            # Skip short candidates early for speed
            if len(pwd) < 9:
                continue

            # Check Rule 1: length>8 & letter+digit
            if PAT_LETTER_DIGIT.match(pwd):
                fout1.write(pwd + "\n")

            # Check Rule 2: length>8 & upper+lower+digit
            if PAT_UPPER_LOWER_DIGIT.match(pwd):
                fout2.write(pwd + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Chinese common password list into two files.")
    parser.add_argument("input_file", help="Path to Chinese-common-password-list.txt")
    parser.add_argument("--out1", default="filtered_alphanum.txt", help="Output file for passwords with letters and numbers (length>8)")
    parser.add_argument("--out2", default="filtered_upperlowernum.txt", help="Output file for passwords with upper & lower case letters and numbers (length>8)")

    args = parser.parse_args()
    filter_passwords(args.input_file, args.out1, args.out2)

    print(f"Done! Results saved to '{args.out1}' and '{args.out2}'.")
