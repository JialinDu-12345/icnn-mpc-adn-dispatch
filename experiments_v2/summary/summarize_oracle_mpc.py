from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              
from experiments_v2.summary.summarize_main_model_based import (              
    read_rows,
    summarize,
    write_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 33-bus Oracle perfect-forecast MPC baselines.")
    parser.add_argument("--input", default="formal/step7_33_oracle_mpc_raw.csv")
    parser.add_argument("--output", default="formal/step7_33_oracle_mpc_summary.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows = read_rows(RESULT_DIR / args.input)
    summary_rows = summarize(rows)
    output_path = RESULT_DIR / args.output
    write_rows(output_path, summary_rows)
    for row in summary_rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

