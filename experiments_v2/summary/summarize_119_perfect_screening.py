from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              
from experiments_v2.global_optimum.select_119_perfect_feasible import (              
    is_selected_perfect_row,
    parse_bool,
    parse_float,
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return mean(clean) if clean else math.nan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 119-bus Perfect screening.")
    parser.add_argument("--input", default="formal/step10_119_inline_screen_log.csv")
    parser.add_argument("--output", default="formal/step10_119_inline_screening_summary.csv")
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = RESULT_DIR / args.input
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw_rows = read_rows(input_path)
    if not raw_rows:
        raise RuntimeError(f"No screening rows found in {input_path}.")
    perfect_rows = [
        row
        for row in raw_rows
        if str(row.get("method", "")).strip() in {"", "Perfect-Global"}
    ]
    rows = perfect_rows or raw_rows

    completed_rows = [row for row in rows if parse_bool(row.get("completed"))]
    certified_rows = [row for row in rows if parse_bool(row.get("certified_global"))]
    selected_rows = [
        row
        for row in rows
        if is_selected_perfect_row(row, mip_gap_target=float(args.mip_gap))
    ]

    summary = {
        "system": "119",
        "input": args.input,
        "mip_gap_target": args.mip_gap,
        "n_scanned": len(rows),
        "n_completed": len(completed_rows),
        "n_certified": len(certified_rows),
        "n_selected_by_rule": len(selected_rows),
        "n_failed_or_incomplete": len(rows) - len(completed_rows),
        "completed_rate": len(completed_rows) / max(len(rows), 1),
        "certified_rate": len(certified_rows) / max(len(rows), 1),
        "selected_rate": len(selected_rows) / max(len(rows), 1),
        "mean_cost_completed": mean_or_nan(parse_float(row.get("cost")) for row in completed_rows),
        "mean_time_completed_s": mean_or_nan(
            parse_float(row.get("solve_time_s")) for row in completed_rows
        ),
        "mean_mip_gap_certified": mean_or_nan(
            parse_float(row.get("mean_mip_gap")) for row in certified_rows
        ),
        "selected_scenarios": ",".join(str(row.get("scenario_id")) for row in selected_rows),
    }

    output_path = RESULT_DIR / args.output
    write_rows(output_path, [summary])
    print(summary)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
