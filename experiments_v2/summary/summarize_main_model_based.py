from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


METHOD_ORDER = {
    "Oracle-H3": 0,
    "Oracle-H7": 1,
    "Oracle-H12": 2,
    "MPC-3": 10,
    "MPC-5": 11,
    "MPC-7": 12,
    "StdNN-H3-MIP": 13,
    "SAC": 20,
    "SAC-ICNN": 21,
    "SACLag": 22,
    "Proposed-H3": 30,
}

GROUP_KEYS = ["system", "actual_case", "forecast_case", "method", "horizon", "gamma_Q"]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    if not clean:
        return math.nan
    return mean(clean)


def std_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    if len(clean) < 2:
        return math.nan
    return stdev(clean)


def ci95(values: Iterable[float]) -> float:
    clean = finite(values)
    if len(clean) < 2:
        return math.nan
    return 1.96 * stdev(clean) / math.sqrt(len(clean))


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


def row_completed(row: Dict[str, str]) -> bool:
    if row.get("completed") not in (None, ""):
        return parse_bool(row.get("completed"))
    return (
        int(float(row.get("executed_steps") or 0)) == 48
        and int(float(row.get("solver_fail_count") or 0)) == 0
    )


def group_key(row: Dict[str, str]) -> tuple:
    return tuple(row.get(key, "") for key in GROUP_KEYS)


def summary_sort_key(key: tuple) -> tuple:
    key_dict = dict(zip(GROUP_KEYS, key))
    return (
        key_dict.get("system", ""),
        key_dict.get("actual_case", ""),
        key_dict.get("forecast_case", ""),
        METHOD_ORDER.get(key_dict.get("method", ""), 999),
        parse_float(key_dict.get("horizon"), 0.0),
        parse_float(key_dict.get("gamma_Q"), -1.0),
    )


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=summary_sort_key):
        group = grouped[key]
        key_dict = dict(zip(GROUP_KEYS, key))
        completed_flags = [row_completed(row) for row in group]
        completed_rows = [row for row, done in zip(group, completed_flags) if done]

        costs = [parse_float(row.get("cost")) for row in completed_rows]
        solve_times = [parse_float(row.get("solve_time_s")) for row in completed_rows]
        avg_step_times = [parse_float(row.get("avg_step_time_s")) for row in completed_rows]
        max_step_times = [parse_float(row.get("max_step_time_s")) for row in completed_rows]
        executed_steps = [parse_float(row.get("executed_steps")) for row in group]
        fail_counts = [int(float(row.get("solver_fail_count") or 0)) for row in group]

        n_runs = len(group)
        completed_count = sum(1 for done in completed_flags if done)
        summary_rows.append(
            {
                **key_dict,
                "n_runs": n_runs,
                "completed_count": completed_count,
                "completed_rate": completed_count / max(n_runs, 1),
                "total_fail_count": sum(fail_counts),
                "mean_executed_steps": mean_or_nan(executed_steps),
                "n_completed": len(completed_rows),
                "mean_cost": mean_or_nan(costs),
                "std_cost": std_or_nan(costs),
                "ci95_cost": ci95(costs),
                "mean_solve_time_s": mean_or_nan(solve_times),
                "std_solve_time_s": std_or_nan(solve_times),
                "ci95_solve_time_s": ci95(solve_times),
                "mean_avg_step_time_s": mean_or_nan(avg_step_times),
                "std_avg_step_time_s": std_or_nan(avg_step_times),
                "mean_max_step_time_s": mean_or_nan(max_step_times),
                "std_max_step_time_s": std_or_nan(max_step_times),
            }
        )
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 33-bus model-based main experiment.")
    parser.add_argument("--input", default="formal/step4_33_main_model_based_raw.csv")
    parser.add_argument("--output", default="formal/step4_33_main_model_based_summary.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    raw_path = RESULT_DIR / args.input
    rows = read_rows(raw_path)
    summary_rows = summarize(rows)

    output_path = RESULT_DIR / args.output
    write_rows(output_path, summary_rows)
    for row in summary_rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
