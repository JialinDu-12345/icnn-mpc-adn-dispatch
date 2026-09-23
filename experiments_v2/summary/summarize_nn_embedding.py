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
    "MPC-3": 0,
    "StdNN-H3-MIP": 1,
    "Proposed-H3": 2,
}

GROUP_KEYS = ["system", "actual_case", "forecast_case", "method", "horizon", "gamma_Q"]
MODEL_SIZE_FIELDS = [
    "mean_num_vars",
    "mean_num_bin_vars",
    "mean_num_constrs",
    "mean_num_qconstrs",
    "mean_num_genconstrs",
    "mean_model_runtime_s",
    "mean_mip_gap",
    "max_mip_gap",
    "mean_std_nn_binary_count",
    "mean_model_status",
    "min_model_sol_count",
    "num_optimal_steps",
    "num_timelimit_steps",
]


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
    return mean(clean) if clean else math.nan


def max_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return max(clean) if clean else math.nan


def min_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return min(clean) if clean else math.nan


def std_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return stdev(clean) if len(clean) >= 2 else math.nan


def ci95(values: Iterable[float]) -> float:
    clean = finite(values)
    return 1.96 * stdev(clean) / math.sqrt(len(clean)) if len(clean) >= 2 else math.nan


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


def summarize(rows: List[Dict[str, str]], allow_partial: bool = False) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=summary_sort_key):
        group = grouped[key]
        key_dict = dict(zip(GROUP_KEYS, key))
        completed_flags = [row_completed(row) for row in group]
        completed_rows = [row for row, completed in zip(group, completed_flags) if completed]
        perf_rows = group if allow_partial else completed_rows

        costs = [parse_float(row.get("cost")) for row in perf_rows]
        solve_times = [parse_float(row.get("solve_time_s")) for row in perf_rows]
        avg_step_times = [parse_float(row.get("avg_step_time_s")) for row in perf_rows]
        max_step_times = [parse_float(row.get("max_step_time_s")) for row in perf_rows]
        executed_steps = [parse_float(row.get("executed_steps")) for row in group]
        fail_counts = [int(float(row.get("solver_fail_count") or 0)) for row in group]
        timeout_counts = [int(float(row.get("timeout_count") or 0)) for row in group]

        row_out: Dict[str, object] = {
            **key_dict,
            "n_runs": len(group),
            "completed_count": sum(1 for completed in completed_flags if completed),
            "completed_rate": sum(1 for completed in completed_flags if completed) / max(len(group), 1),
            "total_fail_count": sum(fail_counts),
            "total_timeout_count": sum(timeout_counts),
            "mean_executed_steps": mean_or_nan(executed_steps),
            "n_completed": len(completed_rows),
            "n_perf_rows": len(perf_rows),
            "allow_partial": allow_partial,
            "mean_cost": mean_or_nan(costs),
            "std_cost": std_or_nan(costs),
            "ci95_cost": ci95(costs),
            "mean_solve_time_s": mean_or_nan(solve_times),
            "std_solve_time_s": std_or_nan(solve_times),
            "ci95_solve_time_s": ci95(solve_times),
            "mean_avg_step_time_s": mean_or_nan(avg_step_times),
            "mean_max_step_time_s": mean_or_nan(max_step_times),
        }

        for field in MODEL_SIZE_FIELDS:
            values = [parse_float(row.get(field)) for row in group]
            if field == "max_mip_gap":
                row_out[field] = max_or_nan(values)
            elif field == "min_model_sol_count":
                row_out[field] = min_or_nan(values)
            elif field in {"num_optimal_steps", "num_timelimit_steps"}:
                row_out[field] = sum(finite(values))
            else:
                row_out[field] = mean_or_nan(values)

        summary_rows.append(row_out)

    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize standard NN-MPC embedding comparison.")
                                                                                    
    parser.add_argument("--input", default="formal/step6_33_stdnn_only_raw.csv")
                                                                                         
    parser.add_argument("--output", default="formal/step6_33_stdnn_only_summary.csv")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows = read_rows(RESULT_DIR / args.input)
    summary_rows = summarize(rows, allow_partial=args.allow_partial)
    output_path = RESULT_DIR / args.output
    write_rows(output_path, summary_rows)

    for row in summary_rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
