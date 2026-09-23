from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No summary rows to write.")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def std_or_nan(values: List[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return math.nan
    return stdev(finite)


def mean_or_nan(values: List[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return mean(finite)


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[float, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        gamma_q = parse_float(row.get("gamma_Q"))
        if not math.isfinite(gamma_q):
            continue
        grouped[gamma_q].append(row)

    summary_rows: List[Dict[str, object]] = []
    for gamma_q in sorted(grouped):
        group = grouped[gamma_q]
        completed_values = []
        costs = []
        times = []
        avg_step_times = []
        voltage_violations = []
        completed_costs = []
        completed_times = []
        completed_avg_step_times = []
        completed_voltage_violations = []
        fail_counts = []
        validation_models = []
        for row in group:
            completed = row.get("completed")
            if completed in (None, ""):
                completed_flag = (
                    int(float(row.get("executed_steps") or 0)) == 48
                    and int(float(row.get("solver_fail_count") or 0)) == 0
                )
            else:
                completed_flag = parse_bool(completed)
            completed_values.append(completed_flag)
            cost_value = parse_float(row.get("cost"))
            time_value = parse_float(row.get("solve_time_s"))
            avg_step_time_value = parse_float(row.get("avg_step_time_s"))
            voltage_violation_value = parse_float(row.get("voltage_violation"), 0.0)
            costs.append(cost_value)
            times.append(time_value)
            avg_step_times.append(avg_step_time_value)
            voltage_violations.append(voltage_violation_value)
            if completed_flag:
                completed_costs.append(cost_value)
                completed_times.append(time_value)
                completed_avg_step_times.append(avg_step_time_value)
                completed_voltage_violations.append(voltage_violation_value)
            fail_counts.append(int(float(row.get("solver_fail_count") or 0)))
            validation_model = str(row.get("gamma_validation_model") or row.get("method") or "")
            if validation_model and validation_model not in validation_models:
                validation_models.append(validation_model)

        n_scenarios = len(group)
        completed_count = sum(1 for value in completed_values if value)
        summary_rows.append(
            {
                "gamma_Q": gamma_q,
                "n_scenarios": n_scenarios,
                "completed_count": completed_count,
                "completed_rate": completed_count / max(n_scenarios, 1),
                "mean_cost": mean_or_nan(costs),
                "std_cost": std_or_nan(costs),
                "mean_completed_cost": mean_or_nan(completed_costs),
                "std_completed_cost": std_or_nan(completed_costs),
                "mean_completed_time_s": mean_or_nan(completed_times),
                "mean_time_s": mean_or_nan(times),
                "std_time_s": std_or_nan(times),
                "mean_completed_avg_step_time_s": mean_or_nan(completed_avg_step_times),
                "mean_avg_step_time_s": mean_or_nan(avg_step_times),
                "mean_completed_cum_voltage_violation": mean_or_nan(
                    completed_voltage_violations
                ),
                "mean_cum_voltage_violation": mean_or_nan(voltage_violations),
                "total_fail_count": sum(fail_counts),
                "gamma_validation_model": ",".join(validation_models),
            }
        )
    return summary_rows


def select_gamma(
    summary_rows: List[Dict[str, object]],
    raw_rows: List[Dict[str, str]],
    allow_incomplete_selection: bool = False,
) -> Optional[float]:
    feasible_rows = [
        row
        for row in summary_rows
        if float(row["completed_rate"]) == 1.0 and int(row["total_fail_count"]) == 0
    ]
    if feasible_rows:
        feasible_rows.sort(
            key=lambda row: (
                float(row["mean_completed_cost"]),
                float(row["mean_time_s"]),
            )
        )
        return float(feasible_rows[0]["gamma_Q"])

    if not allow_incomplete_selection:
        return None

    ranked = sorted(
        summary_rows,
        key=lambda row: (
            -float(row["completed_rate"]),
            int(row["total_fail_count"]),
            float(row["mean_completed_cost"])
            if math.isfinite(float(row["mean_completed_cost"]))
            else math.inf,
            float(row["mean_time_s"]) if math.isfinite(float(row["mean_time_s"])) else math.inf,
        ),
    )
    if not ranked:
        return None
    return float(ranked[0]["gamma_Q"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize and select 33-bus gamma_Q.")
    parser.add_argument("--input", default="formal/step3_33_gamma_validation_raw.csv")
    parser.add_argument(
        "--summary-output",
        "--output",
        dest="summary_output",
        default="formal/step3_33_gamma_validation_summary.csv",
    )
    parser.add_argument(
        "--selected-output",
        "--selected-json",
        dest="selected_output",
        default="selected_gamma_33.json",
    )
    parser.add_argument("--allow-incomplete-selection", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    raw_path = RESULT_DIR / args.input
    rows = read_rows(raw_path)
    summary_rows = summarize(rows)
    selected_gamma = select_gamma(
        summary_rows,
        rows,
        allow_incomplete_selection=args.allow_incomplete_selection,
    )

    summary_path = RESULT_DIR / args.summary_output
    write_rows(summary_path, summary_rows)

    selected_path = RESULT_DIR / args.selected_output
    selection_rule = (
        "Choose the gamma_Q with the lowest validation mean completed cost among candidates "
        "with 100% completed validation scenarios and zero solver failures. If no "
        "candidate satisfies this, return None by default. With --allow-incomplete-selection, "
        "rank by completed_rate, fail_count, mean_completed_cost, and mean_time_s for diagnostic use."
    )
    with selected_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "system": "33",
                "selected_gamma_Q": selected_gamma,
                "allow_incomplete_selection": args.allow_incomplete_selection,
                "selection_rule": selection_rule,
                "raw_path": args.input,
                "summary_path": args.summary_output,
            },
            f,
            indent=2,
        )

    for row in summary_rows:
        print(row)
    print("Selected gamma_Q:", selected_gamma)
    print("Saved:", summary_path)
    print("Saved:", selected_path)


if __name__ == "__main__":
    main()
