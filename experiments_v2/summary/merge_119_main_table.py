from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


DEFAULT_INPUTS = [
    "formal/step10_119_scalability_inline20_summary_with_gap.csv",
]

DEFAULT_REQUIRED_METHODS = [
    "Perfect-Global",
    "MPC-3",
    "SAC",
    "SAC-ICNN",
    "Proposed-H3",
]

POLICY_METHODS = {"SAC", "SAC-ICNN"}

METHOD_ORDER = {
    "Perfect-Global": -10,
    "MPC-3": 10,
    "SAC": 20,
    "SAC-ICNN": 21,
    "StdNN-H3-MIP": 25,
    "Proposed-H3": 30,
}

OUTPUT_COLUMNS = [
    "system",
    "actual_case",
    "forecast_case",
    "display_forecast_case",
    "method",
    "horizon",
    "gamma_Q",
    "n_runs",
    "completed_count",
    "completed_rate",
    "certified_global_count",
    "certified_global_rate",
    "total_fail_count",
    "total_timeout_count",
    "icnn_relax_on_infeasible",
    "icnn_relax_enabled_count",
    "mean_icnn_relax_count",
    "max_icnn_relax_count",
    "total_icnn_relax_count",
    "icnn_exogenous_clip_enabled",
    "icnn_clip_enabled_count",
    "mean_total_icnn_clip_count",
    "max_total_icnn_clip_count",
    "sum_total_icnn_clip_count",
    "max_icnn_clip_violation",
    "mean_executed_steps",
    "n_completed",
    "scenario_selection_file",
    "scenario_selection_rule",
    "cost_accounting",
    "mean_cost",
    "std_cost",
    "ci95_cost",
    "gap_to_perfect_percent",
    "n_gap_matched",
    "mean_gap_to_perfect_percent",
    "std_gap_to_perfect_percent",
    "ci95_gap_to_perfect_percent",
    "mean_economic_generation_cost",
    "std_economic_generation_cost",
    "ci95_economic_generation_cost",
    "mean_penalized_cost",
    "std_penalized_cost",
    "ci95_penalized_cost",
    "mean_balance_penalty_cost",
    "std_balance_penalty_cost",
    "mean_constraint_penalty",
    "std_constraint_penalty",
    "mean_solve_time_s",
    "std_solve_time_s",
    "ci95_solve_time_s",
    "mean_avg_step_time_s",
    "std_avg_step_time_s",
    "mean_max_step_time_s",
    "mean_num_vars",
    "mean_num_bin_vars",
    "mean_std_nn_binary_count",
    "mean_num_constrs",
    "mean_num_qconstrs",
    "mean_num_genconstrs",
    "mean_model_runtime_s",
    "mean_mip_gap",
    "max_mip_gap",
    "mean_model_status",
    "mean_gurobi_status",
    "min_model_sol_count",
    "mean_cost_constraint",
    "policy_training_source",
    "policy_retrained_under_v2_protocol",
    "policy_eval_protocol",
    "source_summary",
]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display_forecast_case(row: Dict[str, object]) -> str:
    method = str(row.get("method", ""))
    forecast_case = str(row.get("forecast_case", ""))
    if method == "Perfect-Global":
        return "full actual trajectory"
    if method in POLICY_METHODS or forecast_case in {"", "not_applicable"}:
        return "-"
    return forecast_case


def validate_required_methods(rows: List[Dict[str, object]], required_methods: List[str]) -> None:
    methods = {str(row.get("method", "")) for row in rows}
    missing = [method for method in required_methods if method not in methods]
    if missing:
        raise RuntimeError(f"119 main table is missing methods: {missing}")


def row_sort_key(row: Dict[str, object]) -> tuple:
    method = str(row.get("method", ""))
    try:
        horizon = float(row.get("horizon") or 0.0)
    except ValueError:
        horizon = 0.0
    return (
        str(row.get("system", "")),
        str(row.get("actual_case", "")),
        METHOD_ORDER.get(method, 999),
        horizon,
        method,
    )


def add_gap_to_perfect(rows: List[Dict[str, object]]) -> None:
    perfect_by_case: Dict[tuple, float] = {}
    for row in rows:
        if str(row.get("method", "")) != "Perfect-Global":
            continue
        cost = parse_float(row.get("mean_cost"))
        if math.isfinite(cost) and cost != 0:
            key = (str(row.get("system", "")), str(row.get("actual_case", "")))
            perfect_by_case[key] = cost

    for row in rows:
        method = str(row.get("method", ""))
        matched_gap = parse_float(row.get("mean_gap_to_perfect_percent"))
        if math.isfinite(matched_gap):
            row["gap_to_perfect_percent"] = matched_gap
            continue

        key = (str(row.get("system", "")), str(row.get("actual_case", "")))
        perfect_cost = perfect_by_case.get(key, math.nan)
        cost = parse_float(row.get("mean_cost"))
        if method == "Perfect-Global" and math.isfinite(cost):
            row["gap_to_perfect_percent"] = 0.0
        elif math.isfinite(cost) and math.isfinite(perfect_cost) and perfect_cost != 0:
            row["gap_to_perfect_percent"] = 100.0 * (cost - perfect_cost) / perfect_cost
        else:
            row["gap_to_perfect_percent"] = ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge 119-bus summaries into the scalability table.")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    parser.add_argument("--output", default="formal/step10_119_main_table_inline20.csv")
    parser.add_argument("--strict", action="store_true", help="Fail if any input summary is missing.")
    parser.add_argument(
        "--required-methods",
        default=",".join(DEFAULT_REQUIRED_METHODS),
        help="Comma-separated methods required when --strict is used. Use an empty value to disable.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows: List[Dict[str, object]] = []
    for input_name in parse_csv_list(args.inputs):
        path = RESULT_DIR / input_name
        if not path.exists():
            message = f"Missing input summary: {path}"
            if args.strict:
                raise FileNotFoundError(message)
            print("[Skipped]", message)
            continue
        for row in read_rows(path):
            normalized = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            normalized["source_summary"] = input_name
            normalized["display_forecast_case"] = display_forecast_case(normalized)
            rows.append(normalized)

    rows = sorted(rows, key=row_sort_key)
    add_gap_to_perfect(rows)
    if args.strict:
        required = parse_csv_list(args.required_methods)
        if required:
            validate_required_methods(rows, required)

    output_path = RESULT_DIR / args.output
    write_rows(output_path, rows)
    for row in rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
