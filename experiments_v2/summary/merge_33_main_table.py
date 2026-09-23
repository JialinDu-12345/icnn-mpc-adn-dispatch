from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


DEFAULT_INPUTS = [
    "formal/step7b_33_perfect_global_summary.csv",
    "formal/step7_33_oracle_mpc_summary.csv",
    "formal/step4_33_main_model_based_summary.csv",
    "formal/step6_33_stdnn_only_summary.csv",
    "formal/step6_33_nn_embedding_summary.csv",
    "formal/step7_33_policy_baselines_summary.csv",
    "formal/step7_33_saclag_stochastic_seed0_test_summary.csv",
]

OPTIONAL_INPUTS = {"formal/step6_33_stdnn_only_summary.csv"}

DEFAULT_REQUIRED_METHODS = [
    "Perfect-Global",
    "Oracle-H3",
    "Oracle-H7",
    "MPC-3",
    "MPC-5",
    "MPC-7",
    "StdNN-H3-MIP",
    "SAC",
    "SAC-ICNN",
    "SACLag",
    "Proposed-H3",
]

POLICY_METHODS = {"SAC", "SAC-ICNN", "SACLag"}

METHOD_ORDER = {
    "Perfect-Global": -10,
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
    "mean_executed_steps",
    "n_completed",
    "policy_eval_mode",
    "policy_eval_seed",
    "policy_stochastic_eval",
    "cost_accounting",
    "voltage_violation_metric",
    "mean_cost",
    "std_cost",
    "ci95_cost",
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
    "mean_max_step_time_s",
    "mean_num_vars",
    "mean_num_bin_vars",
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
    "mean_voltage_violation",
    "std_voltage_violation",
    "ci95_voltage_violation",
    "legacy_mean_cost_constraint",
    "source_summary",
]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        raise RuntimeError(f"Final main table is missing methods: {missing}")


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


def duplicate_key(row: Dict[str, object]) -> tuple:
    return (
        str(row.get("system", "")),
        str(row.get("actual_case", "")),
        str(row.get("forecast_case", "")),
        str(row.get("method", "")),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge 33-bus summaries into the main paper table.")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    parser.add_argument("--output", default="formal/step8_33_main_table.csv")
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
            if args.strict and input_name not in OPTIONAL_INPUTS:
                raise FileNotFoundError(message)
            print("[Skipped]", message)
            continue
        if "step7_33_saclag_stochastic_seed0_test_summary" in input_name:
            source_rank = -1
        elif "step6_33_stdnn_only_summary" in input_name:
            source_rank = 1
        elif "step6_33_nn_embedding_summary" in input_name:
            source_rank = 2
        else:
            source_rank = 0
        for row in read_rows(path):
            if source_rank == -1 and str(row.get("method", "")) != "SACLag":
                continue
            if source_rank in {1, 2} and str(row.get("method", "")) != "StdNN-H3-MIP":
                continue
            normalized = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            normalized["source_summary"] = input_name
            normalized["display_forecast_case"] = display_forecast_case(normalized)
            normalized["_source_rank"] = source_rank
            rows.append(normalized)

    rows = sorted(rows, key=lambda row: (duplicate_key(row), int(row.get("_source_rank", 0))))
    deduped: Dict[tuple, Dict[str, object]] = {}
    for row in rows:
        deduped.setdefault(duplicate_key(row), row)
    rows = sorted(deduped.values(), key=row_sort_key)
    if args.strict:
        validate_required_methods(rows, parse_csv_list(args.required_methods))

    output_path = RESULT_DIR / args.output
    write_rows(output_path, rows)
    for row in rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
