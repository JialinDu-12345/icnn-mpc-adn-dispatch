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
    "Perfect-Global": -10,
    "MPC-3": 10,
    "SAC": 20,
    "SAC-ICNN": 21,
    "StdNN-H3-MIP": 25,
    "Proposed-H3": 30,
}

GROUP_KEYS = ["system", "actual_case", "forecast_case", "method", "horizon", "gamma_Q"]


def parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


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


def std_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return stdev(clean) if len(clean) >= 2 else math.nan


def ci95(values: Iterable[float]) -> float:
    clean = finite(values)
    return 1.96 * stdev(clean) / math.sqrt(len(clean)) if len(clean) >= 2 else math.nan


def max_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return max(clean) if clean else math.nan


def min_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return min(clean) if clean else math.nan


def first_nonempty(rows: Iterable[Dict[str, str]], key: str, default: str = "") -> str:
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return default


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
        METHOD_ORDER.get(key_dict.get("method", ""), 999),
        parse_float(key_dict.get("horizon"), 0.0),
        parse_float(key_dict.get("gamma_Q"), -1.0),
    )


def summarize(rows: List[Dict[str, str]], allow_partial: bool = False) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("system") == "119":
            grouped[group_key(row)].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=summary_sort_key):
        group = grouped[key]
        key_dict = dict(zip(GROUP_KEYS, key))
        completed_flags = [row_completed(row) for row in group]
        completed_rows = [row for row, done in zip(group, completed_flags) if done]
        perf_rows = group if allow_partial else completed_rows

        costs = [parse_float(row.get("cost")) for row in perf_rows]
        solve_times = [parse_float(row.get("solve_time_s")) for row in perf_rows]
        avg_step_times = [parse_float(row.get("avg_step_time_s")) for row in perf_rows]
        max_step_times = [parse_float(row.get("max_step_time_s")) for row in perf_rows]
        executed_steps = [parse_float(row.get("executed_steps")) for row in group]
        fail_counts = [int(float(row.get("solver_fail_count") or 0)) for row in group]
        timeout_counts = [int(float(row.get("timeout_count") or 0)) for row in group]
        certified_flags = [parse_bool(row.get("certified_global")) for row in group]
        icnn_relax_flags = [parse_bool(row.get("icnn_relax_on_infeasible")) for row in group]
        icnn_relax_counts = [
            parse_float(row.get("icnn_relax_count"), 0.0) for row in group
        ]
        icnn_clip_enabled_flags = [
            parse_bool(row.get("icnn_exogenous_clip_enabled")) for row in group
        ]
        icnn_clip_counts = [
            parse_float(row.get("total_icnn_clip_count"), 0.0) for row in group
        ]
        icnn_clip_violations = [
            parse_float(row.get("max_icnn_clip_violation"), 0.0) for row in group
        ]

        economic_costs = [parse_float(row.get("economic_generation_cost")) for row in perf_rows]
        penalized_costs = [parse_float(row.get("penalized_cost")) for row in perf_rows]
        balance_penalty_costs = [
            parse_float(row.get("balance_penalty_cost")) for row in perf_rows
        ]
        constraint_penalties = [parse_float(row.get("constraint_penalty")) for row in perf_rows]
        cost_constraints = [parse_float(row.get("mean_cost_constraint")) for row in perf_rows]

        num_vars = [parse_float(row.get("mean_num_vars")) for row in perf_rows]
        num_bin_vars = [parse_float(row.get("mean_num_bin_vars")) for row in perf_rows]
        std_nn_binary_counts = [
            parse_float(row.get("mean_std_nn_binary_count")) for row in perf_rows
        ]
        num_constrs = [parse_float(row.get("mean_num_constrs")) for row in perf_rows]
        num_qconstrs = [parse_float(row.get("mean_num_qconstrs")) for row in perf_rows]
        num_genconstrs = [parse_float(row.get("mean_num_genconstrs")) for row in perf_rows]
        model_runtimes = [parse_float(row.get("mean_model_runtime_s")) for row in perf_rows]
        mip_gaps = [parse_float(row.get("mean_mip_gap")) for row in perf_rows]
        model_statuses = [parse_float(row.get("mean_model_status")) for row in perf_rows]
        gurobi_statuses = [parse_float(row.get("gurobi_status")) for row in perf_rows]
        sol_counts = [parse_float(row.get("min_model_sol_count")) for row in perf_rows]

        n_runs = len(group)
        completed_count = sum(1 for done in completed_flags if done)
        certified_count = sum(1 for done in certified_flags if done)
        summary_rows.append(
            {
                **key_dict,
                "n_runs": n_runs,
                "completed_count": completed_count,
                "completed_rate": completed_count / max(n_runs, 1),
                "certified_global_count": certified_count,
                "certified_global_rate": certified_count / max(n_runs, 1),
                "total_fail_count": sum(fail_counts),
                "total_timeout_count": sum(timeout_counts),
                "icnn_relax_on_infeasible": any(icnn_relax_flags),
                "icnn_relax_enabled_count": sum(1 for flag in icnn_relax_flags if flag),
                "mean_icnn_relax_count": mean_or_nan(icnn_relax_counts),
                "max_icnn_relax_count": max_or_nan(icnn_relax_counts),
                "total_icnn_relax_count": sum(
                    value for value in icnn_relax_counts if math.isfinite(value)
                ),
                "icnn_exogenous_clip_enabled": any(icnn_clip_enabled_flags),
                "icnn_clip_enabled_count": sum(
                    1 for flag in icnn_clip_enabled_flags if flag
                ),
                "mean_total_icnn_clip_count": mean_or_nan(icnn_clip_counts),
                "max_total_icnn_clip_count": max_or_nan(icnn_clip_counts),
                "sum_total_icnn_clip_count": sum(
                    value for value in icnn_clip_counts if math.isfinite(value)
                ),
                "max_icnn_clip_violation": max_or_nan(icnn_clip_violations),
                "mean_executed_steps": mean_or_nan(executed_steps),
                "n_completed": len(completed_rows),
                "n_perf_rows": len(perf_rows),
                "allow_partial": allow_partial,
                "scenario_selection_file": first_nonempty(group, "scenario_selection_file"),
                "scenario_selection_rule": first_nonempty(group, "scenario_selection_rule"),
                "cost_accounting": first_nonempty(perf_rows, "cost_accounting", "legacy_or_unknown"),
                "policy_training_source": first_nonempty(perf_rows, "policy_training_source"),
                "policy_retrained_under_v2_protocol": first_nonempty(
                    perf_rows, "policy_retrained_under_v2_protocol"
                ),
                "policy_eval_protocol": first_nonempty(perf_rows, "policy_eval_protocol"),
                "mean_cost": mean_or_nan(costs),
                "std_cost": std_or_nan(costs),
                "ci95_cost": ci95(costs),
                "mean_economic_generation_cost": mean_or_nan(economic_costs),
                "std_economic_generation_cost": std_or_nan(economic_costs),
                "ci95_economic_generation_cost": ci95(economic_costs),
                "mean_penalized_cost": mean_or_nan(penalized_costs),
                "std_penalized_cost": std_or_nan(penalized_costs),
                "ci95_penalized_cost": ci95(penalized_costs),
                "mean_balance_penalty_cost": mean_or_nan(balance_penalty_costs),
                "std_balance_penalty_cost": std_or_nan(balance_penalty_costs),
                "mean_constraint_penalty": mean_or_nan(constraint_penalties),
                "std_constraint_penalty": std_or_nan(constraint_penalties),
                "mean_solve_time_s": mean_or_nan(solve_times),
                "std_solve_time_s": std_or_nan(solve_times),
                "ci95_solve_time_s": ci95(solve_times),
                "mean_avg_step_time_s": mean_or_nan(avg_step_times),
                "std_avg_step_time_s": std_or_nan(avg_step_times),
                "mean_max_step_time_s": mean_or_nan(max_step_times),
                "std_max_step_time_s": std_or_nan(max_step_times),
                "mean_num_vars": mean_or_nan(num_vars),
                "mean_num_bin_vars": mean_or_nan(num_bin_vars),
                "mean_std_nn_binary_count": mean_or_nan(std_nn_binary_counts),
                "mean_num_constrs": mean_or_nan(num_constrs),
                "mean_num_qconstrs": mean_or_nan(num_qconstrs),
                "mean_num_genconstrs": mean_or_nan(num_genconstrs),
                "mean_model_runtime_s": mean_or_nan(model_runtimes),
                "mean_mip_gap": mean_or_nan(mip_gaps),
                "max_mip_gap": max_or_nan(mip_gaps),
                "mean_model_status": mean_or_nan(model_statuses),
                "mean_gurobi_status": mean_or_nan(gurobi_statuses),
                "min_model_sol_count": min_or_nan(sol_counts),
                "mean_cost_constraint": mean_or_nan(cost_constraints),
            }
        )
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 119-bus scalability raw results.")
    parser.add_argument(
        "--inputs",
        default="formal/step10_119_main_inline20_raw.csv",
        help="Comma-separated raw CSV files under results_v2.",
    )
    parser.add_argument("--output", default="formal/step10_119_scalability_inline20_summary.csv")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows: List[Dict[str, str]] = []
    for input_name in parse_csv_list(args.inputs):
        path = RESULT_DIR / input_name
        if not path.exists():
            message = f"Missing input raw file: {path}"
            if args.strict:
                raise FileNotFoundError(message)
            print("[Skipped]", message)
            continue
        rows.extend(read_rows(path))

    summary_rows = summarize(rows, allow_partial=args.allow_partial)
    if not summary_rows:
        raise RuntimeError(
            "No 119 rows to summarize. Please check --inputs or run 119 experiments first."
        )
    output_path = RESULT_DIR / args.output
    write_rows(output_path, summary_rows)
    for row in summary_rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
