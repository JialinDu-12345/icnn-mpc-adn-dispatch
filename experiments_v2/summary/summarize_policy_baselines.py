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
    "SAC": 0,
    "SAC-ICNN": 1,
    "SACLag": 2,
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
    return mean(clean) if clean else math.nan


def first_nonempty(rows: Iterable[Dict[str, str]], key: str, default: str = "") -> str:
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return default


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
        completed_rows = [row for row, done in zip(group, completed_flags) if done]
        perf_rows = group if allow_partial else completed_rows

        costs = [parse_float(row.get("cost")) for row in perf_rows]
        solve_times = [parse_float(row.get("solve_time_s")) for row in perf_rows]
        avg_step_times = [parse_float(row.get("avg_step_time_s")) for row in perf_rows]
        max_step_times = [parse_float(row.get("max_step_time_s")) for row in perf_rows]
        executed_steps = [parse_float(row.get("executed_steps")) for row in group]
        fail_counts = [int(float(row.get("solver_fail_count") or 0)) for row in group]
        economic_costs = [
            parse_float(row.get("economic_generation_cost")) for row in perf_rows
        ]
        penalized_costs = [parse_float(row.get("penalized_cost")) for row in perf_rows]
        balance_penalty_costs = [
            parse_float(row.get("balance_penalty_cost")) for row in perf_rows
        ]
        constraint_penalties = [
            parse_float(row.get("constraint_penalty")) for row in perf_rows
        ]
        cost_constraints = [parse_float(row.get("mean_cost_constraint")) for row in perf_rows]
        voltage_violations = [parse_float(row.get("voltage_violation")) for row in perf_rows]
        legacy_cost_constraints = [
            parse_float(row.get("legacy_mean_cost_constraint")) for row in perf_rows
        ]
        policy_eval_modes = sorted(
            {
                str(row.get("policy_eval_mode", "")).strip()
                for row in perf_rows
                if str(row.get("policy_eval_mode", "")).strip()
            }
        )
        policy_eval_seeds = sorted(
            {
                str(row.get("policy_eval_seed", "")).strip()
                for row in perf_rows
                if str(row.get("policy_eval_seed", "")).strip()
            }
        )
        stochastic_flags = [
            parse_bool(row.get("policy_stochastic_eval"))
            for row in perf_rows
            if str(row.get("policy_stochastic_eval", "")).strip()
        ]

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
                "n_perf_rows": len(perf_rows),
                "allow_partial": allow_partial,
                "policy_eval_mode": ",".join(policy_eval_modes),
                "policy_eval_seed": ",".join(policy_eval_seeds),
                "policy_stochastic_eval": (
                    ""
                    if not stochastic_flags
                    else all(stochastic_flags)
                    if all(flag == stochastic_flags[0] for flag in stochastic_flags)
                    else "mixed"
                ),
                "cost_accounting": first_nonempty(
                    perf_rows,
                    "cost_accounting",
                    "legacy_or_unknown",
                ),
                "voltage_violation_metric": first_nonempty(
                    perf_rows,
                    "voltage_violation_metric",
                    "",
                ),
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
                "mean_cost_constraint": mean_or_nan(cost_constraints),
                "mean_voltage_violation": mean_or_nan(voltage_violations),
                "std_voltage_violation": std_or_nan(voltage_violations),
                "ci95_voltage_violation": ci95(voltage_violations),
                "legacy_mean_cost_constraint": mean_or_nan(legacy_cost_constraints),
            }
        )
    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize 33-bus RL policy baselines.")
    parser.add_argument("--input", default="formal/step7_33_policy_baselines_raw.csv")
    parser.add_argument("--output", default="formal/step7_33_policy_baselines_summary.csv")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    rows = read_rows(RESULT_DIR / args.input)
    summary_rows = summarize(rows, allow_partial=args.allow_partial)
    output_path = RESULT_DIR / args.output
    write_rows(output_path, summary_rows)
    if any(row.get("cost_accounting") == "legacy_or_unknown" for row in summary_rows):
        print(
            "[Warning] Some policy rows do not declare cost_accounting. "
            "If the raw file was generated before the cost-accounting patch, "
            "rerun policy_baselines before using mean_cost as economic cost."
        )
    for row in summary_rows:
        print(row)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
