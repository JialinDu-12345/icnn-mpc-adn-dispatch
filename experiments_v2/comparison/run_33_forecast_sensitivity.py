from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import (              
    FORECAST_CASES,
    SPLITS,
    validate_actual_case,
    validate_forecast_case,
)
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.run_33_mpc import load_selected_gamma, run_one_scenario_33              


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "MPC-3": {
        "method": "MPC-3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "MPC-5": {
        "method": "MPC-5",
        "solver_kind": "pure_mpc_clean",
        "horizon": 5,
        "gamma_Q": None,
    },
    "MPC-7": {
        "method": "MPC-7",
        "solver_kind": "pure_mpc_clean",
        "horizon": 7,
        "gamma_Q": None,
    },
    "Proposed-H3": {
        "method": "Proposed-H3",
        "solver_kind": "proposed",
        "horizon": 3,
        "gamma_Q": "auto",
    },
}

TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "mpc5": "MPC-5",
    "mpc-5": "MPC-5",
    "mpc7": "MPC-7",
    "mpc-7": "MPC-7",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
}

DEFAULT_FORECAST_CASES = ["id_reliable", "high_error", "biased_ood"]
DEFAULT_TASKS = ["MPC-3", "MPC-7", "Proposed-H3"]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str, scenario_count: Optional[int]) -> List[int]:
    if raw.strip().lower() == "all":
        scenario_ids = list(SPLITS["33"]["test_id"])
    else:
        scenario_ids = [int(item) for item in parse_csv_list(raw)]

    if scenario_count is not None:
        scenario_ids = scenario_ids[:scenario_count]
    return scenario_ids


def parse_tasks(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_TASKS)

    names: List[str] = []
    seen = set()
    for item in parse_csv_list(raw):
        name = TASK_ALIASES.get(item.lower(), item)
        if name not in TASK_SPECS:
            raise ValueError(f"Unknown task '{item}'. Available: {list(TASK_SPECS)}")
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def parse_forecast_cases(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        cases = list(DEFAULT_FORECAST_CASES)
    else:
        cases = parse_csv_list(raw)

    for case in cases:
        validate_forecast_case(case)
    return cases


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 33-bus forecast-assumption sensitivity experiment."
    )
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--forecast-cases", default=",".join(DEFAULT_FORECAST_CASES))
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step5_33_forecast_sensitivity_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    task_names = parse_tasks(args.tasks)
    forecast_cases = parse_forecast_cases(args.forecast_cases)
    requires_gamma = any(TASK_SPECS[name]["solver_kind"] == "proposed" for name in task_names)
    gamma_q = load_selected_gamma("33") if requires_gamma else None

    print("Scenarios:", scenario_ids)
    print("Tasks:", task_names)
    print("Forecast cases:", forecast_cases)
    print("Actual case:", args.actual_case)
    if requires_gamma:
        print("Selected gamma_Q:", gamma_q)
    else:
        print("Selected gamma_Q: not required")
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)

    if args.dry_run:
        return

    logger = ResultLogger(args.output)
    for forecast_case in forecast_cases:
        for scenario_id in scenario_ids:
            for task_name in task_names:
                spec = TASK_SPECS[task_name]
                gamma_Q = gamma_q if spec["solver_kind"] == "proposed" else None

                row = run_one_scenario_33(
                    method=str(spec["method"]),
                    solver_kind=str(spec["solver_kind"]),
                    scenario_id=scenario_id,
                    forecast_case=forecast_case,
                    actual_case=args.actual_case,
                    horizon=int(spec["horizon"]),
                    gamma_Q=gamma_Q,
                    split="test_id",
                    max_steps=args.max_steps,
                    include_tail=not args.no_tail,
                    quiet_solver=not args.verbose_solver,
                )
                row["experiment_tag"] = "forecast_sensitivity"
                logger.add_row(row)

                print(
                    f"{forecast_case}, {row['method']}, scenario={scenario_id}, "
                    f"completed={row.get('completed')}, "
                    f"steps={row.get('executed_steps')}, "
                    f"cost={float(row.get('cost') or 0.0):.6f}, "
                    f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
                    f"fail={row.get('solver_fail_count')}, "
                    f"reason={row.get('incomplete_reason')}"
                )

    path = logger.save()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
