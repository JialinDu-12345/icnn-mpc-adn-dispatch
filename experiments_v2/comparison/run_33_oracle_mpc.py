from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import SPLITS, validate_actual_case              
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.run_33_mpc import run_one_scenario_33              


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "Oracle-H3": {
        "method": "Oracle-H3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "Oracle-H7": {
        "method": "Oracle-H7",
        "solver_kind": "pure_mpc_clean",
        "horizon": 7,
        "gamma_Q": None,
    },
    "Oracle-H12": {
        "method": "Oracle-H12",
        "solver_kind": "pure_mpc_clean",
        "horizon": 12,
        "gamma_Q": None,
    },
}

TASK_ALIASES = {
    "h3": "Oracle-H3",
    "oracle-h3": "Oracle-H3",
    "oracle3": "Oracle-H3",
    "h7": "Oracle-H7",
    "oracle-h7": "Oracle-H7",
    "oracle7": "Oracle-H7",
    "h12": "Oracle-H12",
    "oracle-h12": "Oracle-H12",
    "oracle12": "Oracle-H12",
}


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
        return list(TASK_SPECS)

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 33-bus perfect-forecast Oracle MPC baselines.")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--tasks", default="Oracle-H3,Oracle-H7")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step7_33_oracle_mpc_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    task_names = parse_tasks(args.tasks)

    print("Scenarios:", scenario_ids)
    print("Tasks:", task_names)
    print("forecast_case: perfect")
    print("actual_case:", args.actual_case)
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)

    if args.dry_run:
        return

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for task_name in task_names:
            spec = TASK_SPECS[task_name]
            row = run_one_scenario_33(
                method=str(spec["method"]),
                solver_kind=str(spec["solver_kind"]),
                scenario_id=scenario_id,
                forecast_case="perfect",
                actual_case=args.actual_case,
                horizon=int(spec["horizon"]),
                gamma_Q=spec["gamma_Q"],
                split="test_id",
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
            )
            row["experiment_tag"] = "oracle_mpc"
            logger.add_row(row)

            print(
                f"{row['method']}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.6f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
                f"fail={row.get('solver_fail_count')}, "
                f"reason={row.get('incomplete_reason')}"
            )

    path = logger.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()

