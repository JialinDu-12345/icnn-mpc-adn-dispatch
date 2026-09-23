from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import SPLITS, validate_actual_case, validate_forecast_case              
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.run_33_mpc import load_selected_gamma, run_one_scenario_33              


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "MPC-3": {
        "method": "MPC-3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "Proposed-H3": {
        "method": "Proposed-H3",
        "solver_kind": "proposed",
        "horizon": 3,
        "gamma_Q": "auto",
    },
    "StdNN-H3-MIP": {
        "method": "StdNN-H3-MIP",
        "solver_kind": "std_nn_mpc",
        "horizon": 3,
        "gamma_Q": "auto",
    },
}

TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
    "stdnn": "StdNN-H3-MIP",
    "std-nn": "StdNN-H3-MIP",
    "stdnn-h3": "StdNN-H3-MIP",
    "stdnn-h3-mip": "StdNN-H3-MIP",
}

DEFAULT_TASKS = ["MPC-3", "Proposed-H3", "StdNN-H3-MIP"]
                                  


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 33-bus standard NN-MPC embedding comparison.")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step6_33_nn_embedding_raw.csv")
                                                                                   
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    task_names = parse_tasks(args.tasks)
    requires_gamma = any(TASK_SPECS[name]["gamma_Q"] == "auto" for name in task_names)
    gamma_q = load_selected_gamma("33") if requires_gamma else None

    print("Scenarios:", scenario_ids)
    print("Tasks:", task_names)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("selected gamma_Q:", gamma_q if requires_gamma else "not required")
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)
    print("time_limit_s:", args.time_limit_s)
    print("mip_gap:", args.mip_gap)

    if args.dry_run:
        return

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for task_name in task_names:
            spec = TASK_SPECS[task_name]
            gamma_Q = gamma_q if spec["gamma_Q"] == "auto" else spec["gamma_Q"]

            row = run_one_scenario_33(
                method=str(spec["method"]),
                solver_kind=str(spec["solver_kind"]),
                scenario_id=scenario_id,
                forecast_case=args.forecast_case,
                actual_case=args.actual_case,
                horizon=int(spec["horizon"]),
                gamma_Q=gamma_Q,
                split="test_id",
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
                time_limit_s=args.time_limit_s,
                mip_gap=args.mip_gap,
            )
            row["experiment_tag"] = "standard_nn_embedding"
            logger.add_row(row)

            print(
                f"{row['method']}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.6f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
                f"bin={row.get('mean_num_bin_vars')}, "
                f"fail={row.get('solver_fail_count')}, "
                f"reason={row.get('incomplete_reason')}"
            )

    path = logger.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()
