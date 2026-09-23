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


TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "mpc5": "MPC-5",
    "mpc-5": "MPC-5",
    "mpc7": "MPC-7",
    "mpc-7": "MPC-7",
    "stdnn": "StdNN-H3-MIP",
    "std-nn": "StdNN-H3-MIP",
    "stdnn-h3": "StdNN-H3-MIP",
    "stdnn-h3-mip": "StdNN-H3-MIP",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
}

TASK_ORDER = ["MPC-3", "MPC-5", "MPC-7", "StdNN-H3-MIP", "Proposed-H3"]


def resolve_gamma(raw: str) -> float:
    if raw.strip().lower() == "auto":
        return load_selected_gamma("33")
    return float(raw)


def task_names_from_raw(raw: Optional[str]) -> List[str]:
    if raw is None or raw.strip().lower() == "all":
        return list(TASK_ORDER)

    task_names: List[str] = []
    seen = set()
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        name = TASK_ALIASES.get(item.lower(), item)
        if name not in TASK_ORDER:
            raise ValueError(f"Unknown task '{item}'. Available tasks: {TASK_ORDER}")
        if name in seen:
            continue
        task_names.append(name)
        seen.add(name)
    return task_names


def get_task_specs_from_names(
    task_names: List[str], gamma_q: Optional[float]
) -> List[Dict[str, Any]]:
    task_map: Dict[str, Dict[str, Any]] = {
        "MPC-3": {"method": "MPC-3", "solver_kind": "pure_mpc_clean", "horizon": 3, "gamma_Q": None},
        "MPC-5": {"method": "MPC-5", "solver_kind": "pure_mpc_clean", "horizon": 5, "gamma_Q": None},
        "MPC-7": {"method": "MPC-7", "solver_kind": "pure_mpc_clean", "horizon": 7, "gamma_Q": None},
        "StdNN-H3-MIP": {
            "method": "StdNN-H3-MIP",
            "solver_kind": "std_nn_mpc",
            "horizon": 3,
            "gamma_Q": gamma_q,
        },
        "Proposed-H3": {
            "method": "Proposed-H3",
            "solver_kind": "proposed",
            "horizon": 3,
            "gamma_Q": gamma_q,
        },
    }

    tasks: List[Dict[str, Any]] = []
    for name in task_names:
        if name not in task_map:
            raise ValueError(f"Unknown task '{name}'. Available tasks: {TASK_ORDER}")
        if name in {"Proposed-H3", "StdNN-H3-MIP"} and gamma_q is None:
            raise ValueError(f"{name} requires gamma_Q. Use --gamma-Q auto or a float value.")
        tasks.append(task_map[name])
    return tasks


def parse_scenarios(raw: Optional[str], scenario_count: Optional[int]) -> List[int]:
    if raw is None or raw.strip().lower() == "all":
        scenario_ids = list(SPLITS["33"]["test_id"])
    else:
        scenario_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]

    if scenario_count is not None:
        return scenario_ids[:scenario_count]
    return scenario_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 33-bus formal model-based main experiment and horizon ablation."
    )
    parser.add_argument("--scenarios", default="all", help="Comma-separated test IDs or 'all'.")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or 'all'.")
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default="auto", help="Float value or 'auto'.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step4_33_main_model_based_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    task_names = task_names_from_raw(args.tasks)
    requires_gamma = any(name in {"Proposed-H3", "StdNN-H3-MIP"} for name in task_names)
    gamma_q = resolve_gamma(args.gamma_Q) if requires_gamma else None

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    tasks = get_task_specs_from_names(task_names, gamma_q)

    if requires_gamma:
        print("Selected gamma_Q for Proposed-H3:", gamma_q)
    else:
        print("Selected gamma_Q for Proposed-H3: not required")
    print("Scenarios:", scenario_ids)
    print("Tasks:", [task["method"] for task in tasks])
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)
    print("time_limit_s:", args.time_limit_s)
    print("mip_gap:", args.mip_gap)

    if args.dry_run:
        return

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for task in tasks:
            row = run_one_scenario_33(
                method=str(task["method"]),
                solver_kind=str(task["solver_kind"]),
                scenario_id=scenario_id,
                forecast_case=args.forecast_case,
                actual_case=args.actual_case,
                horizon=int(task["horizon"]),
                gamma_Q=task["gamma_Q"],
                split="test_id",
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
                time_limit_s=args.time_limit_s,
                mip_gap=args.mip_gap,
            )
            row["experiment_tag"] = "main_model_based"
            logger.add_row(row)
            print(
                f"{row['method']}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"reason={row.get('incomplete_reason')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.6f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
                f"fail={row.get('solver_fail_count')}"
            )

    path = logger.save()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
