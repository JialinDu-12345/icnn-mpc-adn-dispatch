from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import DEFAULT_BASE_SEEDS              
from experiments_v2.policy_baselines.policy_adapter_119 import PolicyAdapter119              
from experiments_v2.result_logger import ResultLogger, make_result_row              
from experiments_v2.run_119_mpc import run_one_scenario_119              


TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "sac": "SAC",
    "sac-icnn": "SAC-ICNN",
    "sacicnn": "SAC-ICNN",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
}

DEFAULT_TASKS = ["Proposed-H3", "SAC", "MPC-3"]


def parse_tasks(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_TASKS)

    tasks: List[str] = []
    seen = set()
    allowed = set(DEFAULT_TASKS + ["SAC-ICNN"])
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        task = TASK_ALIASES.get(item.lower(), item)
        if task not in allowed:
            raise ValueError(f"Unknown task '{item}'. Expected one of {sorted(allowed)}.")
        if task in seen:
            continue
        tasks.append(task)
        seen.add(task)
    return tasks


def resolve_gamma(raw: str) -> float:
    if str(raw).strip().lower() == "auto":
        return 0.3
    return float(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one 119-bus scenario for paper dispatch-trajectory plots."
    )
    parser.add_argument("--scenario", type=int, default=801)
    parser.add_argument("--tasks", default="Proposed-H3,SAC,MPC-3")
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default="0.3")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--verbose-legacy-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step13_119_dispatch_trajectory_raw.csv")
    return parser


def failure_row(task: str, args: argparse.Namespace, exc: BaseException) -> Dict[str, Any]:
    horizon = 3 if task in {"MPC-3", "Proposed-H3"} else 0
    forecast_case = (
        args.forecast_case
        if task in {"MPC-3", "Proposed-H3"}
        else "not_applicable"
    )
    gamma_q = resolve_gamma(args.gamma_Q) if task == "Proposed-H3" else None
    return make_result_row(
        system="119",
        method=task,
        scenario_id=args.scenario,
        split="test_id",
        actual_case=args.actual_case,
        forecast_case=forecast_case,
        horizon=horizon,
        gamma_Q=gamma_q,
        cost=0.0,
        solve_time_s=0.0,
        avg_step_time_s=0.0,
        max_step_time_s=0.0,
        solver_fail_count=1,
        timeout_count=0,
        completed=False,
        incomplete_reason="runner_failure",
        seed=DEFAULT_BASE_SEEDS["119"] + int(args.scenario),
        executed_steps=0,
        include_tail=not args.no_tail,
        max_steps=args.max_steps,
        failure_message=f"dispatch trajectory task={task}, scenario={args.scenario}: {exc}",
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    tasks = parse_tasks(args.tasks)
    gamma_q = resolve_gamma(args.gamma_Q) if "Proposed-H3" in tasks else None

    print("Dispatch trajectory scenario:", args.scenario)
    print("Tasks:", tasks)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("gamma_Q:", gamma_q if gamma_q is not None else "not required")
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)
    print("output:", args.output)

    if args.dry_run:
        return

    logger = ResultLogger(args.output)
    policy_adapters: Dict[str, PolicyAdapter119] = {}

    for task in tasks:
        try:
            if task == "MPC-3":
                row = run_one_scenario_119(
                    method="MPC-3",
                    solver_kind="pure_mpc_clean",
                    scenario_id=args.scenario,
                    horizon=3,
                    gamma_Q=None,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    split="test_id",
                    max_steps=args.max_steps,
                    include_tail=not args.no_tail,
                    quiet_solver=not args.verbose_solver,
                    time_limit_s=args.time_limit_s,
                    mip_gap=args.mip_gap,
                )
            elif task == "Proposed-H3":
                row = run_one_scenario_119(
                    method="Proposed-H3",
                    solver_kind="proposed",
                    scenario_id=args.scenario,
                    horizon=3,
                    gamma_Q=gamma_q,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    split="test_id",
                    max_steps=args.max_steps,
                    include_tail=not args.no_tail,
                    quiet_solver=not args.verbose_solver,
                    time_limit_s=args.time_limit_s,
                    mip_gap=args.mip_gap,
                )
            elif task in {"SAC", "SAC-ICNN"}:
                if task not in policy_adapters:
                    policy_adapters[task] = PolicyAdapter119(
                        method=task,
                        device=args.device,
                        quiet_legacy_init=not args.verbose_legacy_init,
                    )
                row = policy_adapters[task].run_one_day(
                    scenario_id=args.scenario,
                    actual_case=args.actual_case,
                    max_steps=args.max_steps,
                    stochastic=args.stochastic_policy,
                )
            else:
                raise ValueError(f"Unknown task: {task}")
        except Exception as exc:
            print(f"[Failed] dispatch trajectory task={task}, scenario={args.scenario}: {exc}")
            traceback.print_exc()
            row = failure_row(task, args, exc)

        row["experiment_tag"] = "dispatch_trajectory_plot_119"
        row["dispatch_plot_scenario"] = int(args.scenario)
        logger.add_row(row)
        print(
            f"{row['method']}, scenario={args.scenario}, "
            f"completed={row.get('completed')}, "
            f"steps={row.get('executed_steps')}, "
            f"cost={float(row.get('cost') or 0.0):.6f}, "
            f"fail={row.get('solver_fail_count')}"
        )

    logger.save()


if __name__ == "__main__":
    main()
