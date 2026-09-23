from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import (              
    DEFAULT_119,
    validate_actual_case,
    validate_forecast_case,
)
from experiments_v2.global_optimum.run_119_perfect_global import (              
    run_one_perfect_global_119,
)
from experiments_v2.policy_baselines.policy_adapter_119 import PolicyAdapter119              
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.run_119_mpc import run_one_scenario_119              
from experiments_v2.utils.scenario_selection import (              
    resolve_result_path,
    resolve_scenarios,
)


TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "sac": "SAC",
    "sac-icnn": "SAC-ICNN",
    "sacicnn": "SAC-ICNN",
    "stdnn": "StdNN-H3-MIP",
    "std-nn": "StdNN-H3-MIP",
    "stdnn-h3": "StdNN-H3-MIP",
    "stdnn-h3-mip": "StdNN-H3-MIP",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
    "perfect": "Perfect-Global",
    "perfect-global": "Perfect-Global",
}

TASK_ORDER = ["MPC-3", "SAC", "SAC-ICNN", "StdNN-H3-MIP", "Proposed-H3"]
OPTIONAL_TASKS = ["Perfect-Global"]


def resolve_gamma(raw: str) -> float:
    if raw.strip().lower() == "auto":
        return 0.3
    return float(raw)


def selection_rule_from_file(raw_path: Optional[str]) -> str:
    if not raw_path:
        return ""
    name = Path(raw_path).name.lower()
    if "operable" in name:
        return "operable_perfect_global_and_rolling_mpc3_subset"
    if "perfect_feasible" in name:
        return "perfect_global_feasible_certified_subset"
    return "external_selected_scenario_subset"


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def baseline_gate_passed(row: Dict[str, Any], *, n_steps: int) -> bool:
                                                                                  

    completed = parse_bool_value(row.get("completed"))
    solver_fail_count = int_value(row.get("solver_fail_count"), default=0)
    executed_steps = int_value(row.get("executed_steps"), default=0)
    incomplete_reason = str(row.get("incomplete_reason") or "").strip()

    return (
        completed
        and solver_fail_count == 0
        and executed_steps == int(n_steps)
        and incomplete_reason == ""
        and finite_float(row.get("cost"))
    )


def save_selected_json(
    raw_path: str,
    selected_scenarios: List[int],
    args: argparse.Namespace,
) -> None:
    if not raw_path:
        return

    path = resolve_result_path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": "119",
        "selected_scenarios": [int(value) for value in selected_scenarios],
        "selected_count": len(selected_scenarios),
        "target_count": int(args.target_count),
        "candidate_scenarios": [
            int(value) for value in getattr(args, "_resolved_candidate_scenarios", [])
        ],
        "selection_rule": "inline_gate_by_mpc3_and_perfect_global",
        "gate_baselines": ["MPC-3", "Perfect-Global"],
        "accept_time_limit_with_incumbent": True,
        "forecast_case": args.forecast_case,
        "actual_case": args.actual_case,
        "time_limit_s": args.time_limit_s,
        "mip_gap": args.mip_gap,
        "screen_log_output": args.screen_log_output,
        "main_output": args.output,
        "note": (
            "Candidates are retained only after MPC-3 and Perfect-Global both "
            "return completed=True with a finite cost. Gurobi TIME_LIMIT steps "
            "with an incumbent are accepted when the runner reports the day as completed."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved selected scenarios to: {path}")


def task_names_from_raw(raw: Optional[str]) -> List[str]:
    normalized = "" if raw is None else raw.strip().lower()
    if normalized in {"", "all"}:
        return list(TASK_ORDER)
    if normalized in {"all_with_perfect", "all-with-perfect", "allwithperfect"}:
        return list(OPTIONAL_TASKS + TASK_ORDER)

    allowed = set(TASK_ORDER + OPTIONAL_TASKS)
    task_names: List[str] = []
    seen = set()
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        name = TASK_ALIASES.get(item.lower(), item)
        if name not in allowed:
            raise ValueError(f"Unknown task '{item}'. Available tasks: {sorted(allowed)}")
        if name in seen:
            continue
        task_names.append(name)
        seen.add(name)
    return task_names


def get_task_specs_from_names(
    task_names: List[str], gamma_q: Optional[float]
) -> List[Dict[str, Any]]:
    task_map: Dict[str, Dict[str, Any]] = {
        "MPC-3": {
            "runner": "model_based",
            "method": "MPC-3",
            "solver_kind": "pure_mpc_clean",
            "horizon": 3,
            "gamma_Q": None,
        },
        "Proposed-H3": {
            "runner": "model_based",
            "method": "Proposed-H3",
            "solver_kind": "proposed",
            "horizon": 3,
            "gamma_Q": gamma_q,
        },
        "StdNN-H3-MIP": {
            "runner": "model_based",
            "method": "StdNN-H3-MIP",
            "solver_kind": "std_nn_mpc",
            "horizon": 3,
            "gamma_Q": gamma_q,
        },
        "SAC": {
            "runner": "policy",
            "method": "SAC",
        },
        "SAC-ICNN": {
            "runner": "policy",
            "method": "SAC-ICNN",
        },
        "Perfect-Global": {
            "runner": "perfect_global",
            "method": "Perfect-Global",
        },
    }

    tasks: List[Dict[str, Any]] = []
    for name in task_names:
        if name not in task_map:
            raise ValueError(f"Unknown task '{name}'.")
        if name in {"Proposed-H3", "StdNN-H3-MIP"} and gamma_q is None:
            raise ValueError(f"{name} requires gamma_Q.")
        tasks.append(task_map[name])
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 119-bus scalability main experiment."
    )
    parser.add_argument("--scenarios", default="all", help="Comma-separated test IDs or 'all'.")
    parser.add_argument(
        "--scenario-file",
        default="formal/selected_119_inline_operable20.json",
        help="JSON/CSV file under results_v2 containing selected scenarios. Overrides --scenarios.",
    )
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument(
        "--tasks",
        default="StdNN-H3-MIP",
        help="Comma-separated tasks, 'all', or 'all_with_perfect'.",
    )
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default="0.3", help="Float value or 'auto'.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--write-iis", action="store_true")
    parser.add_argument("--iis-dir", default="debug/iis_119_main")
    parser.add_argument(
        "--no-clip-icnn-exogenous-inputs",
        action="store_true",
        help=(
            "Disable clipping of exogenous ICNN terminal inputs. This is only for "
            "diagnostics and is not recommended for formal 119-bus runs."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--verbose-legacy-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gate-by-baselines",
        action="store_true",
        default=False,
        help=(
            "Inline scenario screening mode. For each candidate scenario, run "
            "MPC-3 and Perfect-Global first. Only if both complete, continue "
            "with the other methods."
        ),
    )
    parser.add_argument(
        "--no-gate-by-baselines",
        action="store_false",
        dest="gate_by_baselines",
        help="Disable inline screening and use --scenario-file / --scenarios directly.",
    )
    parser.add_argument(
        "--candidate-scenarios",
        default="800:899",
        help=(
            "Candidate scenarios used in --gate-by-baselines mode. "
            "Examples: 800:899 or 800,801,802."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Number of baseline-passed scenarios to collect in --gate-by-baselines mode.",
    )
    parser.add_argument(
        "--screen-log-output",
        default="formal/step10_119_inline_screen_log.csv",
        help="CSV file for all MPC-3 / Perfect-Global screening attempts.",
    )
    parser.add_argument(
        "--selected-json-output",
        default="formal/selected_119_inline_operable20.json",
        help="JSON file storing the scenarios that passed inline screening.",
    )
    parser.add_argument("--output", default="formal/step12_119_stdnn_raw.csv")
    return parser


def run_task_for_scenario(
    *,
    task: Dict[str, Any],
    scenario_id: int,
    args: argparse.Namespace,
    policy_adapters: Dict[str, PolicyAdapter119],
) -> Dict[str, Any]:
    runner = task["runner"]

    if runner == "model_based":
        row = run_one_scenario_119(
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
            write_iis=args.write_iis,
            iis_dir=args.iis_dir,
            clip_icnn_exogenous_inputs=not args.no_clip_icnn_exogenous_inputs,
        )
        row["experiment_tag"] = "main_scalability_119"

    elif runner == "policy":
        method = str(task["method"])
        if method not in policy_adapters:
            policy_adapters[method] = PolicyAdapter119(
                method=method,
                device=args.device,
                quiet_legacy_init=not args.verbose_legacy_init,
            )
        adapter = policy_adapters[method]
        row = adapter.run_one_day(
            scenario_id=scenario_id,
            actual_case=args.actual_case,
            max_steps=args.max_steps,
            stochastic=args.stochastic_policy,
        )
        row["experiment_tag"] = "main_scalability_119"

    elif runner == "perfect_global":
        row = run_one_perfect_global_119(
            scenario_id=scenario_id,
            actual_case=args.actual_case,
            quiet_solver=not args.verbose_solver,
            time_limit_s=args.time_limit_s,
            mip_gap=1e-3 if args.mip_gap is None else args.mip_gap,
        )
        row["experiment_tag"] = "main_scalability_119"

    else:
        raise ValueError(f"Unknown runner: {runner}")

    return row


def annotate_main_row(
    row: Dict[str, Any],
    *,
    args: argparse.Namespace,
    selection_rule: str,
    selection_order: Optional[int] = None,
    gate_stage: str = "",
) -> Dict[str, Any]:
    row["scenario_selection_file"] = (
        args.selected_json_output if args.gate_by_baselines else (args.scenario_file or "")
    )
    row["scenario_selection_rule"] = selection_rule
    row["inline_gate_by_baselines"] = bool(args.gate_by_baselines)
    row["gate_stage"] = gate_stage
    if selection_order is not None:
        row["gate_selection_order"] = int(selection_order)
    return row


def print_row_summary(row: Dict[str, Any]) -> None:
    print(
        f"{row['method']}, scenario={row['scenario_id']}, "
        f"completed={row.get('completed')}, "
        f"reason={row.get('incomplete_reason')}, "
        f"steps={row.get('executed_steps')}, "
        f"cost={float(row.get('cost') or 0.0):.6f}, "
        f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
        f"fail={row.get('solver_fail_count')}, "
        f"icnn_clip={row.get('total_icnn_clip_count', '')}"
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    if args.gate_by_baselines and str(args.tasks).strip().lower() == "all":
        task_names = [
            "MPC-3",
            "Perfect-Global",
            "SAC",
            "SAC-ICNN",
            "StdNN-H3-MIP",
            "Proposed-H3",
        ]
    else:
        task_names = task_names_from_raw(args.tasks)
    requires_gamma = any(name in {"Proposed-H3", "StdNN-H3-MIP"} for name in task_names)
    gamma_q = resolve_gamma(args.gamma_Q) if requires_gamma else None
    if args.gate_by_baselines:
        scenario_ids = resolve_scenarios(
            system="119",
            scenarios=args.candidate_scenarios or args.scenarios,
            scenario_file=None,
            scenario_count=args.scenario_count,
        )
    else:
        try:
            scenario_ids = resolve_scenarios(
                system="119",
                scenarios=args.scenarios,
                scenario_file=args.scenario_file,
                scenario_count=args.scenario_count,
            )
        except FileNotFoundError as exc:
            if args.scenario_file and "selected_119_operable_20" in args.scenario_file:
                raise FileNotFoundError(
                    f"{exc}. Generate the formal operable selected20 first with: "
                    "python -m experiments_v2.global_optimum.select_119_operable_scenarios "
                    "--candidate-scenarios all --target-count 20 --actual-case id_actual "
                    "--forecast-case id_reliable --resume"
                ) from exc
            raise
    args._resolved_candidate_scenarios = list(scenario_ids)

    if (
        not args.gate_by_baselines
        and args.scenario_file
        and args.scenario_count is None
        and len(scenario_ids) < 20
    ):
        print(
            f"[WARN] scenario_file={args.scenario_file} contains only "
            f"{len(scenario_ids)} scenarios. If this is a formal selected20 run, "
            "please wait until selected_count reaches 20 or run validate_119_selected20 first."
        )
    tasks = get_task_specs_from_names(task_names, gamma_q)

    print("Selected gamma_Q for value-augmented methods:", gamma_q if requires_gamma else "not required")
    print("Scenarios:", scenario_ids)
    print("Tasks:", [task["method"] for task in tasks])
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("gate_by_baselines:", args.gate_by_baselines)
    if args.gate_by_baselines:
        print("scenario_file: ignored in gate mode")
        print("candidate_scenarios:", args.candidate_scenarios or args.scenarios)
        print("target_count:", args.target_count)
        print("screen_log_output:", args.screen_log_output)
        print("selected_json_output:", args.selected_json_output)
    else:
        print("scenario_file:", args.scenario_file or "")
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)
    print("write_iis:", args.write_iis)
    print("iis_dir:", args.iis_dir)
    print("clip_icnn_exogenous_inputs:", not args.no_clip_icnn_exogenous_inputs)
    selection_rule = (
        "inline_gate_by_mpc3_and_perfect_global"
        if args.gate_by_baselines
        else selection_rule_from_file(args.scenario_file)
    )

    if args.dry_run:
        return

    policy_adapters: Dict[str, PolicyAdapter119] = {}

    logger = ResultLogger(args.output)
    task_by_method = {str(task["method"]): task for task in tasks}

    if args.gate_by_baselines:
        required = {"MPC-3", "Perfect-Global"}
        missing = required - set(task_by_method)
        if missing:
            raise ValueError(
                f"--gate-by-baselines requires tasks to include {sorted(required)}. "
                f"Missing: {sorted(missing)}"
            )

        screen_logger = ResultLogger(args.screen_log_output)
        n_steps = int(DEFAULT_119["n_steps"])
        selected_scenarios: List[int] = []

        for scenario_id in scenario_ids:
            if len(selected_scenarios) >= int(args.target_count):
                break

            print(f"\n[Gate] scenario={scenario_id}: running MPC-3 first...")
            mpc3_row = run_task_for_scenario(
                task=task_by_method["MPC-3"],
                scenario_id=scenario_id,
                args=args,
                policy_adapters=policy_adapters,
            )
            annotate_main_row(
                mpc3_row,
                args=args,
                selection_rule=selection_rule,
                gate_stage="gate_mpc3",
            )
            screen_logger.add_row(mpc3_row)
            print_row_summary(mpc3_row)

            if not baseline_gate_passed(mpc3_row, n_steps=n_steps):
                print(f"[Skip] scenario={scenario_id}: MPC-3 did not pass the gate.")
                screen_logger.save()
                continue

            print(f"[Gate] scenario={scenario_id}: running Perfect-Global...")
            perfect_row = run_task_for_scenario(
                task=task_by_method["Perfect-Global"],
                scenario_id=scenario_id,
                args=args,
                policy_adapters=policy_adapters,
            )
            annotate_main_row(
                perfect_row,
                args=args,
                selection_rule=selection_rule,
                gate_stage="gate_perfect_global",
            )
            screen_logger.add_row(perfect_row)
            print_row_summary(perfect_row)

            if not baseline_gate_passed(perfect_row, n_steps=n_steps):
                print(
                    f"[Skip] scenario={scenario_id}: "
                    "Perfect-Global did not pass the gate."
                )
                screen_logger.save()
                continue

            selected_scenarios.append(int(scenario_id))
            selection_order = len(selected_scenarios)
            print(
                f"[Selected] scenario={scenario_id} passed MPC-3 and Perfect-Global "
                f"({selection_order}/{args.target_count})."
            )

            annotate_main_row(
                mpc3_row,
                args=args,
                selection_rule=selection_rule,
                selection_order=selection_order,
                gate_stage="formal_selected",
            )
            annotate_main_row(
                perfect_row,
                args=args,
                selection_rule=selection_rule,
                selection_order=selection_order,
                gate_stage="formal_selected",
            )
            logger.add_row(mpc3_row)
            logger.add_row(perfect_row)

            for task in tasks:
                method = str(task["method"])
                if method in {"MPC-3", "Perfect-Global"}:
                    continue

                row = run_task_for_scenario(
                    task=task,
                    scenario_id=scenario_id,
                    args=args,
                    policy_adapters=policy_adapters,
                )
                annotate_main_row(
                    row,
                    args=args,
                    selection_rule=selection_rule,
                    selection_order=selection_order,
                    gate_stage="formal_selected",
                )
                logger.add_row(row)
                print_row_summary(row)

            logger.save()
            screen_logger.save()
            save_selected_json(args.selected_json_output, selected_scenarios, args)

        if logger.rows:
            path = logger.save()
            print(f"Saved: {path}")
        if screen_logger.rows:
            screen_logger.save()
        save_selected_json(args.selected_json_output, selected_scenarios, args)
        print(f"Selected scenarios: {selected_scenarios}")
        print(f"Selected count: {len(selected_scenarios)}")
        return

    for scenario_id in scenario_ids:
        for task in tasks:
            row = run_task_for_scenario(
                task=task,
                scenario_id=scenario_id,
                args=args,
                policy_adapters=policy_adapters,
            )
            annotate_main_row(
                row,
                args=args,
                selection_rule=selection_rule,
                gate_stage="not_gated",
            )
            logger.add_row(row)
            print_row_summary(row)

    if logger.rows:
        path = logger.save()
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
