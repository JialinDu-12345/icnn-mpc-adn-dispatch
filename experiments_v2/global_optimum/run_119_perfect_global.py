from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import (              
    DEFAULT_119,
    DEFAULT_BASE_SEEDS,
    METHOD_DIRS_119,
    SPLITS,
    validate_actual_case,
)
from experiments_v2.result_logger import ResultLogger, make_result_row              
from experiments_v2.run_119_mpc import (              
    _legacy_bounds_and_network,
    load_legacy_mpc1_definitions,
)
from experiments_v2.scenario_protocol import (              
    get_solver_inputs_119,
    load_actual_scenario,
)


def build_perfect_solver_119():
    method_dir = METHOD_DIRS_119["perfect_global_clean"]
    module = load_legacy_mpc1_definitions(
        module_name="perfect_global_clean_119",
        file_path=method_dir / "MPC_1.py",
    )
    (
        load_min,
        load_max,
        list_r_x_pu,
        list_p_q_pu,
        load_min_Q,
        load_max_Q,
    ) = _legacy_bounds_and_network(module)

    solver = module.Solve_MPC(
        load_min,
        load_max,
        list_r_x_pu,
        list_p_q_pu,
        load_min_Q,
        load_max_Q,
    )
    solver.time_limit_s = None
    solver.mip_gap = 1e-3
    solver.output_flag = 0
    solver.last_model_stats = {}
    return solver


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str, scenario_count: Optional[int]) -> List[int]:
    if raw.strip().lower() == "all":
        scenario_ids = list(SPLITS["119"]["test_id"])
    else:
        scenario_ids = [int(item) for item in parse_csv_list(raw)]

    if scenario_count is not None:
        return scenario_ids[:scenario_count]
    return scenario_ids


def split_for_scenario(system: str, scenario_id: int) -> str:
    sid = int(scenario_id)
    if sid in set(SPLITS[system].get("test_id", [])):
        return "test_id"
    if sid in set(SPLITS[system].get("val", [])):
        return "val"
    if sid in set(SPLITS[system].get("historical", [])):
        return "historical"
    if sid in set(SPLITS[system].get("profile", [])):
        return "historical"
    return "custom"


def run_one_perfect_global_119(
    *,
    scenario_id: int,
    actual_case: str = "id_actual",
    quiet_solver: bool = True,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = 1e-3,
) -> dict[str, Any]:
    validate_actual_case(actual_case)

    system = "119"
    n_steps = int(DEFAULT_119["n_steps"])
    seed = int(DEFAULT_BASE_SEEDS[system] + int(scenario_id))
    split = split_for_scenario(system, scenario_id)

    solver = None
    solver_fail_count = 0
    failure_message = ""
    total_cost = np.nan
    step_costs: List[float] = []
    solve_time = np.nan
    final_values = {
        "P_DG_29": np.nan,
        "P_DG_64": np.nan,
        "SOC_BSS_12": np.nan,
        "SOC_BSS_34": np.nan,
        "SOC_BSS_68": np.nan,
        "SOC_BSS_103": np.nan,
    }
    stats: dict[str, Any] = {}

    try:
        actual = load_actual_scenario(system, scenario_id, actual_case=actual_case)
        (
            load_list,
            load_list_Q,
            wind_11_avail_list,
            wind_32_avail_list,
            wind_66_avail_list,
            wind_101_avail_list,
            pv_22_avail_list,
            pv_36_avail_list,
            pv_70_avail_list,
            pv_107_avail_list,
        ) = get_solver_inputs_119(actual)
        solver = build_perfect_solver_119()

        solver.time_limit_s = time_limit_s
        solver.mip_gap = mip_gap
        solver.output_flag = 0 if quiet_solver else 1

        solve_kwargs = dict(
            window_time=n_steps,
            load_list=load_list,
            load_list_Q=load_list_Q,
            P_DG_29_init=float(DEFAULT_119["P_DG_29_init"]),
            P_DG_64_init=float(DEFAULT_119["P_DG_64_init"]),
            SOC_BSS_12_init=float(DEFAULT_119["SOC_BSS_12_init"]),
            SOC_BSS_34_init=float(DEFAULT_119["SOC_BSS_34_init"]),
            SOC_BSS_68_init=float(DEFAULT_119["SOC_BSS_68_init"]),
            SOC_BSS_103_init=float(DEFAULT_119["SOC_BSS_103_init"]),
            wind_11_avail_list=wind_11_avail_list,
            wind_32_avail_list=wind_32_avail_list,
            wind_66_avail_list=wind_66_avail_list,
            wind_101_avail_list=wind_101_avail_list,
            pv_22_avail_list=pv_22_avail_list,
            pv_36_avail_list=pv_36_avail_list,
            pv_70_avail_list=pv_70_avail_list,
            pv_107_avail_list=pv_107_avail_list,
        )

        if quiet_solver:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink):
                    result = solver.sol_pro_all_day(**solve_kwargs)
        else:
            result = solver.sol_pro_all_day(**solve_kwargs)

        if not isinstance(result, dict):
            raise TypeError(f"Perfect solver returned {type(result)!r}; expected dict.")

        total_cost = float(result["obj"])
        step_costs = [float(value) for value in result.get("cost_hour", [])]
        solve_time = float(result.get("runtime", np.nan))
        stats = dict(result.get("model_stats") or getattr(solver, "last_model_stats", {}) or {})
        for key in final_values:
            values = [float(value) for value in result.get(key, [])]
            if values:
                final_values[key] = values[-1]

    except Exception as exc:
        solver_fail_count = 1
        stats = dict(getattr(solver, "last_model_stats", {}) or {})
        solve_time = float(stats.get("runtime", np.nan))
        failure_message = (
            f"method=Perfect-Global, scenario={scenario_id}, actual_case={actual_case}: {exc}"
        )
        print(f"[Failed] {failure_message}")
        traceback.print_exc()

    executed_steps = len(step_costs) if solver_fail_count == 0 else 0
    solver_has_solution = solver_fail_count == 0 and executed_steps == n_steps
    status = stats.get("status", np.nan)
    mip_gap_value = stats.get("mip_gap", np.nan)
    try:
        status_code = int(float(status))
    except (TypeError, ValueError):
        status_code = -1
    try:
        mip_gap_float = float(mip_gap_value)
    except (TypeError, ValueError):
        mip_gap_float = np.nan
    mip_gap_target = 1e-3 if mip_gap is None else float(mip_gap)
    certified_global = (
        solver_has_solution
        and status_code == 2
        and (not np.isfinite(mip_gap_float) or mip_gap_float <= mip_gap_target + 1e-9)
    )
    timeout_count = 1 if status_code == 9 else 0
    if np.isfinite(solve_time) and executed_steps > 0:
        step_times = [solve_time / executed_steps for _ in range(executed_steps)]
    else:
        step_times = []

    return make_result_row(
        system=system,
        method="Perfect-Global",
        scenario_id=scenario_id,
        split=split,
        actual_case=actual_case,
        forecast_case="perfect_full_horizon",
        horizon=n_steps,
        gamma_Q=None,
        cost=total_cost if solver_has_solution else None,
        perfect_cost=total_cost if solver_has_solution else None,
        gap_percent=0.0 if solver_has_solution else None,
        voltage_violation=None,
        solve_time_s=solve_time,
        avg_step_time_s=solve_time / max(executed_steps, 1)
        if np.isfinite(solve_time)
        else np.nan,
        max_step_time_s=solve_time,
        solver_fail_count=solver_fail_count,
        timeout_count=timeout_count,
        completed=solver_has_solution,
        incomplete_reason="" if solver_has_solution else "solver_failure",
        seed=seed,
        solver_kind="perfect_global_clean",
        solver_window=n_steps,
        executed_steps=executed_steps,
        include_tail=True,
        cost_accounting="full_day_optimization_objective",
        final_P_DG_29=final_values["P_DG_29"],
        final_P_DG_64=final_values["P_DG_64"],
        final_SOC_BSS_12=final_values["SOC_BSS_12"],
        final_SOC_BSS_34=final_values["SOC_BSS_34"],
        final_SOC_BSS_68=final_values["SOC_BSS_68"],
        final_SOC_BSS_103=final_values["SOC_BSS_103"],
        step_costs_json=json.dumps(step_costs),
        step_times_json=json.dumps(step_times),
        certified_global=certified_global,
        gurobi_status=status_code,
        mean_num_vars=stats.get("num_vars", np.nan),
        mean_num_bin_vars=stats.get("num_bin_vars", np.nan),
        mean_num_constrs=stats.get("num_constrs", np.nan),
        mean_num_qconstrs=stats.get("num_qconstrs", np.nan),
        mean_num_genconstrs=stats.get("num_genconstrs", np.nan),
        mean_model_runtime_s=stats.get("runtime", np.nan),
        mean_mip_gap=mip_gap_float,
        max_mip_gap=mip_gap_float,
        mean_model_status=status_code,
        min_model_sol_count=stats.get("sol_count", np.nan),
        failure_message=failure_message,
        experiment_tag="perfect_global_119",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 119-bus full-day Perfect-Global benchmark.")
    parser.add_argument("--scenarios", default="800,801,802,803,804,805,806,807,808,809")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--time-limit-s", type=float, default=3600)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step10_119_perfect_global_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    validate_actual_case(args.actual_case)

    print("Method: Perfect-Global")
    print("Scenarios:", scenario_ids)
    print("actual_case:", args.actual_case)
    print("time_limit_s:", args.time_limit_s)
    print("mip_gap:", args.mip_gap)

    if args.dry_run:
        for scenario_id in scenario_ids:
            actual = load_actual_scenario("119", scenario_id, actual_case=args.actual_case)
            solver_inputs = get_solver_inputs_119(actual)
            print(
                f"dry-run scenario={scenario_id}, "
                f"load={solver_inputs[0].shape}, load_Q={solver_inputs[1].shape}, "
                f"renewables={[item.shape for item in solver_inputs[2:]]}"
            )
        return

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        row = run_one_perfect_global_119(
            scenario_id=scenario_id,
            actual_case=args.actual_case,
            quiet_solver=not args.verbose_solver,
            time_limit_s=args.time_limit_s,
            mip_gap=args.mip_gap,
        )
        logger.add_row(row)
        print(
            f"Perfect-Global, scenario={scenario_id}, "
            f"completed={row.get('completed')}, "
            f"cost={row.get('cost')}, "
            f"time={row.get('solve_time_s')}, "
            f"gap={row.get('mean_mip_gap')}, "
            f"bin={row.get('mean_num_bin_vars')}, "
            f"fail={row.get('solver_fail_count')}"
        )

    path = logger.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()
