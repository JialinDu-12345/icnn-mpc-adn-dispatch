from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import (              
    DEFAULT_33,
    DEFAULT_BASE_SEEDS,
    METHOD_DIRS_33,
    SPLITS,
    validate_actual_case,
)
from experiments_v2.result_logger import ResultLogger, make_result_row              
from experiments_v2.run_33_mpc import collect_gurobi_model_stats              
from experiments_v2.scenario_protocol import (              
    get_solver_inputs_33,
    load_actual_scenario,
)


def load_perfect_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find Perfect-Global solver: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Perfect-Global solver from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.collect_gurobi_model_stats = collect_gurobi_model_stats
    return module


def build_perfect_solver_33():
    method_dir = METHOD_DIRS_33["perfect_global_clean"]
    module = load_perfect_module(
        module_name="perfect_global_clean_33",
        file_path=method_dir / "MPC_1.py",
    )

    list_r_x_pu, list_p_q_pu = module._get_network()
    base_load = np.asarray(list_p_q_pu, dtype=float)
    load_min = 0.6 * base_load[:, 0]
    load_max = 1.2 * base_load[:, 0]
    load_min_Q = 0.6 * base_load[:, 1]
    load_max_Q = 1.2 * base_load[:, 1]

    return module.Solve_MPC(
        load_min,
        load_max,
        list_r_x_pu,
        list_p_q_pu,
        load_min_Q,
        load_max_Q,
    )


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str, scenario_count: Optional[int]) -> List[int]:
    normalized = raw.strip().lower()
    if normalized == "all":
        scenario_ids = list(SPLITS["33"]["test_id"])
    elif normalized in {"val", "validation"}:
        scenario_ids = list(SPLITS["33"]["val"])
    else:
        scenario_ids = [int(item) for item in parse_csv_list(raw)]

    if scenario_count is not None:
        scenario_ids = scenario_ids[:scenario_count]
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


def run_one_perfect_global_33(
    *,
    scenario_id: int,
    actual_case: str = "id_actual",
    quiet_solver: bool = True,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = 1e-3,
    solver_customizer: Optional[Callable[[Any], None]] = None,
) -> dict[str, Any]:
    validate_actual_case(actual_case)

    system = "33"
    n_steps = int(DEFAULT_33["n_steps"])
    seed = int(DEFAULT_BASE_SEEDS[system] + int(scenario_id))
    split = split_for_scenario(system, scenario_id)

    solver = None
    solver_fail_count = 0
    failure_message = ""
    total_cost = np.nan
    step_costs: List[float] = []
    solve_time = np.nan
    final_P_DG_5 = np.nan
    final_SOC_BSS_9 = np.nan
    stats: dict[str, Any] = {}

    try:
        actual = load_actual_scenario(system, scenario_id, actual_case=actual_case)
        load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = get_solver_inputs_33(actual)
        solver = build_perfect_solver_33()
        if solver_customizer is not None:
            solver_customizer(solver)

        solver.time_limit_s = time_limit_s
        solver.mip_gap = mip_gap
        solver.output_flag = 0 if quiet_solver else 1

        solve_kwargs = dict(
            P_DG_5_init=float(DEFAULT_33["P_DG_5_init"]),
            SOC_BSS_9_init=float(DEFAULT_33["SOC_BSS_9_init"]),
            load_list=load_list,
            pv_11_avail_list=pv_11_avail_list,
            wind_26_avail_list=wind_26_avail_list,
            window_time=n_steps,
            load_list_Q=load_list_Q,
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

        P_DG_5_values = [float(value) for value in result.get("P_DG_5", [])]
        SOC_BSS_9_values = [float(value) for value in result.get("SOC_BSS_9", [])]
        if P_DG_5_values:
            final_P_DG_5 = P_DG_5_values[-1]
        if SOC_BSS_9_values:
            final_SOC_BSS_9 = SOC_BSS_9_values[-1]

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
        final_P_DG_5=final_P_DG_5,
        final_SOC_BSS_9=final_SOC_BSS_9,
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
        experiment_tag="perfect_global",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 33-bus full-day Perfect-Global benchmark.")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--time-limit-s", type=float, default=1800)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step7b_33_perfect_global_raw.csv")
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
        solver = build_perfect_solver_33()
        for scenario_id in scenario_ids:
            actual = load_actual_scenario("33", scenario_id, actual_case=args.actual_case)
            load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = get_solver_inputs_33(actual)
            print(
                f"dry-run scenario={scenario_id}, "
                f"load={load_list.shape}, load_Q={load_list_Q.shape}, "
                f"pv={pv_11_avail_list.shape}, wind={wind_26_avail_list.shape}, "
                f"solver={solver.__class__.__name__}"
            )
        return

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        row = run_one_perfect_global_33(
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
