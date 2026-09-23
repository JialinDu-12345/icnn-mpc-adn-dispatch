from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments_v2.config import (              
    DEFAULT_33,
    DEFAULT_BASE_SEEDS,
    FORECAST_CASES,
    METHOD_DIRS_33,
    RESULT_DIR,
    SPLITS,
    validate_actual_case,
    validate_forecast_case,
)
from experiments_v2.result_logger import ResultLogger, make_result_row              
from experiments_v2.scenario_protocol import (              
    ArrayDict,
    get_solver_inputs_33,
    load_actual_scenario,
    make_forecaster,
)


TASK_PRESETS = {
    "mpc3": {
        "method": "MPC-3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "mpc5": {
        "method": "MPC-5",
        "solver_kind": "pure_mpc_clean",
        "horizon": 5,
        "gamma_Q": None,
    },
    "mpc7": {
        "method": "MPC-7",
        "solver_kind": "pure_mpc_clean",
        "horizon": 7,
        "gamma_Q": None,
    },
    "proposed3": {
        "method": "Proposed-H3",
        "solver_kind": "proposed",
        "horizon": 3,
        "gamma_Q": "auto",
    },
}


def load_selected_gamma(system: str) -> float:
    path = RESULT_DIR / f"selected_gamma_{system}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run gamma validation first.")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gamma_q = data.get("selected_gamma_Q")
    if gamma_q is None:
        raise RuntimeError(
            f"selected_gamma_Q is None in {path}. "
            "Gamma validation has not produced a valid selected value."
        )
    return float(gamma_q)


def resolve_gamma_Q(gamma_Q: Any, system: str = "33") -> Optional[float]:
    if gamma_Q is None:
        return None
    if isinstance(gamma_Q, str):
        value = gamma_Q.strip().lower()
        if not value:
            return None
        if value == "auto":
            return load_selected_gamma(system)
        return float(value)
    return float(gamma_Q)


def finite_values(values: Iterable[float]) -> List[float]:
    clean: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return clean


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def max_or_nan(values: Iterable[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(max(clean))


def min_or_nan(values: Iterable[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(min(clean))


def _model_attr(model: Any, attr_name: str, default: Any = np.nan) -> Any:
    try:
        return getattr(model, attr_name)
    except Exception:
        return default


def collect_gurobi_model_stats(model: Any) -> Dict[str, Any]:
                                                                 

    sol_count = _model_attr(model, "SolCount", 0)
    try:
        sol_count = int(sol_count)
    except (TypeError, ValueError):
        sol_count = 0

    try:
        num_bin_vars = sum(1 for var in model.getVars() if getattr(var, "VType", "") == "B")
    except Exception:
        num_bin_vars = np.nan

    mip_gap = np.nan
    try:
        if bool(_model_attr(model, "IsMIP", False)) and sol_count > 0:
            mip_gap = float(_model_attr(model, "MIPGap", np.nan))
    except Exception:
        mip_gap = np.nan

    return {
        "num_vars": _model_attr(model, "NumVars", np.nan),
        "num_bin_vars": num_bin_vars,
        "num_constrs": _model_attr(model, "NumConstrs", np.nan),
        "num_qconstrs": _model_attr(model, "NumQConstrs", np.nan),
        "num_genconstrs": _model_attr(model, "NumGenConstrs", np.nan),
        "runtime": _model_attr(model, "Runtime", np.nan),
        "mip_gap": mip_gap,
        "status": _model_attr(model, "Status", np.nan),
        "sol_count": sol_count,
    }


def _safe_dict_value(dict_value: Dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(dict_value[key])
    except Exception:
        return float(default)


def _extract_indexed_series(dict_value: Dict[str, Any], prefix: str) -> List[float]:
    values: List[Tuple[int, float]] = []
    marker = f"{prefix}["
    for key, value in dict_value.items():
        if not str(key).startswith(marker) or not str(key).endswith("]"):
            continue
        index_text = str(key)[len(marker) : -1]
        if "," in index_text:
            continue
        try:
            index = int(index_text)
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        values.append((index, numeric))
    return [value for _index, value in sorted(values)]


def extract_dispatch_33(dict_value: Dict[str, Any]) -> Dict[str, Any]:
                                                                                  

    p_ch = _safe_dict_value(dict_value, "P_BSS_ch_9[0]", 0.0)
    p_dch = _safe_dict_value(dict_value, "P_BSS_dch_9[0]", 0.0)
    p_bss = -1.02 * p_ch + 0.98 * p_dch
                                                                       
                                                                           
                                     
    p_grid_signed = _safe_dict_value(dict_value, "P_ij[0,0,0]")

    p_dg_window = _extract_indexed_series(dict_value, "P_DG_5")
    soc_window = _extract_indexed_series(dict_value, "SOC_BSS_9")
    p_ch_window = _extract_indexed_series(dict_value, "P_BSS_ch_9")
    p_dch_window = _extract_indexed_series(dict_value, "P_BSS_dch_9")
    cost_hour_window = _extract_indexed_series(dict_value, "cost_hour")
    p_balance_window = _extract_indexed_series(dict_value, "P_balance")
    loss_window = _extract_indexed_series(dict_value, "Loss")
    p_bss_window = [
        -1.02 * ch + 0.98 * dch for ch, dch in zip(p_ch_window, p_dch_window)
    ]

    return {
        "P_DG_5": _safe_dict_value(dict_value, "P_DG_5[0]"),
        "SOC_BSS_9": _safe_dict_value(dict_value, "SOC_BSS_9[0]"),
        "P_BSS": p_bss,
        "P_BSS_ch_9": p_ch,
        "P_BSS_dch_9": p_dch,
        "P_grid": p_grid_signed,
        "P_balance": _safe_dict_value(dict_value, "P_balance[0]"),
        "Loss": _safe_dict_value(dict_value, "Loss[0]"),
        "window_P_DG_5": p_dg_window,
        "window_SOC_BSS_9": soc_window,
        "window_P_BSS": p_bss_window,
        "window_P_BSS_ch_9": p_ch_window,
        "window_P_BSS_dch_9": p_dch_window,
        "window_cost_hour": cost_hour_window,
        "window_P_balance": p_balance_window,
        "window_Loss": loss_window,
        "terminal_P_DG_5": p_dg_window[-1] if p_dg_window else np.nan,
        "terminal_SOC_BSS_9": soc_window[-1] if soc_window else np.nan,
        "Q_f": _safe_dict_value(dict_value, "Q_f", np.nan),
        "Q_critic_1": _safe_dict_value(dict_value, "z_2_1", np.nan),
        "Q_critic_2": _safe_dict_value(dict_value, "z_2_2", np.nan),
        "terminal_actor_5": _safe_dict_value(dict_value, "Actor_5_st", np.nan),
        "terminal_actor_9": _safe_dict_value(dict_value, "Actor_9_st", np.nan),
        "terminal_actor_7": _safe_dict_value(dict_value, "Actor_7_st", np.nan),
        "terminal_actor_11": _safe_dict_value(dict_value, "Actor_11_st", np.nan),
        "terminal_actor_26": _safe_dict_value(dict_value, "Actor_26_st", np.nan),
    }


def _read_legacy_definitions(file_path: Path) -> str:
    source = file_path.read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)
    cut_index = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("env = PowerSystemEnv()") or stripped.startswith("objlist = []"):
            cut_index = idx
            break

    skipped_import_prefixes = (
        "import matplotlib",
        "from SAC_largesystem import",
        "import rl_utils",
        "import pandas",
    )
    safe_lines = [
        line
        for line in lines[:cut_index]
        if not line.lstrip().startswith(skipped_import_prefixes)
    ]
    safe_source = "".join(safe_lines).replace("\r\n", "\n")
    safe_source = safe_source.replace("model.objVal", "model.ObjVal")
    safe_source = safe_source.replace("model.objval", "model.ObjVal")
    safe_source = safe_source.replace(
        "            model.optimize()\n\n",
        "            model.optimize()\n"
        "            self.last_model_stats = collect_gurobi_model_stats(model)\n\n",
    )
    safe_source = safe_source.replace(
        "            print('obj=', model.ObjVal)",
        "            if model.SolCount == 0:\n"
        "                raise RuntimeError(f'Gurobi produced no solution: status={model.Status}, sol_count={model.SolCount}')\n"
        "            print('obj=', model.ObjVal)",
    )
    safe_source = safe_source.replace(
        "        except AttributeError:\n            print('Encountered an attribute error')",
        "        except AttributeError:\n            raise",
    )
    safe_source = safe_source.replace(
        "        except GurobiError as e:\n            print('Error code ' + str(e.errno) + ':' + str(e))",
        "        except GurobiError as e:\n            raise",
    )
    safe_source = safe_source.replace(
        "        return obj, dict_value['P_DG_5[0]'], dict_value['SOC_BSS_9[0]'], dict_value['cost_hour[0]'], model.Runtime",
        "        self.last_dispatch = extract_dispatch_33(dict_value)\n"
        "        return obj, dict_value['P_DG_5[0]'], dict_value['SOC_BSS_9[0]'], dict_value['cost_hour[0]'], model.Runtime",
    )
    return safe_source


def load_legacy_mpc1_definitions(module_name: str, file_path: Path) -> ModuleType:
                                                                                 
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find legacy module file: {file_path}")

    source = _read_legacy_definitions(file_path)
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    module.collect_gurobi_model_stats = collect_gurobi_model_stats
    module.extract_dispatch_33 = extract_dispatch_33
    sys.modules[module_name] = module

    old_cwd = Path.cwd()
    sys.path.insert(0, str(file_path.parent))
    try:
        os.chdir(file_path.parent)
        exec(compile(source, str(file_path), "exec"), module.__dict__)
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)

    module.dict_plot = {
        "P_DG_5": [],
        "SOC_BSS_9": [],
        "Loss": [],
        "P_BSS": [],
        "P_G": [],
    }
    return module


def _legacy_bounds_and_network(module: ModuleType):
    np_state = module.np.random.get_state()
    try:
        module.np.random.seed(0)
        (
            _load_list,
            _pv_11_avail_list,
            _wind_26_avail_list,
            load_min,
            load_max,
            _load_list_Q,
            load_min_Q,
            load_max_Q,
        ) = module._get_scenario()
    finally:
        module.np.random.set_state(np_state)

    list_r_x_pu, list_p_q_pu = module._get_network()
    return load_min, load_max, list_r_x_pu, list_p_q_pu, load_min_Q, load_max_Q


def _legacy_q_matrices(module: ModuleType, method_dir: Path):
    matrix_keys = ["As.0", "Ws.0", "bs.0", "As.1", "Ws.1", "bs.1"]
    q_matrices = []
    for filename in ("critic_1_model.pth", "critic_2_model.pth"):
        state_dict = module.torch.load(str(method_dir / filename), map_location="cpu")
        q_matrices.append(
            [state_dict[key].cpu().detach().numpy() for key in matrix_keys]
        )
    return q_matrices[0], q_matrices[1]


def build_solver_33(
    solver_kind: str,
    gamma_Q: Optional[float] = None,
):
                                                                        
    if solver_kind not in METHOD_DIRS_33:
        raise ValueError(f"Unknown solver_kind: {solver_kind}")
    if solver_kind in {"proposed", "std_nn_mpc"} and gamma_Q is None:
        raise ValueError(
            f"gamma_Q must be provided for {solver_kind} solver. "
            "Use gamma_Q='auto' after validation."
        )

    method_dir = METHOD_DIRS_33[solver_kind]
    module = load_legacy_mpc1_definitions(
        module_name=f"legacy_33_{solver_kind}",
        file_path=method_dir / "MPC_1.py",
    )

    if solver_kind in {"pure_mpc_clean", "std_nn_mpc"}:
        q_1_mat, q_2_mat = None, None
    else:
        q_1_mat, q_2_mat = _legacy_q_matrices(module, method_dir)
    (
        load_min,
        load_max,
        list_r_x_pu,
        list_p_q_pu,
        load_min_Q,
        load_max_Q,
    ) = _legacy_bounds_and_network(module)

    solver = module.Solve_MPC(
        q_1_mat,
        q_2_mat,
        load_min,
        load_max,
        list_r_x_pu,
        list_p_q_pu,
        load_min_Q,
        load_max_Q,
    )

    if solver_kind in {"pure_mpc", "pure_mpc_clean"}:
        solver.scale_Q = 0.0
    elif solver_kind in {"proposed", "std_nn_mpc"}:
        solver.scale_Q = resolve_gamma_Q(gamma_Q)

    return solver


def call_sol_pro_33(
    solver: Any,
    current_time: int,
    solver_window: int,
    P_DG_5_init: float,
    SOC_BSS_9_init: float,
    forecast: ArrayDict,
    remainder: bool = False,
):
    load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = get_solver_inputs_33(
        forecast
    )
    solve_func = solver.sol_pro_remainder if remainder else solver.sol_pro
    return solve_func(
        current_time,
        P_DG_5_init,
        SOC_BSS_9_init,
        load_list,
        pv_11_avail_list,
        wind_26_avail_list,
        solver_window,
        load_list_Q,
    )


def _run_solver_call(quiet_solver: bool, *args: Any, **kwargs: Any):
    if quiet_solver:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                return call_sol_pro_33(*args, **kwargs)
    return call_sol_pro_33(*args, **kwargs)


def _split_for_scenario(system: str, scenario_id: int) -> str:
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


def run_one_scenario_33(
    *,
    method: str,
    solver_kind: str,
    scenario_id: int,
    forecast_case: str,
    horizon: int,
    gamma_Q: Optional[float] = None,
    actual_case: str = "id_actual",
    split: Optional[str] = None,
    max_steps: Optional[int] = None,
    include_tail: bool = True,
    quiet_solver: bool = True,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    solver_customizer: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    validate_forecast_case(forecast_case)
    validate_actual_case(actual_case)

    system = "33"
    seed = DEFAULT_BASE_SEEDS[system] + int(scenario_id)
    actual = load_actual_scenario(system, scenario_id, actual_case=actual_case)
    forecaster = make_forecaster(
        system=system,
        forecast_case_config=FORECAST_CASES[forecast_case],
        seed=seed,
    )
    solver = build_solver_33(solver_kind=solver_kind, gamma_Q=gamma_Q)
    if solver_customizer is not None:
        solver_customizer(solver)
    if time_limit_s is not None and hasattr(solver, "time_limit_s"):
        solver.time_limit_s = float(time_limit_s)
    if mip_gap is not None and hasattr(solver, "mip_gap"):
        solver.mip_gap = float(mip_gap)
    gamma_Q_value = resolve_gamma_Q(gamma_Q)

    n_steps = DEFAULT_33["n_steps"]
    solver_window = int(horizon) + 1
    if solver_window < 2:
        raise ValueError("horizon must be at least 1.")
    if solver_window > n_steps:
        raise ValueError(f"horizon={horizon} is too long for {n_steps} steps.")

    P_DG_5_init = float(DEFAULT_33["P_DG_5_init"])
    SOC_BSS_9_init = float(DEFAULT_33["SOC_BSS_9_init"])

    total_cost = 0.0
    total_solve_time = 0.0
    max_step_time = 0.0
    solver_fail_count = 0
    timeout_count = 0
    step_costs: List[float] = []
    step_times: List[float] = []
    step_num_vars: List[float] = []
    step_num_bin_vars: List[float] = []
    step_num_constrs: List[float] = []
    step_num_qconstrs: List[float] = []
    step_num_genconstrs: List[float] = []
    step_model_runtimes: List[float] = []
    step_mip_gaps: List[float] = []
    step_std_nn_binary_counts: List[float] = []
    step_model_statuses: List[float] = []
    step_model_sol_counts: List[float] = []
    step_P_DG_5: List[float] = []
    step_SOC_BSS_9: List[float] = []
    step_P_BSS: List[float] = []
    step_P_grid: List[float] = []
    step_loss: List[float] = []
    step_load_total: List[float] = []
    step_pv_11: List[float] = []
    step_wind_26: List[float] = []
    failure_message = ""

    def reached_step_limit() -> bool:
        return max_steps is not None and len(step_costs) >= max_steps

    def append_model_stats() -> None:
        stats = getattr(solver, "last_model_stats", {}) or {}
        step_num_vars.append(stats.get("num_vars", np.nan))
        step_num_bin_vars.append(stats.get("num_bin_vars", np.nan))
        step_num_constrs.append(stats.get("num_constrs", np.nan))
        step_num_qconstrs.append(stats.get("num_qconstrs", np.nan))
        step_num_genconstrs.append(stats.get("num_genconstrs", np.nan))
        step_model_runtimes.append(stats.get("runtime", np.nan))
        step_mip_gaps.append(stats.get("mip_gap", np.nan))
        step_std_nn_binary_counts.append(getattr(solver, "std_nn_binary_count_last", np.nan))
        step_model_statuses.append(stats.get("status", np.nan))
        step_model_sol_counts.append(stats.get("sol_count", np.nan))

    def append_dispatch_trajectory(t: int) -> None:
        dispatch = getattr(solver, "last_dispatch", {}) or {}
        step_P_DG_5.append(float(dispatch.get("P_DG_5", np.nan)))
        step_SOC_BSS_9.append(float(dispatch.get("SOC_BSS_9", np.nan)))
        step_P_BSS.append(float(dispatch.get("P_BSS", np.nan)))
        step_P_grid.append(float(dispatch.get("P_grid", np.nan)))
        step_loss.append(float(dispatch.get("Loss", np.nan)))
        step_load_total.append(float(np.sum(actual["load"][t])))
        step_pv_11.append(float(actual["pv_11"][t]))
        step_wind_26.append(float(actual["wind_26"][t]))

    regular_stop = n_steps - solver_window + 1
    current_time = 0

    try:
        for current_time in range(regular_stop):
            if reached_step_limit():
                break

            forecast = forecaster.make_forecast_arrays(
                actual=actual,
                current_time=current_time,
                window_time=solver_window,
            )
            (
                _obj,
                new_P_DG_5_init,
                new_SOC_BSS_9_init,
                cost_hour_now,
                solve_time,
            ) = _run_solver_call(
                quiet_solver,
                solver,
                current_time,
                solver_window,
                P_DG_5_init,
                SOC_BSS_9_init,
                forecast,
                remainder=False,
            )
            append_model_stats()
            append_dispatch_trajectory(current_time)

            cost_hour_now = float(cost_hour_now)
            solve_time = float(solve_time)
            total_cost += cost_hour_now
            total_solve_time += solve_time
            max_step_time = max(max_step_time, solve_time)
            step_costs.append(cost_hour_now)
            step_times.append(solve_time)
            P_DG_5_init = float(new_P_DG_5_init)
            SOC_BSS_9_init = float(new_SOC_BSS_9_init)

        if include_tail and not reached_step_limit():
            for current_time in range(max(regular_stop, 0), n_steps):
                if reached_step_limit():
                    break

                tail_window = n_steps - current_time
                forecast = forecaster.make_forecast_arrays(
                    actual=actual,
                    current_time=current_time,
                    window_time=tail_window,
                )
                (
                    _obj,
                    new_P_DG_5_init,
                    new_SOC_BSS_9_init,
                    cost_hour_now,
                    solve_time,
                ) = _run_solver_call(
                    quiet_solver,
                    solver,
                    current_time,
                    tail_window,
                    P_DG_5_init,
                    SOC_BSS_9_init,
                    forecast,
                    remainder=True,
                )
                append_model_stats()
                append_dispatch_trajectory(current_time)

                cost_hour_now = float(cost_hour_now)
                solve_time = float(solve_time)
                total_cost += cost_hour_now
                total_solve_time += solve_time
                max_step_time = max(max_step_time, solve_time)
                step_costs.append(cost_hour_now)
                step_times.append(solve_time)
                P_DG_5_init = float(new_P_DG_5_init)
                SOC_BSS_9_init = float(new_SOC_BSS_9_init)

    except Exception as exc:
        solver_fail_count += 1
        failure_message = (
            f"method={method}, scenario={scenario_id}, forecast_case={forecast_case}, "
            f"actual_case={actual_case}, horizon={horizon}, t={current_time}: {exc}"
        )
        print(f"[Failed] {failure_message}")
        traceback.print_exc()

    executed_steps = len(step_costs)
    if (
        include_tail
        and executed_steps < n_steps
        and solver_fail_count == 0
        and max_steps is None
    ):
        timeout_count = 1

    completed = executed_steps == n_steps and solver_fail_count == 0
    if completed:
        incomplete_reason = ""
    elif solver_fail_count > 0:
        incomplete_reason = "solver_failure"
    elif max_steps is not None:
        incomplete_reason = "max_steps_debug"
    elif not include_tail:
        incomplete_reason = "tail_disabled"
    elif timeout_count > 0:
        incomplete_reason = "timeout_or_incomplete"
    else:
        incomplete_reason = "incomplete_unknown"

    return make_result_row(
        system=system,
        method=method,
        scenario_id=scenario_id,
        split=split or _split_for_scenario(system, scenario_id),
        actual_case=actual_case,
        forecast_case=forecast_case,
        horizon=horizon,
        gamma_Q=gamma_Q_value,
        cost=total_cost,
        perfect_cost=None,
        gap_percent=None,
        voltage_violation=None,
        solve_time_s=total_solve_time,
        avg_step_time_s=total_solve_time / max(executed_steps, 1),
        max_step_time_s=max_step_time,
        solver_fail_count=solver_fail_count,
        timeout_count=timeout_count,
        completed=completed,
        incomplete_reason=incomplete_reason,
        seed=seed,
        solver_kind=solver_kind,
        solver_window=solver_window,
        executed_steps=executed_steps,
        include_tail=include_tail,
        max_steps=max_steps,
        final_P_DG_5=P_DG_5_init,
        final_SOC_BSS_9=SOC_BSS_9_init,
        step_costs_json=json.dumps(step_costs),
        step_times_json=json.dumps(step_times),
        step_P_DG_5_json=json.dumps(step_P_DG_5),
        step_SOC_BSS_9_json=json.dumps(step_SOC_BSS_9),
        step_P_BSS_json=json.dumps(step_P_BSS),
        step_P_grid_json=json.dumps(step_P_grid),
        step_loss_json=json.dumps(step_loss),
        step_load_total_json=json.dumps(step_load_total),
        step_pv_11_json=json.dumps(step_pv_11),
        step_wind_26_json=json.dumps(step_wind_26),
        mean_num_vars=mean_or_nan(step_num_vars),
        mean_num_bin_vars=mean_or_nan(step_num_bin_vars),
        mean_num_constrs=mean_or_nan(step_num_constrs),
        mean_num_qconstrs=mean_or_nan(step_num_qconstrs),
        mean_num_genconstrs=mean_or_nan(step_num_genconstrs),
        mean_model_runtime_s=mean_or_nan(step_model_runtimes),
        mean_mip_gap=mean_or_nan(step_mip_gaps),
        max_mip_gap=max_or_nan(step_mip_gaps),
        mean_std_nn_binary_count=mean_or_nan(step_std_nn_binary_counts),
        step_model_statuses_json=json.dumps(step_model_statuses),
        step_model_sol_counts_json=json.dumps(step_model_sol_counts),
        mean_model_status=mean_or_nan(step_model_statuses),
        min_model_sol_count=min_or_nan(step_model_sol_counts),
        num_optimal_steps=sum(1 for status in finite_values(step_model_statuses) if int(status) == 2),
        num_timelimit_steps=sum(1 for status in finite_values(step_model_statuses) if int(status) == 9),
        failure_message=failure_message,
    )


def parse_scenario_ids(raw: Optional[str], scenario_count: int) -> List[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return SPLITS["33"]["test_id"][:scenario_count]


def parse_tasks(raw: str) -> List[Dict[str, Any]]:
    tasks = []
    for task_name in [item.strip().lower() for item in raw.split(",") if item.strip()]:
        if task_name not in TASK_PRESETS:
            raise ValueError(
                f"Unknown task '{task_name}'. Expected one of {sorted(TASK_PRESETS)}."
            )
        tasks.append(dict(TASK_PRESETS[task_name]))
    return tasks


def run_batch(args: argparse.Namespace) -> Path:
    scenario_ids = parse_scenario_ids(args.scenarios, args.scenario_count)
    tasks = parse_tasks(args.tasks)

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for task in tasks:
            print(
                f"Running {task['method']} scenario={scenario_id} "
                f"forecast={args.forecast_case} actual={args.actual_case}"
            )
            row = run_one_scenario_33(
                method=task["method"],
                solver_kind=task["solver_kind"],
                scenario_id=scenario_id,
                forecast_case=args.forecast_case,
                actual_case=args.actual_case,
                horizon=task["horizon"],
                gamma_Q=args.gamma_Q
                if task["solver_kind"] == "proposed" and args.gamma_Q is not None
                else task["gamma_Q"],
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
                time_limit_s=args.time_limit_s,
                mip_gap=args.mip_gap,
            )
            logger.add_row(row)
            print(
                {
                    "method": row["method"],
                    "scenario_id": row["scenario_id"],
                    "cost": row["cost"],
                    "solve_time_s": row["solve_time_s"],
                    "executed_steps": row["executed_steps"],
                    "solver_fail_count": row["solver_fail_count"],
                }
            )

    return logger.save()


def dry_run(args: argparse.Namespace) -> None:
    scenario_ids = parse_scenario_ids(args.scenarios, args.scenario_count)
    tasks = parse_tasks(args.tasks)
    print("Dry-run plan")
    print("scenarios:", scenario_ids)
    print("tasks:", tasks)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("max_steps:", args.max_steps)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 33-bus pure MPC and MPC-ICNN through experiments_v2 protocols."
    )
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario IDs.")
    parser.add_argument("--scenario-count", type=int, default=2)
    parser.add_argument(
        "--tasks",
        default="mpc3,mpc5,mpc7,proposed3",
        help="Comma-separated task names: mpc3,mpc5,mpc7,proposed3.",
    )
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default=None, help="Float value or 'auto'.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="step2_33_mpc_smoke_test.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    random.seed(0)
    np.random.seed(0)

    if args.dry_run:
        dry_run(args)
        return

    path = run_batch(args)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
