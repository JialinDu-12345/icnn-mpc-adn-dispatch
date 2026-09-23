from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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
from experiments_v2.global_optimum.run_33_perfect_global import (              
    build_perfect_solver_33,
)
from experiments_v2.run_33_mpc import (              
    build_solver_33,
    call_sol_pro_33,
    load_legacy_mpc1_definitions,
)
from experiments_v2.scenario_protocol import (              
    ArrayDict,
    get_solver_inputs_33,
    load_actual_scenario,
    make_forecaster,
)


HORIZON = 3
SOLVER_WINDOW = HORIZON + 1
N_STEPS = int(DEFAULT_33["n_steps"])
REGULAR_STOP = N_STEPS - SOLVER_WINDOW + 1
STATE_DIM = 69
ACTION_DIM = 5
Z_DIM = STATE_DIM + ACTION_DIM
TERMINAL_SOURCES = ["MPC-3", "Proposed-H3", "Perfect-Global"]


@dataclass(frozen=True)
class TerminalCandidate:
    scenario_id: int
    current_time: int
    terminal_start: int
    terminal_pre_step: int
    source: str
    terminal_P_DG_5: float
    terminal_SOC_BSS_9: float
    z_state: np.ndarray
    forecast_terminal_load_sum: float
    actual_terminal_load_sum: float
    source_completed: bool = True
    source_failure_message: str = ""
    source_cost_so_far: float = math.nan


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
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


def parse_sources(raw: str) -> List[str]:
    aliases = {
        "mpc3": "MPC-3",
        "mpc-3": "MPC-3",
        "proposed": "Proposed-H3",
        "proposed-h3": "Proposed-H3",
        "perfect": "Perfect-Global",
        "perfect-global": "Perfect-Global",
        "perfect_global": "Perfect-Global",
        "all": "all",
    }
    selected: List[str] = []
    seen = set()
    for item in parse_csv_list(raw):
        value = aliases.get(item.lower(), item)
        if value == "all":
            values = TERMINAL_SOURCES
        else:
            values = [value]
        for source in values:
            if source not in TERMINAL_SOURCES:
                raise ValueError(f"Unknown terminal source '{item}'. Expected {TERMINAL_SOURCES}.")
            if source in seen:
                continue
            selected.append(source)
            seen.add(source)
    return selected


def selected_current_times(
    *,
    start_current_time: int,
    terminal_step_stride: int,
    max_terminal_steps: Optional[int],
) -> List[int]:
    if start_current_time < 0 or start_current_time >= REGULAR_STOP:
        raise ValueError(f"start_current_time must be in [0, {REGULAR_STOP - 1}].")
    if terminal_step_stride <= 0:
        raise ValueError("terminal_step_stride must be positive.")

    times = list(range(int(start_current_time), REGULAR_STOP, int(terminal_step_stride)))
    if max_terminal_steps is not None:
        times = times[: int(max_terminal_steps)]
    return times


def finite_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def finite(values: Iterable[float]) -> List[float]:
    clean = []
    for value in values:
        numeric = finite_float(value)
        if math.isfinite(numeric):
            clean.append(numeric)
    return clean


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return mean(clean) if clean else math.nan


def std_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    return stdev(clean) if len(clean) >= 2 else math.nan


def rmse_or_nan(errors: Iterable[float]) -> float:
    clean = finite(errors)
    if not clean:
        return math.nan
    return math.sqrt(mean([value * value for value in clean]))


def ci95(values: Iterable[float]) -> float:
    clean = finite(values)
    if len(clean) < 2:
        return math.nan
    return 1.96 * stdev(clean) / math.sqrt(len(clean))


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return math.nan
    x_mean = mean(xs)
    y_mean = mean(ys)
    dx = [x - x_mean for x in xs]
    dy = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0:
        return math.nan
    return sum(x * y for x, y in zip(dx, dy)) / denom


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    clean = [(finite_float(x), finite_float(y)) for x, y in zip(xs, ys)]
    clean = [(x, y) for x, y in clean if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 2:
        return math.nan
    x_values, y_values = zip(*clean)
    return pearson(rankdata(x_values), rankdata(y_values))


def affine_fit_predict(xs: Sequence[float], ys: Sequence[float]) -> Tuple[List[float], float, float]:
    clean = [(finite_float(x), finite_float(y)) for x, y in zip(xs, ys)]
    clean = [(x, y) for x, y in clean if math.isfinite(x) and math.isfinite(y)]
    if not clean:
        return [math.nan for _ in xs], math.nan, math.nan
    x_values, y_values = zip(*clean)
    if len(clean) < 2:
        slope = 0.0
        intercept = float(y_values[0])
    else:
        x_mean = mean(x_values)
        y_mean = mean(y_values)
        denom = sum((x - x_mean) ** 2 for x in x_values)
        slope = 0.0 if abs(denom) <= 1e-12 else sum(
            (x - x_mean) * (y - y_mean) for x, y in clean
        ) / denom
        intercept = y_mean - slope * x_mean
    return [
        slope * finite_float(x) + intercept if math.isfinite(finite_float(x)) else math.nan
        for x in xs
    ], float(slope), float(intercept)


def pairwise_ranking_accuracy(rows: List[Dict[str, Any]]) -> Tuple[float, int]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["scenario_id"]), int(row["current_time"]))].append(row)

    correct = 0
    total = 0
    for group in grouped.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                oracle_diff = finite_float(a.get("oracle_tail_cost")) - finite_float(
                    b.get("oracle_tail_cost")
                )
                icnn_diff = finite_float(a.get("icnn_terminal_cost")) - finite_float(
                    b.get("icnn_terminal_cost")
                )
                if not math.isfinite(oracle_diff) or not math.isfinite(icnn_diff):
                    continue
                if abs(oracle_diff) <= 1e-9 or abs(icnn_diff) <= 1e-9:
                    continue
                total += 1
                if oracle_diff * icnn_diff > 0:
                    correct += 1
    if total == 0:
        return math.nan, 0
    return 100.0 * correct / total, total


def shifted_tail_array(array: np.ndarray, start: int) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    out = np.zeros_like(arr)
    tail = arr[start:]
    out[: len(tail)] = tail
    if len(tail) < len(out):
        out[len(tail) :] = arr[-1]
    return out


def build_tail_actual(actual: ArrayDict, terminal_start: int) -> ArrayDict:
    return {key: shifted_tail_array(value, terminal_start) for key, value in actual.items()}


def build_shifted_prices(base_prices: np.ndarray, terminal_start: int) -> np.ndarray:
    prices = np.asarray(base_prices, dtype=float)
    tail = prices[terminal_start:]
    if len(tail) == 0:
        return prices.copy()
    out = np.zeros_like(prices)
    out[: len(tail)] = tail
    if len(tail) < len(out):
        out[len(tail) :] = tail[-1]
    return out


def run_solver_call(quiet_solver: bool, *args: Any, **kwargs: Any):
    if quiet_solver:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                return call_sol_pro_33(*args, **kwargs)
    return call_sol_pro_33(*args, **kwargs)


def q_matrices() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    method_dir = METHOD_DIRS_33["proposed"]
    module = load_legacy_mpc1_definitions(
        module_name="legacy_33_icnn_value_diagnostic",
        file_path=method_dir / "MPC_1.py",
    )
    keys = ["As.0", "Ws.0", "bs.0", "As.1", "Ws.1", "bs.1"]
    matrices = []
    for filename in ("critic_1_model.pth", "critic_2_model.pth"):
        state_dict = module.torch.load(str(method_dir / filename), map_location="cpu")
        matrices.append([np.asarray(state_dict[key].cpu().detach().numpy(), dtype=float) for key in keys])
    return matrices[0], matrices[1]


def evaluate_q_numpy(q_mat: List[np.ndarray], z0: np.ndarray) -> float:
    a0, w0, b0, a1, w1, b1 = q_mat
    z0_2d = np.asarray(z0, dtype=float).reshape(1, -1)
    hidden = z0_2d @ a0.T + z0_2d @ w0.T + b0
    relu = np.maximum(hidden, 0.0)
    out = z0_2d @ a1.T + relu @ w1.T + b1
    return float(np.asarray(out).reshape(-1)[0])


def sample_icnn_q(
    q1: List[np.ndarray],
    q2: List[np.ndarray],
    z_state: np.ndarray,
    *,
    n_samples: int,
    seed: int,
) -> Tuple[float, List[float], str]:
    rng = np.random.default_rng(seed)
    action_candidates = [
        np.zeros(ACTION_DIM),
        np.ones(ACTION_DIM),
        -np.ones(ACTION_DIM),
    ]
    if n_samples > len(action_candidates):
        action_candidates.extend(
            rng.uniform(-1.0, 1.0, size=(n_samples - len(action_candidates), ACTION_DIM))
        )

    best_q = -math.inf
    best_action = np.zeros(ACTION_DIM)
    for action in action_candidates:
        z0 = np.concatenate([z_state, np.asarray(action, dtype=float)])
        q_value = min(evaluate_q_numpy(q1, z0), evaluate_q_numpy(q2, z0))
        if q_value > best_q:
            best_q = q_value
            best_action = np.asarray(action, dtype=float)
    return float(best_q), [float(value) for value in best_action], "sampled"


def exact_icnn_q(
    q1: List[np.ndarray],
    q2: List[np.ndarray],
    z_state: np.ndarray,
    *,
    time_limit_s: Optional[float],
) -> Tuple[float, List[float], str]:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("ICNNTerminalValue")
    model.setParam("OutputFlag", 0)
    if time_limit_s is not None:
        model.setParam(GRB.Param.TimeLimit, float(time_limit_s))

    actor = model.addMVar(shape=(ACTION_DIM,), lb=-1.0, ub=1.0, name="actor")
    q_vars = []
    for critic_idx, q_mat in enumerate((q1, q2), start=1):
        a0, w0, b0, a1, w1, b1 = q_mat
        z1 = model.addMVar(shape=(a0.shape[0],), lb=-GRB.INFINITY, name=f"z1_{critic_idx}")
        relu = model.addMVar(shape=(a0.shape[0],), lb=0.0, name=f"relu_{critic_idx}")
        q_out = model.addVar(lb=-GRB.INFINITY, name=f"q_{critic_idx}")

        for j in range(a0.shape[0]):
            const_part = float(np.dot(z_state, a0[j, :STATE_DIM] + w0[j, :STATE_DIM]) + b0[j])
            expr = const_part + gp.quicksum(
                float(a0[j, STATE_DIM + k] + w0[j, STATE_DIM + k]) * actor[k]
                for k in range(ACTION_DIM)
            )
            model.addConstr(z1[j] == expr)
            model.addConstr(relu[j] == gp.max_(z1[j], 0.0))

        const_out = float(np.dot(z_state, a1.reshape(-1)[:STATE_DIM]) + np.asarray(b1).reshape(-1)[0])
        expr_out = const_out + gp.quicksum(
            float(a1.reshape(-1)[STATE_DIM + k]) * actor[k] for k in range(ACTION_DIM)
        )
        expr_out += gp.quicksum(float(w1.reshape(-1)[j]) * relu[j] for j in range(a0.shape[0]))
        model.addConstr(q_out == expr_out)
        q_vars.append(q_out)

    q_terminal = model.addVar(lb=-GRB.INFINITY, name="q_terminal")
    model.addConstr(q_terminal <= q_vars[0])
    model.addConstr(q_terminal <= q_vars[1])
    model.setObjective(q_terminal, GRB.MAXIMIZE)
    model.optimize()

    if int(getattr(model, "SolCount", 0) or 0) <= 0:
        raise RuntimeError(f"ICNN actor optimization failed: status={model.Status}")
    return (
        float(q_terminal.X),
        [float(actor[k].X) for k in range(ACTION_DIM)],
        "exact",
    )


def evaluate_icnn_terminal(
    q1: List[np.ndarray],
    q2: List[np.ndarray],
    z_state: np.ndarray,
    *,
    mode: str,
    n_samples: int,
    seed: int,
    time_limit_s: Optional[float],
) -> Tuple[float, float, List[float], str]:
    if mode == "exact":
        q_value, action, status = exact_icnn_q(q1, q2, z_state, time_limit_s=time_limit_s)
    elif mode == "sampled":
        q_value, action, status = sample_icnn_q(q1, q2, z_state, n_samples=n_samples, seed=seed)
    else:
        raise ValueError("mode must be 'exact' or 'sampled'.")
    return q_value, -100.0 * q_value, action, status


def build_z_state(
    *,
    bound_solver: Any,
    forecast: ArrayDict,
    current_time: int,
    terminal_start: int,
    terminal_P_DG_5: float,
    terminal_SOC_BSS_9: float,
) -> np.ndarray:
    z = np.zeros(STATE_DIM, dtype=float)
    z[0] = terminal_start / 47.0
    z[1:33] = (
        np.asarray(forecast["load"][terminal_start], dtype=float) - bound_solver.load_min
    ) / (bound_solver.load_max - bound_solver.load_min + 1e-6)
    z[33:65] = (
        np.asarray(forecast["load_Q"][terminal_start], dtype=float) - bound_solver.load_min_Q
    ) / (bound_solver.load_max_Q - bound_solver.load_min_Q + 1e-6)
    z[65] = (float(forecast["pv_11"][terminal_start]) - bound_solver.pv_11_min) / (
        bound_solver.pv_11_max - bound_solver.pv_11_min + 1e-6
    )
    z[66] = (float(forecast["wind_26"][terminal_start]) - bound_solver.wind_26_min) / (
        bound_solver.wind_26_max - bound_solver.wind_26_min + 1e-6
    )
    z[67] = (float(terminal_P_DG_5) - bound_solver.gen_min) / (
        bound_solver.gen_max - bound_solver.gen_min + 1e-6
    )
    z[68] = (float(terminal_SOC_BSS_9) - bound_solver.SOC_min) / (
        bound_solver.SOC_max - bound_solver.SOC_min + 1e-6
    )
    return z


def collect_model_based_terminals(
    *,
    source: str,
    scenario_id: int,
    actual: ArrayDict,
    forecast_case: str,
    actual_case: str,
    bound_solver: Any,
    current_times: Sequence[int],
    quiet_solver: bool,
    mpc_time_limit_s: Optional[float],
    mip_gap: Optional[float],
) -> List[TerminalCandidate]:
    if source == "MPC-3":
        solver_kind = "pure_mpc_clean"
        gamma_q = None
    elif source == "Proposed-H3":
        solver_kind = "proposed"
        from experiments_v2.run_33_mpc import load_selected_gamma

        gamma_q = load_selected_gamma("33")
    else:
        raise ValueError(source)

    seed = int(DEFAULT_BASE_SEEDS["33"] + scenario_id)
    forecaster = make_forecaster("33", FORECAST_CASES[forecast_case], seed=seed)
    solver = build_solver_33(solver_kind=solver_kind, gamma_Q=gamma_q)
    if mpc_time_limit_s is not None and hasattr(solver, "time_limit_s"):
        solver.time_limit_s = float(mpc_time_limit_s)
    if mip_gap is not None and hasattr(solver, "mip_gap"):
        solver.mip_gap = float(mip_gap)

    P_DG_5_init = float(DEFAULT_33["P_DG_5_init"])
    SOC_BSS_9_init = float(DEFAULT_33["SOC_BSS_9_init"])
    total_cost = 0.0
    candidates: List[TerminalCandidate] = []
    selected_times = set(int(value) for value in current_times)
    stop = max(selected_times) + 1 if selected_times else 0

    for current_time in range(stop):
        terminal_start = current_time + HORIZON
        terminal_pre_step = terminal_start - 1
        forecast = forecaster.make_forecast_arrays(
            actual=actual,
            current_time=current_time,
            window_time=SOLVER_WINDOW,
        )
        try:
            (
                _obj,
                new_P_DG_5_init,
                new_SOC_BSS_9_init,
                cost_hour_now,
                _solve_time,
            ) = run_solver_call(
                quiet_solver,
                solver,
                current_time,
                SOLVER_WINDOW,
                P_DG_5_init,
                SOC_BSS_9_init,
                forecast,
                remainder=False,
            )
            dispatch = getattr(solver, "last_dispatch", {}) or {}
            terminal_P_DG_5 = finite_float(dispatch.get("terminal_P_DG_5"))
            terminal_SOC_BSS_9 = finite_float(dispatch.get("terminal_SOC_BSS_9"))
            if not math.isfinite(terminal_P_DG_5) or not math.isfinite(terminal_SOC_BSS_9):
                raise RuntimeError("missing terminal dispatch in solver.last_dispatch")
            total_cost += float(cost_hour_now)
            if current_time in selected_times:
                z_state = build_z_state(
                    bound_solver=bound_solver,
                    forecast=forecast,
                    current_time=current_time,
                    terminal_start=terminal_start,
                    terminal_P_DG_5=terminal_P_DG_5,
                    terminal_SOC_BSS_9=terminal_SOC_BSS_9,
                )
                candidates.append(
                    TerminalCandidate(
                        scenario_id=scenario_id,
                        current_time=current_time,
                        terminal_start=terminal_start,
                        terminal_pre_step=terminal_pre_step,
                        source=source,
                        terminal_P_DG_5=terminal_P_DG_5,
                        terminal_SOC_BSS_9=terminal_SOC_BSS_9,
                        z_state=z_state,
                        forecast_terminal_load_sum=float(np.sum(forecast["load"][terminal_start])),
                        actual_terminal_load_sum=float(np.sum(actual["load"][terminal_start])),
                        source_cost_so_far=total_cost,
                    )
                )
            P_DG_5_init = float(new_P_DG_5_init)
            SOC_BSS_9_init = float(new_SOC_BSS_9_init)
        except Exception as exc:
            message = f"{source}, scenario={scenario_id}, t={current_time}: {exc}"
            if current_time in selected_times:
                candidates.append(
                    TerminalCandidate(
                        scenario_id=scenario_id,
                        current_time=current_time,
                        terminal_start=terminal_start,
                        terminal_pre_step=terminal_pre_step,
                        source=source,
                        terminal_P_DG_5=math.nan,
                        terminal_SOC_BSS_9=math.nan,
                        z_state=np.full(STATE_DIM, math.nan),
                        forecast_terminal_load_sum=math.nan,
                        actual_terminal_load_sum=float(np.sum(actual["load"][terminal_start])),
                        source_completed=False,
                        source_failure_message=message,
                        source_cost_so_far=total_cost,
                    )
                )
            else:
                for failed_time in sorted(t for t in selected_times if t >= current_time):
                    failed_terminal_start = failed_time + HORIZON
                    candidates.append(
                        TerminalCandidate(
                            scenario_id=scenario_id,
                            current_time=failed_time,
                            terminal_start=failed_terminal_start,
                            terminal_pre_step=failed_terminal_start - 1,
                            source=source,
                            terminal_P_DG_5=math.nan,
                            terminal_SOC_BSS_9=math.nan,
                            z_state=np.full(STATE_DIM, math.nan),
                            forecast_terminal_load_sum=math.nan,
                            actual_terminal_load_sum=float(
                                np.sum(actual["load"][failed_terminal_start])
                            ),
                            source_completed=False,
                            source_failure_message=message,
                            source_cost_so_far=total_cost,
                        )
                    )
            traceback.print_exc()
            break

    return candidates


def collect_perfect_global_terminals(
    *,
    scenario_id: int,
    actual: ArrayDict,
    forecast_case: str,
    bound_solver: Any,
    current_times: Sequence[int],
    quiet_solver: bool,
    perfect_time_limit_s: Optional[float],
    mip_gap: Optional[float],
) -> List[TerminalCandidate]:
    seed = int(DEFAULT_BASE_SEEDS["33"] + scenario_id)
    forecaster = make_forecaster("33", FORECAST_CASES[forecast_case], seed=seed)
    solver = build_perfect_solver_33()
    solver.time_limit_s = perfect_time_limit_s
    solver.mip_gap = mip_gap
    solver.output_flag = 0 if quiet_solver else 1
    load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = get_solver_inputs_33(actual)

    try:
        if quiet_solver:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink):
                    result = solver.sol_pro_all_day(
                        P_DG_5_init=float(DEFAULT_33["P_DG_5_init"]),
                        SOC_BSS_9_init=float(DEFAULT_33["SOC_BSS_9_init"]),
                        load_list=load_list,
                        pv_11_avail_list=pv_11_avail_list,
                        wind_26_avail_list=wind_26_avail_list,
                        window_time=N_STEPS,
                        load_list_Q=load_list_Q,
                    )
        else:
            result = solver.sol_pro_all_day(
                P_DG_5_init=float(DEFAULT_33["P_DG_5_init"]),
                SOC_BSS_9_init=float(DEFAULT_33["SOC_BSS_9_init"]),
                load_list=load_list,
                pv_11_avail_list=pv_11_avail_list,
                wind_26_avail_list=wind_26_avail_list,
                window_time=N_STEPS,
                load_list_Q=load_list_Q,
            )
        p_dg_values = [float(value) for value in result.get("P_DG_5", [])]
        soc_values = [float(value) for value in result.get("SOC_BSS_9", [])]
    except Exception as exc:
        traceback.print_exc()
        return [
            TerminalCandidate(
                scenario_id=scenario_id,
                current_time=current_time,
                terminal_start=current_time + HORIZON,
                terminal_pre_step=current_time + HORIZON - 1,
                source="Perfect-Global",
                terminal_P_DG_5=math.nan,
                terminal_SOC_BSS_9=math.nan,
                z_state=np.full(STATE_DIM, math.nan),
                forecast_terminal_load_sum=math.nan,
                actual_terminal_load_sum=float(np.sum(actual["load"][current_time + HORIZON])),
                source_completed=False,
                source_failure_message=f"Perfect-Global, scenario={scenario_id}: {exc}",
            )
            for current_time in current_times
        ]

    candidates: List[TerminalCandidate] = []
    for current_time in current_times:
        terminal_start = current_time + HORIZON
        terminal_pre_step = terminal_start - 1
        forecast = forecaster.make_forecast_arrays(
            actual=actual,
            current_time=current_time,
            window_time=SOLVER_WINDOW,
        )
        terminal_P_DG_5 = p_dg_values[terminal_pre_step]
        terminal_SOC_BSS_9 = soc_values[terminal_pre_step]
        z_state = build_z_state(
            bound_solver=bound_solver,
            forecast=forecast,
            current_time=current_time,
            terminal_start=terminal_start,
            terminal_P_DG_5=terminal_P_DG_5,
            terminal_SOC_BSS_9=terminal_SOC_BSS_9,
        )
        candidates.append(
            TerminalCandidate(
                scenario_id=scenario_id,
                current_time=current_time,
                terminal_start=terminal_start,
                terminal_pre_step=terminal_pre_step,
                source="Perfect-Global",
                terminal_P_DG_5=terminal_P_DG_5,
                terminal_SOC_BSS_9=terminal_SOC_BSS_9,
                z_state=z_state,
                forecast_terminal_load_sum=float(np.sum(forecast["load"][terminal_start])),
                actual_terminal_load_sum=float(np.sum(actual["load"][terminal_start])),
            )
        )
    return candidates


def evaluate_tail_cost(
    *,
    tail_solver: Any,
    base_prices: np.ndarray,
    actual: ArrayDict,
    candidate: TerminalCandidate,
    quiet_solver: bool,
    tail_time_limit_s: Optional[float],
    mip_gap: Optional[float],
) -> Tuple[float, float, int, str]:
    if not candidate.source_completed:
        return math.nan, math.nan, 0, candidate.source_failure_message
    terminal_start = int(candidate.terminal_start)
    tail_len = N_STEPS - terminal_start
    if tail_len <= 0:
        return 0.0, 0.0, 0, ""

    tail_actual = build_tail_actual(actual, terminal_start)
    load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = get_solver_inputs_33(tail_actual)
    tail_solver.Prices = build_shifted_prices(base_prices, terminal_start)
    tail_solver.time_limit_s = tail_time_limit_s
    tail_solver.mip_gap = mip_gap
    tail_solver.output_flag = 0 if quiet_solver else 1

    try:
        if quiet_solver:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink):
                    result = tail_solver.sol_pro_all_day(
                        P_DG_5_init=float(candidate.terminal_P_DG_5),
                        SOC_BSS_9_init=float(candidate.terminal_SOC_BSS_9),
                        load_list=load_list,
                        pv_11_avail_list=pv_11_avail_list,
                        wind_26_avail_list=wind_26_avail_list,
                        window_time=tail_len,
                        load_list_Q=load_list_Q,
                    )
        else:
            result = tail_solver.sol_pro_all_day(
                P_DG_5_init=float(candidate.terminal_P_DG_5),
                SOC_BSS_9_init=float(candidate.terminal_SOC_BSS_9),
                load_list=load_list,
                pv_11_avail_list=pv_11_avail_list,
                wind_26_avail_list=wind_26_avail_list,
                window_time=tail_len,
                load_list_Q=load_list_Q,
            )
        stats = dict(result.get("model_stats") or getattr(tail_solver, "last_model_stats", {}) or {})
        status = finite_float(stats.get("status"))
        status_code = int(status) if math.isfinite(status) else 0
        return float(result["obj"]), float(result.get("runtime", math.nan)), status_code, ""
    except Exception as exc:
        traceback.print_exc()
        return math.nan, math.nan, 0, (
            f"tail OPF failed: source={candidate.source}, scenario={candidate.scenario_id}, "
            f"t={candidate.current_time}, terminal_start={terminal_start}: {exc}"
        )


def candidate_to_row(
    candidate: TerminalCandidate,
    *,
    q_value: float,
    icnn_terminal_cost: float,
    actor: List[float],
    icnn_status: str,
    oracle_tail_cost: float,
    oracle_tail_runtime_s: float,
    oracle_tail_status: int,
    failure_message: str,
) -> Dict[str, Any]:
    return {
        "system": "33",
        "scenario_id": candidate.scenario_id,
        "current_time": candidate.current_time,
        "terminal_start": candidate.terminal_start,
        "terminal_pre_step": candidate.terminal_pre_step,
        "terminal_source": candidate.source,
        "terminal_P_DG_5": candidate.terminal_P_DG_5,
        "terminal_SOC_BSS_9": candidate.terminal_SOC_BSS_9,
        "forecast_terminal_load_sum": candidate.forecast_terminal_load_sum,
        "actual_terminal_load_sum": candidate.actual_terminal_load_sum,
        "source_completed": candidate.source_completed,
        "source_cost_so_far": candidate.source_cost_so_far,
        "icnn_q_value": q_value,
        "icnn_terminal_cost": icnn_terminal_cost,
        "icnn_actor_json": json.dumps(actor),
        "icnn_status": icnn_status,
        "oracle_tail_cost": oracle_tail_cost,
        "oracle_tail_runtime_s": oracle_tail_runtime_s,
        "oracle_tail_status": oracle_tail_status,
        "closed_loop_violation_pu": 0.0 if candidate.source_completed else math.nan,
        "failure_message": failure_message or candidate.source_failure_message,
        "z_state_json": json.dumps([float(value) for value in candidate.z_state]),
        "experiment_tag": "icnn_value_diagnostic",
    }


def summarize_value_error(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {source: [] for source in TERMINAL_SOURCES}
    groups["All terminals"] = []
    for row in rows:
        if math.isfinite(finite_float(row.get("oracle_tail_cost"))) and math.isfinite(
            finite_float(row.get("icnn_terminal_cost"))
        ):
            groups.setdefault(str(row["terminal_source"]), []).append(row)
            groups["All terminals"].append(row)

    for source, group in groups.items():
        if not group:
            summary_rows.append(
                {
                    "evaluation_terminal_states": source,
                    "n_samples": 0,
                    "mae_raw": math.nan,
                    "rmse_raw": math.nan,
                    "mae_affine": math.nan,
                    "rmse_affine": math.nan,
                    "affine_slope": math.nan,
                    "affine_intercept": math.nan,
                    "spearman_rho": math.nan,
                    "pairwise_accuracy_percent": math.nan,
                    "n_pairwise": 0,
                }
            )
            continue

        icnn = [finite_float(row["icnn_terminal_cost"]) for row in group]
        oracle = [finite_float(row["oracle_tail_cost"]) for row in group]
        raw_errors = [pred - true for pred, true in zip(icnn, oracle)]
        affine_pred, slope, intercept = affine_fit_predict(icnn, oracle)
        affine_errors = [pred - true for pred, true in zip(affine_pred, oracle)]
        pair_acc, n_pair = pairwise_ranking_accuracy(group)
        summary_rows.append(
            {
                "evaluation_terminal_states": source,
                "n_samples": len(group),
                "mae_raw": mean_or_nan(abs(error) for error in raw_errors),
                "rmse_raw": rmse_or_nan(raw_errors),
                "mae_affine": mean_or_nan(abs(error) for error in affine_errors),
                "rmse_affine": rmse_or_nan(affine_errors),
                "affine_slope": slope,
                "affine_intercept": intercept,
                "spearman_rho": spearman(icnn, oracle),
                "pairwise_accuracy_percent": pair_acc,
                "n_pairwise": n_pair,
            }
        )
    return summary_rows


def summarize_decision_impact(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not math.isfinite(finite_float(row.get("oracle_tail_cost"))):
            continue
        by_key[(int(row["scenario_id"]), int(row["current_time"]))][
            str(row["terminal_source"])
        ] = row

    source_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    reductions: Dict[str, List[float]] = defaultdict(list)
    matched_costs: Dict[str, List[float]] = defaultdict(list)

    for methods in by_key.values():
        mpc = methods.get("MPC-3")
        if mpc is None:
            continue
        mpc_cost = finite_float(mpc.get("oracle_tail_cost"))
        if not math.isfinite(mpc_cost) or abs(mpc_cost) <= 1e-9:
            continue
        for source in TERMINAL_SOURCES:
            row = methods.get(source)
            if row is None:
                continue
            tail_cost = finite_float(row.get("oracle_tail_cost"))
            if not math.isfinite(tail_cost):
                continue
            source_rows[source].append(row)
            matched_costs[source].append(tail_cost)
            reductions[source].append(100.0 * (mpc_cost - tail_cost) / mpc_cost)

    summary_rows: List[Dict[str, Any]] = []
    for source in TERMINAL_SOURCES:
        costs = matched_costs.get(source, [])
        reduction_values = reductions.get(source, [])
        violations = [
            finite_float(row.get("closed_loop_violation_pu"))
            for row in source_rows.get(source, [])
            if math.isfinite(finite_float(row.get("closed_loop_violation_pu")))
        ]
        summary_rows.append(
            {
                "terminal_state_source": source,
                "n_matched_with_mpc3": len(costs),
                "mean_oracle_tail_cost": mean_or_nan(costs),
                "std_oracle_tail_cost": std_or_nan(costs),
                "ci95_oracle_tail_cost": ci95(costs),
                "mean_tail_cost_reduction_vs_mpc3_percent": mean_or_nan(reduction_values),
                "ci95_tail_cost_reduction_vs_mpc3_percent": ci95(reduction_values),
                "mean_closed_loop_violation_pu": mean_or_nan(violations),
            }
        )
    return summary_rows


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


def readable_number(value: object, digits: int = 2) -> str:
    numeric = finite_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
        value = value.replace(old, new)
    return value


def latex_table(headers: List[str], rows: List[List[str]]) -> str:
    colspec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_table_bundle(output_dir: Path, stem: str, headers: List[str], rows: List[List[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    md_path.write_text(markdown_table(headers, rows), encoding="utf-8")
    tex_path.write_text(latex_table(headers, rows), encoding="utf-8")
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("Saved:", tex_path)


def write_paper_tables(
    value_summary: List[Dict[str, Any]],
    impact_summary: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    value_headers = [
        "Evaluation terminal states",
        "N",
        "MAE ($)",
        "RMSE ($)",
        "Spearman rho",
        "Pairwise acc. (%)",
    ]
    value_rows = [
        [
            str(row["evaluation_terminal_states"]),
            str(row["n_samples"]),
            readable_number(row["mae_affine"], 2),
            readable_number(row["rmse_affine"], 2),
            readable_number(row["spearman_rho"], 3),
            readable_number(row["pairwise_accuracy_percent"], 2),
        ]
        for row in value_summary
    ]
    write_table_bundle(output_dir, "table_33_icnn_value_error", value_headers, value_rows)

    impact_headers = [
        "Terminal state source",
        "Oracle tail cost ($)",
        "Tail-cost reduction vs. MPC-3 (%)",
        "Closed-loop vio. (p.u.)",
    ]
    impact_rows = [
        [
            str(row["terminal_state_source"]),
            readable_number(row["mean_oracle_tail_cost"], 2),
            readable_number(row["mean_tail_cost_reduction_vs_mpc3_percent"], 2),
            readable_number(row["mean_closed_loop_violation_pu"], 4),
        ]
        for row in impact_summary
    ]
    write_table_bundle(output_dir, "table_33_icnn_decision_impact", impact_headers, impact_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 33-bus ICNN terminal-value approximation and decision-impact diagnostic."
        )
    )
    parser.add_argument(
        "--scenarios",
        default="80,84,88,92,96",
        help=(
            "Comma-separated IDs, 'val', or 'all'. Default is a sampled "
            "sampled diagnostic to avoid the prohibitively expensive full grid."
        ),
    )
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument(
        "--sources",
        default="MPC-3,Proposed-H3",
        help="Terminal sources: all, MPC-3, Proposed-H3, Perfect-Global.",
    )
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-terminal-steps", type=int, default=6)
    parser.add_argument("--start-current-time", type=int, default=0)
    parser.add_argument("--terminal-step-stride", type=int, default=8)
    parser.add_argument("--icnn-action-mode", choices=["exact", "sampled"], default="exact")
    parser.add_argument("--icnn-action-samples", type=int, default=256)
    parser.add_argument("--icnn-time-limit-s", type=float, default=10.0)
    parser.add_argument("--mpc-time-limit-s", type=float, default=300.0)
    parser.add_argument("--perfect-time-limit-s", type=float, default=1800.0)
    parser.add_argument("--tail-time-limit-s", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step13_33_icnn_value_diagnostic_sample_raw.csv")
    parser.add_argument(
        "--value-summary-output",
        default="formal/step13_33_icnn_value_error_sample_summary.csv",
    )
    parser.add_argument(
        "--impact-summary-output",
        default="formal/step13_33_icnn_decision_impact_sample_summary.csv",
    )
    parser.add_argument("--table-output-dir", default="paper_tables/33")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    sources = parse_sources(args.sources)
    current_times = selected_current_times(
        start_current_time=args.start_current_time,
        terminal_step_stride=args.terminal_step_stride,
        max_terminal_steps=args.max_terminal_steps,
    )
    print("Scenarios:", scenario_ids)
    print("Terminal sources:", sources)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("current_times:", current_times)
    print(
        "terminal candidates:",
        len(scenario_ids) * len(sources) * len(current_times),
        "(sampled diagnostic; pass explicit full-grid arguments only if needed)",
    )
    print("icnn_action_mode:", args.icnn_action_mode)
    print("tail_time_limit_s:", args.tail_time_limit_s)
    print("mip_gap:", args.mip_gap)

    if args.dry_run:
        return

    q1, q2 = q_matrices()
    bound_solver = build_solver_33("proposed", gamma_Q=0.3)
    tail_solver = build_perfect_solver_33()
    base_prices = np.asarray(tail_solver.Prices, dtype=float).copy()

    rows: List[Dict[str, Any]] = []
    for scenario_id in scenario_ids:
        actual = load_actual_scenario("33", scenario_id, actual_case=args.actual_case)
        candidates: List[TerminalCandidate] = []
        if "MPC-3" in sources:
            candidates.extend(
                collect_model_based_terminals(
                    source="MPC-3",
                    scenario_id=scenario_id,
                    actual=actual,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    bound_solver=bound_solver,
                    current_times=current_times,
                    quiet_solver=not args.verbose_solver,
                    mpc_time_limit_s=args.mpc_time_limit_s,
                    mip_gap=args.mip_gap,
                )
            )
        if "Proposed-H3" in sources:
            candidates.extend(
                collect_model_based_terminals(
                    source="Proposed-H3",
                    scenario_id=scenario_id,
                    actual=actual,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    bound_solver=bound_solver,
                    current_times=current_times,
                    quiet_solver=not args.verbose_solver,
                    mpc_time_limit_s=args.mpc_time_limit_s,
                    mip_gap=args.mip_gap,
                )
            )
        if "Perfect-Global" in sources:
            candidates.extend(
                collect_perfect_global_terminals(
                    scenario_id=scenario_id,
                    actual=actual,
                    forecast_case=args.forecast_case,
                    bound_solver=bound_solver,
                    current_times=current_times,
                    quiet_solver=not args.verbose_solver,
                    perfect_time_limit_s=args.perfect_time_limit_s,
                    mip_gap=args.mip_gap,
                )
            )

        for candidate in candidates:
            if candidate.source_completed and np.all(np.isfinite(candidate.z_state)):
                try:
                    q_value, icnn_cost, actor, icnn_status = evaluate_icnn_terminal(
                        q1,
                        q2,
                        candidate.z_state,
                        mode=args.icnn_action_mode,
                        n_samples=args.icnn_action_samples,
                        seed=int(
                            DEFAULT_BASE_SEEDS["33"]
                            + scenario_id * 1000
                            + candidate.current_time * 10
                            + TERMINAL_SOURCES.index(candidate.source)
                        ),
                        time_limit_s=args.icnn_time_limit_s,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    q_value = math.nan
                    icnn_cost = math.nan
                    actor = []
                    icnn_status = f"failed: {exc}"
            else:
                q_value = math.nan
                icnn_cost = math.nan
                actor = []
                icnn_status = "source_failed"

            tail_cost, tail_runtime, tail_status, tail_failure = evaluate_tail_cost(
                tail_solver=tail_solver,
                base_prices=base_prices,
                actual=actual,
                candidate=candidate,
                quiet_solver=not args.verbose_solver,
                tail_time_limit_s=args.tail_time_limit_s,
                mip_gap=args.mip_gap,
            )
            row = candidate_to_row(
                candidate,
                q_value=q_value,
                icnn_terminal_cost=icnn_cost,
                actor=actor,
                icnn_status=icnn_status,
                oracle_tail_cost=tail_cost,
                oracle_tail_runtime_s=tail_runtime,
                oracle_tail_status=tail_status,
                failure_message=tail_failure,
            )
            rows.append(row)
            print(
                f"{row['terminal_source']}, scenario={scenario_id}, t={row['current_time']}, "
                f"Pdg={finite_float(row['terminal_P_DG_5']):.4f}, "
                f"SOC={finite_float(row['terminal_SOC_BSS_9']):.4f}, "
                f"ICNN={finite_float(row['icnn_terminal_cost']):.4f}, "
                f"tail={finite_float(row['oracle_tail_cost']):.4f}, "
                f"status={row['icnn_status']}"
            )

    raw_path = RESULT_DIR / args.output
    write_rows(raw_path, rows)
    value_summary = summarize_value_error(rows)
    impact_summary = summarize_decision_impact(rows)
    write_rows(RESULT_DIR / args.value_summary_output, value_summary)
    write_rows(RESULT_DIR / args.impact_summary_output, impact_summary)
    write_paper_tables(value_summary, impact_summary, RESULT_DIR / args.table_output_dir)
    print("Raw results:", raw_path)


if __name__ == "__main__":
    main()
