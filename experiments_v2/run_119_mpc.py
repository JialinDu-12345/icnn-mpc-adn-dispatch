from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments_v2.config import (              
    DEFAULT_119,
    DEFAULT_BASE_SEEDS,
    FORECAST_CASES,
    METHOD_DIRS_119,
    RESULT_DIR,
    SPLITS,
    validate_actual_case,
    validate_forecast_case,
)
from experiments_v2.result_logger import ResultLogger, make_result_row              
from experiments_v2.run_33_mpc import (              
    collect_gurobi_model_stats,
    finite_values,
    max_or_nan,
    mean_or_nan,
    min_or_nan,
)
from experiments_v2.scenario_protocol import (              
    ArrayDict,
    get_solver_inputs_119,
    load_actual_scenario,
    make_forecaster,
)
from experiments_v2.utils.scenario_selection import resolve_scenarios              


TASK_PRESETS = {
    "mpc3": {
        "method": "MPC-3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "proposed3": {
        "method": "Proposed-H3",
        "solver_kind": "proposed",
        "horizon": 3,
        "gamma_Q": 0.3,
    },
    "stdnn3": {
        "method": "StdNN-H3-MIP",
        "solver_kind": "std_nn_mpc",
        "horizon": 3,
        "gamma_Q": 0.3,
    },
}


def resolve_gamma_Q_119(gamma_Q: Any) -> Optional[float]:
    if gamma_Q is None:
        return None
    if isinstance(gamma_Q, str):
        value = gamma_Q.strip().lower()
        if not value:
            return None
        if value == "auto":
            return 0.3
        return float(value)
    return float(gamma_Q)


def _safe_dict_value(values: Dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _safe_window(values: Dict[str, Any], prefix: str, n: int) -> List[float]:
    return [_safe_dict_value(values, f"{prefix}[{idx}]") for idx in range(int(n))]


def extract_dispatch_119(dict_value: Dict[str, Any]) -> Dict[str, Any]:
                                                                                      

    p_dg_29 = _safe_dict_value(dict_value, "P_DG_29[0]")
    p_dg_64 = _safe_dict_value(dict_value, "P_DG_64[0]")
    soc_values = [
        _safe_dict_value(dict_value, "SOC_BSS_12[0]"),
        _safe_dict_value(dict_value, "SOC_BSS_34[0]"),
        _safe_dict_value(dict_value, "SOC_BSS_68[0]"),
        _safe_dict_value(dict_value, "SOC_BSS_103[0]"),
    ]

    p_ch = sum(
        _safe_dict_value(dict_value, key, 0.0)
        for key in [
            "P_BSS_ch_12[0]",
            "P_BSS_ch_34[0]",
            "P_BSS_ch_68[0]",
            "P_BSS_ch_103[0]",
        ]
    )
    p_dch = sum(
        _safe_dict_value(dict_value, key, 0.0)
        for key in [
            "P_BSS_dch_12[0]",
            "P_BSS_dch_34[0]",
            "P_BSS_dch_68[0]",
            "P_BSS_dch_103[0]",
        ]
    )

                                                                       
                                                                                  
    p_grid_signed = _safe_dict_value(dict_value, "P_ij[0,0,0]")
    if not np.isfinite(p_grid_signed):
        p_grid_signed = _safe_dict_value(dict_value, "P_balance[0]")

    return {
        "P_DG_total": p_dg_29 + p_dg_64,
        "P_DG_29": p_dg_29,
        "P_DG_64": p_dg_64,
        "SOC_total": float(np.sum(soc_values)),
        "SOC_mean": float(np.mean(soc_values)),
        "P_BSS": p_dch * 0.98 - p_ch * 1.02,
        "P_BSS_ch": p_ch,
        "P_BSS_dch": p_dch,
        "P_grid": p_grid_signed,
        "P_balance": _safe_dict_value(dict_value, "P_balance[0]"),
        "Loss": _safe_dict_value(dict_value, "Loss[0]"),
        "P_DG_29_window": _safe_window(dict_value, "P_DG_29", 3),
        "P_DG_64_window": _safe_window(dict_value, "P_DG_64", 3),
        "SOC_BSS_12_window": _safe_window(dict_value, "SOC_BSS_12", 3),
        "SOC_BSS_34_window": _safe_window(dict_value, "SOC_BSS_34", 3),
        "SOC_BSS_68_window": _safe_window(dict_value, "SOC_BSS_68", 3),
        "SOC_BSS_103_window": _safe_window(dict_value, "SOC_BSS_103", 3),
    }


def _is_119_std_nn_path(file_path: Path) -> bool:
    normalized_path = str(file_path).replace("\\", "/")
    return "solvers/119/std_nn_mpc" in normalized_path


def _ensure_119_icnn_exogenous_clipping(safe_source: str, file_path: Path) -> str:
                                                                             

    normalized_path = str(file_path).replace("\\", "/")
    if "solvers/119/mpc_icnn" not in normalized_path:
        return safe_source

    required_tokens = (
        "clip_icnn_exogenous_inputs",
        "_clip_icnn_exogenous_value",
        "_reset_icnn_clip_stats",
        "_add_icnn_input_var",
        "_add_icnn_input_mvar",
    )
    missing_tokens = [token for token in required_tokens if token not in safe_source]
    forbidden_tokens = (
        "relax_icnn_input_bounds",
        "icnn_input_lb_relaxed",
        "icnn_input_ub_relaxed",
    )
    found_forbidden = [token for token in forbidden_tokens if token in safe_source]
    if missing_tokens or found_forbidden:
        raise RuntimeError(
            "The 119 Proposed-H3 solver must keep ICNN inputs bounded and clip "
            "only exogenous terminal constants. "
            f"Missing={missing_tokens}; forbidden={found_forbidden}; file={file_path}"
        )
    return safe_source


def _inject_119_std_nn_source(safe_source: str, file_path: Path) -> str:
                                                                                

    if not _is_119_std_nn_path(file_path):
        return safe_source

    if "standard_sac_critics_119.npz" in safe_source:
        return safe_source

    import_block = (
        "import numpy as np\n"
        "from pathlib import Path\n"
        "from experiments_v2.nn_embedding.standard_nn_embedding_utils import (\n"
        "    add_standard_relu_network,\n"
        "    load_standard_critic_layers,\n"
        ")\n"
    )
    if "add_standard_relu_network" not in safe_source:
        safe_source = safe_source.replace("import numpy as np\n", import_block, 1)

    init_old = "        self.scale_Q = 0\n"
    init_new = (
        "        self.scale_Q = 0.3\n"
        "        self.standard_nn_path = Path(__file__).resolve().parent / \"standard_sac_critics_119.npz\"\n"
        "        if not self.standard_nn_path.exists():\n"
        "            raise FileNotFoundError(\n"
        "                \"Missing 119 standard SAC critic export: \"\n"
        "                f\"{self.standard_nn_path}. Export ordinary SAC critics with: \"\n"
        "                \"python -m experiments_v2.nn_embedding.export_standard_sac_critic \"\n"
        "                \"--critic-1 experiment_assets/solvers/119/std_nn_mpc/standard_critic_1_model.pth \"\n"
        "                \"--critic-2 experiment_assets/solvers/119/std_nn_mpc/standard_critic_2_model.pth \"\n"
        "                \"--output experiment_assets/solvers/119/std_nn_mpc/standard_sac_critics_119.npz\"\n"
        "            )\n"
        "        self.standard_q1_layers, self.standard_q2_layers = load_standard_critic_layers(self.standard_nn_path)\n"
        "        self.std_nn_binary_count_last = 0\n"
    )
    if init_old not in safe_source:
        raise RuntimeError(f"Cannot inject StdNN init block into {file_path}")
    safe_source = safe_source.replace(init_old, init_new, 1)

    terminal_bounds_method = '''
    def _terminal_feature_bounds(self, current_time, window_time):
        z0_lb = np.zeros(269, dtype=float)
        z0_ub = np.ones(269, dtype=float)

        terminal_idx = window_time - 1 + current_time
        t_normal_st = terminal_idx / 47.0
        z0_lb[0] = t_normal_st
        z0_ub[0] = t_normal_st

        for i in range(118):
            load_p = (
                self.list_p_q_pu_t[terminal_idx][i][0] - self.load_min[i]
            ) / (self.load_max[i] - self.load_min[i] + 1e-6)
            load_q = (
                self.list_p_q_pu_t[terminal_idx][i][1] - self.load_min_Q[i]
            ) / (self.load_max_Q[i] - self.load_min_Q[i] + 1e-6)
            load_p = float(np.clip(load_p, -1.0, 1.0))
            load_q = float(np.clip(load_q, -1.0, 1.0))

            z0_lb[i + 1] = load_p
            z0_ub[i + 1] = load_p
            z0_lb[i + 119] = load_q
            z0_ub[i + 119] = load_q

        wt_values = [
            self.wind_11_avail_list[terminal_idx],
            self.wind_32_avail_list[terminal_idx],
            self.wind_66_avail_list[terminal_idx],
            self.wind_101_avail_list[terminal_idx],
        ]
        pv_values = [
            self.pv_22_avail_list[terminal_idx],
            self.pv_36_avail_list[terminal_idx],
            self.pv_70_avail_list[terminal_idx],
            self.pv_107_avail_list[terminal_idx],
        ]

        for k, value in enumerate(wt_values):
            norm = (value - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
            norm = float(np.clip(norm, -1.0, 1.0))
            z0_lb[237 + k] = norm
            z0_ub[237 + k] = norm

        for k, value in enumerate(pv_values):
            norm = (value - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)
            norm = float(np.clip(norm, -1.0, 1.0))
            z0_lb[241 + k] = norm
            z0_ub[241 + k] = norm

        z0_lb[245:251] = 0.0
        z0_ub[245:251] = 1.0
        z0_lb[251:269] = -1.0
        z0_ub[251:269] = 1.0
        return z0_lb, z0_ub

'''
    method_anchor = "        return list_p_q_pu_t\n\n    def sol_pro("
    if method_anchor not in safe_source:
        raise RuntimeError(f"Cannot inject StdNN terminal bounds into {file_path}")
    safe_source = safe_source.replace(
        method_anchor,
        "        return list_p_q_pu_t\n\n" + terminal_bounds_method + "    def sol_pro(",
        1,
    )

    cost_var_anchor = (
        "            cost_genera = model.addMVar(shape=(window_time - 1,), "
        "vtype=GRB.CONTINUOUS, name='cost_genera')\n"
    )
    std_var_block = (
        cost_var_anchor
        + "            Q_f = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Q_f')\n"
        + "            z0 = model.addMVar(shape=(269,), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z0')\n"
        + "            terminal_actor = model.addMVar(shape=(18,), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='terminal_actor')\n"
    )
    if cost_var_anchor not in safe_source:
        raise RuntimeError(f"Cannot inject StdNN variables into {file_path}")
    safe_source = safe_source.replace(cost_var_anchor, std_var_block, 1)

    objective_old = (
        "            model.addConstr(cost_day == cost_window)\n"
        "            # model.addConstr(cost_day == cost_window)\n"
    )
    objective_new = (
        "            model.addConstr(cost_day == cost_window - Q_f * 100 * self.scale_Q)\n"
        "            # StdNN-H3-MIP subtracts the ordinary SAC critic's reward-to-go estimate.\n"
    )
    if objective_old not in safe_source:
        raise RuntimeError(f"Cannot inject StdNN objective into {file_path}")
    safe_source = safe_source.replace(objective_old, objective_new, 1)

    flow_anchor = "            for t in range(window_time - 1):\n                # print(\"t\", t)\n"
    std_constraint_block = '''
            z0_lb, z0_ub = self._terminal_feature_bounds(current_time, window_time)
            terminal_idx = window_time - 1 + current_time
            model.addConstr(z0[0] == terminal_idx / 47.0)

            for i in range(118):
                load_p = (
                    self.list_p_q_pu_t[terminal_idx][i][0] - self.load_min[i]
                ) / (self.load_max[i] - self.load_min[i] + 1e-6)
                load_q = (
                    self.list_p_q_pu_t[terminal_idx][i][1] - self.load_min_Q[i]
                ) / (self.load_max_Q[i] - self.load_min_Q[i] + 1e-6)
                model.addConstr(z0[i + 1] == float(np.clip(load_p, -1.0, 1.0)))
                model.addConstr(z0[i + 119] == float(np.clip(load_q, -1.0, 1.0)))

            for k, value in enumerate([
                self.wind_11_avail_list[terminal_idx],
                self.wind_32_avail_list[terminal_idx],
                self.wind_66_avail_list[terminal_idx],
                self.wind_101_avail_list[terminal_idx],
            ]):
                norm = (value - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
                model.addConstr(z0[237 + k] == float(np.clip(norm, -1.0, 1.0)))

            for k, value in enumerate([
                self.pv_22_avail_list[terminal_idx],
                self.pv_36_avail_list[terminal_idx],
                self.pv_70_avail_list[terminal_idx],
                self.pv_107_avail_list[terminal_idx],
            ]):
                norm = (value - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)
                model.addConstr(z0[241 + k] == float(np.clip(norm, -1.0, 1.0)))

            model.addConstr(z0[245] == (P_DG_29[window_time - 2] - self.gen_min) / (self.gen_max - self.gen_min + 1e-6))
            model.addConstr(z0[246] == (P_DG_64[window_time - 2] - self.gen_min) / (self.gen_max - self.gen_min + 1e-6))
            model.addConstr(z0[247] == (SOC_BSS_12[window_time - 2] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6))
            model.addConstr(z0[248] == (SOC_BSS_34[window_time - 2] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6))
            model.addConstr(z0[249] == (SOC_BSS_68[window_time - 2] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6))
            model.addConstr(z0[250] == (SOC_BSS_103[window_time - 2] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6))
            for i in range(18):
                model.addConstr(z0[251 + i] == terminal_actor[i])

            q1_out, q1_bins, _ = add_standard_relu_network(
                model=model,
                x_vars=[z0[i] for i in range(269)],
                layers=self.standard_q1_layers,
                x_lb=z0_lb,
                x_ub=z0_ub,
                name="std119_q1",
            )
            q2_out, q2_bins, _ = add_standard_relu_network(
                model=model,
                x_vars=[z0[i] for i in range(269)],
                layers=self.standard_q2_layers,
                x_lb=z0_lb,
                x_ub=z0_ub,
                name="std119_q2",
            )
            model.addGenConstrMin(Q_f, [q1_out, q2_out], name="std119_q_min")
            self.std_nn_binary_count_last = len(q1_bins) + len(q2_bins)

'''
    if flow_anchor not in safe_source:
        raise RuntimeError(f"Cannot inject StdNN constraints into {file_path}")
    safe_source = safe_source.replace(flow_anchor, std_constraint_block + flow_anchor, 1)
    return safe_source


def _inject_119_dispatch_capture(safe_source: str) -> str:
    return_old = (
        "        return (obj, dict_value['P_DG_29[0]'], dict_value['P_DG_64[0]'],\n"
        "                dict_value['SOC_BSS_12[0]'], dict_value['SOC_BSS_34[0]'], dict_value['SOC_BSS_68[0]'], dict_value['SOC_BSS_103[0]'],\n"
        "                dict_value['cost_hour[0]'], model.Runtime)\n"
    )
    return_new = (
        "        self.last_dispatch = extract_dispatch_119(dict_value)\n"
        + return_old
    )
    if return_old not in safe_source:
        raise RuntimeError("Cannot inject 119 dispatch capture into legacy source.")
    return safe_source.replace(return_old, return_new)


def _read_legacy_definitions(file_path: Path) -> str:
    source = file_path.read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)
    cut_index = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("env = PowerSystemEnv()")
            or stripped.startswith("objlist = []")
            or stripped.startswith("list_obj = []")
            or stripped.startswith("seedlist = ")
            or stripped.startswith("for seed in ")
            or (line == line.lstrip() and stripped.startswith("seed = "))
        ):
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
        "            model.setParam('MIPFocus',0)\n",
        "            model.setParam('MIPFocus', 0)\n"
        "            model.setParam('OutputFlag', int(getattr(self, 'output_flag', 0)))\n"
        "            model.setParam('NonConvex', 2)\n",
    )
    safe_source = safe_source.replace(
        "            model.setParam('outPutFlag', 0)",
        "            model.setParam('OutputFlag', int(getattr(self, 'output_flag', 0)))",
    )
    for limit in ("300", "432"):
        safe_source = safe_source.replace(
            f"            model.setParam(GRB.Param.TimeLimit, {limit})\n",
            "            if getattr(self, 'time_limit_s', None) is not None:\n"
            "                model.setParam(GRB.Param.TimeLimit, float(self.time_limit_s))\n",
        )
    safe_source = safe_source.replace(
        "            model.Params.MIPGap = 0.001\n",
        "            if getattr(self, 'mip_gap', None) is not None:\n"
        "                model.Params.MIPGap = float(self.mip_gap)\n",
    )
    safe_source = safe_source.replace(
        "            model.optimize()\n",
        "            model.optimize()\n"
        "            self.last_model_stats = collect_gurobi_model_stats(model)\n"
        "            if int(getattr(model, 'SolCount', 0)) == 0:\n"
        "                status = int(getattr(model, 'Status', 0))\n"
        "                if status == 3 and getattr(self, 'write_iis', False):\n"
        "                    from pathlib import Path as _Path\n"
        "                    iis_dir = _Path(getattr(self, 'iis_dir', '.'))\n"
        "                    iis_dir.mkdir(parents=True, exist_ok=True)\n"
        "                    tag = str(getattr(self, 'debug_tag', 'proposed119'))\n"
        "                    current_t = locals().get('current_time', 'unknown')\n"
        "                    iis_path = iis_dir / f'{tag}_t{current_t}.ilp'\n"
        "                    print(f'[IIS] Computing IIS: {iis_path}')\n"
        "                    model.computeIIS()\n"
        "                    model.write(str(iis_path))\n"
        "                raise RuntimeError(f'Gurobi produced no solution: status={model.Status}, sol_count={model.SolCount}')\n",
    )
    safe_source = safe_source.replace(
        "        except AttributeError:\n            print('Encountered an attribute error')",
        "        except AttributeError:\n            raise",
    )
    safe_source = safe_source.replace(
        "        except GurobiError as e:\n            print('Error code ' + str(e.errno) + ':' + str(e))",
        "        except GurobiError as e:\n            raise",
    )
    safe_source = _ensure_119_icnn_exogenous_clipping(safe_source, file_path)
    safe_source = _inject_119_std_nn_source(safe_source, file_path)
    if file_path.parent.name in {"pure_mpc_clean", "mpc_icnn", "std_nn_mpc"}:
        safe_source = _inject_119_dispatch_capture(safe_source)
    return safe_source


def load_legacy_mpc1_definitions(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find copied 119 solver file: {file_path}")

    source = _read_legacy_definitions(file_path)
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    module.collect_gurobi_model_stats = collect_gurobi_model_stats
    module.extract_dispatch_119 = extract_dispatch_119
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
        "P_DG_29": [],
        "P_DG_64": [],
        "SOC_BSS_12": [],
        "SOC_BSS_34": [],
        "SOC_BSS_68": [],
        "SOC_BSS_103": [],
        "Loss": [],
        "P_BSS": [],
        "P_G": [],
    }
    return module


def _legacy_bounds_and_network(module: ModuleType):
    list_r_x_pu, list_p_q_pu = module._get_network()
    base_load = np.asarray(list_p_q_pu, dtype=float)
    load_min = 0.6 * base_load[:, 0]
    load_max = 1.2 * base_load[:, 0]
    load_min_Q = 0.6 * base_load[:, 1]
    load_max_Q = 1.2 * base_load[:, 1]
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


def build_solver_119(
    solver_kind: str,
    gamma_Q: Optional[float] = None,
):
    if solver_kind not in METHOD_DIRS_119:
        raise ValueError(f"Unknown 119 solver_kind: {solver_kind}")
    if solver_kind in {"proposed", "std_nn_mpc"} and gamma_Q is None:
        raise ValueError(f"gamma_Q must be provided for {solver_kind}.")

    method_dir = METHOD_DIRS_119[solver_kind]
    module = load_legacy_mpc1_definitions(
        module_name=f"legacy_119_{solver_kind}",
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

    if solver_kind == "proposed":
        q_1_mat, q_2_mat = _legacy_q_matrices(module, method_dir)
    else:
        q_1_mat, q_2_mat = None, None

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
    solver.time_limit_s = None
    solver.mip_gap = 1e-3
    solver.output_flag = 0
    solver.last_model_stats = {}
    solver.last_dispatch = {}
    solver.write_iis = False
    solver.iis_dir = str(RESULT_DIR / "debug" / "iis_119_main")
    solver.debug_tag = f"119_{solver_kind}"

    if solver_kind == "pure_mpc_clean":
        solver.scale_Q = 0.0
    elif solver_kind == "proposed":
        solver.scale_Q = resolve_gamma_Q_119(gamma_Q)
        solver.clip_icnn_exogenous_inputs = True
    elif solver_kind == "std_nn_mpc":
        solver.scale_Q = resolve_gamma_Q_119(gamma_Q)

    return solver


def initial_state_119() -> Dict[str, float]:
    return {
        "P_DG_29": float(DEFAULT_119["P_DG_29_init"]),
        "P_DG_64": float(DEFAULT_119["P_DG_64_init"]),
        "SOC_BSS_12": float(DEFAULT_119["SOC_BSS_12_init"]),
        "SOC_BSS_34": float(DEFAULT_119["SOC_BSS_34_init"]),
        "SOC_BSS_68": float(DEFAULT_119["SOC_BSS_68_init"]),
        "SOC_BSS_103": float(DEFAULT_119["SOC_BSS_103_init"]),
    }


def call_sol_pro_119(
    solver: Any,
    current_time: int,
    solver_window: int,
    state: Dict[str, float],
    forecast: ArrayDict,
    remainder: bool = False,
):
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
    ) = get_solver_inputs_119(forecast)

    solve_func = solver.sol_pro_remainder if remainder else solver.sol_pro
    return solve_func(
        current_time,
        solver_window,
        load_list,
        load_list_Q,
        state["P_DG_29"],
        state["P_DG_64"],
        state["SOC_BSS_12"],
        state["SOC_BSS_34"],
        state["SOC_BSS_68"],
        state["SOC_BSS_103"],
        wind_11_avail_list,
        wind_32_avail_list,
        wind_66_avail_list,
        wind_101_avail_list,
        pv_22_avail_list,
        pv_36_avail_list,
        pv_70_avail_list,
        pv_107_avail_list,
    )


def _run_solver_call(quiet_solver: bool, *args: Any, **kwargs: Any):
    if quiet_solver:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink):
                return call_sol_pro_119(*args, **kwargs)
    return call_sol_pro_119(*args, **kwargs)


def is_gurobi_infeasible_error(exc: Exception) -> bool:
    text = str(exc)
    upper_text = text.upper()
    return (
        "status=3" in text
        or "INFEASIBLE" in upper_text
        or "INF_OR_UNBD" in upper_text
    )


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


def run_one_scenario_119(
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
    write_iis: bool = False,
    iis_dir: str = "debug/iis_119_main",
    clip_icnn_exogenous_inputs: bool = True,
) -> Dict[str, Any]:
    validate_forecast_case(forecast_case)
    validate_actual_case(actual_case)

    system = "119"
    seed = DEFAULT_BASE_SEEDS[system] + int(scenario_id)
    actual = load_actual_scenario(system, scenario_id, actual_case=actual_case)
    forecaster = make_forecaster(
        system=system,
        forecast_case_config=FORECAST_CASES[forecast_case],
        seed=seed,
    )
    gamma_Q_value = resolve_gamma_Q_119(gamma_Q)
    solver = build_solver_119(solver_kind=solver_kind, gamma_Q=gamma_Q_value)
    if solver_kind == "proposed":
        solver.clip_icnn_exogenous_inputs = bool(clip_icnn_exogenous_inputs)
    debug_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method)).strip("_")
    solver.write_iis = bool(write_iis)
    solver.iis_dir = str(RESULT_DIR / iis_dir)
    solver.debug_tag = (
        f"{debug_method or 'method'}_s{int(scenario_id)}_"
        f"{forecast_case}_{actual_case}"
    )
    if time_limit_s is not None:
        solver.time_limit_s = float(time_limit_s)
    if mip_gap is not None:
        solver.mip_gap = float(mip_gap)

    n_steps = int(DEFAULT_119["n_steps"])
    solver_window = int(horizon) + 1
    if solver_window < 2:
        raise ValueError("horizon must be at least 1.")
    if solver_window > n_steps:
        raise ValueError(f"horizon={horizon} is too long for {n_steps} steps.")

    state = initial_state_119()
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
    step_model_statuses: List[float] = []
    step_model_sol_counts: List[float] = []
    step_std_nn_binary_counts: List[float] = []
    step_icnn_clip_count: List[int] = []
    step_icnn_clip_max_violation: List[float] = []
    step_P_DG_total: List[float] = []
    step_SOC_mean: List[float] = []
    step_SOC_total: List[float] = []
    step_P_BSS: List[float] = []
    step_P_grid: List[float] = []
    step_loss: List[float] = []
    step_load_total: List[float] = []
    step_wt_total: List[float] = []
    step_pv_total: List[float] = []
    total_icnn_clip_count = 0
    max_icnn_clip_violation = 0.0
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
        step_model_statuses.append(stats.get("status", np.nan))
        step_model_sol_counts.append(stats.get("sol_count", np.nan))
        step_std_nn_binary_counts.append(
            getattr(solver, "std_nn_binary_count_last", np.nan)
        )

    def append_icnn_clip_stats() -> None:
        nonlocal total_icnn_clip_count, max_icnn_clip_violation

        clip_count_now = int(getattr(solver, "icnn_clip_count_last_solve", 0))
        clip_violation_now = float(
            getattr(solver, "icnn_clip_max_violation_last_solve", 0.0)
        )
        step_icnn_clip_count.append(clip_count_now)
        step_icnn_clip_max_violation.append(clip_violation_now)
        total_icnn_clip_count += clip_count_now
        max_icnn_clip_violation = max(max_icnn_clip_violation, clip_violation_now)

    def append_dispatch_trajectory(t: int) -> None:
        dispatch = getattr(solver, "last_dispatch", {}) or {}
        step_P_DG_total.append(float(dispatch.get("P_DG_total", np.nan)))
        step_SOC_mean.append(float(dispatch.get("SOC_mean", np.nan)))
        step_SOC_total.append(float(dispatch.get("SOC_total", np.nan)))
        step_P_BSS.append(float(dispatch.get("P_BSS", np.nan)))
        step_P_grid.append(float(dispatch.get("P_grid", np.nan)))
        step_loss.append(float(dispatch.get("Loss", np.nan)))
        step_load_total.append(float(np.sum(actual["load"][t])))
        step_wt_total.append(
            float(
                actual["wind_11"][t]
                + actual["wind_32"][t]
                + actual["wind_66"][t]
                + actual["wind_101"][t]
            )
        )
        step_pv_total.append(
            float(
                actual["pv_22"][t]
                + actual["pv_36"][t]
                + actual["pv_70"][t]
                + actual["pv_107"][t]
            )
        )

    def solve_step(
        *,
        current_time: int,
        window: int,
        forecast: ArrayDict,
        remainder: bool,
    ):
        return _run_solver_call(
            quiet_solver,
            solver,
            current_time,
            window,
            state,
            forecast,
            remainder=remainder,
        )

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
                new_P_DG_29,
                new_P_DG_64,
                new_SOC_BSS_12,
                new_SOC_BSS_34,
                new_SOC_BSS_68,
                new_SOC_BSS_103,
                cost_hour_now,
                solve_time,
            ) = solve_step(
                current_time=current_time,
                window=solver_window,
                forecast=forecast,
                remainder=False,
            )
            append_model_stats()
            append_icnn_clip_stats()
            append_dispatch_trajectory(current_time)

            cost_hour_now = float(cost_hour_now)
            solve_time = float(solve_time)
            total_cost += cost_hour_now
            total_solve_time += solve_time
            max_step_time = max(max_step_time, solve_time)
            step_costs.append(cost_hour_now)
            step_times.append(solve_time)
            state = {
                "P_DG_29": float(new_P_DG_29),
                "P_DG_64": float(new_P_DG_64),
                "SOC_BSS_12": float(new_SOC_BSS_12),
                "SOC_BSS_34": float(new_SOC_BSS_34),
                "SOC_BSS_68": float(new_SOC_BSS_68),
                "SOC_BSS_103": float(new_SOC_BSS_103),
            }

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
                    new_P_DG_29,
                    new_P_DG_64,
                    new_SOC_BSS_12,
                    new_SOC_BSS_34,
                    new_SOC_BSS_68,
                    new_SOC_BSS_103,
                    cost_hour_now,
                    solve_time,
                ) = solve_step(
                    current_time=current_time,
                    window=tail_window,
                    forecast=forecast,
                    remainder=True,
                )
                append_model_stats()
                append_icnn_clip_stats()
                append_dispatch_trajectory(current_time)

                cost_hour_now = float(cost_hour_now)
                solve_time = float(solve_time)
                total_cost += cost_hour_now
                total_solve_time += solve_time
                max_step_time = max(max_step_time, solve_time)
                step_costs.append(cost_hour_now)
                step_times.append(solve_time)
                state = {
                    "P_DG_29": float(new_P_DG_29),
                    "P_DG_64": float(new_P_DG_64),
                    "SOC_BSS_12": float(new_SOC_BSS_12),
                    "SOC_BSS_34": float(new_SOC_BSS_34),
                    "SOC_BSS_68": float(new_SOC_BSS_68),
                    "SOC_BSS_103": float(new_SOC_BSS_103),
                }

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
        final_P_DG_29=state["P_DG_29"],
        final_P_DG_64=state["P_DG_64"],
        final_SOC_BSS_12=state["SOC_BSS_12"],
        final_SOC_BSS_34=state["SOC_BSS_34"],
        final_SOC_BSS_68=state["SOC_BSS_68"],
        final_SOC_BSS_103=state["SOC_BSS_103"],
        step_costs_json=json.dumps(step_costs),
        step_times_json=json.dumps(step_times),
        step_P_DG_total_json=json.dumps(step_P_DG_total),
        step_SOC_mean_json=json.dumps(step_SOC_mean),
        step_SOC_total_json=json.dumps(step_SOC_total),
        step_P_BSS_json=json.dumps(step_P_BSS),
        step_P_grid_json=json.dumps(step_P_grid),
        step_loss_json=json.dumps(step_loss),
        step_load_total_json=json.dumps(step_load_total),
        step_wt_total_json=json.dumps(step_wt_total),
        step_pv_total_json=json.dumps(step_pv_total),
        mean_num_vars=mean_or_nan(step_num_vars),
        mean_num_bin_vars=mean_or_nan(step_num_bin_vars),
        mean_std_nn_binary_count=mean_or_nan(step_std_nn_binary_counts),
        mean_num_constrs=mean_or_nan(step_num_constrs),
        mean_num_qconstrs=mean_or_nan(step_num_qconstrs),
        mean_num_genconstrs=mean_or_nan(step_num_genconstrs),
        mean_model_runtime_s=mean_or_nan(step_model_runtimes),
        mean_mip_gap=mean_or_nan(step_mip_gaps),
        max_mip_gap=max_or_nan(step_mip_gaps),
        step_model_statuses_json=json.dumps(step_model_statuses),
        step_model_sol_counts_json=json.dumps(step_model_sol_counts),
        icnn_relax_on_infeasible=False,
        icnn_relax_count=0,
        icnn_relax_steps_json=json.dumps([]),
        step_icnn_relaxed_json=json.dumps([0] * executed_steps),
        icnn_exogenous_clip_enabled=bool(
            getattr(solver, "clip_icnn_exogenous_inputs", False)
        ),
        total_icnn_clip_count=int(total_icnn_clip_count),
        max_icnn_clip_violation=float(max_icnn_clip_violation),
        step_icnn_clip_count_json=json.dumps(step_icnn_clip_count),
        step_icnn_clip_max_violation_json=json.dumps(step_icnn_clip_max_violation),
        mean_model_status=mean_or_nan(step_model_statuses),
        min_model_sol_count=min_or_nan(step_model_sol_counts),
        num_optimal_steps=sum(1 for status in finite_values(step_model_statuses) if int(status) == 2),
        num_timelimit_steps=sum(1 for status in finite_values(step_model_statuses) if int(status) == 9),
        failure_message=failure_message,
    )


def parse_scenario_ids(raw: Optional[str], scenario_count: int) -> List[int]:
    if raw:
        if raw.strip().lower() == "all":
            return SPLITS["119"]["test_id"][:scenario_count]
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return SPLITS["119"]["test_id"][:scenario_count]


def parse_tasks(raw: str) -> List[Dict[str, Any]]:
    tasks = []
    aliases = {
        "mpc-3": "mpc3",
        "stdnn": "stdnn3",
        "std-nn": "stdnn3",
        "stdnn-h3": "stdnn3",
        "stdnn-h3-mip": "stdnn3",
        "proposed": "proposed3",
        "proposed-h3": "proposed3",
    }
    for task_name in [item.strip().lower() for item in raw.split(",") if item.strip()]:
        key = aliases.get(task_name, task_name)
        if key not in TASK_PRESETS:
            raise ValueError(
                f"Unknown task '{task_name}'. Expected one of {sorted(TASK_PRESETS)}."
            )
        tasks.append(dict(TASK_PRESETS[key]))
    return tasks


def selection_rule_from_file(raw_path: Optional[str]) -> str:
    if not raw_path:
        return ""
    name = Path(raw_path).name.lower()
    if "operable" in name:
        return "operable_perfect_global_and_rolling_mpc3_subset"
    if "perfect_feasible" in name:
        return "perfect_global_feasible_certified_subset"
    return "external_selected_scenario_subset"


def run_batch(args: argparse.Namespace) -> Path:
    scenario_ids = resolve_scenarios(
        system="119",
        scenarios=args.scenarios,
        scenario_file=args.scenario_file,
        scenario_count=args.scenario_count,
    )
    tasks = parse_tasks(args.tasks)
    selection_rule = selection_rule_from_file(args.scenario_file)

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for task in tasks:
            print(
                f"Running {task['method']} scenario={scenario_id} "
                f"forecast={args.forecast_case} actual={args.actual_case}"
            )
            row = run_one_scenario_119(
                method=task["method"],
                solver_kind=task["solver_kind"],
                scenario_id=scenario_id,
                forecast_case=args.forecast_case,
                actual_case=args.actual_case,
                horizon=task["horizon"],
                gamma_Q=args.gamma_Q
                if task["solver_kind"] in {"proposed", "std_nn_mpc"}
                and args.gamma_Q is not None
                else task["gamma_Q"],
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
                time_limit_s=args.time_limit_s,
                mip_gap=args.mip_gap,
                write_iis=args.write_iis,
                iis_dir=args.iis_dir,
                clip_icnn_exogenous_inputs=not args.no_clip_icnn_exogenous_inputs,
            )
            row["scenario_selection_file"] = args.scenario_file or ""
            row["scenario_selection_rule"] = selection_rule
            logger.add_row(row)
            print(
                {
                    "method": row["method"],
                    "scenario_id": row["scenario_id"],
                    "cost": row["cost"],
                    "solve_time_s": row["solve_time_s"],
                    "executed_steps": row["executed_steps"],
                    "solver_fail_count": row["solver_fail_count"],
                    "total_icnn_clip_count": row.get("total_icnn_clip_count", ""),
                }
            )

    return logger.save()


def dry_run(args: argparse.Namespace) -> None:
    scenario_ids = resolve_scenarios(
        system="119",
        scenarios=args.scenarios,
        scenario_file=args.scenario_file,
        scenario_count=args.scenario_count,
    )
    tasks = parse_tasks(args.tasks)
    print("Dry-run plan")
    print("scenarios:", scenario_ids)
    print("tasks:", tasks)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("scenario_file:", args.scenario_file or "")
    print("max_steps:", args.max_steps)
    print("write_iis:", args.write_iis)
    print("iis_dir:", args.iis_dir)
    print("clip_icnn_exogenous_inputs:", not args.no_clip_icnn_exogenous_inputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 119-bus pure MPC and MPC-ICNN through experiments_v2 protocols."
    )
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario IDs or 'all'.")
    parser.add_argument(
        "--scenario-file",
        default=None,
        help="JSON/CSV file under results_v2 containing selected scenarios. Overrides --scenarios.",
    )
    parser.add_argument("--scenario-count", type=int, default=2)
    parser.add_argument(
        "--tasks",
        default="mpc3,proposed3",
        help="Comma-separated task names: mpc3,stdnn3,proposed3.",
    )
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default=None, help="Float value or 'auto'.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
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
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="step10_119_mpc_smoke_test.csv")
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
