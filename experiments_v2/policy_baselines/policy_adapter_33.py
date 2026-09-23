from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from experiments_v2.config import DEFAULT_BASE_SEEDS
from experiments_v2.result_logger import make_result_row
from experiments_v2.scenario_protocol import load_actual_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_ASSET_ROOT = PROJECT_ROOT / "experiment_assets" / "policies" / "33"

POLICY_METHODS = {
    "SAC": {
        "asset_dir": POLICY_ASSET_ROOT / "sac",
        "legacy_module_name": "legacy_policy_33_sac",
        "saclag": False,
    },
    "SAC-ICNN": {
        "asset_dir": POLICY_ASSET_ROOT / "sac_icnn",
        "legacy_module_name": "legacy_policy_33_sac_icnn",
        "saclag": False,
    },
    "SACLag": {
        "asset_dir": POLICY_ASSET_ROOT / "saclag",
        "legacy_module_name": "legacy_policy_33_saclag",
        "saclag": True,
    },
}


class PolicyDependencyError(RuntimeError):
    pass                                                                                 


def load_actor_state_dict(model_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find copied actor checkpoint: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict):
        for nested_key in ("actor_state_dict", "actor", "model_state_dict", "state_dict"):
            nested = checkpoint.get(nested_key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported actor checkpoint format: {model_path}")
    return checkpoint


def infer_hidden_dim_from_actor_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for preferred_key in ("fc1.weight", "net.0.weight", "actor.fc1.weight", "module.fc1.weight"):
        value = state_dict.get(preferred_key)
        if hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])

    for key, value in state_dict.items():
        if not hasattr(value, "shape") or len(value.shape) < 1:
            continue
        if key.endswith("fc1.weight") or key.endswith("net.0.weight"):
            return int(value.shape[0])

    raise RuntimeError(
        "Cannot infer hidden_dim from actor state_dict. "
        f"Available keys: {list(state_dict)[:20]}"
    )


def _read_legacy_policy_definitions(file_path: Path) -> str:
    source = file_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    lines = source.splitlines(keepends=True)
    cut_index = len(lines)
    for idx, line in enumerate(lines):
        if line.strip().startswith("env = PowerSystemEnv()"):
            cut_index = idx
            break
    return "".join(lines[:cut_index])


def load_legacy_policy_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find copied policy source: {file_path}")

    source = _read_legacy_policy_definitions(file_path)
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    sys.modules[module_name] = module

    old_cwd = Path.cwd()
    sys.path.insert(0, str(file_path.parent))
    try:
        os.chdir(file_path.parent)
        exec(compile(source, str(file_path), "exec"), module.__dict__)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise PolicyDependencyError(
            f"Cannot load {file_path.name} because dependency '{missing}' is not installed "
            "in the active Python environment. The copied policy assets are present under "
            f"{file_path.parent}, so install the missing dependency or run with a Python "
            "environment that has the old policy dependencies."
        ) from exc
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)

    return module


def reset_legacy_runtime_globals(module: ModuleType) -> None:
                                                                       

                                                                        
                                                                              
                                                                             
                           
       
    dict_plot = defaultdict(list)
    for key in ["P_DG_5", "SOC_BSS_9", "Loss", "P_BSS", "P_G"]:
        dict_plot[key] = []
    module.dict_plot = dict_plot

    for name in [
        "list_cost",
        "list_cost_c",
        "list_time_",
        "objlist",
        "costlist",
        "rewardlist",
        "return_list",
        "cost_c_list",
        "voltage_violation_list",
    ]:
        setattr(module, name, [])


def time_of_use_price(hour: int) -> float:
    if hour < 12:
        return 64.0
    if 24 <= hour < 36:
        return 185.0
    return 123.0


def generator_economic_cost(p_mw: float) -> float:
    return float((0.00240 * float(p_mw) ** 2 + 12.3299 * float(p_mw)) / 2.0)


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if np.isfinite(value)]


def sum_or_nan(values: Iterable[float]) -> float:
    clean = finite_values(values)
    return float(np.sum(clean)) if clean else np.nan


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite_values(values)
    return float(np.mean(clean)) if clean else np.nan


def voltage_violation_from_env(env: Any, vmin: float = 0.94, vmax: float = 1.06) -> float:
                                                                                           
    try:
        vm = np.asarray(env.net.res_bus["vm_pu"].values, dtype=float).reshape(-1)
    except Exception:
        return np.nan
    if vm.size == 0:
        return np.nan

    upper = np.maximum(vm - float(vmax), 0.0)
    lower = np.maximum(float(vmin) - vm, 0.0)
    return float(np.sum(upper + lower))


def set_policy_eval_seed(seed: int, device: torch.device) -> None:
                                                                              
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def latest_plot_value(module: ModuleType, key: str, default: float = np.nan) -> float:
    try:
        values = getattr(module, "dict_plot", {}).get(key, [])
        if not values:
            return float(default)
        return float(values[-1])
    except Exception:
        return float(default)


def policy_step_cost_components(
    *,
    reward: float,
    info: Dict[str, Any],
    hour: int,
) -> Dict[str, float]:
                                                                                  

                                       
                                                                    

                                                                               
                                                                              
       
    penalized_cost = -100.0 * float(reward)
    economic_cost = np.nan
    balance_penalty = np.nan
    balance_error = np.nan

    if isinstance(info, dict):
        sgen_p_raw = info.get("sgen_p")
        if sgen_p_raw is not None:
            sgen_p = np.asarray(sgen_p_raw, dtype=float).reshape(-1)
            if sgen_p.size > 0:
                economic_cost = generator_economic_cost(float(sgen_p[0]))

        total_loss = info.get("total_loss")
        load_p_raw = info.get("load_p")
        total_sgen = info.get("total_sgen")
        if total_loss is not None and load_p_raw is not None and total_sgen is not None:
            total_load = float(np.sum(np.asarray(load_p_raw, dtype=float)))
            balance_error = total_load + float(total_loss) - float(total_sgen)
            balance_penalty = time_of_use_price(hour) * max(0.0, balance_error) / 2.0

    if not np.isfinite(economic_cost) and np.isfinite(balance_penalty):
        economic_cost = penalized_cost - balance_penalty
    if not np.isfinite(balance_penalty) and np.isfinite(economic_cost):
        balance_penalty = penalized_cost - economic_cost

    reward_penalty = (
        penalized_cost - economic_cost if np.isfinite(economic_cost) else np.nan
    )
    return {
        "economic_cost": float(economic_cost),
        "penalized_cost": float(penalized_cost),
        "balance_penalty": float(balance_penalty),
        "balance_error": float(balance_error),
        "reward_penalty": float(reward_penalty),
    }


def _split_for_scenario(scenario_id: int) -> str:
    if 80 <= int(scenario_id) < 100:
        return "test_id"
    if 70 <= int(scenario_id) < 80:
        return "val"
    if 0 <= int(scenario_id) < 70:
        return "historical"
    return "custom"


class PolicyAdapter33:
                                                                    

    def __init__(
        self,
        method: str,
        device: str = "cpu",
        quiet_legacy_init: bool = True,
    ):
        if method not in POLICY_METHODS:
            raise ValueError(f"Unknown policy method '{method}'. Expected {list(POLICY_METHODS)}.")

        self.method = method
        self.spec = POLICY_METHODS[method]
        self.asset_dir = Path(self.spec["asset_dir"])
        self.source_path = self.asset_dir / "SAC_largesystem.py"
        self.model_path = self.asset_dir / "model.pth"
        self.device = torch.device(device)
        self.quiet_legacy_init = bool(quiet_legacy_init)

        self.module = load_legacy_policy_module(
            module_name=str(self.spec["legacy_module_name"]),
            file_path=self.source_path,
        )
        reset_legacy_runtime_globals(self.module)
        self.env = self._build_env()
        self.agent = self._build_agent()

    def _build_env(self):
        if self.quiet_legacy_init:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink):
                    return self.module.PowerSystemEnv()
        return self.module.PowerSystemEnv()

    def _build_agent(self):
        state_dim = int(self.env.observation_space.shape[0])
        action_dim = int(self.env.action_space.shape[0])
        state_dict = load_actor_state_dict(self.model_path, self.device)
        hidden_dim = infer_hidden_dim_from_actor_state_dict(state_dict)
        actor_lr = 3e-4
        critic_lr = 3e-3
        alpha_lr = 3e-4
        target_entropy = -action_dim
        tau = 0.005
        gamma = 0.99

        if bool(self.spec["saclag"]):
            agent = self.module.SACContinuous(
                state_dim,
                hidden_dim,
                action_dim,
                actor_lr,
                critic_lr,
                alpha_lr,
                target_entropy,
                tau,
                gamma,
                self.device,
                0.99,
                0.0,
            )
        else:
            agent = self.module.SACContinuous(
                state_dim,
                hidden_dim,
                action_dim,
                actor_lr,
                critic_lr,
                alpha_lr,
                target_entropy,
                tau,
                gamma,
                self.device,
            )

        agent.actor.load_state_dict(state_dict)
        agent.actor.eval()
        return agent

    def _reset_with_actual(self, actual: Dict[str, np.ndarray]) -> np.ndarray:
        return self.env.new_reset(
            actual["pv_11"].copy(),
            actual["wind_26"].copy(),
            actual["load"].copy(),
            actual["load_Q"].copy(),
        )

    def _policy_step(self, state: np.ndarray, stochastic: bool) -> tuple:
        current_hour = int(getattr(self.env, "hour", 0))
        action = self.agent.take_action(state, stochastic=stochastic)
        step_fn = self.env.step if bool(self.spec["saclag"]) else getattr(self.env, "step_1", self.env.step)
        output = step_fn(action)
        if len(output) == 4:
            next_state, reward, done, info = output
            cost_c = np.nan
        elif len(output) == 5:
            next_state, reward, cost_c, done, info = output
        else:
            raise RuntimeError(f"Unexpected env.step output length: {len(output)}")
        components = policy_step_cost_components(
            reward=float(reward),
            info=info if isinstance(info, dict) else {},
            hour=current_hour,
        )
        return next_state, float(reward), float(cost_c), bool(done), info, components

    def run_one_day(
        self,
        scenario_id: int,
        actual_case: str = "id_actual",
        max_steps: Optional[int] = None,
        stochastic: bool = False,
        eval_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        reset_legacy_runtime_globals(self.module)
        if eval_seed is not None:
            set_policy_eval_seed(int(eval_seed), self.device)

        actual = load_actual_scenario("33", scenario_id, actual_case=actual_case)
        state = self._reset_with_actual(actual)

        n_steps = 48 if max_steps is None else min(48, int(max_steps))
        step_costs: List[float] = []
        step_economic_costs: List[float] = []
        step_penalized_costs: List[float] = []
        step_balance_penalties: List[float] = []
        step_reward_penalties: List[float] = []
        step_balance_errors: List[float] = []
        step_times: List[float] = []
        step_cost_constraints: List[float] = []
        step_legacy_cost_constraints: List[float] = []
        step_P_DG_5: List[float] = []
        step_SOC_BSS_9: List[float] = []
        step_P_BSS: List[float] = []
        step_P_grid: List[float] = []
        step_loss: List[float] = []
        step_load_total: List[float] = []
        step_pv_11: List[float] = []
        step_wind_26: List[float] = []
        fail_count = 0
        incomplete_reason = ""
        failure_message = ""

        for _step in range(n_steps):
            try:
                start = time.perf_counter()
                state, reward, cost_c, done, _info, cost_components = self._policy_step(
                    state,
                    stochastic=stochastic,
                )
                elapsed = time.perf_counter() - start

                economic_cost = float(cost_components["economic_cost"])
                penalized_cost = float(cost_components["penalized_cost"])
                balance_penalty = float(cost_components["balance_penalty"])
                reward_penalty = float(cost_components["reward_penalty"])
                balance_error = float(cost_components["balance_error"])

                if not np.isfinite(economic_cost):
                    economic_cost = penalized_cost

                step_costs.append(penalized_cost)
                step_economic_costs.append(economic_cost)
                step_penalized_costs.append(penalized_cost)
                step_balance_penalties.append(balance_penalty)
                step_reward_penalties.append(reward_penalty)
                step_balance_errors.append(balance_error)
                step_times.append(float(elapsed))
                legacy_cost_c = float(cost_c)
                vio_pu = voltage_violation_from_env(self.env, vmin=0.94, vmax=1.06)
                step_cost_constraints.append(float(vio_pu))
                step_legacy_cost_constraints.append(legacy_cost_c)

                step_index = len(step_costs) - 1
                step_P_DG_5.append(latest_plot_value(self.module, "P_DG_5"))
                step_SOC_BSS_9.append(latest_plot_value(self.module, "SOC_BSS_9"))
                step_P_BSS.append(latest_plot_value(self.module, "P_BSS"))
                step_P_grid.append(latest_plot_value(self.module, "P_G"))
                step_loss.append(latest_plot_value(self.module, "Loss"))
                step_load_total.append(float(np.sum(actual["load"][step_index])))
                step_pv_11.append(float(actual["pv_11"][step_index]))
                step_wind_26.append(float(actual["wind_26"][step_index]))

                if done:
                    break
            except Exception as exc:
                fail_count += 1
                incomplete_reason = "policy_failure"
                if isinstance(exc, NameError) and "dict_plot" in str(exc):
                    failure_message = (
                        f"method={self.method}, scenario={scenario_id}, actual_case={actual_case}: "
                        "legacy global variable 'dict_plot' is missing. "
                        "Call reset_legacy_runtime_globals(self.module) before policy evaluation."
                    )
                else:
                    failure_message = (
                        f"method={self.method}, scenario={scenario_id}, "
                        f"actual_case={actual_case}: {exc}"
                    )
                print(f"[Failed] {failure_message}")
                break

        executed_steps = len(step_costs)
        completed = executed_steps == 48 and fail_count == 0
        if completed:
            incomplete_reason = ""
        elif incomplete_reason:
            pass
        elif max_steps is not None:
            incomplete_reason = "max_steps_debug"
        else:
            incomplete_reason = "incomplete"

        total_time = float(np.sum(step_times)) if step_times else 0.0
        primary_cost_total = sum_or_nan(step_costs) if step_costs else 0.0
        economic_total = sum_or_nan(step_economic_costs) if step_economic_costs else 0.0
        penalized_total = sum_or_nan(step_penalized_costs) if step_penalized_costs else 0.0
        balance_penalty_total = sum_or_nan(step_balance_penalties)
        reward_penalty_total = sum_or_nan(step_reward_penalties)
        voltage_violation_total = sum_or_nan(step_cost_constraints)
        return make_result_row(
            system="33",
            method=self.method,
            scenario_id=scenario_id,
            split=_split_for_scenario(scenario_id),
            actual_case=actual_case,
            forecast_case="not_applicable",
            horizon=0,
            gamma_Q=None,
            cost=primary_cost_total,
            perfect_cost=None,
            gap_percent=None,
            voltage_violation=voltage_violation_total,
            solve_time_s=total_time,
            avg_step_time_s=total_time / max(executed_steps, 1),
            max_step_time_s=float(np.max(step_times)) if step_times else 0.0,
            solver_fail_count=fail_count,
            timeout_count=0,
            completed=completed,
            incomplete_reason=incomplete_reason,
            seed=DEFAULT_BASE_SEEDS["33"] + int(scenario_id),
            solver_kind="policy",
            solver_window=0,
            executed_steps=executed_steps,
            include_tail=False,
            max_steps=max_steps,
            policy_asset_dir=str(self.asset_dir.relative_to(PROJECT_ROOT)),
            policy_stochastic_eval=bool(stochastic),
            policy_eval_seed=eval_seed,
            policy_eval_mode="stochastic_sampled_action"
            if stochastic
            else "deterministic_mean_action",
            cost_accounting="legacy_penalized_operating_cost",
            voltage_violation_metric="env.net.res_bus.vm_pu, vmin=0.94, vmax=1.06",
            penalized_cost=penalized_total,
            economic_generation_cost=economic_total,
            balance_penalty_cost=balance_penalty_total,
            constraint_penalty=reward_penalty_total,
            step_costs_json=json.dumps(step_costs),
            step_economic_costs_json=json.dumps(step_economic_costs),
            step_penalized_costs_json=json.dumps(step_penalized_costs),
            step_balance_penalties_json=json.dumps(step_balance_penalties),
            step_reward_penalties_json=json.dumps(step_reward_penalties),
            step_balance_errors_json=json.dumps(step_balance_errors),
            step_times_json=json.dumps(step_times),
            step_P_DG_5_json=json.dumps(step_P_DG_5),
            step_SOC_BSS_9_json=json.dumps(step_SOC_BSS_9),
            step_P_BSS_json=json.dumps(step_P_BSS),
            step_P_grid_json=json.dumps(step_P_grid),
            step_loss_json=json.dumps(step_loss),
            step_load_total_json=json.dumps(step_load_total),
            step_pv_11_json=json.dumps(step_pv_11),
            step_wind_26_json=json.dumps(step_wind_26),
            step_cost_constraints_json=json.dumps(step_cost_constraints),
            step_voltage_violations_pu_json=json.dumps(step_cost_constraints),
            step_legacy_cost_constraints_json=json.dumps(step_legacy_cost_constraints),
            mean_economic_generation_cost=mean_or_nan(step_economic_costs),
            mean_penalized_cost=mean_or_nan(step_penalized_costs),
            mean_balance_penalty=mean_or_nan(step_balance_penalties),
            mean_constraint_penalty=mean_or_nan(step_reward_penalties),
            mean_cost_constraint=mean_or_nan(step_cost_constraints),
            mean_voltage_violation_pu=mean_or_nan(step_cost_constraints),
            legacy_mean_cost_constraint=mean_or_nan(step_legacy_cost_constraints),
            legacy_cum_cost_constraint=sum_or_nan(step_legacy_cost_constraints),
            failure_message=failure_message,
            experiment_tag="policy_baseline",
        )


def available_policy_methods() -> List[str]:
    return list(POLICY_METHODS)
