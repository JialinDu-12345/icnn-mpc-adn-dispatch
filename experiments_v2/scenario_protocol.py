from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from experiments_v2.config import (
    ACTUAL_CASES,
    DATA_DIRS,
    DEVICE_ACTIVE_CAPS,
    FORECAST_CASES,
    SPLITS,
    SYSTEM_KEYS,
    validate_actual_case,
    validate_system,
)


ArrayDict = Dict[str, np.ndarray]


def _as_tuple(scenario_ids: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(scenario_id) for scenario_id in scenario_ids)


def stable_int_hash(*items: object) -> int:
    text = "|".join(str(item) for item in items)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _scenario_path(system: str, scenario_id: int):
    validate_system(system)
    return DATA_DIRS[system] / f"results_{int(scenario_id)}.npz"


def load_scenario(system: str, scenario_id: int) -> ArrayDict:
                                                             
    path = _scenario_path(system, scenario_id)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with np.load(path) as data:
        expected_keys = SYSTEM_KEYS[system]["all_keys"]
        missing = [key for key in expected_keys if key not in data.files]
        if missing:
            raise KeyError(
                f"Missing keys {missing} in {path}. Existing keys: {list(data.files)}"
            )
        scenario = {key: np.asarray(data[key], dtype=float) for key in expected_keys}

    scenario = apply_system_exogenous_envelope(system, scenario)
    validate_scenario_shapes(system, scenario, source=str(path))
    return scenario


def apply_actual_case(system: str, scenario: ArrayDict, actual_case_config: dict) -> ArrayDict:
                                                                                  
    validate_system(system)
    load_scale = float(actual_case_config.get("load_scale", 1.0))
    renewable_scale = float(actual_case_config.get("renewable_scale", 1.0))

    adjusted = {key: value.copy() for key, value in scenario.items()}
    for key in SYSTEM_KEYS[system]["load_keys"]:
        adjusted[key] = clip_nonnegative(adjusted[key] * load_scale)
    for key in SYSTEM_KEYS[system]["renewable_keys"]:
        adjusted[key] = clip_nonnegative(adjusted[key] * renewable_scale)

    adjusted = apply_system_exogenous_envelope(system, adjusted)
    validate_scenario_shapes(system, adjusted, source="actual_case scenario")
    return adjusted


def load_actual_scenario(
    system: str,
    scenario_id: int,
    actual_case: str = "id_actual",
) -> ArrayDict:
    validate_actual_case(actual_case)
    scenario = load_scenario(system, scenario_id)
    return apply_actual_case(system, scenario, ACTUAL_CASES[actual_case])


def validate_scenario_shapes(system: str, scenario: ArrayDict, source: str = "scenario") -> None:
                                                                                 
    validate_system(system)
    n_steps = SYSTEM_KEYS[system]["n_steps"]
    n_load_buses = SYSTEM_KEYS[system]["n_load_buses"]

    for key in SYSTEM_KEYS[system]["all_keys"]:
        if key not in scenario:
            raise KeyError(f"Missing key '{key}' in {source}.")

    for key in SYSTEM_KEYS[system]["renewable_keys"]:
        if scenario[key].shape != (n_steps,):
            raise ValueError(
                f"{source}: key '{key}' has shape {scenario[key].shape}, "
                f"expected {(n_steps,)}."
            )

    for key in SYSTEM_KEYS[system]["load_keys"]:
        if scenario[key].shape != (n_steps, n_load_buses):
            raise ValueError(
                f"{source}: key '{key}' has shape {scenario[key].shape}, "
                f"expected {(n_steps, n_load_buses)}."
            )

    validate_device_caps(system, scenario, source=source)


def clip_device_active_caps(system: str, scenario: ArrayDict, margin: float = 1e-6) -> ArrayDict:
                                                                           
    validate_system(system)
    clipped = {key: value.copy() for key, value in scenario.items()}

    for key, cap in DEVICE_ACTIVE_CAPS.get(system, {}).items():
        if key not in clipped:
            continue
        upper = float(cap) * (1.0 - margin)
        clipped[key] = np.clip(clipped[key], 0.0, upper)

    return clipped


def clip_device_active_value(system: str, key: str, value: np.ndarray) -> np.ndarray:
    clipped = clip_nonnegative(np.asarray(value, dtype=float))
    cap = DEVICE_ACTIVE_CAPS.get(system, {}).get(key)
    if cap is None:
        return clipped
    return np.minimum(clipped, float(cap) * (1.0 - 1e-6))


@lru_cache(maxsize=4)
def load_legacy_load_envelope(system: str) -> Optional[Dict[str, np.ndarray]]:
                                                                            

    validate_system(system)
    if system != "119":
        return None

    path = DATA_DIRS[system] / "legacy_119_load_envelope.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing 119 load envelope asset: {path}. "
            "Run: python -m experiments_v2.utils.export_119_legacy_envelope"
        )

    with np.load(path) as data:
        return {
            "load_min": np.asarray(data["load_min"], dtype=float),
            "load_max": np.asarray(data["load_max"], dtype=float),
            "load_min_Q": np.asarray(data["load_min_Q"], dtype=float),
            "load_max_Q": np.asarray(data["load_max_Q"], dtype=float),
        }


def clip_legacy_load_envelope_value(
    system: str,
    key: str,
    value: np.ndarray,
) -> np.ndarray:
    clipped = clip_nonnegative(np.asarray(value, dtype=float))
    envelope = load_legacy_load_envelope(system)
    if envelope is None:
        return clipped
    if key == "load":
        return np.clip(clipped, envelope["load_min"], envelope["load_max"])
    if key == "load_Q":
        return np.clip(clipped, envelope["load_min_Q"], envelope["load_max_Q"])
    return clipped


def clip_system_exogenous_value(system: str, key: str, value: np.ndarray) -> np.ndarray:
    clipped = clip_legacy_load_envelope_value(system, key, value)
    return clip_device_active_value(system, key, clipped)


def apply_system_exogenous_envelope(system: str, scenario: ArrayDict) -> ArrayDict:
    validate_system(system)
    return {
        key: clip_system_exogenous_value(system, key, value)
        for key, value in scenario.items()
    }


def validate_device_caps(system: str, scenario: ArrayDict, source: str = "scenario") -> None:
    for key, cap in DEVICE_ACTIVE_CAPS.get(system, {}).items():
        if key not in scenario:
            continue

        max_value = float(np.max(scenario[key]))
        min_value = float(np.min(scenario[key]))
        if min_value < -1e-8:
            raise ValueError(f"{source}: key '{key}' has negative value {min_value}.")
        if max_value > float(cap) + 1e-8:
            raise ValueError(
                f"{source}: key '{key}' exceeds device cap. "
                f"max={max_value}, cap={cap}."
            )


@lru_cache(maxsize=16)
def _build_typical_profile_cached(system: str, scenario_ids: Tuple[int, ...]) -> ArrayDict:
    buckets = {key: [] for key in SYSTEM_KEYS[system]["all_keys"]}
    for scenario_id in scenario_ids:
        scenario = load_scenario(system, scenario_id)
        for key, value in scenario.items():
            buckets[key].append(value)

    typical = {
        key: np.mean(np.stack(values, axis=0), axis=0)
        for key, values in buckets.items()
    }
    validate_scenario_shapes(system, typical, source=f"{system} typical profile")
    return typical


def build_typical_profile(system: str, scenario_ids: List[int]) -> ArrayDict:
                                                                        
    validate_system(system)
    if not scenario_ids:
        raise ValueError("scenario_ids cannot be empty.")

    cached = _build_typical_profile_cached(system, _as_tuple(scenario_ids))
    return {key: value.copy() for key, value in cached.items()}


def build_historical_typical_profile(system: str) -> ArrayDict:
                                                                    

                                                                          
                                                                     
       
    return build_typical_profile(system, SPLITS[system]["historical"])


def build_training_typical_profile(system: str) -> ArrayDict:
                                      

                                                   
       
    return build_historical_typical_profile(system)


def clip_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass(frozen=True)
class ForecastConfig:
    error_multiplier: float = 1.0
    load_bias: float = 0.0
    renewable_bias: float = 0.0
    error_base: float = 0.03
    error_slope: float = 0.04
    eps: float = 1e-6
    ratio_min: float = 0.3
    ratio_max: float = 3.0
    min_reference: float = 1e-4
    forecast_min_scale: float = 0.5
    forecast_max_scale: float = 1.5
    use_actual_future: bool = False


class RollingForecaster:
                                                                             

                                          
                                                                        
                                                             
       

    def __init__(
        self,
        system: str,
        typical_profile: ArrayDict,
        config: ForecastConfig,
        seed: int,
    ):
        validate_system(system)
        validate_scenario_shapes(system, typical_profile, source="typical_profile")

        self.system = system
        self.typical = {key: value.copy() for key, value in typical_profile.items()}
        self.config = config
        self.seed = int(seed)

        self.load_keys = set(SYSTEM_KEYS[system]["load_keys"])
        self.renewable_keys = set(SYSTEM_KEYS[system]["renewable_keys"])
        self.all_keys = SYSTEM_KEYS[system]["all_keys"]
        self.n_steps = SYSTEM_KEYS[system]["n_steps"]

    def _bias_for_key(self, key: str) -> float:
        if key in self.load_keys:
            return self.config.load_bias
        if key in self.renewable_keys:
            return self.config.renewable_bias
        return 0.0

    def predict_one_key(
        self,
        actual: ArrayDict,
        key: str,
        current_time: int,
        horizon_offset: int,
    ) -> np.ndarray:
        if key not in self.all_keys:
            raise KeyError(f"Unknown key '{key}' for system {self.system}.")
        if current_time < 0 or current_time >= self.n_steps:
            raise ValueError(f"current_time must be in [0, {self.n_steps - 1}].")
        if horizon_offset < 0:
            raise ValueError("horizon_offset must be nonnegative.")

        target_time = min(current_time + horizon_offset, self.n_steps - 1)

        if self.config.use_actual_future:
            return clip_system_exogenous_value(
                self.system,
                key,
                np.asarray(actual[key][target_time], dtype=float).copy(),
            )

        if horizon_offset == 0:
            return clip_system_exogenous_value(
                self.system,
                key,
                np.asarray(actual[key][current_time], dtype=float).copy(),
            )

        actual_now = np.asarray(actual[key][current_time], dtype=float)
        typical_now = self.typical[key][current_time]
        typical_future = self.typical[key][target_time]
        raw_ratio = actual_now / (typical_now + self.config.eps)
        ratio = np.where(
            typical_now > self.config.min_reference,
            np.clip(raw_ratio, self.config.ratio_min, self.config.ratio_max),
            1.0,
        )
        baseline = typical_future * ratio

        sigma = self.config.error_multiplier * (
            self.config.error_base + self.config.error_slope * horizon_offset
        )
        noise_seed = stable_int_hash(
            self.system,
            self.seed,
            key,
            current_time,
            horizon_offset,
        )
        rng = np.random.default_rng(noise_seed)
        error = rng.normal(
            loc=self._bias_for_key(key),
            scale=sigma,
            size=np.shape(baseline),
        )
        forecast = baseline * (1.0 + error)
        lower = self.config.forecast_min_scale * typical_future
        upper = self.config.forecast_max_scale * typical_future
        forecast = np.where(
            typical_future > self.config.min_reference,
            np.clip(forecast, lower, upper),
            forecast,
        )
        return clip_system_exogenous_value(self.system, key, forecast)

    def make_forecast_arrays(
        self,
        actual: ArrayDict,
        current_time: int,
        window_time: int,
    ) -> ArrayDict:
                                                                        

                                                                            
                                                                               
                                         
           
        validate_scenario_shapes(self.system, actual, source="actual")
        if window_time <= 0:
            raise ValueError("window_time must be positive.")
        if current_time < 0 or current_time >= self.n_steps:
            raise ValueError(f"current_time must be in [0, {self.n_steps - 1}].")

        forecast = {key: value.copy() for key, value in actual.items()}
        for h in range(window_time):
            t = current_time + h
            if t >= self.n_steps:
                break
            for key in self.all_keys:
                forecast[key][t] = self.predict_one_key(
                    actual=actual,
                    key=key,
                    current_time=current_time,
                    horizon_offset=h,
                )

        validate_scenario_shapes(self.system, forecast, source="forecast")
        return forecast


def make_forecaster(
    system: str,
    forecast_case_config: dict,
    seed: int,
) -> RollingForecaster:
    validate_system(system)
    typical_profile = build_historical_typical_profile(system)
    config = ForecastConfig(**forecast_case_config)
    return RollingForecaster(
        system=system,
        typical_profile=typical_profile,
        config=config,
        seed=seed,
    )


def make_forecaster_by_case(system: str, forecast_case: str, seed: int) -> RollingForecaster:
    if forecast_case not in FORECAST_CASES:
        raise ValueError(
            f"Unknown forecast case '{forecast_case}'. "
            f"Expected one of {sorted(FORECAST_CASES)}."
        )
    return make_forecaster(system, FORECAST_CASES[forecast_case], seed)


def get_solver_inputs_33(forecast: ArrayDict):
                                                                  
    validate_scenario_shapes("33", forecast, source="forecast")
    return (
        forecast["load"],
        forecast["pv_11"],
        forecast["wind_26"],
        forecast["load_Q"],
    )


def get_solver_inputs_119(forecast: ArrayDict):
                                                                   
    validate_scenario_shapes("119", forecast, source="forecast")
    return (
        forecast["load"],
        forecast["load_Q"],
        forecast["wind_11"],
        forecast["wind_32"],
        forecast["wind_66"],
        forecast["wind_101"],
        forecast["pv_22"],
        forecast["pv_36"],
        forecast["pv_70"],
        forecast["pv_107"],
    )
