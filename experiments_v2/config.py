from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("EXPERIMENT_ASSET_ROOT", PROJECT_ROOT / "experiment_assets"))
RESULT_DIR = PROJECT_ROOT / "results_v2"

DATA_DIRS = {
    "33": ASSET_ROOT / "data" / "33",
    "119": ASSET_ROOT / "data" / "119",
}

METHOD_DIRS_33 = {
    "pure_mpc": ASSET_ROOT / "solvers" / "33" / "pure_mpc",
    "pure_mpc_clean": ASSET_ROOT / "solvers" / "33" / "pure_mpc_clean",
    "proposed": ASSET_ROOT / "solvers" / "33" / "mpc_icnn",
    "std_nn_mpc": ASSET_ROOT / "solvers" / "33" / "std_nn_mpc",
    "perfect_global_clean": ASSET_ROOT / "solvers" / "33" / "perfect_global_clean",
}

METHOD_DIRS_119 = {
    "pure_mpc_clean": ASSET_ROOT / "solvers" / "119" / "pure_mpc_clean",
    "proposed": ASSET_ROOT / "solvers" / "119" / "mpc_icnn",
    "std_nn_mpc": ASSET_ROOT / "solvers" / "119" / "std_nn_mpc",
    "perfect_global_clean": ASSET_ROOT / "solvers" / "119" / "perfect_global_clean",
}

POLICY_DIRS_119 = {
    "SAC": ASSET_ROOT / "policies" / "119" / "sac",
    "SAC-ICNN": ASSET_ROOT / "policies" / "119" / "sac_icnn",
}

DEFAULT_33 = {
    "n_steps": 48,
    "default_horizon": 3,
    "P_DG_5_init": 0.0,
    "SOC_BSS_9_init": 1.25,
}

DEFAULT_119 = {
    "n_steps": 48,
    "default_horizon": 3,
    "P_DG_29_init": 0.0,
    "P_DG_64_init": 0.0,
    "SOC_BSS_12_init": 1.25,
    "SOC_BSS_34_init": 1.25,
    "SOC_BSS_68_init": 1.25,
    "SOC_BSS_103_init": 1.25,
}

GAMMA_Q_CANDIDATES_33 = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

SELECTED_GAMMA_Q = {
    "33": None,
    "119": None,
}

DEVICE_ACTIVE_CAPS = {
    "33": {
        "pv_11": 3.0,
        "wind_26": 2.0,
    },
    "119": {
        "wind_11": 2.0,
        "wind_32": 2.0,
        "wind_66": 2.0,
        "wind_101": 2.0,
        "pv_22": 3.0,
        "pv_36": 3.0,
        "pv_70": 3.0,
        "pv_107": 3.0,
    },
}

SYSTEM_KEYS = {
    "33": {
        "renewable_keys": ["pv_11", "wind_26"],
        "load_keys": ["load", "load_Q"],
        "all_keys": ["pv_11", "wind_26", "load", "load_Q"],
        "n_steps": 48,
        "n_load_buses": 32,
        "solver_input_order": ["load", "pv_11", "wind_26", "load_Q"],
    },
    "119": {
        "renewable_keys": [
            "wind_11",
            "wind_32",
            "wind_66",
            "wind_101",
            "pv_22",
            "pv_36",
            "pv_70",
            "pv_107",
        ],
        "load_keys": ["load", "load_Q"],
        "all_keys": [
            "wind_11",
            "wind_32",
            "wind_66",
            "wind_101",
            "pv_22",
            "pv_36",
            "pv_70",
            "pv_107",
            "load",
            "load_Q",
        ],
        "n_steps": 48,
        "n_load_buses": 118,
        "solver_input_order": [
            "load",
            "load_Q",
            "wind_11",
            "wind_32",
            "wind_66",
            "wind_101",
            "pv_22",
            "pv_36",
            "pv_70",
            "pv_107",
        ],
    },
}

SPLITS: Dict[str, Dict[str, List[int]]] = {
    "33": {
        "historical": list(range(0, 70)),
        "profile": list(range(0, 70)),
        "val": list(range(70, 80)),
        "test_id": list(range(80, 100)),
                                                                      
                                                                        
        "train": list(range(0, 70)),
    },
    "119": {
        "historical": list(range(0, 700)),
        "profile": list(range(0, 700)),
        "val": list(range(700, 800)),
        "test_id": list(range(800, 850)),
                                         
        "train": list(range(0, 700)),
    },
}

FORECAST_CASES = {
    "id_reliable": {
        "error_multiplier": 1.0,
        "load_bias": 0.0,
        "renewable_bias": 0.0,
        "error_base": 0.01,
        "error_slope": 0.015,
        "forecast_min_scale": 0.5,
        "forecast_max_scale": 1.5,
    },
    "high_error": {
        "error_multiplier": 2.5,
        "load_bias": 0.0,
        "renewable_bias": 0.0,
        "error_base": 0.03,
        "error_slope": 0.04,
        "forecast_min_scale": 0.4,
        "forecast_max_scale": 2.0,
    },
    "biased_ood": {
        "error_multiplier": 2.0,
        "load_bias": 0.10,
        "renewable_bias": -0.10,
        "error_base": 0.03,
        "error_slope": 0.04,
        "forecast_min_scale": 0.4,
        "forecast_max_scale": 2.5,
    },
    "perfect": {
        "error_multiplier": 0.0,
        "load_bias": 0.0,
        "renewable_bias": 0.0,
        "error_base": 0.0,
        "error_slope": 0.0,
        "forecast_min_scale": 1.0,
        "forecast_max_scale": 1.0,
        "use_actual_future": True,
    },
}

ACTUAL_CASES = {
    "id_actual": {
        "load_scale": 1.0,
        "renewable_scale": 1.0,
    },
    "high_load_actual": {
        "load_scale": 1.10,
        "renewable_scale": 1.0,
    },
    "low_renewable_actual": {
        "load_scale": 1.0,
        "renewable_scale": 0.90,
    },
    "stress_actual": {
        "load_scale": 1.10,
        "renewable_scale": 0.90,
    },
}

DEFAULT_BASE_SEEDS = {
    "33": 100000,
    "119": 200000,
}

RAW_RESULT_COLUMNS = [
    "system",
    "method",
    "scenario_id",
    "split",
    "actual_case",
    "forecast_case",
    "horizon",
    "gamma_Q",
    "cost",
    "perfect_cost",
    "gap_percent",
    "voltage_violation",
    "solve_time_s",
    "avg_step_time_s",
    "max_step_time_s",
    "solver_fail_count",
    "timeout_count",
    "completed",
    "incomplete_reason",
    "seed",
]


def ensure_result_dir() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR


def validate_system(system: str) -> None:
    if system not in SYSTEM_KEYS:
        raise ValueError(f"Unknown system '{system}'. Expected one of {sorted(SYSTEM_KEYS)}.")


def validate_forecast_case(forecast_case: str) -> None:
    if forecast_case not in FORECAST_CASES:
        raise ValueError(
            f"Unknown forecast case '{forecast_case}'. "
            f"Expected one of {sorted(FORECAST_CASES)}."
        )


def validate_actual_case(actual_case: str) -> None:
    if actual_case not in ACTUAL_CASES:
        raise ValueError(
            f"Unknown actual case '{actual_case}'. "
            f"Expected one of {sorted(ACTUAL_CASES)}."
        )
