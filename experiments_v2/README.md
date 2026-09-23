# experiments_v2 Code Map

This package contains the formal experiment runners, solver adapters, scenario
protocol, post-processing scripts, and plotting scripts used by the repository.
It uses local data and copied legacy-compatible assets from `../experiment_assets`.

## Core Files

| Path | Purpose |
| --- | --- |
| `config.py` | Central paths, scenario splits, forecast cases, actual cases, gamma candidates, and result columns. |
| `scenario_protocol.py` | Loads realized scenarios, applies the 119-bus legacy load envelope, builds historical forecast profiles, and creates deterministic rolling forecasts. |
| `result_logger.py` | Writes unified scenario-level CSV result rows. |
| `run_33_mpc.py` | Adapter for 33-bus MPC-based solvers and Gurobi model statistics. |
| `run_119_mpc.py` | Adapter for 119-bus `MPC-3`, `Proposed-H3`, and optional StdNN solvers. |

## Formal Runners

| Path | Purpose |
| --- | --- |
| `comparison/run_33_main.py` | Main 33-bus comparison for `MPC-3`, `MPC-5`, `MPC-7`, `StdNN-H3-MIP`, and `Proposed-H3`. |
| `comparison/run_33_forecast_sensitivity.py` | Forecast-assumption sensitivity experiments. |
| `comparison/run_33_oracle_mpc.py` | Rolling perfect-forecast Oracle MPC baselines. |
| `comparison/run_33_dispatch_trajectory.py` | Representative dispatch-trajectory runner for scheduling figures. |
| `comparison/run_33_mpc_diagnostic.py` | Controlled MPC failure/sufficiency diagnostic with configurable horizon, perfect forecasts, and Perfect-Global cache reuse. |
| `comparison/run_33_icnn_value_diagnostic.py` | Sampled ICNN terminal-value approximation and decision-impact diagnostic. |
| `comparison/run_119_main.py` | 119-bus scalability runner with optional inline baseline gating. |
| `comparison/run_119_dispatch_trajectory.py` | Representative 119-bus dispatch runner for `Proposed-H3`, `SAC`, and `MPC-3`. |
| `global_optimum/run_33_perfect_global.py` | Full-day 33-bus `Perfect-Global` benchmark. |
| `global_optimum/run_119_perfect_global.py` | Full-day 119-bus `Perfect-Global` benchmark. |
| `global_optimum/select_119_perfect_feasible.py` | 119-bus feasible-scenario screening utility. |
| `global_optimum/select_119_operable_scenarios.py` | Backup 119-bus screening path. |
| `mpc_icnn/h3_gamma_validation.py` | Gamma validation for `Proposed-H3`. |
| `nn_embedding/run_33_nn_embedding.py` | Standard ReLU critic embedding comparison. |
| `policy_baselines/run_33_policy_baselines.py` | 33-bus SAC, SAC-ICNN, and SACLag policy-baseline evaluation. |
| `policy_baselines/run_119_policy_baselines.py` | 119-bus SAC and SAC-ICNN policy-baseline evaluation. |
| `pure_mpc/run_h3.py`, `run_h5.py`, `run_h7.py` | Thin entry points for individual pure-MPC horizons. |

## Summary And Plot Scripts

The `summary/` package converts raw scenario-level CSV files into paper tables,
matched-pair comparisons, model-complexity summaries, and refreshed final
outputs. The `plots/` package regenerates the 33-bus paper figures, the three
119-bus dispatch PDFs, and the local `images/*.pdf` files used by manuscript
builds. `plots/make_119_paper_outputs.py` intentionally creates dispatch
figures only.

Typical post-processing commands are documented in the repository-level
`README.md` and in `docs/reproduction.md`.
