# Data Description

The repository is self-contained. Scenario data and solver/policy assets are
stored under `experiment_assets`.

## Scenario Files

| System | Folder | Count | Notes |
| --- | --- | ---: | --- |
| 33-bus | `experiment_assets/data/33` | 100 | Realized daily scenarios `results_0.npz` to `results_99.npz`. |
| 119-bus | `experiment_assets/data/119` | 1000 | Realized daily scenarios `results_0.npz` to `results_999.npz`. |

The 119-bus data folder also contains `legacy_119_load_envelope.npz`, which is
used to keep the scenario protocol inside the load domain assumed by the
119-bus solver implementations.

## Splits

| System | Historical/profile | Validation | Held-out test |
| --- | --- | --- | --- |
| 33-bus | `0:69` | `70:79` | `80:99` |
| 119-bus | `0:699` | `700:799` | candidate range starts at `800` |

Historical/profile scenarios are used to construct deterministic forecast
profiles for rolling MPC evaluation. They are not used to retrain the provided
policy baselines.

## Forecast And Actual Cases

Forecast cases are defined in `experiments_v2/config.py`:

- `id_reliable`
- `high_error`
- `biased_ood`
- `perfect`

Actual cases are also defined in `config.py`:

- `id_actual`
- `high_load_actual`
- `low_renewable_actual`
- `stress_actual`

## Assets

The solver and policy sources under `experiment_assets` are self-contained
modules. Experiment runners load them through adapters under `experiments_v2`,
which avoids machine-specific paths and unused top-level run loops.
