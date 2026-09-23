# Method Mapping

This repository uses the following method names consistently in code, tables,
figures, and documentation.

| Name | Implementation entry point | Description |
| --- | --- | --- |
| `MPC-3` | `experiments_v2.comparison.run_33_main`, `run_119_main` | Rolling MPC with horizon 3 and no terminal value. |
| `MPC-5` | `experiments_v2.comparison.run_33_main` | Rolling MPC with horizon 5. |
| `MPC-7` | `experiments_v2.comparison.run_33_main` | Rolling MPC with horizon 7. |
| `Proposed-H3` | `experiments_v2.mpc_icnn.run_h3`, main comparison runners | MPC with SAC-trained ICNN terminal value and horizon 3. |
| `StdNN-H3-MIP` | `experiments_v2.nn_embedding.run_33_nn_embedding` | MPC with a standard ReLU critic embedded through mixed-integer constraints. |
| `SAC` | `experiments_v2.policy_baselines.run_33_policy_baselines`, `run_119_policy_baselines` | Fixed pretrained SAC policy baseline. |
| `SAC-ICNN` | policy-baseline runners | Fixed pretrained SAC policy with ICNN critic architecture. |
| `SACLag` | `experiments_v2.policy_baselines.run_33_policy_baselines` | 33-bus fixed pretrained constrained policy baseline. |
| `Oracle-H3/H7/H12` | `experiments_v2.comparison.run_33_oracle_mpc` | Rolling MPC with actual future trajectories available inside each window. |
| `Perfect-Global` | `experiments_v2.global_optimum.run_33_perfect_global`, `run_119_perfect_global` | Full-day optimization benchmark with all 48 future steps visible. |

`Perfect-Global` and `Oracle-H*` should be discussed separately. `Oracle-H*`
is still a rolling-window controller, while `Perfect-Global` solves a full-day
optimization problem.

## Diagnostic Naming

`run_33_mpc_diagnostic.py` accepts `MPC-3` in `--methods` as the MPC task
selector. The actual output name is determined by `--mpc-horizon`, so a run
with `--mpc-horizon 7` is written as `MPC-7` in raw rows and paper tables.

The four controlled diagnostic cases are:

| Case | Modification |
| --- | --- |
| `original` | Original prices and DG ramp coupling. |
| `flat_price` | Removes time-varying energy prices. |
| `relaxed_ramp` | Weakens DG inter-temporal ramp coupling. |
| `flat_price_relaxed_ramp` | Applies both controlled modifications. |

The 119-bus dispatch-trajectory figures report `Proposed-H3`, `SAC`, and
`MPC-3`.
