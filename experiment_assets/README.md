# experiment_assets

This folder stores the self-contained data, solver sources, pretrained weights,
and policy checkpoints required by the reproduced experiments. The public
experiment code does not read from private local folders.

| Path | Contents |
| --- | --- |
| `data/33` | 100 realized 33-bus scenario files, `results_0.npz` to `results_99.npz`. |
| `data/119` | 1000 realized 119-bus scenario files plus `legacy_119_load_envelope.npz`. |
| `solvers/33/pure_mpc_clean` | Clean 33-bus pure-MPC solver used by `MPC-3`, `MPC-5`, and `MPC-7`. |
| `solvers/33/mpc_icnn` | 33-bus proposed MPC-ICNN solver and ICNN critic weights. |
| `solvers/33/std_nn_mpc` | 33-bus standard ReLU critic embedding solver and exported critic arrays. |
| `solvers/33/perfect_global_clean` | Clean full-day 33-bus `Perfect-Global` solver. |
| `solvers/119/pure_mpc_clean` | Clean 119-bus pure-MPC solver. |
| `solvers/119/mpc_icnn` | 119-bus proposed MPC-ICNN solver and ICNN critic weights. |
| `solvers/119/std_nn_mpc` | Optional 119-bus standard-critic embedding assets; not used by the reported 119-bus comparison. |
| `solvers/119/perfect_global_clean` | Clean full-day 119-bus `Perfect-Global` solver. |
| `policies/33` | 33-bus SAC, SAC-ICNN, and SACLag policy sources and checkpoints. |
| `policies/119` | 119-bus SAC and SAC-ICNN policy sources and checkpoints. |

The `MPC_1.py` and `SAC_largesystem.py` files provide solver and policy assets.
They are loaded through adapters in `experiments_v2`, so their unused top-level
experiment loops are not executed by the experiment runners.

The `solvers/33/pure_mpc` directory is a compatibility asset. The 33-bus MPC
experiments use `solvers/33/pure_mpc_clean`.
