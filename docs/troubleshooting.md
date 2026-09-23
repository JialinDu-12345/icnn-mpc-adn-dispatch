# Troubleshooting

## Gurobi Cannot Start

Check that `gurobipy` imports and that a license is visible:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

If this fails, install Gurobi and configure the license before running MPC-based
experiments.

## `ModuleNotFoundError: experiments_v2`

Run commands from the repository root:

```bash
python -m experiments_v2.comparison.run_33_main --help
```

Avoid running package modules from inside subfolders unless `PYTHONPATH` points
to the repository root.

## Missing `selected_gamma_33.json`

`--gamma-Q auto` reads `results_v2/selected_gamma_33.json`. The file is included
in the repository. If it is removed, rerun gamma validation and summarization as
shown in `docs/reproduction.md`.

## Git Storage

This release stores all data, pretrained weights, workbooks, and figures
directly in Git. Git LFS is not required. The `.gitattributes` file marks
these artifacts as binary and disables LFS filters for them.

When publishing this snapshot, commit the extracted repository files, not
the outer ZIP archive. If future releases introduce substantially larger
assets, reassess the storage method before committing them.

## Slow 119-Bus Runs

The 119-bus experiments are much heavier than the 33-bus smoke tests. Start with
a short dry run or a single 33-bus scenario before launching full experiments.

## Slow MPC Diagnostic Runs

Use `--reuse-perfect-global` with
`formal/step12_33_mpc_diagnostic_raw.csv` when comparing perfect forecasts or
different MPC horizons. Perfect-Global is independent of the rolling forecast
case, so rerunning it is unnecessary when the diagnostic case, scenario, and
actual case are unchanged.

The H7 diagnostic can require a longer per-step limit than H3. The recommended
setting is `--mpc-time-limit-s 600`.

## Slow ICNN Oracle-Tail Labeling

Use the sampled defaults in `run_33_icnn_value_diagnostic.py`. A full grid over
20 scenarios, three terminal sources, and all terminal steps can take many days
because each terminal state requires an oracle-tail optimization.

## Pandapower Or Gym Import Errors

The policy-baseline environments require `pandapower` and `gym`. Reinstall from
`requirements.txt` or recreate the conda environment if either package is
missing.
