# Reproduction Guide

This document lists the recommended command order for reproducing the reported
experiments. Run commands from the repository root.

## Environment Check

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python -c "import numpy, pandas, pandapower, torch; print('core dependencies ok')"
```

All MPC-based runs require a valid Gurobi license. Runtime can vary with CPU,
thread settings, and license configuration.

## 33-Bus Gamma Validation

```bash
python -m experiments_v2.mpc_icnn.h3_gamma_validation \
  --scenarios val \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-candidates 0,0.1,0.3,0.5,0.7,0.9,1.0 \
  --output formal/step3_33_gamma_validation_raw.csv

python -m experiments_v2.summary.summarize_gamma_validation \
  --input formal/step3_33_gamma_validation_raw.csv \
  --output formal/step3_33_gamma_validation_summary.csv
```

The repository includes `results_v2/selected_gamma_33.json`.

## 33-Bus Main And Sensitivity Experiments

```bash
python -m experiments_v2.comparison.run_33_main \
  --scenarios all \
  --tasks MPC-3,MPC-5,MPC-7,StdNN-H3-MIP,Proposed-H3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q auto \
  --time-limit-s 300 \
  --mip-gap 0.001

python -m experiments_v2.comparison.run_33_forecast_sensitivity \
  --scenarios all \
  --tasks MPC-3,MPC-7,Proposed-H3 \
  --forecast-cases id_reliable,high_error,biased_ood

python -m experiments_v2.nn_embedding.run_33_nn_embedding \
  --scenarios all \
  --tasks StdNN-H3-MIP \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --time-limit-s 300 \
  --mip-gap 0.001 \
  --output formal/step6_33_stdnn_only_raw.csv
```

## 33-Bus Benchmarks And Policy Baselines

```bash
python -m experiments_v2.global_optimum.run_33_perfect_global \
  --scenarios all \
  --actual-case id_actual \
  --time-limit-s 1800 \
  --mip-gap 0.001

python -m experiments_v2.comparison.run_33_oracle_mpc \
  --scenarios all \
  --tasks Oracle-H3,Oracle-H7,Oracle-H12 \
  --actual-case id_actual

python -m experiments_v2.policy_baselines.run_33_policy_baselines \
  --scenarios all \
  --methods SAC,SAC-ICNN \
  --actual-case id_actual \
  --output formal/step7_33_policy_baselines_raw.csv

python -m experiments_v2.policy_baselines.run_33_policy_baselines \
  --scenarios all \
  --methods SACLag \
  --actual-case id_actual \
  --stochastic \
  --eval-seed 0 \
  --output formal/step7_33_saclag_stochastic_seed0_test_raw.csv

python -m experiments_v2.summary.summarize_policy_baselines \
  --input formal/step7_33_policy_baselines_raw.csv \
  --output formal/step7_33_policy_baselines_summary.csv

python -m experiments_v2.summary.summarize_policy_baselines \
  --input formal/step7_33_saclag_stochastic_seed0_test_raw.csv \
  --output formal/step7_33_saclag_stochastic_seed0_test_summary.csv

python -m experiments_v2.summary.refresh_33_outputs --strict
```

## 119-Bus Scalability Experiment

```bash
python -m experiments_v2.utils.export_119_legacy_envelope
python -m experiments_v2.comparison.run_119_main \
  --no-gate-by-baselines \
  --scenario-file formal/selected_119_inline_operable20.json \
  --tasks MPC-3,SAC,SAC-ICNN,Proposed-H3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q 0.3 \
  --time-limit-s 300 \
  --mip-gap 0.001 \
  --output formal/step10_119_main_inline20_raw.csv
python -m experiments_v2.summary.refresh_119_outputs --strict
```

The selected scenario file and its Perfect-Global screening records are
included in `results_v2/formal`. The task list matches the methods reported in
the paper's 119-bus scalability study.

## Paper Figures And Tables

After formal raw outputs are available:

```bash
python -m experiments_v2.comparison.run_33_dispatch_trajectory \
  --scenario 80 \
  --tasks MPC-3,SAC,Proposed-H3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q auto \
  --output formal/step9_33_dispatch_trajectory_raw.csv

python -m experiments_v2.plots.make_33_paper_outputs
```

This refreshes `results_v2/paper_figures`, `results_v2/paper_tables`, and
`images`.

## MPC Failure and Sufficiency Diagnostic

Run the controlled H3 diagnostic with the standard rolling forecast first. The
command below solves each Perfect-Global variant and creates the cache used by
the perfect-forecast comparisons:

```bash
python -m experiments_v2.comparison.run_33_mpc_diagnostic \
  --cases all \
  --methods MPC-3,Perfect-Global \
  --scenarios all \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --rerun-original \
  --rerun-perfect-global \
  --mpc-horizon 3 \
  --relaxed-ramp-value 0.8 \
  --mpc-time-limit-s 300 \
  --perfect-time-limit-s 1800 \
  --mip-gap 0.001 \
  --output formal/step12_33_mpc_diagnostic_raw.csv \
  --summary-output formal/step12_33_mpc_diagnostic_summary.csv \
  --table-output-dir paper_tables/33/mpc_diagnostic
```

Isolate rolling-forecast error at H3 while reusing Perfect-Global:

```bash
python -m experiments_v2.comparison.run_33_mpc_diagnostic \
  --cases all \
  --methods MPC-3,Perfect-Global \
  --scenarios all \
  --forecast-case perfect \
  --actual-case id_actual \
  --rerun-original \
  --reuse-perfect-global \
  --perfect-cache-input formal/step12_33_mpc_diagnostic_raw.csv \
  --mpc-horizon 3 \
  --relaxed-ramp-value 0.8 \
  --mpc-time-limit-s 300 \
  --mip-gap 0.001 \
  --output formal/step12_33_mpc_diagnostic_perfect_forecast_H3_raw.csv \
  --summary-output formal/step12_33_mpc_diagnostic_perfect_forecast_H3_summary.csv \
  --table-output-dir paper_tables/33/mpc_diagnostic_perfect_forecast_H3
```

Test whether a longer prediction window is sufficient on the two
price-flattened variants:

```bash
python -m experiments_v2.comparison.run_33_mpc_diagnostic \
  --cases flat_price,flat_price_relaxed_ramp \
  --methods MPC-3,Perfect-Global \
  --scenarios all \
  --forecast-case perfect \
  --actual-case id_actual \
  --rerun-original \
  --reuse-perfect-global \
  --perfect-cache-input formal/step12_33_mpc_diagnostic_raw.csv \
  --mpc-horizon 7 \
  --relaxed-ramp-value 0.8 \
  --mpc-time-limit-s 600 \
  --mip-gap 0.001 \
  --output formal/step12_33_mpc_diagnostic_perfect_forecast_H7_raw.csv \
  --summary-output formal/step12_33_mpc_diagnostic_perfect_forecast_H7_summary.csv \
  --table-output-dir paper_tables/33/mpc_diagnostic_perfect_forecast_H7
```

The raw and summary tables include completion/failure counts, timeout counts,
matched cost gaps, runtime statistics, and 95% confidence intervals.

## Sampled ICNN Value Diagnostic

The default configuration is intentionally sampled because full oracle-tail
labeling over every scenario, source, and time step is prohibitively slow:

```bash
python -m experiments_v2.comparison.run_33_icnn_value_diagnostic
```

It evaluates scenarios `80,84,88,92,96`, terminal sources `MPC-3` and
`Proposed-H3`, and six terminal times with stride eight. Pass explicit
`--scenarios`, `--sources`, and terminal-step options only when a denser run is
required.

## 119-Bus Dispatch Trajectories

```bash
python -m experiments_v2.comparison.run_119_dispatch_trajectory \
  --scenario 801 \
  --tasks Proposed-H3,SAC,MPC-3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q 0.3 \
  --time-limit-s 300 \
  --mip-gap 0.001 \
  --output formal/step13_119_dispatch_trajectory_raw.csv

python -m experiments_v2.plots.make_119_paper_outputs
```

The plotting command produces only the three 119-bus scheduling trajectories;
it does not generate a 119-bus cost figure or runtime box plot.
