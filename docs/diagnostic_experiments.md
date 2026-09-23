# Diagnostic Experiments

This document summarizes the diagnostic studies provided with the main
33-bus and 119-bus comparisons.

## MPC Failure and Sufficiency

`experiments_v2.comparison.run_33_mpc_diagnostic` compares finite-horizon MPC
against case-matched Perfect-Global rows under four controlled variants:

- original prices and DG ramp coupling;
- flat prices;
- relaxed DG ramp coupling;
- flat prices with relaxed DG ramp coupling.

The runner supports rolling or perfect within-window forecasts, configurable
MPC horizons, Perfect-Global cache reuse, failure and timeout accounting, and
95% confidence intervals for cost, gap, and runtime statistics.

## ICNN Terminal-Value and Decision Impact

`experiments_v2.comparison.run_33_icnn_value_diagnostic` compares the ICNN
terminal-cost estimates with deterministic offline tail-OPF values. The default
configuration evaluates 30 paired terminal decisions, corresponding to 60
sampled terminal states, and reports rank correlation, pairwise preference
accuracy, and paired tail-cost reduction.

## 119-Bus Dispatch Trajectories

`experiments_v2.comparison.run_119_dispatch_trajectory` records full-day
dispatch details for `Proposed-H3`, `SAC`, and `MPC-3` on scenario 801.
`experiments_v2.plots.make_119_paper_outputs` creates the three corresponding
scheduling PDFs and the 119-bus scalability tables.

## Included Results

The result snapshot contains the rolling-forecast and perfect-forecast H3 MPC
diagnostics, the sampled ICNN terminal-state diagnostics, and the full 119-bus
dispatch rows used to generate the trajectory figures. The longer-horizon
diagnostic command is provided in `docs/reproduction.md`; its output is created
when that experiment is run.
