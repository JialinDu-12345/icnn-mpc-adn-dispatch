# Result Snapshot

This directory contains the scenario-level results and paper artifacts produced
by the experiment modules. Relative output paths passed to the command-line
runners are resolved under `results_v2`.

| Path | Contents |
| --- | --- |
| `formal/` | Raw scenario runs, summaries, matched comparisons, and selected-scenario records. |
| `paper_tables/33/` | 33-bus main, sensitivity, complexity, and diagnostic tables. |
| `paper_tables/119/` | 119-bus scalability and model-complexity tables. |
| `paper_figures/33/` | 33-bus figures and source workbooks. |
| `paper_figures/119/` | 119-bus dispatch-trajectory figures. |
| `selected_gamma_33.json` | Selected terminal-value coefficient used by `--gamma-Q auto`. |

The principal processed inputs are:

- `step8_33_main_table_with_perfect_gap.csv` for the 33-bus main comparison;
- `step5_33_pairwise_mpc3_vs_proposed_by_case.csv` for forecast sensitivity;
- `step10_119_main_table_inline20.csv` for the 119-bus scalability comparison;
- `step12_33_mpc_diagnostic_*` for the controlled MPC diagnostic;
- `step13_33_icnn_*` for the sampled terminal-state diagnostic;
- `step13_119_dispatch_trajectory_raw.csv` for the 119-bus dispatch figures.

Run the refresh and plotting modules documented in `docs/reproduction.md` to
regenerate the processed tables and figures from the formal raw files.
