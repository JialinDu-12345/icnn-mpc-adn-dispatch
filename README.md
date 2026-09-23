# Learning-Augmented MPC with an ICNN-Based Value Function for Economic Dispatch of Active Distribution Networks

Official implementation and experiment snapshot for:

Jialin Du, Di Cao, Weihao Hu, Pengfei Zhao, Zhenyuan Zhang, and Zhe Chen,
"Learning-Augmented MPC with an ICNN-Based Value Function for Economic Dispatch
of Active Distribution Networks."

The framework trains input-convex neural-network (ICNN) critics with soft
actor-critic (SAC) and embeds the learned terminal-cost proxy into a
short-horizon model predictive control (MPC) problem. The ICNN supplies
long-term value guidance, while the online dispatch action is still obtained
from the network-constrained MPC model. The implementation uses auxiliary
continuous variables and Gurobi general constraints for the ICNN embedding and
does not explicitly introduce activation binaries. The complete online problem
remains nonconvex because it retains the nonlinear DistFlow equations and the
embedded terminal term.

## Main Features

- Modified IEEE 33-bus and 119-bus active distribution network studies.
- Rolling MPC with 3-, 5-, and 7-step horizons.
- SAC-trained ICNN terminal value embedded in horizon-3 MPC.
- Standard ReLU critic embedding through a mixed-integer benchmark model.
- SAC, SAC-ICNN, and SACLag policy baselines.
- Perfect-information full-day and rolling-window oracle benchmarks.
- Forecast sensitivity, terminal-state quality, scale-factor, model-complexity,
  and dispatch-trajectory analyses.
- Scenario-level raw outputs, processed tables, figures, pretrained weights,
  and all data required by the provided experiment runners.

## Reported Results

The tables below reproduce the principal results in the paper. Values are means
with 95% confidence intervals over 20 held-out test scenarios. Runtime is the
average online time per dispatch step.

### IEEE 33-Bus System

| Method | Cost ($) | Gap to Perfect-Global (%) | Time (s) | Voltage violation (p.u.) |
| --- | ---: | ---: | ---: | ---: |
| MPC-3 | 1682.98 +/- 26.97 | 28.18 +/- 0.65 | 1.462 +/- 0.189 | 0 |
| MPC-5 | 1625.74 +/- 25.75 | 23.83 +/- 1.01 | 10.438 +/- 1.736 | 0 |
| MPC-7 | 1594.15 +/- 27.64 | 21.42 +/- 1.30 | 35.154 +/- 4.221 | 0 |
| StdNN-H3-MIP | 1545.96 +/- 48.66 | 17.76 +/- 3.45 | 2.051 +/- 0.107 | 0 |
| SAC-ICNN | 1941.22 +/- 19.62 | 47.90 +/- 0.82 | 0.011 +/- 0.001 | 1.0973 +/- 0.0365 |
| SACLag | 1458.49 +/- 21.06 | 11.09 +/- 0.39 | 0.008 +/- 0.001 | 0.0009 +/- 0.0004 |
| SAC | 1406.27 +/- 19.71 | 7.11 +/- 0.31 | 0.011 +/- 0.001 | 0.2656 +/- 0.0177 |
| Proposed-H3 | 1334.67 +/- 20.73 | 1.65 +/- 0.23 | 0.691 +/- 0.024 | 0 |
| Perfect-Global | 1313.01 +/- 19.73 | - | - | 0 |

### IEEE 119-Bus System

| Method | Cost ($) | Gap to Perfect-Global (%) | Time (s) | Voltage violation (p.u.) |
| --- | ---: | ---: | ---: | ---: |
| MPC-3 | 3331.89 +/- 22.49 | 77.50 +/- 1.73 | 34.181 +/- 2.774 | 0 |
| SAC-ICNN | 4645.74 +/- 27.23 | 147.49 +/- 2.03 | 0.015 +/- 0.001 | 0 |
| SAC | 2797.33 +/- 26.59 | 49.00 +/- 1.22 | 0.015 +/- 0.001 | 0 |
| Proposed-H3 | 2033.30 +/- 26.13 | 8.28 +/- 0.88 | 16.197 +/- 1.198 | 0 |
| Perfect-Global | 1878.10 +/- 24.56 | - | - | 0 |

## Repository Layout

| Path | Contents |
| --- | --- |
| `experiments_v2/` | Experiment runners, solver adapters, summaries, and figure builders. |
| `experiment_assets/` | Scenario data, solver sources, pretrained critics, and policy checkpoints. |
| `results_v2/formal/` | Scenario-level results, summaries, and selected-scenario records. |
| `results_v2/paper_tables/` | CSV, Markdown, and LaTeX tables. |
| `results_v2/paper_figures/` | Paper figures and source workbooks. |
| `images/` | Manuscript-facing figure copies. |
| `paper_assets/` | Static network and training figures used by the manuscript. |
| `docs/` | Data, methods, reproduction, diagnostics, and troubleshooting guides. |

`results_v2` is the configured output root used by the experiment modules.

## Requirements

- Python 3.9
- Gurobi 13.0 with a valid license for MPC-based experiments
- PyTorch 2.0 or newer
- NumPy, pandas, Matplotlib, pandapower, Gym, tqdm, and openpyxl

The data, pretrained weights, workbooks, and figures in this release are
stored directly in Git. Git LFS is not required to clone or download this
snapshot.

## Installation

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate lamc-adn
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

Verify the core dependencies and Gurobi installation:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python -c "import numpy, pandas, pandapower, torch; print('dependencies available')"
```

## Quick Start

Run commands from the repository root. This one-step smoke test checks the
33-bus data, solver adapters, and ICNN assets without running a full day:

```bash
python -m experiments_v2.comparison.run_33_main \
  --scenarios 80 \
  --tasks MPC-3,Proposed-H3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q auto \
  --max-steps 1 \
  --no-tail \
  --time-limit-s 300 \
  --mip-gap 0.001 \
  --output debug/smoke_33_main.csv
```

## Reproducing the Paper

Run the main 33-bus model-based comparison:

```bash
python -m experiments_v2.comparison.run_33_main \
  --scenarios all \
  --tasks MPC-3,MPC-5,MPC-7,StdNN-H3-MIP,Proposed-H3 \
  --forecast-case id_reliable \
  --actual-case id_actual \
  --gamma-Q auto \
  --time-limit-s 300 \
  --mip-gap 0.001
```

Run the matched 119-bus scalability comparison:

```bash
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
```

Regenerate the processed outputs after the raw experiments finish:

```bash
python -m experiments_v2.summary.refresh_33_outputs --strict
python -m experiments_v2.summary.refresh_119_outputs --strict
python -m experiments_v2.plots.make_33_paper_outputs
python -m experiments_v2.plots.make_119_paper_outputs
```

The complete command order, including policy baselines, gamma validation,
forecast sensitivity, offline benchmarks, diagnostic experiments, and dispatch
trajectories, is provided in [`docs/reproduction.md`](docs/reproduction.md).

## Data and Evaluation Protocol

The 33-bus scenarios are split into historical/profile scenarios `0:69`,
validation scenarios `70:79`, and held-out test scenarios `80:99`. The 119-bus
historical and validation ranges are `0:699` and `700:799`; the tested scenarios
are selected from the candidate range beginning at `800` using the stored
scenario-selection records. Historical scenarios construct forecast profiles,
validation scenarios select the terminal-value scale factor, and held-out test
scenarios are reserved for the final comparisons.

See [`docs/data_description.md`](docs/data_description.md) and
[`docs/method_mapping.md`](docs/method_mapping.md) for the complete protocol and
method definitions.

## Outputs

Experiment commands write relative output paths under `results_v2`. The
repository includes the scenario-level snapshot used to produce the reported
tables and figures. Regenerated paper artifacts are written to
`results_v2/paper_tables`, `results_v2/paper_figures`, and `images`.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The source-code
URL reported in the paper is:

<https://github.com/JialinDu-12345/icnn-mpc-adn-dispatch>

## License

Original source code in this repository is licensed under the MIT License.
Third-party software, datasets, pretrained models, and external dependencies
remain subject to their respective licenses and terms. Gurobi requires a
separate license.
