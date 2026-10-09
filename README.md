# Wind turbine fault detection with PCA MSPC

## Quick Start

From the repository root, with Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/phase1_healthy_model.py
python src/phase2_fault_detection.py
python src/phase3_diagnostics.py
```

The three `phase*.py` commands are the complete current modelling pipeline, in the required order. They read the included workbook at `resources/wind_turbine_fault_diagnosis_data.xlsx` and write CSV results and PNG figures to `outputs/`. Start with `outputs/phase1_full_healthy_selection.csv`, `outputs/phase2_alarm_summary.csv`, the two `outputs/phase2_control_charts_No.*WT.png` figures, and `outputs/phase3_sensor_ranking.csv`. No exploratory script or pre-existing output is needed to run Phase 1. The `outputs/` directory must exist; it is included in this repository.

If your system calls Python 3.12 `python3` rather than `python3.12`, substitute that name in the first line. On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`. Run all Python commands from the repository root, not from `src/`.

## Project Overview

This Level A project uses a healthy turbine as the reference for multivariate statistical process control (MSPC). A linear principal component analysis (PCA) model fitted on No.2WT produces two monitoring statistics: Hotelling's T² for variation within the retained PC space and squared prediction error (SPEx, also called Q) for variation outside it. The fixed healthy model is then used to evaluate No.14WT and No.39WT. Variable contributions and biplots describe the largest statistical departures; they do not identify physical fault causes.

## Repository Structure and Execution Order

| Path | Role |
| --- | --- |
| `resources/wind_turbine_fault_diagnosis_data.xlsx` | Required input workbook; tracked in the repository. |
| `requirements.txt` | Pinned Python dependencies. |
| `src/pretreatment.py` | Supporting module. Loads and aligns the workbook, interpolates one missing value, and removes healthy-constant variables. It does **not** save a prepared dataset or need a separate run. |
| `src/pca_monitoring.py` | Supporting module. Implements autoscaling, PCA fitting/projection, T² and Q limits, model CSV serialization, and contributions. Do **not** run it directly. |
| `src/phase1_healthy_model.py` | **Run first.** Fits the final healthy model and saves its parameters, PC selection, limits, and healthy charts. |
| `src/phase2_fault_detection.py` | **Run second.** Loads the Phase 1 model and limits, projects the healthy and two faulty recordings, and saves monitoring results and faulty charts. |
| `src/phase3_diagnostics.py` | **Run third.** Loads the Phase 1 model and Phase 2 monitoring CSV, checks consistency, then produces fault-period, contribution, biplot, ranking, and alarm-proportion outputs. |
| `src/diagnose_healthy_variograms.py` | Optional exploratory investigation of changing healthy PC-score variation. It recomputes its own healthy PCA and is **not** a prerequisite for the final model. |
| `src/inspect_wind_turbine_data.py` | Optional initial raw-data inspection and descriptive plots. Not part of the current three-phase pipeline. |
| `src/pca_healthy_turbine.py` | Earlier pretreatment/healthy-PCA exploration. Useful background, but **not** the final Kaiser-selected monitoring model and not a pipeline step. |
| `outputs/` | Only the 22 current Submission 3 outputs from Phases 1–3. Running a phase regenerates its corresponding files. |
| `archive/legacy_70_30/` | Superseded chronological-split results and investigation notes; retained as methodological evidence, not pipeline inputs. |
| `archive/exploratory/` | Earlier raw-data/PCA exploration and temporal variogram results. Optional scripts write here, not to `outputs/`. |
| `archive/README.md` | Inventory and interpretation of the archived work. |

The dependency chain is explicit in the code: Phase 2 reads `outputs/phase1_final_model_{parameters,eigenvalues,limits}.csv`; Phase 3 reads those same model files **and** `outputs/phase2_monitoring_statistics.csv`. Phase 3 verifies its recomputed T², Q, and alarm flags against Phase 2 before writing diagnostics. Thus Phase 3 cannot be run on a fresh checkout before Phases 1 and 2. The checked-in outputs are examples, not required inputs for a fresh full run.

## Requirements and Input Data

Python 3.12 was used for an end-to-end verification run. The project does not enforce a minimum version in code; use a Python version for which the pinned versions in `requirements.txt` can be installed. Required packages are NumPy, pandas, SciPy, Matplotlib, and openpyxl. The scripts use Matplotlib's non-interactive `Agg` backend, so no graphical desktop session is required. A virtual environment keeps these dependencies separate from other Python installations.

The workbook must remain at the path shown above and retain the exact sheet names `No.2WT`, `No.14WT`, and `No.39WT`, with a header row of numbered variables. The workbook also contains `No.3`, which is excluded because its variable structure is not compatible. The loader uses common variables 1–27, preserves row order, linearly interpolates the single missing No.14WT value at variable 9 / observation 358, and removes variables 12 and 15 because they have zero healthy variance. This yields 25 PCA variables and 1,570, 686, and 1,405 observations for No.2WT, No.14WT, and No.39WT, respectively. No manual spreadsheet edits or external data download are needed.

## Current Modelling Methodology

Phase 1 implements the TA's Option 2: **all** healthy No.2WT observations supply the sample mean and sample standard deviation used for autoscaling and the PCA reference model. The Kaiser rule retains every healthy covariance eigenvalue strictly greater than 1; the supplied data select **six PCs**, explaining **84.01%** of standardized healthy variance. No faulty observation influences scaling, PCA, PC selection, or control limits. Phase 2 applies the saved healthy means, scales, loadings, and six-PC count to both faulty turbines.

The main alarm rule uses separate 99% theoretical limits: an F-distribution T² limit and a Jackson–Mudholkar approximation for Q based on excluded eigenvalues. With the current data these are **T² = 16.946** and **Q = 12.067** (rounded). An observation alarms if either statistic exceeds its limit. Charts also show mean-plus-three-standard-deviation lines for comparison; those lines do not determine the reported main alarms.

The earlier chronological 70/30 healthy split and its five-PC monitoring files are retained in `archive/legacy_70_30/` as an investigation, **not** as the final calibration or validation strategy. That split produced 422/471 alarms in the later healthy block (89.6%), consistent with healthy variation not covered by its early calibration block. The temporal variogram findings in `archive/exploratory/` document PC1 drift and additional late PC2 variation using observation lags; they do not change the final model.

## Expected Outputs

| Stage | Main files in `outputs/` | Meaning |
| --- | --- | --- |
| Phase 1 | `phase1_full_healthy_selection.csv`, `phase1_full_healthy_eigenvalues.csv`, `phase1_full_healthy_scree_plot.png` | Kaiser PC selection and healthy eigenvalue spectrum. |
| Phase 1 | `phase1_final_model_parameters.csv`, `phase1_final_model_eigenvalues.csv`, `phase1_final_model_limits.csv` | Reloadable healthy scaling/loadings, spectrum, component count, and limits; **required by Phases 2 and 3**. |
| Phase 1 | `phase1_final_model_control_charts.png` | Healthy **in-sample** T² and Q charts. |
| Phase 2 | `phase2_monitoring_statistics.csv`, `phase2_alarm_summary.csv` | Per-observation faulty statistics/flags and per-turbine alarm counts (including an in-sample healthy summary). The statistics CSV is **required by Phase 3**. |
| Phase 2 | `phase2_control_charts_No.14WT.png`, `phase2_control_charts_No.39WT.png` | Faulty-turbine T² and Q control charts under the fixed healthy limits. |
| Phase 3 | `phase3_fault_periods.csv`, `phase3_alarm_proportions.csv` | Descriptive transition/episode segments and their alarm proportions. |
| Phase 3 | `phase3_contributions.csv`, `phase3_sensor_ranking.csv`, `phase3_constant_variable_check.csv` | Variable contributions/ranks and a separate check of excluded healthy-constant variables 12 and 15. |
| Phase 3 | `phase3_contributions_No.*WT.png`, `phase3_contribution_heatmap_No.*WT.png`, `phase3_biplots_No.*WT.png` | Mean contributions, individual-observation contributions, and time-coloured score/loading views for each faulty turbine. |

For a quick numerical check, `phase2_alarm_summary.csv` should report **15/1,570 (0.96%)** No.2WT alarms, **378/686 (55.10%)** No.14WT alarms, and **521/1,405 (37.08%)** No.39WT alarms under the T²-or-Q rule. `phase3_sensor_ranking.csv` puts variables **9, 13, and 22** in the top five for both statistics in both faulty recordings. Small last-decimal-place differences in floating-point limits across numerical-library versions are possible; compare rounded values and alarm counts, not byte-for-byte CSV identity.

Files such as `phase1_alarm_summary.csv`, `phase1_control_charts_n4.png`, `phase1_control_charts_n5.png`, `phase1_scree_plot.png`, and `phase1_calibration_eigenvalues.csv` in `archive/legacy_70_30/` describe the **superseded 70/30 experiment**. The `pca_healthy_*`, `pretreatment_*`, and `step2_*` files in `archive/exploratory/` come from optional earlier exploration. Do not use these as final six-PC monitoring results. The main `outputs/` directory contains no historical results.

## Optional Diagnostics

These commands can be run from the repository root after setup, in any order. Neither feeds the three-phase model:

```bash
python src/inspect_wind_turbine_data.py
python src/pca_healthy_turbine.py
python src/diagnose_healthy_variograms.py
```

The variogram script writes `archive/exploratory/healthy_temporal_variogram.png` and `archive/exploratory/healthy_temporal_variogram_summary.csv`. The other two scripts create raw-data, pretreatment, and initial PCA descriptive files in `archive/exploratory/`. They do not replace Phase 1 or add files to the final `outputs/` directory. Do not run optional scripts merely to reproduce the current monitoring charts and alarm summaries.

## Reproducibility and Interpretation Notes

- The commands assume the repository root is the current directory. Running from elsewhere can make relative shell paths fail, although the Python data/output paths themselves are resolved from the script locations.
- The supplied workbook and `outputs/` directory are tracked. Phase scripts overwrite their own generated files; use a fresh clone or copy if you need to preserve the checked-in examples while testing. Archived evidence is separate and is not read by the production pipeline.
- The 0.96% healthy alarm proportion is **in-sample**, not an independently validated false-alarm rate. Option 2 has no separate healthy validation set. The two marginal 99% limits do not guarantee a 1% combined alarm rate.
- No observation-level fault labels, verified timestamps, or sensor names are supplied. Alarm proportions are not detection accuracy; contribution rankings indicate statistical sensitivity, not physical causality.
- No.14WT observations 359–686 and No.39WT observations 471–798 contain a near-identical shared block, so their corresponding post-transition alarms are not independent evidence. PC6 (eigenvalue about 1.014) is also close to the Kaiser threshold, and the interpolated No.14WT transition row should not be overinterpreted.
- The current repository contains no automated test suite. The complete pipeline was rebuilt in this repository with Python 3.12 and locally available NumPy/pandas/SciPy/Matplotlib/openpyxl versions; it reproduced the PC count, explained variance, alarm counts, and sensor ranking. The exact pinned package set in `requirements.txt` was not separately installed during that check.

