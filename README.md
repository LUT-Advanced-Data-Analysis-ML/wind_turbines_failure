# Wind turbine fault detection with linear and kernel PCA MSPC

## Quick Start

From the repository root, with Python 3.12 available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/phase1_healthy_model.py
python src/phase2_fault_detection.py
python src/phase3_diagnostics.py
python src/phase4_kernel_tuning.py
python src/phase5_kernel_detection.py
python src/phase6_kernel_contributions.py
python src/phase7_sensor_selection.py
```

The seven `phase*.py` commands are the complete current modelling pipeline, in the required order. Phases 1–3 build and evaluate the linear PCA model; Phases 4–7 build and evaluate the kernel PCA model and compare it with the linear one. They read the included workbook at `resources/wind_turbine_fault_diagnosis_data.xlsx` and write CSV results and PNG figures to `outputs/`. Start with `outputs/phase1_full_healthy_selection.csv`, `outputs/phase2_alarm_summary.csv`, the two `outputs/phase2_control_charts_No.*WT.png` figures, `outputs/phase3_sensor_ranking.csv`, `outputs/phase4_tuning_map.png`, `outputs/phase5_kernel_alarm_summary.csv`, `outputs/phase6_contribution_comparison.png` and `outputs/phase7_minimum_sets.csv`. No exploratory script or pre-existing output is needed to run Phase 1. The `outputs/` directory must exist; it is included in this repository.

Phase 4 cross-validates 220 kernel settings and takes about 20 minutes; every other phase finishes in under a minute.

If your system calls Python 3.12 `python3` rather than `python3.12`, substitute that name in the first line. On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`. Python 3.14 (`py -3.14 -m venv .venv`) also works with the pinned packages. Run all Python commands from the repository root, not from `src/`.

## Project Overview

This Level A project uses a healthy turbine as the reference for multivariate statistical process control (MSPC). A linear principal component analysis (PCA) model fitted on No.2WT produces two monitoring statistics: Hotelling's T² for variation within the retained PC space and squared prediction error (SPEx, also called Q) for variation outside it. The fixed healthy model is then used to evaluate No.14WT and No.39WT. Variable contributions and biplots describe the largest statistical departures; they do not identify physical fault causes.

A nonlinear kernel PCA model is then built from the same healthy data. It is used to compare fault detection with the linear model, to check whether the sensors that explain the fault change, and to find the fewest sensors that give the same result as each full 25-sensor model.

## Repository Structure and Execution Order

| Path | Role |
| --- | --- |
| `resources/wind_turbine_fault_diagnosis_data.xlsx` | Required input workbook; tracked in the repository. |
| `requirements.txt` | Pinned Python dependencies. |
| `src/pretreatment.py` | Supporting module. Loads and aligns the workbook, interpolates one missing value, and removes healthy-constant variables. It does **not** save a prepared dataset or need a separate run. |
| `src/pca_monitoring.py` | Supporting module. Implements autoscaling, PCA fitting/projection, T² and Q limits, model CSV serialization, and contributions. Do **not** run it directly. |
| `src/kernel_monitoring.py` | Supporting module. Implements the kernel PCA model (one kernel per sensor, summed), its projection, T² and Q limits, per-sensor T² and Q contributions, and saving of the selected settings. Do **not** run it directly. |
| `src/time_series_cv.py` | Supporting module. Blocked cross-validation of false alarms on the healthy turbine. Do **not** run it directly. |
| `src/phase1_healthy_model.py` | **Run first.** Fits the final healthy model and saves its parameters, PC selection, limits, and healthy charts. |
| `src/phase2_fault_detection.py` | **Run second.** Loads the Phase 1 model and limits, projects the healthy and two faulty recordings, and saves monitoring results and faulty charts. |
| `src/phase3_diagnostics.py` | **Run third.** Loads the Phase 1 model and Phase 2 monitoring CSV, checks consistency, then produces fault-period, contribution, biplot, ranking, and alarm-proportion outputs. |
| `src/phase4_kernel_tuning.py` | **Run fourth.** Chooses the kernel family, width and number of kernel components by blocked cross-validation on the healthy turbine and saves the selected settings. |
| `src/phase5_kernel_detection.py` | **Run fifth.** Refits the selected kernel model, projects the faulty turbines, and saves alarms, fault periods, alarm proportions, kernel/linear agreement, control charts and score plots. |
| `src/phase6_kernel_contributions.py` | **Run sixth.** Splits T² and Q into per-sensor shares for the kernel and linear models and compares the sensor rankings over the fault period. |
| `src/phase7_sensor_selection.py` | **Run seventh.** Adds sensors in the order of their Q shares and finds the fewest sensors that give the same result as each full model. |
| `src/diagnose_healthy_variograms.py` | Optional exploratory investigation of changing healthy PC-score variation. It recomputes its own healthy PCA and is **not** a prerequisite for the final model. |
| `src/inspect_wind_turbine_data.py` | Optional initial raw-data inspection and descriptive plots. Not part of the current pipeline. |
| `src/pca_healthy_turbine.py` | Earlier pretreatment/healthy-PCA exploration. Useful background, but **not** the final Kaiser-selected monitoring model and not a pipeline step. |
| `outputs/` | The 43 current outputs of Phases 1–7 (23 from the linear Phases 1–3 and 20 from the kernel Phases 4–7). Running a phase regenerates its corresponding files. |
| `archive/legacy_70_30/` | Superseded chronological-split results and investigation notes; retained as methodological evidence, not pipeline inputs. |
| `archive/exploratory/` | Earlier raw-data/PCA exploration and temporal variogram results. Optional scripts write here, not to `outputs/`. |
| `archive/README.md` | Inventory and interpretation of the archived work. |

The dependency chain is explicit in the code: Phase 2 reads `outputs/phase1_final_model_{parameters,eigenvalues,limits}.csv`; Phase 3 reads those same model files **and** `outputs/phase2_monitoring_statistics.csv`. Phase 3 verifies its recomputed T², Q, and alarm flags against Phase 2 before writing diagnostics. Thus Phase 3 cannot be run on a fresh checkout before Phases 1 and 2. Phase 4 reads no earlier outputs. Phase 5 reads the Phase 1 model files, `outputs/phase2_monitoring_statistics.csv` and `outputs/phase4_selected_kernel.csv`; Phase 6 reads the Phase 1 model files and the Phase 4 settings; Phase 7 reads the Phase 4 settings, `outputs/phase5_kernel_alarm_proportions.csv` and `outputs/phase6_partial_score_contributions.csv`. The checked-in outputs are examples, not required inputs for a fresh full run.

## Requirements and Input Data

Python 3.12 and Python 3.14 were used for end-to-end runs; the pinned versions in `requirements.txt` install on both. The project does not enforce a minimum version in code; use a Python version for which the pinned versions can be installed. Required packages are NumPy, pandas, SciPy, Matplotlib, and openpyxl. The scripts use Matplotlib's non-interactive `Agg` backend, so no graphical desktop session is required. A virtual environment keeps these dependencies separate from other Python installations.

The workbook must remain at the path shown above and retain the exact sheet names `No.2WT`, `No.14WT`, and `No.39WT`, with a header row of numbered variables. The workbook also contains `No.3`, which is excluded because its variable structure is not compatible. The loader uses common variables 1–27, preserves row order, linearly interpolates the single missing No.14WT value at variable 9 / observation 358, and removes variables 12 and 15 because they have zero healthy variance. This yields 25 PCA variables and 1,570, 686, and 1,405 observations for No.2WT, No.14WT, and No.39WT, respectively. No manual spreadsheet edits or external data download are needed.

## Current Modelling Methodology

Phase 1 calibrates the PCA/MSPC reference model using all healthy No.2WT observations. **All** healthy No.2WT observations supply the sample mean and sample standard deviation used for autoscaling and the PCA reference model. The Kaiser rule retains every healthy covariance eigenvalue strictly greater than 1; the supplied data select **six PCs**, explaining **84.01%** of standardized healthy variance. No faulty observation influences scaling, PCA, PC selection, or control limits. Phase 2 applies the saved healthy means, scales, loadings, and six-PC count to both faulty turbines.

The main alarm rule uses separate 99% theoretical limits: an F-distribution T² limit and a Jackson–Mudholkar approximation for Q based on excluded eigenvalues. With the current data these are **T² = 16.946** and **Q = 12.067** (rounded). An observation alarms if either statistic exceeds its limit. Charts also show mean-plus-three-standard-deviation lines for comparison; those lines do not determine the reported main alarms.

The earlier chronological 70/30 healthy split and its five-PC monitoring files are retained in `archive/legacy_70_30/` as an investigation, **not** as the final calibration or validation strategy. That split produced 422/471 alarms in the later healthy block (89.6%), consistent with healthy variation not covered by its early calibration block. The temporal variogram findings in `archive/exploratory/` document PC1 drift and additional late PC2 variation using observation lags; they do not change the final model.

### Kernel PCA analysis (Phases 4–7)

**Kernel model.** For every sensor, a one-sensor kernel (Gaussian, Cauchy, or Matérn with smoothness ½, ³⁄₂ or ⁵⁄₂) measures how similar two autoscaled values are; the kernel of two observations is the sum of the 25 one-sensor kernels, all with one common width. The kernel matrix is centred in the feature space, and each sensor's piece is centred with its own healthy means, so the per-sensor pieces add up exactly to the full kernel. T² is variance-normalised as in Phase 1 (each squared score divided by its healthy variance), and Q is the squared distance from the retained kernel components. The kernel model, and the linear model wherever it is compared with it in Phases 5–7, use the same theoretical 99% limit formulas: the F-distribution limit for T² and a scaled chi-square limit for Q, fitted to the mean and variance of the healthy Q values.

**Phase 4: tuning on healthy data only.** The healthy recording is cut into blocks of 50 observations that are dealt to five folds in turn, so every fold covers the start, middle and end of the recording; 15 observations on each side of a validation block are left out of training, because neighbouring observations are nearly identical. Autoscaling, kernel, components and limits are re-learned on the training observations of each fold, and the false-alarm rate is the share of held-out healthy observations above the T² or Q limit. Five families × 11 widths (0.25–50 healthy standard deviations) × 4, 6, 8 or 10 components are scored. Settings whose mean false-alarm rate is within 0.25 percentage points of the best distance from 1% are near-ties; among them the fewest components and then the narrowest width are selected. The selected model is **Gaussian, width 5, 4 components** (cross-validated false-alarm rate 0.94%).

**Phase 5: detection.** The selected model is refitted on all healthy observations and the faulty turbines are projected onto it, as in Phases 2 and 3. For a fair comparison, the Phase 1 linear model is re-evaluated with the same limit formulas (F-distribution T² and scaled chi-square Q); the Phase 2 and 3 results are not changed.

**Phase 6: sensor contributions.** T² shares come from the per-sensor partial scores (each sensor's piece of the kernel row projected onto the retained components). Q is split exactly into one non-negative part per sensor, because a sum of one-sensor kernels has a feature space made of one block per sensor; with a linear kernel this split equals the linear squared residuals e_l². Shares (each observation adds up to 100%) are averaged over the fault period before the variable 13 transition for both models.

**Phase 7: fewest sensors.** For each model, sensors are added in the order of their mean Q share and the search stops at the first set that gives the same result as the full model: at least 99% of the fault period alarmed in both turbines, after-transition alarms at most 2 percentage points above the full model's, and a cross-validated healthy false-alarm rate of at most 2%.

## Expected Outputs

| Stage | Main files in `outputs/` | Meaning |
| --- | --- | --- |
| Phase 1 | `phase1_full_healthy_selection.csv`, `phase1_full_healthy_eigenvalues.csv`, `phase1_full_healthy_scree_plot.png` | Kaiser PC selection and healthy eigenvalue spectrum. |
| Phase 1 | `phase1_final_model_parameters.csv`, `phase1_final_model_eigenvalues.csv`, `phase1_final_model_limits.csv` | Reloadable healthy scaling/loadings, spectrum, component count, and limits; **required by Phases 2, 3, 5 and 6**. |
| Phase 1 | `phase1_final_model_control_charts.png` | Healthy **in-sample** T² and Q charts. |
| Phase 2 | `phase2_monitoring_statistics.csv`, `phase2_alarm_summary.csv` | Per-observation faulty statistics/flags and per-turbine alarm counts (including an in-sample healthy summary). The statistics CSV is **required by Phases 3 and 5**. |
| Phase 2 | `phase2_control_charts_No.14WT.png`, `phase2_control_charts_No.39WT.png` | Faulty-turbine T² and Q control charts under the fixed healthy limits. |
| Phase 3 | `phase3_fault_periods.csv`, `phase3_alarm_proportions.csv` | Descriptive transition/episode segments and their alarm proportions. |
| Phase 3 | `phase3_contributions.csv`, `phase3_sensor_ranking.csv`, `phase3_constant_variable_check.csv` | Variable contributions/ranks and a separate check of excluded healthy-constant variables 12 and 15. |
| Phase 3 | `phase3_contributions_No.*WT.png`, `phase3_contribution_heatmap_No.*WT.png`, `phase3_biplots_No.*WT.png`, `phase3_report_biplot_PC1_PC2.png` | Mean contributions, individual-observation contributions, time-coloured score/loading views for each faulty turbine, and one PC1–PC2 biplot of both faulty turbines for the report. |
| Phase 4 | `phase4_cv_results.csv`, `phase4_cv_summary.csv`, `phase4_tuning_map.png` | Cross-validated false-alarm rate of every kernel setting (every round, and mean with standard error), near-ties and the selected setting. |
| Phase 4 | `phase4_selected_kernel.csv` | Selected kernel family, width, number of components and limits; **required by Phases 5, 6 and 7**. |
| Phase 5 | `phase5_kernel_monitoring_statistics.csv`, `phase5_kernel_alarm_summary.csv`, `phase5_kernel_fault_periods.csv`, `phase5_kernel_alarm_proportions.csv`, `phase5_alarm_agreement.csv` | Kernel and linear (same limit formulas) statistics, alarms, fault periods and alarm proportions, and their agreement. The proportions CSV is **required by Phase 7**. |
| Phase 5 | `phase5_kernel_control_charts_No.14WT.png`, `phase5_kernel_control_charts_No.39WT.png`, `phase5_kernel_score_plots.png` | Kernel T² and Q control charts and KPC1–KPC2 score plots of the faulty turbines over the healthy scores. |
| Phase 6 | `phase6_partial_score_contributions.csv`, `phase6_ranking_comparison.csv` | Mean fault-period T² and Q shares of every sensor for both models, and the comparison of their rankings; the contributions CSV is **required by Phase 7**. |
| Phase 6 | `phase6_contribution_comparison.png`, `phase6_contribution_heatmap_No.*WT.png` | Mean shares of both models side by side, and the shares of every observation. |
| Phase 7 | `phase7_sensor_selection_linear.csv`, `phase7_sensor_selection_kernel.csv`, `phase7_minimum_sets.csv` | Every step of the sensor search and the fewest sensors that give the same result as each full model. |

For a quick numerical check, `phase2_alarm_summary.csv` should report **15/1,570 (0.96%)** No.2WT alarms, **378/686 (55.10%)** No.14WT alarms, and **521/1,405 (37.08%)** No.39WT alarms under the T²-or-Q rule. `phase3_sensor_ranking.csv` puts variables **9, 13, and 22** in the top five for both statistics in both faulty recordings. `phase4_selected_kernel.csv` selects a **Gaussian kernel with width 5 and 4 components**. `phase5_kernel_alarm_summary.csv` reports **375/686 (54.7%)** No.14WT and **493/1,405 (35.1%)** No.39WT kernel alarms, with every fault-period observation alarmed in both turbines. `phase7_minimum_sets.csv` gives **sensor 9** for the linear model and **sensors 20, 22 and 21** for the kernel model. Small last-decimal-place differences in floating-point limits across numerical-library versions are possible; compare rounded values and alarm counts, not byte-for-byte CSV identity.

Files such as `phase1_alarm_summary.csv`, `phase1_control_charts_n4.png`, `phase1_control_charts_n5.png`, `phase1_scree_plot.png`, and `phase1_calibration_eigenvalues.csv` in `archive/legacy_70_30/` describe the **superseded 70/30 experiment**. The `pca_healthy_*`, `pretreatment_*`, and `step2_*` files in `archive/exploratory/` come from optional earlier exploration. Do not use these as final six-PC monitoring results. The main `outputs/` directory contains no historical results.

## Optional Diagnostics

These commands can be run from the repository root after setup, in any order. None of these scripts supplies inputs to the modelling pipeline:

```bash
python src/inspect_wind_turbine_data.py
python src/pca_healthy_turbine.py
python src/diagnose_healthy_variograms.py
```

The variogram script writes `archive/exploratory/healthy_temporal_variogram.png` and `archive/exploratory/healthy_temporal_variogram_summary.csv`, plus `healthy_score_trajectories_variograms.png` (all six retained PC scores with moving-window variograms) and `healthy_block_holdout.csv` (each healthy block hidden in turn and projected onto a model of the rest). The other two scripts create raw-data, pretreatment, and initial PCA descriptive files in `archive/exploratory/`. They do not replace Phase 1 or add files to the final `outputs/` directory. Do not run optional scripts merely to reproduce the current monitoring charts and alarm summaries.

## Reproducibility and Interpretation Notes

- The commands assume the repository root is the current directory. Running from elsewhere can make relative shell paths fail, although the Python data/output paths themselves are resolved from the script locations.
- The supplied workbook and `outputs/` directory are tracked. Phase scripts overwrite their own generated files; use a fresh clone or copy if you need to preserve the checked-in examples while testing. Archived evidence is separate and is not read by the production pipeline.
- The 0.96% healthy alarm proportion is **in-sample**, not an independently validated false-alarm rate. Using all healthy observations for calibration leaves no independent healthy validation set. The two marginal 99% limits do not guarantee a 1% combined alarm rate.
- No observation-level fault labels, verified timestamps, or sensor names are supplied. Alarm proportions are not detection accuracy; contribution rankings indicate statistical sensitivity, not physical causality.
- PC6 (eigenvalue about 1.014) is close to the Kaiser threshold, and the interpolated No.14WT transition row should not be overinterpreted.
- The kernel model detects the fault through Q only: its T² does not alarm in the fault period, because a sensor value far outside the healthy range is similar to no healthy value and cannot move the kernel scores.
- Each one-sensor kernel is bounded, so the kernel Q shares are about equal for every sensor far outside the healthy range: they show **which** sensors are out of range, not **how far**. The five top-ranked kernel sensors in Phase 7 (20, 22, 21, 13 and 9) have nearly equal shares, so their order is close to arbitrary.
- The current repository contains no automated test suite; the modules check themselves with `assert` statements (for example, the linear kernel must reproduce linear PCA exactly). The pinned package set in `requirements.txt` was installed on Python 3.14, and a fresh copy of the repository reproduced all Phase 1–7 outputs: alarm counts, selected components, selected kernel and sensor rankings identically, and floating-point values to about 10⁻¹¹ relative.
