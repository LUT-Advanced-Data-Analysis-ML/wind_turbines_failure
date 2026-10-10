# Archived project evidence

These files are retained for traceability. They are **not** inputs to the current Phase 1 → Phase 2 → Phase 3 pipeline, and their numbers should not be substituted for current Submission 3 results in `outputs/`.

- `legacy_70_30/` contains the superseded chronological 70/30, five-PC monitoring results and the contemporaneous modelling investigation. It documents why that healthy split was rejected. The Markdown record is a historical snapshot and includes results and proposals that no longer describe the final six-PC model.
- `exploratory/` contains the initial raw-data inspection, pretreatment and healthy-PCA diagnostics, plus the temporal-variogram investigation. The variogram figures, summary and block hold-out table support the decision to use full-healthy calibration. The pretreatment investigation is retained as source evidence for data handling and the shared faulty-recording segment.

The optional scripts in `src/` write their results to `exploratory/`. They are not required to regenerate the current results. Use the repository-root `README.md` for the supported run commands and the `outputs/` directory for the final six-PC model and monitoring results.
