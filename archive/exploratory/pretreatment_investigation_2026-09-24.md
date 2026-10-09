# Pretreatment investigation record

Wind Turbines Level A — 24 September 2026

This record documents the read-only data investigations and the resulting pretreatment decisions for intermediary submission 2. The source is `resources/wind_turbine_fault_diagnosis_data.xlsx`. Observation numbers below are one-based. The analyses concern the three retained sheets (`No.2WT`, `No.14WT`, and `No.39WT`) and their common variables 1–27, excluding healthy constants 12 and 15. No response variables or fault-detection model were used.

## Extreme-value audit

We calculated Tukey 1.5 × IQR fences separately for each of the 25 retained variables in each turbine. A flag is a prompt to inspect an observation, not a determination that it is erroneous. The reproducible 75-row summary is `archive/exploratory/pretreatment_extreme_value_diagnostics.csv`; the existing figure in that directory illustrates No.14WT variables 5 and 11 only.

| Turbine | Flagged cells | Variables with flags | Notable observations |
| --- | ---: | ---: | --- |
| No.2WT | 746 | 8 | Variable 22 contributes 646 flags because its IQR is extremely narrow. Variable 13 has seven short excursions. |
| No.14WT | 177 | 11 | Observation 358 has nine simultaneously flagged retained variables; observation 288 is another multivariable transient. |
| No.39WT | 635 | 20 | Observation 470 has 17 simultaneous flags. Several other flags form long operating stretches. |

The IQR rule is not a suitable automatic cleaning rule here. It overflags long regimes and tightly quantized variables: healthy variable 22 alone accounts for most healthy flags. Conversely, No.14WT variable 5 reaches 1,021.72 at observation 358 yet has no IQR flags because its distribution spans very different regimes. No.14WT variable 11 reaches 17,052.87 at the same observation and has one IQR flag. The extreme transition rows may be sensor or acquisition anomalies, genuine changes, or both; the workbook does not resolve this. We retain measured values without clipping, replacing, or deleting observations.

## Healthy variable 13 sensitivity

Healthy variable 13 has seven IQR-flagged observations: 1022 (0.279926986), 1023 (0.043260377), 1372 (0.163193703), 1373 (0.045293704), 1423 (0.483960330), 1424 (0.039093703), and 1426 (0.041893721). The other 1,563 readings cluster near 0.0103937. The seven excursions account for 99.79% of the variable's total squared autoscaled variation. Its sample standard deviation is 0.01436366 with all readings, versus 0.00000001879 among the unflagged readings (approximately 764,000 times smaller). This makes the fitted scale dependent on the excursions.

In a temporary sensitivity PCA omitting variable 13, PC1–PC3 score correlations with the current PCA exceeded 0.999 and their leading loading variables were unchanged. The absolute explained variances of PC1–PC3 also changed little. Their *percentages* rose partly because the comparison has 24 rather than 25 unit-variance variables. Later components changed materially: variable 13 has loadings 0.743 on current PC6 and 0.608 on current PC7, and omitting it reorganizes those components. Replacing the seven observed readings by interpolation was considered only as an in-memory sensitivity test, not as a treatment: it leaves an almost constant signal whose minute fluctuations would be magnified by autoscaling.

## Variable 13 across turbines and shared sequence

No.14WT variable 13 is mostly near 88.99 through observation 357, passes through 23.18 at observation 358, and is near 0.010394 from observation 359 onward. No.39WT is near 88.99 through observation 469 and near 0.010394 from observation 471 onward, with a transition at observation 470. The healthy seven excursions are short and below 0.484, so they do not resemble the faulty sheets' long high-valued stretches. These patterns may indicate operating regimes, a quantized measurement, or another data-collection feature; they do not identify the sensor or prove categorical meaning.

The two faulty sheets share a near-identical 328-observation sequence: No.14WT observations 359–686 align with No.39WT observations 471–798. Across the 27 raw common variables, 324 paired rows match exactly. In the remaining four rows, only variable 13 differs, by 0.000006321 (0.0104 versus 0.010393679). This strongly suggests a reused recording or data block, although the workbook does not explain its provenance. The shared portion is not independent evidence from two turbines. It does not affect the healthy PCA fit, which uses only No.2WT, but it limits later comparisons or pooled analyses of the faulty sheets.

## Possible categorical variables

All retained cells are stored numerically. Variable 14 has 11 unique integer levels in the healthy turbine and is a possible state-like signal, but it may instead be a quantized numerical measurement. Variable 13 has a few repeated plateaus and transients, but its physical meaning is also unknown. No variable can be confirmed categorical from its numeric pattern alone. No one-hot encoding is applied.

## Pretreatment decisions and limits

- Keep the existing 25 variables in the same order for all three retained turbines; remove variables 12 and 15 from every turbine because they are constant in the healthy reference.
- Preserve all observation rows. Continue to linearly interpolate only the missing No.14WT variable 9 value at observation 358, yielding 24,782,093. The estimate is uncertain because other variables change abruptly there.
- Fit means and sample standard deviations on healthy No.2WT only, then apply those fixed parameters to all retained turbines. Retain variable 13 provisionally despite its scale sensitivity, and interpret PC6–PC7 cautiously.
- Do not automatically treat IQR flags, healthy variable 13 excursions, or the transition rows as errors. Do not cap, replace, delete, or one-hot encode them without sensor definitions or acquisition metadata.
- Keep the original observation order. Sampling is nominally 10 seconds, but there are no timestamps to verify gaps, elapsed duration, or synchronization. Do not resample or claim simultaneous cross-turbine events.
- Do not deduplicate the shared faulty-sheet segment without provenance. Describe it as a dependence limitation rather than independent confirmation.

The main unresolved questions are variable meanings and units, whether the unusual observations reflect faults or acquisition artefacts, the origin of the shared data block, and the true timing of observations. These uncertainties should be revisited before fault interpretation; they do not currently justify a further pretreatment change.
