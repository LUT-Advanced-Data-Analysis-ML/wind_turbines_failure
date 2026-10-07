# Modelling investigation record

Wind Turbines Level A — 7 October 2026

This record documents the first implementation of the modelling plan of report 2 (Figure 5) and the open questions it raised. The pretreated data are those of `pretreatment_investigation_2026-09-24.md`: No.2WT, No.14WT and No.39WT with the same 25 variables. Phase 1 (`src/phase1_healthy_model.py`), Phase 2 (`src/phase2_fault_detection.py`) and Phase 3 (`src/phase3_diagnostics.py`) write their results to `outputs/phase1_*`, `outputs/phase2_*` and `outputs/phase3_*`. Theoretical limits are 99%: F-distribution for T² and Jackson–Mudholkar for Q. An alarm is an observation above the T² or the Q limit. Observation numbers are one-based.

## Phase 1: healthy reference model

No.2WT was split in time order into calibration (observations 1–1099) and validation (1100–1570). The calibration eigenvalues are 11.37, 4.77, 1.63, 1.38, 1.05 and 0.998. The Kaiser criterion gives 5 components and the scree-plot elbow gives 3, so the reduction started from n0 = 5.

| n | Calibration alarms | Validation alarms | Decision |
| ---: | ---: | ---: | --- |
| 5 | 8 / 1099 (0.7%) | 422 / 471 (89.6%) | initial n0 |
| 4 | 8 / 1099 (0.7%) | 435 / 471 (92.4%) | rejected: alarms added |

The final model therefore has n* = 5. It was refitted on all 1570 observations: T² limit 15.19, Q limit 14.11, with 9 healthy alarms (0.6%). The calibration behaves as expected, but the validation block does not. Validation T² exceeds its limit steadily from about observation 1150, and Q from about observation 1350. Short spikes at observations 1022–1023, 1372–1373 and 1423–1424 coincide with the healthy variable 13 excursions. The mean + 3 SD limits are unstable: the calibration T² 3 SD limit is 58.6 at n = 5 but 12.7 at n = 4, because the single spike at observation 1022 inflates it. The theoretical limits are therefore used for all decisions.

## Validation diagnosis (exploratory)

These checks were run outside the committed scripts and should be reproduced in a script before they are reported.

- **The alarms come mainly from T².** All 422 alarming validation observations exceed the T² limit, 411 also exceed Q, and none exceed Q alone. T² first exceeds its limit at observation 1127 and Q at observation 1157.
- **PC1 follows the order of the observations.** The PC1 scores of the full healthy model correlate with observation order at |r| = 0.92; PC2–PC8 correlate at 0.28 or less. In validation, PC1 carries 48% of T², against 20% for each of the five components in calibration.
- **The drift is in the PC1 group.** Fifteen variables correlate with observation order at |r| ≥ 0.73: 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 20, 21, 22 and 24. The next is variable 26 (0.58), then 0.37 or less. These are the variables with |PC1 loading| of about 0.2 or more, the loading every variable would have if all contributed equally (variable 22: 0.195). In validation, variables 3, 4, 6, 7, 8, 20 and 21 shift by 4.4–6.9 calibration standard deviations, and 91–100% of their values lie outside the calibration range. The largest validation T² contributions come from variables 21, 13, 4, 3, 20, 8, 7 and 6. The largest Q contributions come from 20, 27, 22, 21, 19 and 26.

The first 70% of No.2WT therefore does not cover the operating range of the last 30%. The healthy model fitted on all observations is in control, so the failure reflects incomplete calibration coverage rather than a broken correlation structure. Whether PC1 is a pure time trend or a slowly changing operating condition cannot be determined without sensor metadata.

Two options are under consideration. Neither has been implemented, and the estimates come from quick checks only:

- **Remove the PC1 group from all turbines.** Validation alarms fall to about 47–52% when the 14 variables with |PC1 loading| ≥ 0.2 are removed. They fall to about 19–25% when variables 22 and 26 are also removed. After removal, a new PC1 still correlates with observation order at about 0.5. Variables 9 and 22, which Phase 3 ranks as fault-sensitive, would be lost from diagnostics.
- **Exclude PC1 from monitoring and plot its score separately.** Q never contains PC1, because it is the residual after all retained components, so only T² changes (PC2–PCn). At n = 5, validation T² alarms fall from 89.6% to 38.6%, but Q stays at 87.3%, so alarms remain at 87.5%. A fault that moves only along PC1 would not be detected.

## Phase 2: fault detection

Both faulty turbines were projected onto the final model without refitting.

| Turbine | All observations | Before variable 13 transition | Transition | After transition |
| --- | ---: | ---: | --- | ---: |
| No.14WT | 377 / 686 (55.0%) | 357 / 357 (100%) | obs 358: alarm | 19 / 328 (5.8%): T² 4, Q 15 |
| No.39WT | 508 / 1405 (36.2%) | 469 / 469 (100%) | obs 470: alarm | 38 / 935 (4.1%): T² 4, Q 34 |

Before the transition, T² is about 10⁶–10⁷ and Q about 10⁷–10⁸, against limits near 15. Both turbines drop to near-healthy levels at the transition, consistent with a fault that is later resolved. Both alarm from observation 1, so the recordings start inside the fault and its onset cannot be observed. After the transition the alarm rate (4–6%) is above the nominal 1% and comes mostly from Q.

## Phase 3: fault periods and sensor diagnostics

The fault period was defined in two ways. **Method A** uses the variable 13 transition. **Method B** uses alarm episodes: an episode starts at three consecutive alarms and ends at the last alarm before 30 alarm-free observations.

| Turbine | Method A | Method B episodes |
| --- | --- | --- |
| No.14WT | 1–357 | 1–385 (94% alarms); 474–516 (14%) |
| No.39WT | 1–469 | 1–497 (95%); 586–628 (14%); 884–903 (35%) |

The methods agree on the fault block; method B extends it by 27 observations in both turbines. The later episodes 474–516 and 586–628 lie at the same position inside the shared data block, so they are a single finding. Outside the method B episodes, alarm rates are 3.5% (No.14WT) and 2.5% (No.39WT).

Variables 12 and 15 are constant in No.2WT (1750 and 849) and are not in the PCA. In the fault period, variable 12 is 50 in both turbines. Variable 15 is unchanged (849) in No.14WT but 9.3–11 in No.39WT. Both are 0 at the transition and return exactly to their healthy values afterwards. This independently supports the fault period as a distinct state.

Contributions were divided by each variable's healthy 99th-percentile contribution and ranked by magnitude, because T² contributions can be negative. Variables 9, 13 and 22 are in the top five for T² and SPEx in both turbines; variables 9, 13, 16 and 22 are in at least three of the four rankings. This ranking must be interpreted cautiously. In the fault period, all 25 variables exceed their healthy 99th percentile in about 100% of observations (24 of 25 for T²). The ranking therefore shows which variables move most in healthy standard deviations, which favours variables with a very small healthy spread such as 9 and 13. Variable 13 also defines method A, so its rank is partly circular. After the transition, the remaining alarms are driven in SPEx by variable 26 in both turbines (6.4 and 6.8 times its healthy 99th percentile), while the T² contributions stay below the healthy level.

## Decisions and limits

- Theoretical 99% limits are used for all decisions. The 3 SD limits are reported only for comparison because the healthy spikes make them unstable.
- The current model (25 variables, n* = 5) is kept as the reference result until the validation problem is resolved. Phases 2 and 3 load the saved Phase 1 model, so they update automatically when Phase 1 changes.
- Fault onset cannot be estimated, because both faulty recordings start in alarm. Only the end of the fault is observable.
- No.14WT observations 359–686 and No.39WT observations 471–798 are a near-identical shared block. Their after-transition results and the later method B episodes are not independent evidence.
- The fault-period sensor ranking reflects deviation size in healthy standard deviations, not a confirmed fault cause.

The main open questions are how to handle the healthy drift in validation (removing the PC1 group, excluding PC1 from monitoring, or another approach), whether PC1 is a time trend or an operating condition, and what variable 26 means in the after-transition alarms.
