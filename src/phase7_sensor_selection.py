"""Phase 7: the first passing contribution-ranked sensor prefix for each model.

For each model (kernel and linear) the sensors are ranked by their mean
share of Q over the fault period (Phase 6, averaged over the two faulty
turbines). Models are built on the top 1, top 2, ... sensors, from healthy
No.2WT observations only, and the search stops at the first prefix meeting
these monitoring criteria relative to the full 25-sensor model:
A: at least 99% of the fault period alarmed in both faulty turbines,
B: after-transition alarm rate at most 2 percentage points above the full model's,
C: healthy false-alarm rate in the blocked cross-validation at most 2%.
This ranked-prefix search does not establish the smallest passing subset
among all sensor combinations.

Note: Kernel sets keep the Phase 4 settings. Linear sets follow the Phase 1
recipe (autoscaling and the Kaiser rule) with one component for a single
sensor. Both models use the F-distribution T2 limit and the scaled
chi-square Q limit, as in Phase 5. Models are fitted only on healthy data.
Faulty observations determine the Q-share ranking and assess criteria A and B,
so subset selection is fault-specific.
"""

import numpy as np
import pandas as pd

from kernel_monitoring import fit_kernel_model, kernel_limits, load_kernel_settings, project_kernel
from pca_monitoring import (
    PCAModel,
    Projection,
    fit_model,
    project,
    t2_limit_f,
    verify_reference_projection,
)
from phase1_healthy_model import KAISER_EIGENVALUE_THRESHOLD
from phase3_diagnostics import VARIABLE_13_TRANSITION
from phase4_kernel_tuning import LIMIT_TYPE, SELECTED_KERNEL_FILE
from pretreatment import FAULTY_TURBINES, HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data
from time_series_cv import cross_validate_far

CONTRIBUTIONS_FILE = OUTPUT_DIR / "phase6_partial_score_contributions.csv"
PHASE5_PROPORTIONS_FILE = OUTPUT_DIR / "phase5_kernel_alarm_proportions.csv"
MODELS = ["kernel", "linear"]
# Sensors are ranked by the statistic that detects the fault in both models.
RANKING_STATISTIC = "Q"
MIN_DETECTION_PERCENT = 99.0
AFTER_TRANSITION_MARGIN_PERCENT = 2.0
MAX_FALSE_ALARM_PERCENT = 2.0


def main() -> None:
    pca_x, variables = load_pca_data()
    settings = load_kernel_settings(SELECTED_KERNEL_FILE)
    rankings = rank_sensors()

    # Phase 5 after-transition alarm rates of the full models.
    phase5 = pd.read_csv(PHASE5_PROPORTIONS_FILE)
    phase5 = phase5[
        (phase5["period"] == "A: after transition") & (phase5["statistic"] == "T2 or Q")
    ].set_index(["model", "turbine"])["alarm_percent"]

    print("\n========================================================")
    print(
        f"\nSensors added in the order of their mean {RANKING_STATISTIC} share over the "
        "fault period (Phase 6, mean of both faulty turbines)"
    )
    print(
        f"Passes monitoring criteria relative to the full model: A) >= {MIN_DETECTION_PERCENT:g}% of the fault period "
        f"alarmed in both turbines; B) after-transition alarms <= full model + "
        f"{AFTER_TRANSITION_MARGIN_PERCENT:g} points; C) healthy CV false-alarm rate <= "
        f"{MAX_FALSE_ALARM_PERCENT:g}%"
    )

    minimum_rows = []
    for model_name in MODELS:
        ranking = rankings[model_name]
        full = evaluate_sensor_set(model_name, variables, pca_x, settings)
        for name in FAULTY_TURBINES:
            assert np.isclose(full[f"after_transition_percent_{name}"], phase5[(model_name, name)])
        after_limits = {
            name: full[f"after_transition_percent_{name}"] + AFTER_TRANSITION_MARGIN_PERCENT
            for name in FAULTY_TURBINES
        }

        print(f"\n{model_name} model, ranking: {' '.join(map(str, ranking))}")
        print(
            f"  full model ({len(variables)} sensors): fault period "
            + " / ".join(f"{full[f'fault_detection_percent_{name}']:.1f}%" for name in FAULTY_TURBINES)
            + ", after transition "
            + " / ".join(f"{full[f'after_transition_percent_{name}']:.1f}%" for name in FAULTY_TURBINES)
            + f", CV false alarms {full['cv_false_alarm_percent']:.2f}%"
        )

        steps = []
        for n_sensors in range(1, len(ranking) + 1):
            sensors = ranking[:n_sensors]
            row = evaluate_sensor_set(model_name, sensors, pca_x, settings)
            row["check_a_fault_detected"] = all(
                row[f"fault_detection_percent_{name}"] >= MIN_DETECTION_PERCENT
                for name in FAULTY_TURBINES
            )
            row["check_b_quiet_after_transition"] = all(
                row[f"after_transition_percent_{name}"] <= after_limits[name] + 1e-9
                for name in FAULTY_TURBINES
            )
            row["check_c_few_false_alarms"] = (
                row["cv_false_alarm_percent"] <= MAX_FALSE_ALARM_PERCENT
            )
            row["same_result"] = (
                row["check_a_fault_detected"]
                and row["check_b_quiet_after_transition"]
                and row["check_c_few_false_alarms"]
            )
            steps.append({"n_sensors": n_sensors, "added_sensor": sensors[-1], **row})
            print(
                f"  {n_sensors:2d} sensors (+{sensors[-1]:2d}), {row['n_components']} components: "
                "fault period "
                + " / ".join(f"{row[f'fault_detection_percent_{name}']:5.1f}%" for name in FAULTY_TURBINES)
                + ", after transition "
                + " / ".join(f"{row[f'after_transition_percent_{name}']:4.1f}%" for name in FAULTY_TURBINES)
                + f", CV false alarms {row['cv_false_alarm_percent']:5.2f}% | A "
                + ("pass" if row["check_a_fault_detected"] else "fail")
                + ", B "
                + ("pass" if row["check_b_quiet_after_transition"] else "fail")
                + ", C "
                + ("pass" if row["check_c_few_false_alarms"] else "fail")
            )
            if row["same_result"]:
                break

        steps = pd.DataFrame(steps)
        steps.to_csv(OUTPUT_DIR / f"phase7_sensor_selection_{model_name}.csv", index=False)
        selected = steps.iloc[-1]
        assert selected["same_result"]
        minimum_rows.append(
            {
                "model": model_name,
                "ranking": " ".join(map(str, ranking)),
                "minimum_sensors": int(selected["n_sensors"]),
                "sensors": selected["sensors"],
                "n_components": int(selected["n_components"]),
                "cv_false_alarm_percent": selected["cv_false_alarm_percent"],
                **{
                    f"fault_detection_percent_{name}": selected[f"fault_detection_percent_{name}"]
                    for name in FAULTY_TURBINES
                },
                **{
                    f"after_transition_percent_{name}": selected[f"after_transition_percent_{name}"]
                    for name in FAULTY_TURBINES
                },
            }
        )
        print(
            f"  -> {model_name}: first passing ranked prefix with "
            f"{int(selected['n_sensors'])} sensor(s): {selected['sensors']}"
        )

    pd.DataFrame(minimum_rows).to_csv(OUTPUT_DIR / "phase7_minimum_sets.csv", index=False)


def rank_sensors() -> dict[str, list[int]]:
    """
    Sensors of each model ordered by their mean fault-period share of the ranking statistic.

    The Phase 6 shares are averaged over the two faulty turbines; equal
    shares are ordered by variable number.
    """
    contributions = pd.read_csv(CONTRIBUTIONS_FILE)
    contributions = contributions[contributions["statistic"] == RANKING_STATISTIC]
    rankings = {}
    for model_name in MODELS:
        mean_shares = (
            contributions[contributions["model"] == model_name]
            .groupby("variable")["mean_share_percent"]
            .mean()
            .reset_index()
            .sort_values(["mean_share_percent", "variable"], ascending=[False, True])
        )
        rankings[model_name] = mean_shares["variable"].astype(int).tolist()
    return rankings


def single_sensor_projection(model: PCAModel, x: pd.DataFrame) -> Projection:
    """
    Single-sensor chart: T2 = z^2 of the autoscaled sensor and no residual.

    With one sensor the only component is the sensor itself (eigenvalue 1
    after autoscaling), so Q is zero.
    """
    z = ((x - model.mean) / model.std).to_numpy()
    return Projection(scaled=z, scores=z, residuals=np.zeros_like(z), t2=z[:, 0] ** 2, q=np.zeros(len(z)))


def fit_monitor(model_name: str, healthy_x: pd.DataFrame, settings: dict):
    """
    Fit the model of one sensor set on healthy observations.

    Returns the projection function, the limits and the number of components.
    Kernel sets keep the Phase 4 settings; linear sets use the Kaiser rule
    (at least one component), and a single sensor uses the single-sensor
    chart with the one-component F-distribution T2 limit and no Q limit.
    """
    if model_name == "kernel":
        model = fit_kernel_model(healthy_x, settings["family"], settings["width"])
        n_components = settings["n_components"]

        def project_function(x):
            return project_kernel(model, x, n_components)
    elif healthy_x.shape[1] == 1:
        model = fit_model(healthy_x)
        n_components = 1

        def project_function(x):
            return single_sensor_projection(model, x)
    else:
        model = fit_model(healthy_x)
        n_components = max(1, int(np.sum(model.eigenvalues > KAISER_EIGENVALUE_THRESHOLD)))

        def project_function(x):
            return project(model, x, n_components)

    reference = project_function(healthy_x)
    verify_reference_projection(model, reference)
    if model_name == "linear" and healthy_x.shape[1] == 1:
        limits = {"t2": t2_limit_f(model, 1), "q": np.inf}
    else:
        limits = kernel_limits(model, reference, LIMIT_TYPE)
    return project_function, limits, n_components


def monitor_round(
    train_x: pd.DataFrame, valid_x: pd.DataFrame, model_name: str, settings: dict
) -> list[dict]:
    """False-alarm rate of one cross-validation round: fit on training, score the validation observations."""
    project_function, limits, n_components = fit_monitor(model_name, train_x, settings)
    projection = project_function(valid_x)
    alarms = (projection.t2 > limits["t2"]) | (projection.q > limits["q"])
    return [{"n_components": n_components, "far_percent": 100 * alarms.mean()}]


def evaluate_sensor_set(
    model_name: str, sensors: list[int], pca_x: dict[str, pd.DataFrame], settings: dict
) -> dict:
    """
    Fault-period detection, after-transition alarms and healthy CV false alarms of one sensor set.

    The model is fitted on all healthy observations of the chosen sensors
    and the faulty turbines are projected onto it; the cross-validation
    refits it on the training observations of every round.
    """
    columns = [variable for variable in pca_x[HEALTHY_TURBINE].columns if variable in sensors]
    healthy_x = pca_x[HEALTHY_TURBINE][columns]
    project_function, limits, n_components = fit_monitor(model_name, healthy_x, settings)

    row = {"sensors": " ".join(map(str, sensors)), "n_components": n_components}
    for name in FAULTY_TURBINES:
        projection = project_function(pca_x[name][columns])
        alarms = (projection.t2 > limits["t2"]) | (projection.q > limits["q"])
        observations = np.arange(1, len(alarms) + 1)
        transition = VARIABLE_13_TRANSITION[name]
        row[f"fault_detection_percent_{name}"] = 100 * alarms[observations < transition].mean()
        row[f"after_transition_percent_{name}"] = 100 * alarms[observations > transition].mean()

    rounds = cross_validate_far(healthy_x, monitor_round, model_name=model_name, settings=settings)
    row["cv_false_alarm_percent"] = rounds["far_percent"].mean()
    row["cv_false_alarm_se"] = rounds["far_percent"].std(ddof=1) / np.sqrt(len(rounds))
    return row


if __name__ == "__main__":
    main()
