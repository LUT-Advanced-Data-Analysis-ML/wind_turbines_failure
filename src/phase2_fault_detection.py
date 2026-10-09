"""Phase 2 of the modelling plan: fault detection in No.14WT and No.39WT.

Follows steps 8-10 of the PCA-based fault detection workflow (report 2,
Figure 5): autoscale the faulty turbines with the healthy No.2WT statistics,
project them onto the final healthy model, compute T2 and Q, and chart them
against the healthy control limits.

Note: The final model, its number of components and its limits are loaded
from the Phase 1 outputs. Nothing is refitted on the faulty turbines.
"""

import numpy as np
import pandas as pd

from pca_monitoring import PCAModel, load_model, project
from phase1_healthy_model import FINAL_MODEL_PREFIX, alarm_flags, create_control_charts
from pretreatment import FAULTY_TURBINES, HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data

MONITORING_STATISTICS_FILE = OUTPUT_DIR / "phase2_monitoring_statistics.csv"


def main() -> None:
    pca_x, _ = load_pca_data()
    model, n_components, limits = load_final_model()

    print("\n========================================================")
    print(
        f"\nFinal healthy model: {n_components} components, fitted on "
        f"{model.n_observations} {HEALTHY_TURBINE} observations"
    )
    print(f"Limits: T2 F = {limits['t2_f']:.2f}, Q JM = {limits['q_jm']:.2f}")

    # ============================ Steps 8-9: autoscale, project and compute T2 and Q ===========================
    # The healthy turbine is projected as an in-sample reference row only;
    # its alarm proportion is not an independent validation estimate.
    projections = {
        name: project(model, pca_x[name], n_components)
        for name in [HEALTHY_TURBINE, *FAULTY_TURBINES]
    }

    statistics = []
    summary = []
    for name, projection in projections.items():
        observations = np.arange(1, len(projection.t2) + 1)
        t2_alarms = projection.t2 > limits["t2_f"]
        q_alarms = projection.q > limits["q_jm"]
        either = alarm_flags(projection, limits)
        either_3sd = alarm_flags(projection, limits, three_sigma=True)

        if name in FAULTY_TURBINES:
            statistics.append(
                pd.DataFrame(
                    {
                        "turbine": name,
                        "observation": observations,
                        "t2": projection.t2,
                        "q": projection.q,
                        "t2_alarm": t2_alarms,
                        "q_alarm": q_alarms,
                        "t2_or_q_alarm": either,
                        "t2_alarm_3sd": projection.t2 > limits["t2_3sd"],
                        "q_alarm_3sd": projection.q > limits["q_3sd"],
                        "t2_or_q_alarm_3sd": either_3sd,
                    }
                )
            )
        summary.append(
            {
                "turbine": name,
                "role": "healthy in-sample" if name == HEALTHY_TURBINE else "faulty evaluation",
                "observations": len(projection.t2),
                "n_components": n_components,
                "t2_alarms": int(t2_alarms.sum()),
                "q_alarms": int(q_alarms.sum()),
                "t2_or_q_alarms": int(either.sum()),
                "t2_or_q_alarm_percent": 100 * either.mean(),
                "t2_or_q_alarms_3sd": int(either_3sd.sum()),
                "t2_or_q_alarm_percent_3sd": 100 * either_3sd.mean(),
                "first_alarm_observation": int(observations[either][0]) if either.any() else None,
                "last_alarm_observation": int(observations[either][-1]) if either.any() else None,
            }
        )

    pd.concat(statistics, ignore_index=True).to_csv(
        MONITORING_STATISTICS_FILE, index=False
    )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "phase2_alarm_summary.csv", index=False)

    # ============================ Step 10: T2 and Q charts of the faulty turbines ===========================
    for name in FAULTY_TURBINES:
        projection = projections[name]
        create_control_charts(
            {name: (np.arange(1, len(projection.t2) + 1), projection)},
            limits,
            n_components,
            f"{name} projected onto the final {HEALTHY_TURBINE} model, {n_components} components",
            OUTPUT_DIR / f"phase2_control_charts_{name}.png",
        )

    print("\n========================================================")
    print("\nSteps 8-10: out-of-control observations")
    for row in summary.itertuples():
        print(
            f"{row.turbine} ({row.role}): T2 or Q alarms = {row.t2_or_q_alarms} / "
            f"{row.observations} ({row.t2_or_q_alarm_percent:.1f}%); "
            f"T2 = {row.t2_alarms}, Q = {row.q_alarms}; "
            f"with 3SD limits: {row.t2_or_q_alarm_percent_3sd:.1f}%"
        )
        if row.role == "faulty evaluation":
            print(
                f"  first alarm: observation {row.first_alarm_observation}, "
                f"last alarm: observation {row.last_alarm_observation}"
            )


def load_final_model() -> tuple[PCAModel, int, dict[str, float]]:
    """Load the final healthy model, its number of components and its limits from Phase 1."""
    model = load_model(FINAL_MODEL_PREFIX)
    saved_limits = pd.read_csv(
        f"{FINAL_MODEL_PREFIX}_limits.csv", float_precision="round_trip"
    ).iloc[0]
    limits = {
        "t2_f": saved_limits["t2_limit_f"],
        "q_jm": saved_limits["q_limit_jackson_mudholkar"],
        "t2_3sd": saved_limits["t2_limit_3sd"],
        "q_3sd": saved_limits["q_limit_3sd"],
    }
    return model, int(saved_limits["n_components"]), limits


if __name__ == "__main__":
    main()
