"""Phase 1 of the modelling plan: the healthy reference model (No.2WT).

Follows steps 1-7 of the PCA-based fault detection workflow (report 2,
Figure 5): split No.2WT in time order, fit PCA on the calibration block,
choose the initial number of components, compute control limits, chart T2
and Q for both blocks, reduce the number of components while no alarms are
added, and refit the final model on all healthy observations.

Note: No.14WT and No.39WT are not used for calibration, validation or
choosing the number of components. They are only projected in Phase 2.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca_monitoring import (
    CONFIDENCE,
    PCAModel,
    Projection,
    fit_model,
    project,
    q_limit_jackson_mudholkar,
    save_model,
    t2_limit_f,
    three_sigma_limit,
    verify_reference_projection,
)
from pretreatment import HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data

# The first 70% of No.2WT form the calibration block and the last 30% the
# validation block, so validation lies entirely after calibration in time.
CALIBRATION_FRACTION = 0.70
FINAL_MODEL_PREFIX = OUTPUT_DIR / "phase1_final_model"
BLOCK_COLORS = {"calibration": "tab:blue", "validation": "tab:orange"}


def main() -> None:
    pca_x, variables = load_pca_data()
    healthy_x = pca_x[HEALTHY_TURBINE]

    # ============================ Step 1: split No.2WT in time order ===========================
    n_calibration = round(CALIBRATION_FRACTION * len(healthy_x))
    blocks = {
        "calibration": healthy_x.iloc[:n_calibration],
        "validation": healthy_x.iloc[n_calibration:],
    }

    print("\n========================================================")
    print("\nStep 1: time-ordered split of No.2WT")
    for name, data in blocks.items():
        print(
            f"{name}: observations {data.index[0] + 1}-{data.index[-1] + 1} "
            f"({len(data)} observations x {data.shape[1]} variables)"
        )

    # ============================ Step 2: fit PCA on the calibration block ===========================
    # The calibration block is autoscaled with its own means and standard
    # deviations; the validation block is only projected onto this model.
    calibration_model = fit_model(blocks["calibration"])

    # ============================ Step 3: initial number of components ===========================
    eigenvalues = calibration_model.eigenvalues
    n_kaiser = int(np.sum(eigenvalues > 1))
    n_elbow = scree_elbow(eigenvalues)
    # The reduction loop can only remove components, so it starts from the
    # larger of the two suggestions.
    n_initial = max(n_kaiser, n_elbow)
    save_calibration_eigenvalues(eigenvalues)
    create_scree_plot(eigenvalues, n_kaiser, n_elbow, n_initial)

    print("\n========================================================")
    print("\nStep 3: initial number of components")
    for component, value in enumerate(eigenvalues[:10], start=1):
        print(f"PC{component}: eigenvalue = {value:.3f}")
    print(f"Kaiser criterion (eigenvalue > 1): {n_kaiser} components")
    print(f"Scree-plot elbow: {n_elbow} components")
    print(f"Initial number of components n0 = {n_initial}")

    # ============================ Steps 4-6: limits, charts and reduction of n ===========================
    # At each n, an alarm is an observation above the theoretical T2 or Q
    # limit. n is reduced while n - 1 does not increase the number of alarms
    # in either block.
    results = {n_initial: evaluate_components(calibration_model, blocks, n_initial)}
    decisions = {n_initial: "initial n0"}
    n_components = n_initial
    while n_components > 1:
        candidate = n_components - 1
        results[candidate] = evaluate_components(calibration_model, blocks, candidate)
        alarms_added = any(
            results[candidate]["alarms"][block] > results[n_components]["alarms"][block]
            for block in blocks
        )
        if alarms_added:
            decisions[candidate] = "rejected: alarms added"
            break
        decisions[candidate] = "accepted: no alarms added"
        n_components = candidate
    n_final = n_components

    summary = summarise_alarms(results, decisions, n_final)
    summary.to_csv(OUTPUT_DIR / "phase1_alarm_summary.csv", index=False)
    for n, result in results.items():
        create_control_charts(
            {
                block: (blocks[block].index + 1, result["projections"][block])
                for block in blocks
            },
            result["limits"],
            n,
            f"No.2WT calibration model, {n} components",
            OUTPUT_DIR / f"phase1_control_charts_n{n}.png",
            split_observation=n_calibration + 0.5,
        )

    print("\n========================================================")
    print("\nSteps 4-6: control limits and reduction of the number of components")
    for n, result in results.items():
        limits = result["limits"]
        print(f"\nn = {n} ({decisions[n]})")
        print(
            f"limits: T2 F = {limits['t2_f']:.2f}, Q JM = {limits['q_jm']:.2f}, "
            f"T2 3SD = {limits['t2_3sd']:.2f}, Q 3SD = {limits['q_3sd']:.2f}"
        )
        for block in blocks:
            row = summary[(summary["n_components"] == n) & (summary["block"] == block)].iloc[0]
            print(
                f"  {block}: T2 or Q alarms = {row['t2_or_q_alarms']} / "
                f"{row['observations']} ({row['t2_or_q_alarm_percent']:.1f}%); "
                f"T2 = {row['t2_alarms']}, Q = {row['q_alarms']}; "
                f"with 3SD limits: {row['t2_or_q_alarm_percent_3sd']:.1f}%"
            )
    print(f"\nFinal number of components n* = {n_final}")

    # ============================ Step 7: final healthy model ===========================
    # Scaling, PCA and limits are refitted on all healthy observations with
    # n* components and reused unchanged in Phase 2.
    final_model = fit_model(healthy_x)
    final_projection = project(final_model, healthy_x, n_final)
    verify_reference_projection(final_model, final_projection)
    final_limits = control_limits(final_model, final_projection, n_final)

    save_model(final_model, FINAL_MODEL_PREFIX)
    pd.DataFrame(
        [
            {
                "n_components": n_final,
                "n_observations": final_model.n_observations,
                "confidence": CONFIDENCE,
                "t2_limit_f": final_limits["t2_f"],
                "q_limit_jackson_mudholkar": final_limits["q_jm"],
                "t2_limit_3sd": final_limits["t2_3sd"],
                "q_limit_3sd": final_limits["q_3sd"],
            }
        ]
    ).to_csv(f"{FINAL_MODEL_PREFIX}_limits.csv", index=False)
    create_control_charts(
        {"No.2WT": (healthy_x.index + 1, final_projection)},
        final_limits,
        n_final,
        f"Final No.2WT model (all observations), {n_final} components",
        OUTPUT_DIR / "phase1_final_model_control_charts.png",
    )

    final_alarms = alarm_flags(final_projection, final_limits)
    print("\n========================================================")
    print("\nStep 7: final healthy model")
    print(
        f"Fitted on all {final_model.n_observations} No.2WT observations "
        f"with {n_final} components"
    )
    print(
        f"Limits: T2 F = {final_limits['t2_f']:.2f}, "
        f"Q JM = {final_limits['q_jm']:.2f}"
    )
    print(
        f"No.2WT observations above the T2 or Q limit: {final_alarms.sum()} "
        f"({100 * final_alarms.mean():.1f}%)"
    )
    print(f"Saved model: {FINAL_MODEL_PREFIX.name}_parameters/eigenvalues/limits.csv")


def scree_elbow(eigenvalues: np.ndarray) -> int:
    """
    Number of components at the elbow of the scree plot.

    Both axes are scaled to [0, 1] and the elbow is the component farthest
    from the straight line joining the first and last eigenvalues.
    """
    components = np.arange(1, len(eigenvalues) + 1)
    x = (components - components[0]) / (components[-1] - components[0])
    y = (eigenvalues - eigenvalues[-1]) / (eigenvalues[0] - eigenvalues[-1])
    # The line joins (0, 1) and (1, 0), so the distance is proportional to
    # 1 - x - y for points below the line.
    return int(components[np.argmax(1 - x - y)])


def control_limits(
    model: PCAModel, reference_projection: Projection, n_components: int
) -> dict[str, float]:
    """Theoretical 99% limits and mean + 3 SD limits of the reference T2 and Q."""
    return {
        "t2_f": t2_limit_f(model, n_components),
        "q_jm": q_limit_jackson_mudholkar(model, n_components),
        "t2_3sd": three_sigma_limit(reference_projection.t2),
        "q_3sd": three_sigma_limit(reference_projection.q),
    }


def alarm_flags(
    projection: Projection, limits: dict[str, float], three_sigma: bool = False
) -> np.ndarray:
    """Observations above the T2 or the Q limit."""
    if three_sigma:
        return (projection.t2 > limits["t2_3sd"]) | (projection.q > limits["q_3sd"])
    return (projection.t2 > limits["t2_f"]) | (projection.q > limits["q_jm"])


def evaluate_components(
    model: PCAModel, blocks: dict[str, pd.DataFrame], n_components: int
) -> dict:
    """Project both blocks onto the calibration model with n components."""
    projections = {
        block: project(model, data, n_components) for block, data in blocks.items()
    }
    verify_reference_projection(model, projections["calibration"])
    limits = control_limits(model, projections["calibration"], n_components)
    alarms = {
        block: int(alarm_flags(projection, limits).sum())
        for block, projection in projections.items()
    }
    return {"projections": projections, "limits": limits, "alarms": alarms}


def summarise_alarms(
    results: dict[int, dict], decisions: dict[int, str], n_final: int
) -> pd.DataFrame:
    """Alarm counts and proportions per number of components and block."""
    rows = []
    for n, result in results.items():
        limits = result["limits"]
        for block, projection in result["projections"].items():
            t2_alarms = projection.t2 > limits["t2_f"]
            q_alarms = projection.q > limits["q_jm"]
            either = alarm_flags(projection, limits)
            either_3sd = alarm_flags(projection, limits, three_sigma=True)
            rows.append(
                {
                    "n_components": n,
                    "decision": decisions[n],
                    "selected": n == n_final,
                    "block": block,
                    "observations": len(projection.t2),
                    "t2_limit_f": limits["t2_f"],
                    "q_limit_jackson_mudholkar": limits["q_jm"],
                    "t2_alarms": int(t2_alarms.sum()),
                    "q_alarms": int(q_alarms.sum()),
                    "t2_or_q_alarms": int(either.sum()),
                    "t2_or_q_alarm_percent": 100 * either.mean(),
                    "t2_limit_3sd": limits["t2_3sd"],
                    "q_limit_3sd": limits["q_3sd"],
                    "t2_alarms_3sd": int((projection.t2 > limits["t2_3sd"]).sum()),
                    "q_alarms_3sd": int((projection.q > limits["q_3sd"]).sum()),
                    "t2_or_q_alarms_3sd": int(either_3sd.sum()),
                    "t2_or_q_alarm_percent_3sd": 100 * either_3sd.mean(),
                }
            )
    return pd.DataFrame(rows)


def save_calibration_eigenvalues(eigenvalues: np.ndarray) -> None:
    """Save the calibration eigenvalues used to choose the initial components."""
    explained_ratio = eigenvalues / eigenvalues.sum()
    pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(eigenvalues) + 1)],
            "eigenvalue": eigenvalues,
            "explained_variance_percent": 100 * explained_ratio,
            "cumulative_variance_percent": 100 * np.cumsum(explained_ratio),
            "kaiser_eigenvalue_above_1": eigenvalues > 1,
        }
    ).to_csv(OUTPUT_DIR / "phase1_calibration_eigenvalues.csv", index=False)


def create_scree_plot(
    eigenvalues: np.ndarray, n_kaiser: int, n_elbow: int, n_initial: int
) -> None:
    """Scree plot of the calibration model with the Kaiser line and n0."""
    components = np.arange(1, len(eigenvalues) + 1)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(components, eigenvalues, color="tab:blue", alpha=0.75, label="Eigenvalue")
    axis.plot(components, eigenvalues, color="tab:blue", marker="o", markersize=3)
    axis.axhline(1, color="tab:red", linestyle="--", linewidth=1,
                 label=f"Kaiser criterion (eigenvalue = 1): {n_kaiser} PCs")
    axis.scatter(n_elbow, eigenvalues[n_elbow - 1], color="tab:orange", s=60,
                 zorder=3, label=f"Scree-plot elbow: PC{n_elbow}")
    axis.axvline(n_initial + 0.5, color="0.4", linestyle=":",
                 label=f"Initial number of components n0 = {n_initial}")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Eigenvalue")
    axis.set_xticks(components)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Scree plot of the No.2WT calibration block (observations 1-1099)")
    fig.savefig(OUTPUT_DIR / "phase1_scree_plot.png", dpi=160)
    plt.close(fig)


def create_control_charts(
    blocks: dict[str, tuple[pd.Index, Projection]],
    limits: dict[str, float],
    n_components: int,
    title: str,
    path,
    split_observation: float | None = None,
) -> None:
    """
    T2 and Q charts in observation order with theoretical and 3 SD limits.

    The y-axes are logarithmic because both statistics are positive and
    strongly skewed.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, statistic, label, theoretical, theoretical_name, three_sd in zip(
        axes,
        ["t2", "q"],
        ["Hotelling's T2", "Q (SPEx)"],
        [limits["t2_f"], limits["q_jm"]],
        ["F-distribution", "Jackson-Mudholkar"],
        [limits["t2_3sd"], limits["q_3sd"]],
    ):
        for block, (observations, projection) in blocks.items():
            values = getattr(projection, statistic)
            axis.plot(observations, values, linewidth=0.7,
                      color=BLOCK_COLORS.get(block, "tab:blue"), label=block)
        axis.axhline(theoretical, color="tab:red", linewidth=1.2,
                     label=f"99% limit, {theoretical_name} ({theoretical:.2f})")
        axis.axhline(three_sd, color="0.3", linestyle="--", linewidth=1,
                     label=f"Mean + 3 SD limit ({three_sd:.2f})")
        if split_observation is not None:
            axis.axvline(split_observation, color="0.5", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Observation order")
    fig.suptitle(f"{title}: T2 and Q control charts")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
