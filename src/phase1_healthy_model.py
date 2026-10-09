"""Phase 1: full-healthy No.2WT PCA/MSPC reference model (TA Option 2).

Fit and autoscale on all healthy observations, retain the components with
eigenvalues greater than one, and calculate the existing T2 and Q limits.
The former 70/30 analysis is retained in the historical outputs but is not
used to choose the model: its later block contained new healthy variation.
No.14WT and No.39WT are reserved for evaluation in Phase 2.
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

FINAL_MODEL_PREFIX = OUTPUT_DIR / "phase1_final_model"
KAISER_EIGENVALUE_THRESHOLD = 1.0


def main() -> None:
    pca_x, _ = load_pca_data()
    healthy_x = pca_x[HEALTHY_TURBINE]
    # The means, sample standard deviations and PCA basis are fitted once on
    # all 1570 healthy observations. Kaiser selection uses this same spectrum.
    model = fit_model(healthy_x)
    eigenvalues = model.eigenvalues
    n_components = int(np.sum(eigenvalues > KAISER_EIGENVALUE_THRESHOLD))
    assert 0 < n_components < len(eigenvalues)
    explained_percent = 100 * eigenvalues[:n_components].sum() / eigenvalues.sum()
    save_full_healthy_eigenvalues(eigenvalues)
    create_scree_plot(eigenvalues, n_components)

    projection = project(model, healthy_x, n_components)
    verify_reference_projection(model, projection)
    limits = control_limits(model, projection, n_components)
    save_model(model, FINAL_MODEL_PREFIX)
    pd.DataFrame(
        [
            {
                "n_components": n_components,
                "n_observations": model.n_observations,
                "confidence": CONFIDENCE,
                "t2_limit_f": limits["t2_f"],
                "q_limit_jackson_mudholkar": limits["q_jm"],
                "t2_limit_3sd": limits["t2_3sd"],
                "q_limit_3sd": limits["q_3sd"],
            }
        ]
    ).to_csv(f"{FINAL_MODEL_PREFIX}_limits.csv", index=False)
    alarms = alarm_flags(projection, limits)
    pd.DataFrame(
        [{
            "selection_method": "Kaiser eigenvalue > 1 on full healthy data",
            "eigenvalue_threshold": KAISER_EIGENVALUE_THRESHOLD,
            "n_observations": model.n_observations,
            "n_variables": len(model.variables),
            "n_components": n_components,
            "cumulative_explained_variance_percent": explained_percent,
            "healthy_evaluation": "in-sample, not independent validation",
            "t2_alarms": int((projection.t2 > limits["t2_f"]).sum()),
            "q_alarms": int((projection.q > limits["q_jm"]).sum()),
            "t2_or_q_alarms": int(alarms.sum()),
            "t2_or_q_alarm_percent": 100 * alarms.mean(),
        }]
    ).to_csv(OUTPUT_DIR / "phase1_full_healthy_selection.csv", index=False)
    create_control_charts(
        {"No.2WT (in-sample)": (healthy_x.index + 1, projection)},
        limits,
        n_components,
        f"Full-healthy No.2WT model (in-sample), {n_components} components",
        OUTPUT_DIR / "phase1_final_model_control_charts.png",
    )

    print("\n========================================================")
    print("\nFull-healthy reference model (TA Option 2)")
    print(
        f"Fitted on all {model.n_observations} No.2WT observations; "
        f"Kaiser eigenvalue > {KAISER_EIGENVALUE_THRESHOLD:g} selects "
        f"{n_components} components ({explained_percent:.2f}% variance)"
    )
    print(
        f"99% limits: T2 F = {limits['t2_f']:.2f}, "
        f"Q JM = {limits['q_jm']:.2f}"
    )
    print(
        f"No.2WT in-sample observations above the T2 or Q limit: {alarms.sum()} "
        f"({100 * alarms.mean():.1f}%)"
    )
    print(f"Saved model: {FINAL_MODEL_PREFIX.name}_parameters/eigenvalues/limits.csv")


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


def save_full_healthy_eigenvalues(eigenvalues: np.ndarray) -> None:
    """Save the full-healthy eigenvalues used for Kaiser selection."""
    explained_ratio = eigenvalues / eigenvalues.sum()
    pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(eigenvalues) + 1)],
            "eigenvalue": eigenvalues,
            "explained_variance_percent": 100 * explained_ratio,
            "cumulative_variance_percent": 100 * np.cumsum(explained_ratio),
            "kaiser_eigenvalue_above_1": eigenvalues > 1,
        }
    ).to_csv(OUTPUT_DIR / "phase1_full_healthy_eigenvalues.csv", index=False)


def create_scree_plot(eigenvalues: np.ndarray, n_components: int) -> None:
    """Scree plot of all healthy observations with the Kaiser cutoff."""
    components = np.arange(1, len(eigenvalues) + 1)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(components, eigenvalues, color="tab:blue", alpha=0.75, label="Eigenvalue")
    axis.plot(components, eigenvalues, color="tab:blue", marker="o", markersize=3)
    axis.axhline(1, color="tab:red", linestyle="--", linewidth=1,
                 label=f"Kaiser cutoff: eigenvalue > 1 ({n_components} PCs)")
    axis.axvline(n_components + 0.5, color="0.4", linestyle=":",
                 label=f"Retain PCs 1-{n_components}")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Eigenvalue")
    axis.set_xticks(components)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Scree plot of all 1570 healthy No.2WT observations")
    fig.savefig(OUTPUT_DIR / "phase1_full_healthy_scree_plot.png", dpi=160)
    plt.close(fig)


def create_control_charts(
    blocks: dict[str, tuple[pd.Index, Projection]],
    limits: dict[str, float],
    n_components: int,
    title: str,
    path,
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
                      color="tab:blue", label=block)
        axis.axhline(theoretical, color="tab:red", linewidth=1.2,
                     label=f"99% limit, {theoretical_name} ({theoretical:.2f})")
        axis.axhline(three_sd, color="0.3", linestyle="--", linewidth=1,
                     label=f"Mean + 3 SD limit ({three_sd:.2f})")
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
