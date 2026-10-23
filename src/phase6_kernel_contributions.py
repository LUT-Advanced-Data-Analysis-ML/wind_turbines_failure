"""Phase 6: sensor contributions of the kernel and linear models.

For every observation of the faulty turbines, T2 and Q are split into one
share per sensor, for the kernel model selected in Phase 4 and the Phase 1
linear model. T2 shares come from the partial scores (the per-sensor pieces
of the scores); Q shares come from the split of the residual into one block
per sensor. The shares are averaged over the fault period before the
variable 13 transition, and the sensor rankings of the two models are compared.

Note: Shares (each observation adds up to 100%) are used instead of raw
contributions, so that both models are on the same scale and a few extreme
observations do not dominate the averages. The Phase 3 contributions are
not changed.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from kernel_monitoring import (
    fit_kernel_model,
    load_kernel_settings,
    partial_q_contributions,
    partial_score_contributions,
    project_kernel,
    verify_linear_kernel,
    verify_partial_scores,
)
from pca_monitoring import PCAModel, Projection, project, spe_contributions
from phase2_fault_detection import load_final_model
from phase3_diagnostics import TOP_VARIABLES, VARIABLE_13_TRANSITION
from phase4_kernel_tuning import SELECTED_KERNEL_FILE
from pretreatment import FAULTY_TURBINES, HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data

MODELS = ["kernel", "linear"]
STATISTICS = ["T2", "Q"]
PERIOD = "A: fault period (before variable 13 transition)"
# Shares below this percentage are shown in the lowest colour of the heatmaps.
HEATMAP_MIN_PERCENT = 0.1


def main() -> None:
    pca_x, variables = load_pca_data()
    healthy_x = pca_x[HEALTHY_TURBINE]

    settings = load_kernel_settings(SELECTED_KERNEL_FILE)
    kernel_model = fit_kernel_model(healthy_x, settings["family"], settings["width"])
    linear_model, linear_components, _ = load_final_model()

    # ============================ Shares of every observation of the faulty turbines.
    shares = {}
    for name in FAULTY_TURBINES:
        kernel_projection = project_kernel(kernel_model, pca_x[name], settings["n_components"])
        verify_partial_scores(kernel_model, kernel_projection)
        linear_projection = project(linear_model, pca_x[name], linear_components)
        # The linear kernel reproduces the linear partial scores and the
        # linear Q contributions e_l^2 exactly.
        verify_linear_kernel(healthy_x, pca_x[name], linear_components)

        raw = {
            ("kernel", "T2"): partial_score_contributions(kernel_model, kernel_projection, normalise=False),
            ("kernel", "Q"): partial_q_contributions(kernel_model, kernel_projection),
            ("linear", "T2"): linear_t2_contributions(linear_model, linear_projection),
            ("linear", "Q"): spe_contributions(linear_projection),
        }
        for key, values in raw.items():
            percent = 100 * values / values.sum(axis=1, keepdims=True)
            assert np.allclose(percent.sum(axis=1), 100)
            shares[(name, *key)] = percent

    # ============================ Average over the fault period and rank the sensors.
    contribution_rows = []
    for name in FAULTY_TURBINES:
        transition = VARIABLE_13_TRANSITION[name]
        variable_13 = pca_x[name][13].to_numpy()
        assert np.median(variable_13[: transition - 1]) > 80
        assert variable_13[transition:].max() < 1
        fault_period = np.arange(1, len(pca_x[name]) + 1) < transition

        for model_name in MODELS:
            for statistic in STATISTICS:
                mean_shares = shares[(name, model_name, statistic)][fault_period].mean(axis=0)
                ranks = stats.rankdata(-mean_shares, method="min").astype(int)
                for variable, share, rank in zip(variables, mean_shares, ranks):
                    contribution_rows.append(
                        {
                            "turbine": name,
                            "model": model_name,
                            "statistic": statistic,
                            "period": PERIOD,
                            "observations": int(fault_period.sum()),
                            "variable": variable,
                            "mean_share_percent": share,
                            "rank": rank,
                        }
                    )
    contributions = pd.DataFrame(contribution_rows)
    contributions.to_csv(OUTPUT_DIR / "phase6_partial_score_contributions.csv", index=False)

    # ============================ Compare the kernel and linear rankings.
    comparison_rows = []
    for (name, statistic), group in contributions.groupby(["turbine", "statistic"], sort=False):
        table = group.pivot(index="variable", columns="model", values="mean_share_percent")
        top = {
            model_name: table[model_name].sort_values(ascending=False).index[:TOP_VARIABLES].tolist()
            for model_name in MODELS
        }
        comparison_rows.append(
            {
                "turbine": name,
                "statistic": statistic,
                "period": PERIOD,
                "kernel_top": " ".join(map(str, top["kernel"])),
                "linear_top": " ".join(map(str, top["linear"])),
                "top_overlap": len(set(top["kernel"]) & set(top["linear"])),
                "spearman": stats.spearmanr(table["kernel"], table["linear"]).statistic,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "phase6_ranking_comparison.csv", index=False)

    print("\n========================================================")
    print(
        f"\nKernel model: {settings['family']}, width {settings['width']:g}, "
        f"{settings['n_components']} components; linear model: {linear_components} components"
    )
    print("Mean sensor shares over the fault period (before the variable 13 transition)")
    for row in comparison.itertuples():
        print(f"\n{row.turbine}, {row.statistic}:")
        for model_name in MODELS:
            group = contributions[
                (contributions["turbine"] == row.turbine)
                & (contributions["statistic"] == row.statistic)
                & (contributions["model"] == model_name)
            ].sort_values("rank")
            values = ", ".join(
                f"{item.variable} ({item.mean_share_percent:.1f}%)"
                for item in group.head(TOP_VARIABLES).itertuples()
            )
            print(f"  {model_name:6s} top {TOP_VARIABLES}: {values}")
        print(
            f"  shared top-{TOP_VARIABLES} sensors: {row.top_overlap}; "
            f"Spearman rank correlation of all {len(variables)} sensors: {row.spearman:.3f}"
        )

    create_comparison_plot(contributions, variables)
    for name in FAULTY_TURBINES:
        create_share_heatmap(shares, name, variables)


def linear_t2_contributions(model: PCAModel, projection: Projection) -> np.ndarray:
    """
    Linear counterpart of the kernel T2 contributions, M x K.

    The partial score of sensor l on component a is z_l p_la, and
    r_l = sum_a (z_l p_la)^2 / lambda_a.
    """
    n_components = projection.scores.shape[1]
    partial_scores = projection.scaled[:, :, None] * model.loadings[None, :, :n_components]
    return np.sum(partial_scores**2 / model.eigenvalues[:n_components], axis=2)


def create_comparison_plot(contributions: pd.DataFrame, variables: list[int]) -> None:
    """Mean fault-period share of every sensor, kernel and linear side by side."""
    positions = np.arange(len(variables))
    width = 0.4
    colors = {"kernel": "tab:orange", "linear": "tab:blue"}
    fig, axes = plt.subplots(
        len(FAULTY_TURBINES), len(STATISTICS), figsize=(16, 9), sharex=True, constrained_layout=True
    )
    for row, name in enumerate(FAULTY_TURBINES):
        for column, statistic in enumerate(STATISTICS):
            axis = axes[row, column]
            for offset, model_name in zip([-width / 2, width / 2], MODELS):
                values = (
                    contributions[
                        (contributions["turbine"] == name)
                        & (contributions["statistic"] == statistic)
                        & (contributions["model"] == model_name)
                    ]
                    .set_index("variable")
                    .reindex(variables)["mean_share_percent"]
                )
                axis.bar(positions + offset, values, width, color=colors[model_name],
                         edgecolor="black", linewidth=0.3, label=model_name)
            axis.set_title(f"{name}, {statistic}")
            axis.set_ylabel("Mean share (%)")
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(positions, labels=variables, rotation=90, fontsize=8)
            if row == len(FAULTY_TURBINES) - 1:
                axis.set_xlabel("Variable")
            if row == 0 and column == 0:
                axis.legend()
    fig.suptitle("Mean sensor shares of T2 and Q over the fault period (before the variable 13 transition)")
    fig.savefig(OUTPUT_DIR / "phase6_contribution_comparison.png", dpi=160)
    plt.close(fig)


def create_share_heatmap(
    shares: dict[tuple[str, str, str], np.ndarray], name: str, variables: list[int]
) -> None:
    """
    Sensor shares of every observation in observation order, for both models and statistics.

    The colour scale is logarithmic so that small shares stay visible.
    """
    rows = [(model_name, statistic) for model_name in MODELS for statistic in STATISTICS]
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 14), sharex=True, constrained_layout=True)
    for axis, (model_name, statistic) in zip(axes, rows):
        values = np.clip(shares[(name, model_name, statistic)], HEATMAP_MIN_PERCENT, 100)
        image = axis.imshow(
            values.T,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            norm=LogNorm(vmin=HEATMAP_MIN_PERCENT, vmax=100),
            extent=(0.5, len(values) + 0.5, len(variables) - 0.5, -0.5),
        )
        axis.set_yticks(np.arange(len(variables)), labels=variables, fontsize=7)
        axis.set_ylabel("Variable")
        axis.set_title(f"{model_name}, {statistic} shares")
        fig.colorbar(image, ax=axis, label="Share (%)")
    axes[-1].set_xlabel("Observation order")
    fig.suptitle(f"{name}: sensor shares of T2 and Q for every observation")
    fig.savefig(OUTPUT_DIR / f"phase6_contribution_heatmap_{name}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
