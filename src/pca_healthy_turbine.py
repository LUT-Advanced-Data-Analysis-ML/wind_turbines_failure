"""Initial PCA of the healthy wind-turbine SCADA data.

The analysis follows the course workflow: construct X, resolve essential
data issues, autoscale the healthy data, and compute PCA through SVD.
No response variable, fault label, or supervised method is used.

Note: Selected turbines are based on the ininitial inspection of the raw data. The healthy turbine is used to fit the PCA model, 
        and its statistics are used to autoscale the other turbines for later projection.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# File paths and turbine selection are shared with the modelling scripts.
# The healthy turbine is used to fit the PCA model.
from pretreatment import (
    EXPLORATORY_OUTPUT_DIR as OUTPUT_DIR,
    HEALTHY_TURBINE,
    TURBINES,
    interpolate_missing_value,
    load_aligned_data,
    remove_healthy_constant_variables,
)

# These two No.14WT variables illustrate the audit in one focused plot.
PLOTTED_EXTREME_VALUE_VARIABLES = [5, 11]
LABEL_OFFSETS = {
    2: (10, 8), 3: (8, 18), 4: (8, 8), 5: (8, 20),
    6: (8, 7), 7: (8, -5), 8: (8, -17), 10: (8, 7),
    11: (8, -8), 13: (-16, 10), 14: (-16, -7), 20: (8, -29),
    21: (8, -18), 22: (8, 7), 23: (8, -12), 24: (8, -9),
}


def main() -> None:
    # Load the three structurally compatible turbines and retain variables
    # 1-27. This removes variable 28 from the healthy turbine.
    aligned = load_aligned_data()
    missing_values_before = {
        name: int(data.isna().sum().sum())
        for name, data in aligned.items()
    }

    # Preserve the time-series row and linearly interpolate the single
    # missing No.14WT value, then remove the healthy constant variables.
    interpolation = interpolate_missing_value(aligned)
    pca_x, pca_variables, constant_variables = remove_healthy_constant_variables(aligned)

    # Screen every retained variable for unusual values. The IQR limits are
    # descriptive diagnostics, not rules for changing measured observations.
    create_extreme_value_diagnostics(pca_x)

    healthy_x = pca_x[HEALTHY_TURBINE]

    # Fit pretreatment only on the healthy turbine. Any later turbine must use
    # these fixed means and scales; the PCA model must not be refitted.
    # Autoscaling follows the course guidance that variance acts as a weight.
    healthy_mean = healthy_x.mean()
    healthy_scale = healthy_x.std(ddof=1)
    scaled_x = {
        name: (data - healthy_mean) / healthy_scale
        for name, data in pca_x.items()
    }
    healthy_x_scaled = scaled_x[HEALTHY_TURBINE]

    # Compute PCA through economy SVD: scores T = U*S and loadings P = V.
    u, singular_values, vt = np.linalg.svd(healthy_x_scaled.to_numpy(), full_matrices=False)
    scores = u * singular_values
    loadings = vt.T
    explained_variance = singular_values**2 / (len(healthy_x_scaled) - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    cumulative_ratio = np.cumsum(explained_ratio)

    save_numeric_outputs(
        pca_variables,
        healthy_mean,
        healthy_scale,
        scores,
        loadings,
        explained_variance,
        explained_ratio,
        cumulative_ratio,
    )
    save_pretreatment_summary(
        aligned, pca_x, constant_variables, missing_values_before
    )
    create_pretreated_data_plot(scaled_x, pca_variables)
    create_pca_plots(healthy_x, pca_variables, scores, loadings, explained_ratio, cumulative_ratio)

    # Report the main pretreatment and PCA results needed for the next step.
    print("Aligned X matrices before PCA variable removal")
    for name, data in aligned.items():
        print(f"{name}: {data.shape[0]} observations x {data.shape[1]} variables")
    print("---")
    print(
        f"No.14WT variable {interpolation['variable']}, "
        f"observation {interpolation['observation']}: "
        f"linear interpolation = {interpolation['value']:.0f}; "
        f"rows preserved = {interpolation['rows']}"
    )
    print("---")
    print(f"Excluded zero-variance healthy variables: {constant_variables}")
    print("---")
    print(f"Final variables used for every retained turbine: {pca_variables}")
    print("---")
    print("PCA-ready aligned X matrices:")
    for name, data in pca_x.items():
        print(f"{name}: {data.shape[0]} observations x {data.shape[1]} variables")
    print("---")
    print(f"Final healthy PCA X: {healthy_x.shape[0]} observations x {healthy_x.shape[1]} variables")
    print("---")
    print("Pretreatment: mean-centering and unit-variance scaling using No.2WT statistics")
    variable_9 = healthy_x[9]
    print("---")
    print(
        "Variable 9 retained: it is not constant "
        f"({variable_9.nunique()} unique values, raw range "
        f"{variable_9.max() - variable_9.min():.3g}). Its low relative raw "
        "variation is influenced by its large offset; autoscaling gives it "
        "unit variance like the other retained variables."
    )

    print("\n========================================================")
    print("\nExplained variance")
    for component in range(5):
        print(
            f"PC{component + 1}: {100 * explained_ratio[component]:.2f}% "
            f"(cumulative {100 * cumulative_ratio[component]:.2f}%)"
        )
    for threshold in (0.80, 0.90, 0.95):
        count = int(np.searchsorted(cumulative_ratio, threshold) + 1)
        print(f"Components for {int(threshold * 100)}% cumulative variance: {count}")
    print("---")
    print("\nVariables contributing most to the first three PCs")
    for component in range(3):
        order = np.argsort(np.abs(loadings[:, component]))[::-1][:6]
        values = ", ".join(
            f"{pca_variables[index]} ({loadings[index, component]:+.3f})"
            for index in order
        )
        print(f"PC{component + 1}: {values}")

    # Correlations support the interpretation of variables pointing in similar
    # or opposite directions in the loading plots and biplots.
    print("\n========================================================")
    correlation = healthy_x.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    print("\nStrongest positive healthy-variable correlations")
    for (first, second), value in pairs.sort_values(ascending=False).head(6).items():
        print(f"Variables {first} and {second}: r = {value:.3f}")
    print("---")
    print("Strongest negative healthy-variable correlations")
    for (first, second), value in pairs.sort_values().head(6).items():
        print(f"Variables {first} and {second}: r = {value:.3f}")


def save_numeric_outputs(
    variables: list[int],
    means: pd.Series,
    scales: pd.Series,
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_variance: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_ratio: np.ndarray,
) -> None:
    """
    Save PCA results and autoscaling parameters as CSV files.

    Creates compact output tables containing the explained variance of each
    principal component, PCA loadings, scores for the first five principal
    components, and the mean and standard deviation used to autoscale the
    healthy-turbine data. These outputs support PCA interpretation and allow
    new observations to be projected using the same preprocessing parameters.

    Parameters
    ----------
    variables : list[int]
        Identifiers of the variables included in the PCA model.
    means : pd.Series
        Mean value of each variable in the healthy-turbine data, used for
        centering during autoscaling.
    scales : pd.Series
        Standard deviation of each variable in the healthy-turbine data, used
        for scaling during autoscaling.
    scores : np.ndarray
        PCA score matrix containing the projected observations in principal
        component space.
    loadings : np.ndarray
        PCA loading matrix describing the contribution of each original
        variable to each principal component.
    explained_variance : np.ndarray
        Variance explained by each principal component.
    explained_ratio : np.ndarray
        Fraction of the total variance explained by each principal component.
    cumulative_ratio : np.ndarray
        Cumulative fraction of total variance explained by successive
        principal components.
    """
    component_names = [f"PC{i}" for i in range(1, len(variables) + 1)]

    pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": explained_variance,
            "explained_variance_percent": 100 * explained_ratio,
            "cumulative_variance_percent": 100 * cumulative_ratio,
        }
    ).to_csv(OUTPUT_DIR / "pca_healthy_explained_variance.csv", index=False)

    loading_table = pd.DataFrame(loadings, index=variables, columns=component_names)
    loading_table.index.name = "variable"
    loading_table.to_csv(OUTPUT_DIR / "pca_healthy_loadings.csv")

    score_table = pd.DataFrame(scores[:, :5], columns=component_names[:5])
    score_table.index = np.arange(1, len(score_table) + 1)
    score_table.index.name = "observation"
    score_table.to_csv(OUTPUT_DIR / "pca_healthy_scores_first5.csv")

    pd.DataFrame(
        {
            "variable": variables,
            "healthy_mean": means,
            "healthy_std": scales,
            "standard_deviation_ddof": 1,
            "parameters_fitted_on": HEALTHY_TURBINE,
            "parameters_applied_to": ", ".join(TURBINES),
        }
    ).to_csv(OUTPUT_DIR / "pca_healthy_autoscaling_parameters.csv", index=False)


def save_pretreatment_summary(
    aligned: dict[str, pd.DataFrame],
    pca_x: dict[str, pd.DataFrame],
    constant_variables: list[int],
    missing_values_before: dict[str, int],
) -> None:
    """Save the final cleaning and scaling summary for each retained turbine."""
    rows = []
    for name in TURBINES:
        rows.append(
            {
                "turbine": name,
                "observations": len(pca_x[name]),
                "variables_after_alignment": aligned[name].shape[1],
                "final_variables": pca_x[name].shape[1],
                "missing_values_before": missing_values_before[name],
                "interpolated_values": (
                    missing_values_before[name] if name == "No.14WT" else 0
                ),
                "missing_values_after": int(pca_x[name].isna().sum().sum()),
                "removed_zero_variance_variables": ", ".join(
                    map(str, constant_variables)
                ),
                "centering_scaling_reference": HEALTHY_TURBINE,
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "pretreatment_cleaned_data_summary.csv", index=False
    )


def create_pretreated_data_plot(
    scaled_x: dict[str, pd.DataFrame], variables: list[int]
) -> None:
    """Visualize the cleaned and autoscaled data for the retained turbines."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    for axis, (name, data) in zip(axes, scaled_x.items()):
        image = axis.imshow(
            data.to_numpy().T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-4,
            vmax=4,
            interpolation="nearest",
        )
        axis.set_yticks(np.arange(len(variables)), labels=variables, fontsize=8)
        axis.set_ylabel("Variable")
        axis.set_xlabel("Observation order")
        axis.set_title(f"{name}: cleaned and autoscaled data")

    fig.colorbar(
        image,
        ax=axes,
        label="Autoscaled value (colour display limited to -4 to +4)",
        shrink=0.9,
    )
    fig.suptitle("Pretreated wind-turbine data using healthy-turbine scaling")
    fig.savefig(OUTPUT_DIR / "pretreatment_autoscaled_data.png", dpi=160)
    plt.close(fig)


def create_extreme_value_diagnostics(
    data_by_turbine: dict[str, pd.DataFrame]
) -> None:
    """Audit all retained variables and plot two No.14WT examples."""
    summaries = []

    for name, data in data_by_turbine.items():
        for variable in data.columns:
            values = data[variable]
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            outside_limits = (values < lower_limit) | (values > upper_limit)

            summaries.append(
                {
                    "turbine": name,
                    "variable": variable,
                    "lower_iqr_limit": lower_limit,
                    "upper_iqr_limit": upper_limit,
                    "values_outside_iqr_limits": int(outside_limits.sum()),
                    "99th_percentile": values.quantile(0.99),
                    "maximum": values.max(),
                    "maximum_observation": values.idxmax() + 1,
                    "second_largest": values.nlargest(2).iloc[-1],
                }
            )

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "pretreatment_extreme_value_diagnostics.csv", index=False
    )

    data = data_by_turbine["No.14WT"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    for axis, variable in zip(axes, PLOTTED_EXTREME_VALUE_VARIABLES):
        values = data[variable]
        maximum_observation = values.idxmax() + 1

        observation_order = np.arange(1, len(values) + 1)
        axis.plot(observation_order, values, color="tab:blue", linewidth=0.8)
        axis.axhline(
            values.quantile(0.99),
            color="tab:orange",
            linestyle="--",
            label="99th percentile",
        )
        axis.scatter(
            maximum_observation,
            values.max(),
            color="tab:red",
            s=35,
            zorder=3,
            label=f"Maximum (observation {maximum_observation})",
        )
        axis.set_ylabel(f"Variable {variable}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[-1].set_xlabel("Observation order")
    fig.suptitle("No.14WT extreme-value diagnostics (values retained)")
    fig.savefig(OUTPUT_DIR / "pretreatment_extreme_value_diagnostics.png", dpi=160)
    plt.close(fig)


def create_pca_plots(
    healthy_x: pd.DataFrame,
    variables: list[int],
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_ratio: np.ndarray,
) -> None:
    """Create the essential healthy-PCA figures used in the course workflow."""
    components = np.arange(1, len(explained_ratio) + 1)

    # Scree and cumulative explained-variance plot.
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(components, 100 * explained_ratio, color="tab:blue", alpha=0.75,
             label="Individual variance")
    axis.plot(components, 100 * cumulative_ratio, color="tab:orange", marker="o",
              markersize=3, label="Cumulative variance")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance (%)")
    axis.set_xticks(components)
    axis.set_ylim(0, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Healthy-turbine PCA explained variance")
    fig.savefig(OUTPUT_DIR / "pca_healthy_explained_variance.png", dpi=160)
    plt.close(fig)

    # Score plots for the first three PCs show the main healthy operating
    # trajectory. Colour represents observation order, not a response class.
    score_pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (x_pc, y_pc) in zip(axes, score_pairs):
        points = axis.scatter(
            scores[:, x_pc],
            scores[:, y_pc],
            c=np.arange(1, len(scores) + 1),
            cmap="viridis",
            s=11,
            alpha=0.6,
        )
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set_xlabel(f"PC{x_pc + 1} ({100 * explained_ratio[x_pc]:.2f}%)")
        axis.set_ylabel(f"PC{y_pc + 1} ({100 * explained_ratio[y_pc]:.2f}%)")
        axis.set_title(f"PC{x_pc + 1}-PC{y_pc + 1}")
    fig.colorbar(points, ax=axes, label="Observation order", shrink=0.9)
    fig.suptitle("Healthy-turbine PCA scores")
    fig.savefig(OUTPUT_DIR / "pca_healthy_scores.png", dpi=160)
    plt.close(fig)

    # Loading positions indicate which variables vary together or oppositely.
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    axis.scatter(loadings[:, 0], loadings[:, 1], color="tab:red", s=35)
    for variable, x_value, y_value in zip(variables, loadings[:, 0], loadings[:, 1]):
        offset = LABEL_OFFSETS.get(variable, (4, 4))
        axis.annotate(str(variable), (x_value, y_value), xytext=offset,
                      textcoords="offset points", fontsize=9)
    axis.axhline(0, color="0.6", linewidth=0.8)
    axis.axvline(0, color="0.6", linewidth=0.8)
    axis.set_xlabel(f"PC1 loading ({100 * explained_ratio[0]:.2f}%)")
    axis.set_ylabel(f"PC2 loading ({100 * explained_ratio[1]:.2f}%)")
    axis.set_title("Healthy-turbine PCA loadings")
    axis.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "pca_healthy_loadings.png", dpi=160)
    plt.close(fig)

    # Signed loading bars identify the variables contributing most strongly to
    # each of the first three components.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for component, axis in enumerate(axes):
        order = np.argsort(np.abs(loadings[:, component]))[::-1][:8]
        order = order[np.argsort(loadings[order, component])]
        values = loadings[order, component]
        labels = [str(variables[index]) for index in order]
        colors = ["tab:orange" if value < 0 else "tab:blue" for value in values]
        axis.barh(labels, values, color=colors, alpha=0.8)
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_xlabel("Loading")
        axis.set_ylabel("Variable")
        axis.set_title(f"PC{component + 1} ({100 * explained_ratio[component]:.2f}%)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Largest healthy-turbine loading contributions")
    fig.savefig(OUTPUT_DIR / "pca_healthy_loading_contributions.png", dpi=160)
    plt.close(fig)

    # Biplots combine scores with loading directions. One common scale factor
    # is used for both arrow coordinates so their relative angles are preserved.
    # Correlations are interpreted primarily from the separate loading plot.
    biplot_pairs = [(0, 1), (0, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, (x_pc, y_pc) in zip(axes, biplot_pairs):
        axis.scatter(scores[:, x_pc], scores[:, y_pc], color="tab:blue", s=10, alpha=0.2)
        score_extent = np.percentile(np.abs(scores[:, [x_pc, y_pc]]), 98, axis=0)
        loading_pair = loadings[:, [x_pc, y_pc]]
        component_limits = score_extent / np.max(np.abs(loading_pair), axis=0)
        arrow_scale = 0.78 * component_limits.min()
        if y_pc == 1:
            label_indices = set(
                np.argsort(np.linalg.norm(loading_pair, axis=1))[::-1][:12]
            )
        else:
            # In PC1-PC3, label only the strongest PC3 directions. Many PC1
            # loadings are almost equal and their labels would overlap.
            label_indices = set(np.argsort(np.abs(loadings[:, y_pc]))[::-1][:4])
        for index, (variable, loading) in enumerate(zip(variables, loading_pair)):
            end_x = loading[0] * arrow_scale
            end_y = loading[1] * arrow_scale
            axis.arrow(0, 0, end_x, end_y, color="tab:red", alpha=0.55,
                       width=0.01, head_width=0.15, length_includes_head=True)
            if index in label_indices:
                offset = LABEL_OFFSETS.get(variable, (4, 4))
                axis.annotate(str(variable), (end_x, end_y), xytext=offset,
                              textcoords="offset points", fontsize=8, color="tab:red")
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set_xlabel(f"PC{x_pc + 1} scores ({100 * explained_ratio[x_pc]:.2f}%)")
        axis.set_ylabel(f"PC{y_pc + 1} scores ({100 * explained_ratio[y_pc]:.2f}%)")
        axis.set_title(f"PC{x_pc + 1}-PC{y_pc + 1}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle("Healthy-turbine PCA biplots (loading arrows uniformly scaled)")
    fig.savefig(OUTPUT_DIR / "pca_healthy_biplot.png", dpi=160)
    plt.close(fig)

    # The correlation heatmap supports interpretation of loading directions.
    correlation = healthy_x.corr()
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(variables)), labels=variables, rotation=90)
    axis.set_yticks(np.arange(len(variables)), labels=variables)
    axis.set_xlabel("Variable")
    axis.set_ylabel("Variable")
    axis.set_title("Healthy-turbine variable correlations")
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    fig.savefig(OUTPUT_DIR / "pca_healthy_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # A separate TA-style overview pairs biplots with signed loading bars.
    # Existing PCA figures above are retained unchanged.
    create_biplot_loading_overview(
        variables, scores, loadings, explained_ratio
    )


def create_biplot_loading_overview(
    variables: list[int],
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_ratio: np.ndarray,
) -> None:
    """Pair score/loading biplots with loading bars for PC1-PC3."""
    biplot_pairs = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # Top row: three score plots with loading directions.
    for axis, (x_pc, y_pc) in zip(axes[0], biplot_pairs):
        axis.scatter(
            scores[:, x_pc], scores[:, y_pc],
            color="tab:red", s=5, alpha=0.25
        )
        loading_pair = loadings[:, [x_pc, y_pc]]
        score_extent = np.percentile(np.abs(scores[:, [x_pc, y_pc]]), 98, axis=0)
        component_limits = score_extent / np.max(np.abs(loading_pair), axis=0)
        arrow_scale = 0.72 * component_limits.min()
        label_indices = set(
            np.argsort(np.linalg.norm(loading_pair, axis=1))[::-1][:10]
        )

        for index, (variable, loading) in enumerate(zip(variables, loading_pair)):
            end_x = loading[0] * arrow_scale
            end_y = loading[1] * arrow_scale
            axis.arrow(
                0, 0, end_x, end_y,
                color="tab:blue", alpha=0.65, width=0.008,
                head_width=0.12, length_includes_head=True,
            )
            if index in label_indices:
                axis.annotate(
                    str(variable), (end_x, end_y), xytext=(3, 3),
                    textcoords="offset points", fontsize=7, color="tab:blue",
                )

        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_xlabel(
            f"PC{x_pc + 1} scores ({100 * explained_ratio[x_pc]:.2f}%)"
        )
        axis.set_ylabel(
            f"PC{y_pc + 1} scores ({100 * explained_ratio[y_pc]:.2f}%)"
        )
        axis.set_title(f"Biplot: PC{x_pc + 1}-PC{y_pc + 1}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    # Bottom row: signed loadings for the first three components.
    positions = np.arange(len(variables))
    for component, axis in enumerate(axes[1]):
        axis.bar(
            positions, loadings[:, component],
            color="tab:blue", edgecolor="black", linewidth=0.4,
        )
        axis.axhline(0, color="0.4", linewidth=0.8)
        axis.set_xticks(positions, labels=variables, rotation=90, fontsize=7)
        axis.set_xlabel("Variable")
        axis.set_ylabel("Loading")
        axis.set_title(
            f"Loadings of PC{component + 1} "
            f"({100 * explained_ratio[component]:.2f}%)"
        )
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Healthy-turbine PCA biplots and signed loadings")
    fig.savefig(OUTPUT_DIR / "pca_healthy_biplot_loading_overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
