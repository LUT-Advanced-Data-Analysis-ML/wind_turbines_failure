"""Phase 4: kernel settings chosen by blocked cross-validation of No.2WT.

Every combination of kernel family, shared width and number of kernel
components is scored by its false-alarm rate (FAR) on held-out healthy
observations, with the theoretical 99% limits (F-distribution T2 limit and
scaled chi-square Q limit). Settings are ranked by how close their combined
T2-or-Q FAR is to a 1% design target; two marginal 99% limits do not guarantee
a 1% combined rate. Among near-ties, the fewest components and then the
narrowest width are preferred. The faulty turbines are not used.

Note: Phase 4 saves the CV results and summary as well as the selected settings;
Phase 5 refits the final kernel model from the selected settings.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from kernel_monitoring import (
    KERNEL_FAMILIES,
    fit_kernel_model,
    kernel_limits,
    load_kernel_settings,
    project_kernel,
    save_kernel_settings,
    verify_partial_scores,
)
from pca_monitoring import CONFIDENCE, verify_reference_projection
from pretreatment import HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data
from time_series_cv import (
    N_FOLDS,
    OFFSETS,
    cross_validate_far,
    kernel_round,
    summarise_far,
)

SELECTED_KERNEL_FILE = OUTPUT_DIR / "phase4_selected_kernel.csv"
GRID_FAMILIES = [family for family in KERNEL_FAMILIES if family != "linear"]
# Widths are in healthy standard deviations, because the sensors are autoscaled.
WIDTHS = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 50]
N_COMPONENTS_LIST = [4, 6, 8, 10]
LIMIT_TYPE = "theoretical"
# Use 1% as a heuristic target for the combined alarm rate. Settings whose
# distance from that target is within 0.25 percentage points of the best
# distance are near-ties. This tolerance is roughly the binomial standard
# error for 1570 independent observations at 1%, but temporal dependence
# means it is not a CV uncertainty bound.
TARGET_FAR_PERCENT = 100 * (1 - CONFIDENCE)
TIE_TOLERANCE_PERCENT = 0.25
# Sort order of the near-ties (the first row is selected): fewest
# components, narrowest width, closest to the target, then the fixed family
# order in FAMILY_ORDER.
TIE_BREAK_COLUMNS = ["n_components_min", "width", "distance_from_target_percent", "family_rank"]
FAMILY_ORDER = ["gaussian", "matern5/2", "cauchy", "matern3/2", "matern1/2"]
# Axis ticks of the tuning map (the FAR axis is linear up to 1% and logarithmic above).
X_TICKS = [0.25, 0.5, 1, 2, 5, 10, 20, 50]
Y_TICKS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]


def main() -> None:
    pca_x, _ = load_pca_data()
    healthy_x = pca_x[HEALTHY_TURBINE]
    rounds = len(OFFSETS) * N_FOLDS

    print("\n========================================================")
    print(
        f"\nCross-validating {len(GRID_FAMILIES)} kernel families x {len(WIDTHS)} widths x "
        f"{len(N_COMPONENTS_LIST)} component numbers on {HEALTHY_TURBINE} "
        f"({rounds} rounds each, {LIMIT_TYPE} limits)"
    )
    results = cross_validate_kernels(healthy_x)
    summary = select_setting(summarise_far(results))
    winner = summary[summary["selected"]].iloc[0]

    results.to_csv(OUTPUT_DIR / "phase4_cv_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "phase4_cv_summary.csv", index=False)
    create_tuning_map(summary)

    # ============================ Refit the selected setting on all healthy observations.
    family = winner["family"]
    width = float(winner["width"])
    n_components = int(winner["n_components_min"])
    model = fit_kernel_model(healthy_x, family, width)
    projection = project_kernel(model, healthy_x, n_components)
    verify_reference_projection(model, projection)
    verify_partial_scores(model, projection)
    limits = kernel_limits(model, projection, LIMIT_TYPE)
    save_kernel_settings(model, n_components, LIMIT_TYPE, limits, SELECTED_KERNEL_FILE)
    assert load_kernel_settings(SELECTED_KERNEL_FILE)["limits"] == limits

    alarms = (projection.t2 > limits["t2"]) | (projection.q > limits["q"])
    explained_percent = 100 * model.eigenvalues[:n_components].sum() / model.eigenvalues.sum()

    # ============================ Report the selection.
    print("\n========================================================")
    print(
        f"\nSelection rule: settings whose mean FAR is within {TIE_TOLERANCE_PERCENT:g} "
        f"percentage points of the best distance from {TARGET_FAR_PERCENT:g}% are near-ties; "
        "among them the fewest components, then the narrowest width, then the FAR "
        "closest to the target, then the fixed family order in FAMILY_ORDER"
    )
    ties = summary[summary["near_tie"]].sort_values(TIE_BREAK_COLUMNS)
    print(f"\nNear-ties ({len(ties)} settings), in tie-break order:")
    for row in ties.itertuples():
        print(
            f"  {row.family:10s} width {row.width:5g}, {row.n_components_min:2d} components: "
            f"FAR {row.far_percent_mean:5.2f} +/- {row.far_percent_se:.2f}%"
        )
    if width in (min(WIDTHS), max(WIDTHS)):
        print(f"\nWarning: the selected width {width:g} is at the edge of the width grid.")

    print(
        f"\nSelected kernel: {family}, width {width:g}, {n_components} components, "
        f"{LIMIT_TYPE} limits; CV FAR {winner['far_percent_mean']:.2f} +/- "
        f"{winner['far_percent_se']:.2f}%"
    )
    print(
        f"Refitted on all {model.n_observations} {HEALTHY_TURBINE} observations: "
        f"{explained_percent:.2f}% of the kernel variance in {n_components} components; "
        f"limits T2 = {limits['t2']:.3f}, Q = {limits['q']:.4f}"
    )
    print(
        f"{HEALTHY_TURBINE} in-sample observations above the T2 or Q limit: {alarms.sum()} "
        f"({100 * alarms.mean():.2f}%; in-sample, not validation)"
    )
    print(f"Saved settings: {SELECTED_KERNEL_FILE.name}")


def cross_validate_kernels(healthy_x: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-validate every grid family and width; only the theoretical-limit rows are kept.

    One kernel fit per round serves all numbers of components.
    """
    results = []
    for family in GRID_FAMILIES:
        for width in WIDTHS:
            result = cross_validate_far(
                healthy_x,
                kernel_round,
                family=family,
                width=float(width),
                n_components_list=N_COMPONENTS_LIST,
            )
            result = result[result["limit_type"] == LIMIT_TYPE]
            results.append(result)
            far = result.groupby("n_components")["far_percent"].mean()
            print(
                f"  {family:10s} width {width:5g}: mean FAR {far.min():6.2f}-{far.max():6.2f}% "
                "over the numbers of components",
                flush=True,
            )
    return pd.concat(results, ignore_index=True)


def select_setting(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Mark the near-ties and the selected setting in the CV summary.

    Distance = |mean FAR - target|. Settings within the tie tolerance of the
    smallest distance are near-ties; the selected one has the fewest
    components, then the narrowest width, then the smallest distance, then
    the fixed family order in FAMILY_ORDER.
    """
    table = summary.copy()
    table["distance_from_target_percent"] = (table["far_percent_mean"] - TARGET_FAR_PERCENT).abs()
    table["family_rank"] = table["family"].map(
        {family: rank for rank, family in enumerate(FAMILY_ORDER)}
    )
    best = table["distance_from_target_percent"].min()
    table["near_tie"] = (
        table["distance_from_target_percent"] <= best + TIE_TOLERANCE_PERCENT + 1e-9
    )
    winner = table[table["near_tie"]].sort_values(TIE_BREAK_COLUMNS).index[0]
    table["selected"] = table.index == winner
    assert table["selected"].sum() == 1
    return table


def create_tuning_map(summary: pd.DataFrame) -> None:
    """
    Mean CV FAR against kernel width for every family and number of components.

    The grey band is the near-tie range and the star the selected setting.
    The y-axis is linear up to 1% and logarithmic above.
    """
    best = summary["distance_from_target_percent"].min()
    band = (
        TARGET_FAR_PERCENT - best - TIE_TOLERANCE_PERCENT,
        TARGET_FAR_PERCENT + best + TIE_TOLERANCE_PERCENT,
    )
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(N_COMPONENTS_LIST)))
    y_max = (summary["far_percent_mean"] + summary["far_percent_se"]).max()
    fig, axes = plt.subplots(
        1, len(GRID_FAMILIES), figsize=(19, 5), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, family in zip(axes, GRID_FAMILIES):
        axis.axhspan(*band, color="0.85", label=f"near-ties ({band[0]:.2f}-{band[1]:.2f}%)")
        panel = summary[summary["family"] == family]
        for color, n_components in zip(colors, N_COMPONENTS_LIST):
            line = panel[panel["n_components_min"] == n_components].sort_values("width")
            axis.errorbar(line["width"], line["far_percent_mean"], yerr=line["far_percent_se"],
                          color=color, marker="o", markersize=3, linewidth=1, capsize=2,
                          label=f"{n_components} components")
        selected = panel[panel["selected"]]
        axis.scatter(selected["width"], selected["far_percent_mean"], marker="*", s=250,
                     color="tab:red", edgecolor="black", zorder=5, label="selected")
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1)
        axis.set_xticks(X_TICKS)
        axis.set_yticks([tick for tick in Y_TICKS if tick <= y_max])
        axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.minorticks_off()
        axis.grid(alpha=0.25)
        axis.set_title(family, fontsize=10)
        axis.set_xlabel("Kernel width (healthy SD)")
    axes[0].set_ylabel("Mean CV false-alarm rate, T2 or Q (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(
        f"Blocked CV on {HEALTHY_TURBINE}, {LIMIT_TYPE} limits: false-alarm rate against "
        "kernel width (error bars: standard error over the rounds)"
    )
    fig.savefig(OUTPUT_DIR / "phase4_tuning_map.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
