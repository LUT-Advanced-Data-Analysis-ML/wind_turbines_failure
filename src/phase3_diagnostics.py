"""Phase 3 of the modelling plan: sensor diagnostics for No.14WT and No.39WT.

Follows steps 11-14 of the PCA-based fault detection workflow (report 2,
Figure 5): select the out-of-control observations and the fault period of
each faulty turbine, compute T2 and SPEx contributions of the out-of-control
observations, draw time-coloured biplots against the healthy turbine, check
the alarm proportions, and rank the variables that are most representative
of the faults.

Note: The final healthy model is loaded from the Phase 1 outputs. The faulty
turbines are projected onto it again because contributions need the scaled
values, scores and residuals; the resulting T2, Q and alarms are verified
against the Phase 2 outputs.
"""

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from pca_monitoring import (
    PCAModel,
    Projection,
    project,
    spe_contributions,
    t2_contributions,
)
from phase1_healthy_model import alarm_flags
from phase2_fault_detection import MONITORING_STATISTICS_FILE, load_final_model
from pretreatment import (
    FAULTY_TURBINES,
    HEALTHY_TURBINE,
    OUTPUT_DIR,
    load_aligned_data,
    load_pca_data,
)

# Method A: variable 13 stays near 88.99 until a transition observation and
# near 0.0104 afterwards (pretreatment investigation, 24 September 2026).
VARIABLE_13_TRANSITION = {"No.14WT": 358, "No.39WT": 470}
# Method B: an episode starts at the first of three consecutive alarms and
# ends at the last alarm before 30 alarm-free observations (about 5 minutes).
MIN_CONSECUTIVE_ALARMS = 3
ALARM_FREE_GAP = 30
# Variables 12 and 15 are constant in the healthy turbine and are not part
# of the PCA model; departures from their healthy values are checked here.
HEALTHY_CONSTANT_VARIABLES = [12, 15]
TOP_VARIABLES = 5
BIPLOT_PAIRS = [(0, 1), (0, 2), (1, 2)]


def main() -> None:
    pca_x, variables = load_pca_data()
    model, n_components, limits = load_final_model()
    phase2_statistics = pd.read_csv(MONITORING_STATISTICS_FILE)

    projections = {
        name: project(model, pca_x[name], n_components)
        for name in [HEALTHY_TURBINE, *FAULTY_TURBINES]
    }
    alarms = {}
    for name in FAULTY_TURBINES:
        saved = phase2_statistics[phase2_statistics["turbine"] == name]
        alarms[name] = alarm_flags(projections[name], limits)
        assert np.allclose(projections[name].t2, saved["t2"], rtol=1e-12)
        assert np.allclose(projections[name].q, saved["q"], rtol=1e-12)
        assert np.array_equal(alarms[name], saved["t2_or_q_alarm"].to_numpy())

    # ============================ Step 11: out-of-control observations and fault periods ===========================
    periods = {}
    episodes = {}
    comparison = []
    for name in FAULTY_TURBINES:
        observations = np.arange(1, len(alarms[name]) + 1)
        transition = VARIABLE_13_TRANSITION[name]

        # Confirm the variable 13 levels on both sides of the transition.
        variable_13 = pca_x[name][13].to_numpy()
        assert np.median(variable_13[: transition - 1]) > 80
        assert variable_13[transition:].max() < 1

        episodes[name] = find_alarm_episodes(alarms[name])
        in_episode = np.zeros(len(observations), dtype=bool)
        for start, end in episodes[name]:
            in_episode[start - 1 : end] = True

        periods[name] = {
            "A: fault period (before variable 13 transition)": observations < transition,
            "A: transition observation": observations == transition,
            "A: after transition": observations > transition,
            "B: alarm episodes": in_episode,
            "B: outside alarm episodes": ~in_episode,
        }

        comparison.append(
            {
                "turbine": name,
                "method": "A: variable 13 transition",
                "fault_start": 1,
                "fault_end": transition - 1,
                "observations": transition - 1,
                "alarm_percent": 100 * alarms[name][: transition - 1].mean(),
            }
        )
        for start, end in episodes[name]:
            comparison.append(
                {
                    "turbine": name,
                    "method": "B: alarm episode",
                    "fault_start": start,
                    "fault_end": end,
                    "observations": end - start + 1,
                    "alarm_percent": 100 * alarms[name][start - 1 : end].mean(),
                }
            )
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUTPUT_DIR / "phase3_fault_periods.csv", index=False)

    print("\n========================================================")
    print("\nStep 11: fault periods")
    print(
        f"Method B: start at {MIN_CONSECUTIVE_ALARMS} consecutive alarms, end at the "
        f"last alarm before {ALARM_FREE_GAP} alarm-free observations"
    )
    for row in comparison.itertuples():
        print(
            f"{row.turbine}, {row.method}: observations {row.fault_start}-"
            f"{row.fault_end} ({row.observations} observations, "
            f"{row.alarm_percent:.1f}% alarms)"
        )

    # ============================ Steps 12a-b: T2 and SPEx contributions ===========================
    # Each statistic is explained by its own out-of-control observations: T2
    # contributions are averaged over the T2 alarms and SPEx contributions
    # over the Q alarms. The scaled values already use the healthy statistics,
    # so the contributions are compared without further normalisation.
    contribution_rows = []
    for name in FAULTY_TURBINES:
        projection = projections[name]
        contributions = {
            "T2": (t2_contributions(model, projection), projection.t2 > limits["t2_f"]),
            "SPEx": (spe_contributions(projection), projection.q > limits["q_jm"]),
        }
        for statistic, (values, statistic_alarms) in contributions.items():
            for index, variable in enumerate(variables):
                contribution_rows.append(
                    {
                        "turbine": name,
                        "statistic": statistic,
                        "alarm_observations": int(statistic_alarms.sum()),
                        "variable": variable,
                        "mean_contribution": values[statistic_alarms, index].mean(),
                    }
                )
        create_contribution_heatmap(contributions, name, variables)
    contributions_table = pd.DataFrame(contribution_rows)
    contributions_table.to_csv(OUTPUT_DIR / "phase3_contributions.csv", index=False)
    for name in FAULTY_TURBINES:
        create_contribution_plot(contributions_table, name, variables)

    # ============================ Step 12c: time-coloured biplots ===========================
    for name in FAULTY_TURBINES:
        create_time_coloured_biplots(
            model, projections[HEALTHY_TURBINE], projections[name], name, variables
        )

    # ============================ Extra check: healthy constant variables 12 and 15 ===========================
    aligned = load_aligned_data()
    healthy_values = {
        variable: aligned[HEALTHY_TURBINE][variable].iloc[0]
        for variable in HEALTHY_CONSTANT_VARIABLES
    }
    constant_rows = []
    for name in FAULTY_TURBINES:
        for period, mask in periods[name].items():
            if not mask.any():
                continue
            for variable, healthy_value in healthy_values.items():
                values = aligned[name][variable].to_numpy()[mask]
                constant_rows.append(
                    {
                        "turbine": name,
                        "period": period,
                        "variable": variable,
                        "healthy_value": healthy_value,
                        "observations": int(mask.sum()),
                        "observations_different_from_healthy": int(np.sum(values != healthy_value)),
                        "unique_values": len(np.unique(values)),
                        "minimum": values.min(),
                        "maximum": values.max(),
                    }
                )
    constant_check = pd.DataFrame(constant_rows)
    constant_check.to_csv(OUTPUT_DIR / "phase3_constant_variable_check.csv", index=False)

    print("\n========================================================")
    print("\nExtra check: variables 12 and 15 (constant in No.2WT)")
    for row in constant_check[~constant_check["period"].str.startswith("B:")].itertuples():
        print(
            f"{row.turbine}, {row.period}, variable {row.variable}: "
            f"{row.observations_different_from_healthy} / {row.observations} "
            f"differ from healthy value {row.healthy_value:g} "
            f"(range {row.minimum:g}-{row.maximum:g})"
        )

    # ============================ Step 13: alarm proportions ===========================
    proportion_rows = []
    for name in FAULTY_TURBINES:
        projection = projections[name]
        flags = {
            ("T2", "theoretical"): projection.t2 > limits["t2_f"],
            ("Q", "theoretical"): projection.q > limits["q_jm"],
            ("T2 or Q", "theoretical"): alarm_flags(projection, limits),
            ("T2", "3SD"): projection.t2 > limits["t2_3sd"],
            ("Q", "3SD"): projection.q > limits["q_3sd"],
            ("T2 or Q", "3SD"): alarm_flags(projection, limits, three_sigma=True),
        }
        for period, mask in periods[name].items():
            if not mask.any():
                continue
            for (statistic, limit_type), flag in flags.items():
                proportion_rows.append(
                    {
                        "turbine": name,
                        "period": period,
                        "statistic": statistic,
                        "limit_type": limit_type,
                        "observations": int(mask.sum()),
                        "alarms": int(flag[mask].sum()),
                        "alarm_percent": 100 * flag[mask].mean(),
                    }
                )
    proportions = pd.DataFrame(proportion_rows)
    proportions.to_csv(OUTPUT_DIR / "phase3_alarm_proportions.csv", index=False)

    print("\n========================================================")
    print("\nStep 13: T2 or Q alarm proportions (theoretical 99% limits)")
    selected = proportions[
        (proportions["statistic"] == "T2 or Q") & (proportions["limit_type"] == "theoretical")
    ]
    for row in selected.itertuples():
        print(
            f"{row.turbine}, {row.period}: {row.alarms} / {row.observations} "
            f"({row.alarm_percent:.1f}%)"
        )
    print(
        "Note: No.14WT observations 359-686 and No.39WT observations 471-798 are "
        "a near-identical shared data block, so their after-transition results "
        "are not independent."
    )

    # ============================ Step 14: fault-sensitive sensors ===========================
    # Variables are ranked by the magnitude of their mean contribution over
    # the out-of-control observations; T2 contributions can be negative, so
    # the sign only gives the direction. A variable is reported when it is
    # among the top variables for both statistics in both turbines.
    ranking = contributions_table.copy()
    ranking["absolute_mean_contribution"] = ranking["mean_contribution"].abs()
    ranking["rank"] = ranking.groupby(["turbine", "statistic"])[
        "absolute_mean_contribution"
    ].rank(ascending=False, method="min").astype(int)
    ranking = ranking.sort_values(["turbine", "statistic", "rank"])
    ranking.to_csv(OUTPUT_DIR / "phase3_sensor_ranking.csv", index=False)

    top = ranking[ranking["rank"] <= TOP_VARIABLES]
    appearances = top.groupby("variable").size()
    combinations = len(FAULTY_TURBINES) * 2

    print("\n========================================================")
    print(f"\nStep 14: top {TOP_VARIABLES} variables by mean contribution over the out-of-control observations")
    for (name, statistic), group in top.groupby(["turbine", "statistic"], sort=False):
        values = ", ".join(
            f"{row.variable} ({row.mean_contribution:.3g})" for row in group.itertuples()
        )
        print(f"{name} {statistic} ({group['alarm_observations'].iloc[0]} alarms): {values}")
    print(
        f"\nFault-sensitive variables (top {TOP_VARIABLES} for T2 and SPEx in both turbines): "
        f"{sorted(appearances[appearances == combinations].index.tolist())}"
    )
    print(
        f"Variables in the top {TOP_VARIABLES} of at least {combinations - 1} of "
        f"{combinations} turbine-statistic combinations: "
        f"{sorted(appearances[appearances >= combinations - 1].index.tolist())}"
    )


def find_alarm_episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """
    Fault episodes as (first, last) one-based observation numbers.

    An episode starts at the first of MIN_CONSECUTIVE_ALARMS consecutive
    alarms and ends at the last alarm before ALARM_FREE_GAP consecutive
    alarm-free observations (or at the last alarm of the recording).
    """
    episodes = []
    start = None
    last_alarm = None
    run = 0
    for index, alarm in enumerate(flags):
        if alarm:
            run += 1
            if start is None and run >= MIN_CONSECUTIVE_ALARMS:
                start = index - MIN_CONSECUTIVE_ALARMS + 1
            if start is not None:
                last_alarm = index
        else:
            run = 0
            if start is not None and index - last_alarm >= ALARM_FREE_GAP:
                episodes.append((start + 1, last_alarm + 1))
                start = None
    if start is not None:
        episodes.append((start + 1, last_alarm + 1))
    return episodes


def create_contribution_plot(
    contributions_table: pd.DataFrame, name: str, variables: list[int]
) -> None:
    """
    Mean T2 and SPEx contributions over each statistic's out-of-control observations.

    A symmetric logarithmic axis keeps the very large, the small and the
    negative T2 contributions visible.
    """
    statistics = [("T2", "T2 contribution", "T2"), ("SPEx", "SPEx contribution", "Q")]
    positions = np.arange(len(variables))
    fig, axes = plt.subplots(
        len(statistics), 1, figsize=(14, 9), sharex=True, constrained_layout=True
    )
    for axis, (statistic, label, limit_name) in zip(axes, statistics):
        table = (
            contributions_table[
                (contributions_table["turbine"] == name)
                & (contributions_table["statistic"] == statistic)
            ]
            .set_index("variable")
            .reindex(variables)
        )
        values = table["mean_contribution"].to_numpy()
        colors = ["tab:orange" if value < 0 else "tab:blue" for value in values]
        axis.bar(positions, values, color=colors, edgecolor="black", linewidth=0.3)
        axis.axhline(0, color="0.4", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylabel(f"Mean {label}")
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(
            f"{statistic}: mean over the {table['alarm_observations'].iloc[0]} "
            f"observations above the {limit_name} limit"
        )
    axes[-1].set_xticks(positions, labels=variables, rotation=90, fontsize=8)
    axes[-1].set_xlabel("Variable")

    fig.suptitle(f"{name}: mean variable contributions of the out-of-control observations (symmetric log scale)")
    fig.savefig(OUTPUT_DIR / f"phase3_contributions_{name}.png", dpi=160)
    plt.close(fig)


def create_contribution_heatmap(
    contributions: dict[str, tuple[np.ndarray, np.ndarray]],
    name: str,
    variables: list[int],
) -> None:
    """
    Contributions of every out-of-control observation in observation order.

    Unlike the mean bar charts, each alarm keeps its own column, so the small
    alarms after the transition stay visible next to the fault block.
    Observations without an alarm for that statistic are shown in grey.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    for axis, (statistic, limit_name) in zip(axes, [("T2", "T2"), ("SPEx", "Q")]):
        values, statistic_alarms = contributions[statistic]
        shown = np.where(statistic_alarms[:, None], values, np.nan)
        largest = np.nanmax(np.abs(shown))
        if statistic == "T2":
            # T2 contributions can be negative, so a diverging symmetric log scale is used.
            colormap = matplotlib.colormaps["coolwarm"].copy()
            norm = SymLogNorm(linthresh=1, vmin=-largest, vmax=largest)
        else:
            # SPEx contributions below 0.01 are shown in the lowest colour.
            colormap = matplotlib.colormaps["viridis"].copy()
            norm = LogNorm(vmin=1e-2, vmax=largest)
        colormap.set_bad("0.85")

        image = axis.imshow(
            shown.T,
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            norm=norm,
            extent=(0.5, len(values) + 0.5, len(variables) - 0.5, -0.5),
        )
        axis.set_yticks(np.arange(len(variables)), labels=variables, fontsize=7)
        axis.set_ylabel("Variable")
        axis.set_title(
            f"{statistic} contributions of the {int(statistic_alarms.sum())} observations "
            f"above the {limit_name} limit (grey: no {limit_name} alarm)"
        )
        fig.colorbar(image, ax=axis, label=f"{statistic} contribution")

    axes[-1].set_xlabel("Observation order")
    fig.suptitle(f"{name}: contributions of every out-of-control observation")
    fig.savefig(OUTPUT_DIR / f"phase3_contribution_heatmap_{name}.png", dpi=160)
    plt.close(fig)


def create_time_coloured_biplots(
    model: PCAModel,
    healthy: Projection,
    faulty: Projection,
    name: str,
    variables: list[int],
) -> None:
    """
    Faulty scores coloured by observation order over the healthy No.2WT scores.

    The top row shows every observation. The bottom row zooms in on the
    healthy score range so the after-transition observations and the
    loading directions remain readable.
    """
    explained = 100 * model.eigenvalues / model.eigenvalues.sum()
    healthy_extent = 1.3 * np.percentile(np.abs(healthy.scores), 99.5, axis=0)
    order = np.arange(1, len(faulty.t2) + 1)

    fig, axes = plt.subplots(2, len(BIPLOT_PAIRS), figsize=(17, 11), constrained_layout=True)
    for column, (x_pc, y_pc) in enumerate(BIPLOT_PAIRS):
        for row, zoom in enumerate([False, True]):
            axis = axes[row, column]
            axis.scatter(healthy.scores[:, x_pc], healthy.scores[:, y_pc],
                         color="0.6", s=6, alpha=0.4, label=f"{HEALTHY_TURBINE} (healthy)")
            points = axis.scatter(faulty.scores[:, x_pc], faulty.scores[:, y_pc],
                                  c=order, cmap="viridis", s=9, alpha=0.8, label=name)
            if zoom:
                axis.set_xlim(-healthy_extent[x_pc], healthy_extent[x_pc])
                axis.set_ylim(-healthy_extent[y_pc], healthy_extent[y_pc])

            # One common scale per panel preserves the relative arrow directions.
            x_limits, y_limits = axis.get_xlim(), axis.get_ylim()
            half_range = min(np.ptp(x_limits), np.ptp(y_limits)) / 2
            loading_pair = model.loadings[:, [x_pc, y_pc]]
            arrow_scale = 0.8 * half_range / np.max(np.abs(loading_pair))
            label_indices = set(np.argsort(np.linalg.norm(loading_pair, axis=1))[::-1][:8])
            for index, (variable, loading) in enumerate(zip(variables, loading_pair)):
                end_x, end_y = loading * arrow_scale
                axis.annotate("", xy=(end_x, end_y), xytext=(0, 0),
                              arrowprops={"arrowstyle": "->", "color": "tab:red", "alpha": 0.6})
                if index in label_indices:
                    axis.annotate(str(variable), (end_x, end_y), xytext=(3, 3),
                                  textcoords="offset points", fontsize=7, color="tab:red")
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)

            axis.axhline(0, color="0.5", linewidth=0.8)
            axis.axvline(0, color="0.5", linewidth=0.8)
            axis.set_xlabel(f"PC{x_pc + 1} scores ({explained[x_pc]:.2f}%)")
            axis.set_ylabel(f"PC{y_pc + 1} scores ({explained[y_pc]:.2f}%)")
            view = "zoom on healthy score range" if zoom else "all observations"
            axis.set_title(f"PC{x_pc + 1}-PC{y_pc + 1}, {view}")
            axis.grid(alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(fontsize=8, loc="best")

    fig.colorbar(points, ax=axes, label=f"{name} observation order", shrink=0.8)
    fig.suptitle(f"{name} scores on the final {HEALTHY_TURBINE} model (time-coloured biplots)")
    fig.savefig(OUTPUT_DIR / f"phase3_biplots_{name}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
