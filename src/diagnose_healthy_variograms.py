"""Exploratory observation-lag variograms of the healthy turbine's PCA scores.

This is a descriptive diagnostic. It does not select components, fit control
limits, or alter any of the Phase 1–3 modelling outputs.

It also produces the healthy-behaviour figure of the report appendix (all six
retained PC scores with moving-window variograms) and a block hold-out check
of the healthy turbine, which show why a time-ordered healthy validation
block fails for this recording.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_monitoring import fit_model, project, q_limit_jackson_mudholkar, t2_limit_f
from pretreatment import EXPLORATORY_OUTPUT_DIR, HEALTHY_TURBINE, load_pca_data


MAX_LAG = 60  # Observation steps; timestamps are not available.
SELECTED_LAGS = (1, 10, 30, 60)
FIGURE = EXPLORATORY_OUTPUT_DIR / "healthy_temporal_variogram.png"
SUMMARY = EXPLORATORY_OUTPUT_DIR / "healthy_temporal_variogram_summary.csv"
COLORS = {"early": "#277DA1", "middle": "#F8961E", "late": "#9B5DE5"}

# Report appendix: moving-window variograms of the six retained PC scores.
N_RETAINED = 6
WINDOW = 100  # Observations per window (about 17 min at the nominal 10 s).
WINDOW_STEP = 5
WINDOW_LAG = 20
SCORE_FIGURE = EXPLORATORY_OUTPUT_DIR / "healthy_score_trajectories_variograms.png"
# Block hold-out: each block is hidden in turn, the model is refitted on the
# rest (Kaiser rule) and the hidden block is projected onto it.
N_HOLDOUT_BLOCKS = 5
HOLDOUT_SUMMARY = EXPLORATORY_OUTPUT_DIR / "healthy_block_holdout.csv"


def semivariogram(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean squared score difference / 2 for pairs separated by each lag."""
    return np.array(
        [0.5 * np.mean((values[lag:] - values[:-lag]) ** 2)
         for lag in range(1, max_lag + 1)]
    )


def remove_linear_trend(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove only the within-block linear trend for a comparison curve."""
    order = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(order, values, 1)
    return values - (slope * order + intercept), float(slope)


def main() -> None:
    data, _ = load_pca_data()
    healthy = data[HEALTHY_TURBINE]
    model = fit_model(healthy)  # Same autoscaling and economy SVD as Phase 1.
    scaled = ((healthy - model.mean) / model.std).to_numpy()
    scores = scaled @ model.loadings[:, :3]
    observations = np.arange(1, len(healthy) + 1)

    # PCA signs are arbitrary. Orient PC1 consistently for the trajectory.
    correlation = float(np.corrcoef(observations, scores[:, 0])[0, 1])
    if correlation < 0:
        scores[:, 0] *= -1
        correlation *= -1

    calibration_end = round(0.70 * len(healthy))
    early_end = calibration_end // 2
    blocks = {
        "early": (0, early_end),
        "middle": (early_end, calibration_end),
        "late": (calibration_end, len(healthy)),
    }
    assert min(end - start for start, end in blocks.values()) > MAX_LAG

    rows = []
    curves = {}
    for block, (start, end) in blocks.items():
        for component in (0, 1, 2):
            values = scores[start:end, component]
            detrended, slope = remove_linear_trend(values)
            raw_curve = semivariogram(values, MAX_LAG)
            detrended_curve = semivariogram(detrended, MAX_LAG)
            curves[block, component, "raw"] = raw_curve
            curves[block, component, "detrended"] = detrended_curve
            row = {
                "block": block,
                "component": f"PC{component + 1}",
                "first_observation": start + 1,
                "last_observation": end,
                "observations": end - start,
                "score_mean": float(values.mean()),
                "score_sd": float(values.std(ddof=1)),
                "linear_slope_per_observation": slope,
            }
            for lag in SELECTED_LAGS:
                row[f"raw_gamma_lag_{lag}"] = float(raw_curve[lag - 1])
                row[f"detrended_gamma_lag_{lag}"] = float(detrended_curve[lag - 1])
            rows.append(row)

    pd.DataFrame(rows).to_csv(SUMMARY, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for component, axis in enumerate(axes[0]):
        for block, (start, end) in blocks.items():
            axis.plot(observations[start:end], scores[start:end, component],
                      color=COLORS[block], linewidth=0.8,
                      label=f"{block}: {start + 1}–{end}")
        axis.plot(observations, pd.Series(scores[:, component]).rolling(
                      51, center=True, min_periods=1).mean(),
                  color="black", linewidth=1.4,
                  label="51-observation moving mean")
        axis.axvline(calibration_end + 0.5, color="0.4", linestyle="--",
                     linewidth=1, label="original 70/30 boundary")
        title = (f"PC1 vs observation order (|r| = {correlation:.3f})"
                 if component == 0 else "PC2 vs observation order")
        axis.set(title=title, xlabel="Observation",
                 ylabel=f"PC{component + 1} score")
        axis.legend(fontsize=7, ncol=2)

    lags = np.arange(1, MAX_LAG + 1)
    panels = [
        (axes[1, 0], 0, "raw", "PC1: within-block variogram"),
        (axes[1, 1], 0, "detrended", "PC1: after within-block linear detrending"),
        (axes[2, 0], 1, "detrended", "PC2: after within-block linear detrending"),
        (axes[2, 1], 2, "detrended", "PC3: after within-block linear detrending"),
    ]
    for axis, component, variant, title in panels:
        for block in blocks:
            axis.plot(lags, curves[block, component, variant],
                      color=COLORS[block], linewidth=1.8, label=block)
        axis.set(title=title, xlabel="Lag (observations)",
                 ylabel="Semivariance (score²)")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("No.2WT: descriptive PCA-score variation across the recording")
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    print(f"PC1 |r| with observation order: {correlation:.4f}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved {FIGURE} and {SUMMARY}")

    late_change_start = create_score_trajectory_figure(healthy)
    holdout = block_holdout(healthy)
    print(f"\nLargest PC2 step (start of the late healthy change): observation {late_change_start}")
    print("Block hold-out (Kaiser rule on the remaining healthy observations):")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"Saved {SCORE_FIGURE} and {HOLDOUT_SUMMARY}")


def create_score_trajectory_figure(healthy: pd.DataFrame) -> int:
    """
    PC1-PC6 scores of the full healthy model and their moving-window variograms.

    Scores keep the signs of the Phase 1 model, so they match the biplots.
    Each score is divided by the square root of its eigenvalue, and each
    moving-window semivariance by the eigenvalue, so all PCs share one scale.
    Returns the observation of the largest PC2 step, where the late healthy
    change starts.
    """
    model = fit_model(healthy)
    scores = project(model, healthy, N_RETAINED).scores / np.sqrt(model.eigenvalues[:N_RETAINED])
    observations = np.arange(1, len(healthy) + 1)

    # The late change starts at the largest step of the smoothed PC2 score.
    smoothed_pc2 = pd.Series(scores[:, 1]).rolling(15, center=True).mean().to_numpy()
    late_change_start = int(np.nanargmax(np.abs(np.diff(smoothed_pc2)))) + 2

    starts = np.arange(0, len(healthy) - WINDOW + 1, WINDOW_STEP)
    centres = starts + WINDOW / 2 + 0.5
    # Scores are divided by sqrt(lambda), so this semivariance is gamma / lambda.
    window_gamma = np.array(
        [[semivariogram(scores[start:start + WINDOW, component], WINDOW_LAG)[-1]
          for component in range(N_RETAINED)]
         for start in starts]
    )

    fig, axes = plt.subplots(N_RETAINED, 2, figsize=(13, 13), sharex=True,
                             constrained_layout=True)
    for component in range(N_RETAINED):
        left, right = axes[component]
        for axis in (left, right):
            axis.axvline(1099.5, color="0.4", linestyle="--", linewidth=1,
                         label="Former 70/30 boundary")
            axis.grid(alpha=0.25)
        left.plot(observations, scores[:, component], color="tab:blue", linewidth=0.6)
        left.set_ylabel(f"PC{component + 1} score / √λ")
        right.plot(centres, window_gamma[:, component], color="tab:blue", linewidth=1)
        right.set_yscale("log")
        right.set_ylabel(f"PC{component + 1} γ({WINDOW_LAG}) / λ")

    axes[0, 0].set_title("Scores of the full healthy model (6 PCs)")
    axes[0, 1].set_title(
        f"Moving-window semivariance at lag {WINDOW_LAG} "
        f"(window {WINDOW} observations)"
    )
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[-1, 0].set_xlabel("Observation order")
    axes[-1, 1].set_xlabel("Window centre (observation order)")
    fig.suptitle("No.2WT: retained PC scores and moving-window variograms over the recording")
    fig.savefig(SCORE_FIGURE, dpi=160)
    plt.close(fig)
    return late_change_start


def block_holdout(healthy: pd.DataFrame) -> pd.DataFrame:
    """
    Hide each healthy block in turn, refit on the rest and count its alarms.

    The number of PCs follows the Kaiser rule of each refitted model, and an
    alarm is an observation above the 99% T2 or Q limit, as in Phase 1. The
    original 70/30 split and its reverse are added for comparison.
    """
    n = len(healthy)
    edges = np.linspace(0, n, N_HOLDOUT_BLOCKS + 1).astype(int)
    splits = [
        (f"hide block {block + 1}", np.r_[0:edges[block], edges[block + 1]:n],
         np.arange(edges[block], edges[block + 1]))
        for block in range(N_HOLDOUT_BLOCKS)
    ]
    n_70 = round(0.70 * n)
    splits.append(("calibrate first 70%, validate last 30%",
                   np.arange(n_70), np.arange(n_70, n)))
    splits.append(("calibrate last 70%, validate first 30%",
                   np.arange(n - n_70, n), np.arange(n - n_70)))

    rows = []
    for label, train_rows, test_rows in splits:
        train, test = healthy.iloc[train_rows], healthy.iloc[test_rows]
        model = fit_model(train)
        n_components = int(np.sum(model.eigenvalues > 1))
        t2_limit = t2_limit_f(model, n_components)
        q_limit = q_limit_jackson_mudholkar(model, n_components)
        held_out = project(model, test, n_components)
        fitted = project(model, train, n_components)
        rows.append(
            {
                "split": label,
                "test_first_observation": test_rows[0] + 1,
                "test_last_observation": test_rows[-1] + 1,
                "n_components": n_components,
                "train_alarm_percent": 100 * np.mean(
                    (fitted.t2 > t2_limit) | (fitted.q > q_limit)
                ),
                "test_alarm_percent": 100 * np.mean(
                    (held_out.t2 > t2_limit) | (held_out.q > q_limit)
                ),
                "test_t2_alarm_percent": 100 * np.mean(held_out.t2 > t2_limit),
                "test_q_alarm_percent": 100 * np.mean(held_out.q > q_limit),
            }
        )
    holdout = pd.DataFrame(rows)
    holdout.to_csv(HOLDOUT_SUMMARY, index=False)
    return holdout


if __name__ == "__main__":
    main()
