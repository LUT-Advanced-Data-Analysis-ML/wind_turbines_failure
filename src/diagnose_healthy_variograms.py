"""Exploratory observation-lag variograms of the healthy turbine's PCA scores.

This is a descriptive diagnostic. It does not select components, fit control
limits, or alter any of the Phase 1–3 modelling outputs.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_monitoring import fit_model
from pretreatment import HEALTHY_TURBINE, OUTPUT_DIR, load_pca_data


MAX_LAG = 60  # Observation steps; timestamps are not available.
SELECTED_LAGS = (1, 10, 30, 60)
FIGURE = OUTPUT_DIR / "healthy_temporal_variogram.png"
SUMMARY = OUTPUT_DIR / "healthy_temporal_variogram_summary.csv"
COLORS = {"early": "#277DA1", "middle": "#F8961E", "late": "#9B5DE5"}


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


if __name__ == "__main__":
    main()
