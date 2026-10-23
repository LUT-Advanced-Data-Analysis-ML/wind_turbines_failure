"""Blocked cross-validation of false alarms on the healthy turbine.

The healthy recording is cut into blocks of consecutive observations and the
blocks are dealt to the folds in turn, so every fold covers the start, the
middle and the end of the recording. Observations next to a validation block
are left out of training (a gap), because neighbouring observations are
nearly identical. Each model, its autoscaling and its limits are fitted on
the training observations of a fold only, and the false-alarm rate (FAR) is
the share of validation observations above the T2 or the Q limit.

Note: No turbine data are loaded and nothing is saved here. The phase scripts
decide which observations and settings are cross-validated.
"""

import numpy as np
import pandas as pd

from kernel_monitoring import (
    LIMIT_TYPES,
    KernelModel,
    KernelProjection,
    fit_kernel_model,
    kernel_limits,
    project_kernel,
)
from pca_monitoring import Projection, fit_model, project, verify_reference_projection
from phase1_healthy_model import KAISER_EIGENVALUE_THRESHOLD, control_limits

# Blocks of 50 observations (about 8 minutes) are long compared with the
# memory of the healthy recording (after removing the slow drift, the
# autocorrelation falls below 0.2 within at most 15 observations) and short
# compared with the drift itself (hundreds of observations).
BLOCK_LENGTH = 50
N_FOLDS = 5
# Observations removed from training on each side of a validation block.
GAP = 15
# Block boundaries start at the first observation; one set of boundaries
# with five folds gives five rounds.
OFFSETS = (0,)
# Phase 1 limit names for each limit type of the linear model.
LINEAR_LIMIT_KEYS = {"theoretical": ("t2_f", "q_jm"), "mean_3sd": ("t2_3sd", "q_3sd")}
# Columns that identify one cross-validated setting.
SETTING_COLUMNS = ["model", "family", "width", "n_rule", "n_setting", "limit_type"]


def make_blocked_folds(
    n_obs: int,
    block: int = BLOCK_LENGTH,
    n_folds: int = N_FOLDS,
    gap: int = GAP,
    offset: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Zero-based training and validation observations of each fold.

    Block boundaries are placed at offset, offset + block, ...; a piece
    shorter than block / 2 at either end of the recording joins its
    neighbouring block. Block i is validated in fold i mod n_folds, and the
    gap observations on each side of its validation blocks are used neither
    for training nor for validation in that fold.
    """
    assert 0 <= offset < block
    edges = sorted({0, *range(offset, n_obs, block), n_obs})
    if edges[1] - edges[0] < block / 2:
        del edges[1]
    if edges[-1] - edges[-2] < block / 2:
        del edges[-2]
    blocks = list(zip(edges[:-1], edges[1:]))
    assert len(blocks) >= 3 * n_folds, "Too few blocks for the folds to cover the recording."

    folds = []
    times_validated = np.zeros(n_obs, dtype=int)
    for fold in range(n_folds):
        fold_blocks = blocks[fold::n_folds]
        valid = np.zeros(n_obs, dtype=bool)
        excluded = np.zeros(n_obs, dtype=bool)
        for start, end in fold_blocks:
            valid[start:end] = True
            excluded[max(start - gap, 0) : min(end + gap, n_obs)] = True
        train = ~excluded

        # The nearest training observation is exactly gap + 1 observations
        # away from every validation block, except at the recording ends.
        for start, end in fold_blocks:
            if start > 0:
                assert not train[max(start - gap, 0) : start].any()
                assert start - gap - 1 < 0 or train[start - gap - 1]
            if end < n_obs:
                assert not train[end : end + gap].any()
                assert end + gap >= n_obs or train[end + gap]
        assert not (train & valid).any()
        thirds = np.unique(np.flatnonzero(valid) * 3 // n_obs)
        assert thirds.size == 3, "Every fold must cover the start, middle and end of the recording."

        times_validated += valid
        folds.append((np.flatnonzero(train), np.flatnonzero(valid)))
    assert (times_validated == 1).all(), "Every observation must be validated exactly once."
    return folds


def _false_alarms(
    projection: Projection | KernelProjection, t2_limit: float, q_limit: float
) -> dict[str, float]:
    """Limits and the percentages of observations above the T2, the Q, and the T2 or Q limit."""
    t2_alarms = projection.t2 > t2_limit
    q_alarms = projection.q > q_limit
    return {
        "t2_limit": t2_limit,
        "q_limit": q_limit,
        "far_t2_percent": 100 * t2_alarms.mean(),
        "far_q_percent": 100 * q_alarms.mean(),
        "far_percent": 100 * (t2_alarms | q_alarms).mean(),
    }


def _truncate(model: KernelModel, projection: KernelProjection, n_components: int) -> KernelProjection:
    """
    A kernel projection reduced to its first n components.

    T2 keeps the first n terms of sum_a y_a^2 / lambda_a. The squared scores
    of the dropped components move to Q, because kc(x, x) = Q + sum_a y_a^2.
    """
    scores = projection.scores[:, :n_components]
    return KernelProjection(
        scaled=projection.scaled,
        partial_scores=projection.partial_scores[:, :, :n_components],
        scores=scores,
        t2=np.sum(scores**2 / model.eigenvalues[:n_components], axis=1),
        q=projection.q + np.sum(projection.scores[:, n_components:] ** 2, axis=1),
    )


def kernel_round(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    family: str,
    width: float,
    n_components_list: list[int],
) -> list[dict]:
    """
    FAR of one kernel model for every number of components and limit type.

    The model is fitted once on the training observations, and both sets are
    projected once onto the largest number of components; fewer components
    are obtained by truncation. The limits come from the training projection.
    """
    model = fit_kernel_model(train_x, family, width)
    largest = max(n_components_list)
    train_full = project_kernel(model, train_x, largest)
    valid_full = project_kernel(model, valid_x, largest)

    # Truncation must agree with a direct projection onto fewer components.
    smallest = min(n_components_list)
    direct = project_kernel(model, valid_x, smallest)
    truncated = _truncate(model, valid_full, smallest)
    assert np.allclose(truncated.t2, direct.t2) and np.allclose(truncated.q, direct.q)

    rows = []
    for n_components in n_components_list:
        train_projection = _truncate(model, train_full, n_components)
        valid_projection = _truncate(model, valid_full, n_components)
        verify_reference_projection(model, train_projection)
        for limit_type in LIMIT_TYPES:
            limits = kernel_limits(model, train_projection, limit_type)
            rows.append(
                {
                    "model": "kernel",
                    "family": family,
                    "width": model.width,
                    "n_rule": "fixed",
                    "n_components": n_components,
                    "limit_type": limit_type,
                    **_false_alarms(valid_projection, limits["t2"], limits["q"]),
                }
            )
    return rows


def linear_round(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    n_components_list: list[int] | str,
) -> list[dict]:
    """
    FAR of the Phase 1 linear PCA recipe fitted on the training observations.

    n_components_list holds fixed numbers of components, or is "kaiser" to
    retain the training eigenvalues greater than one as in Phase 1. The
    theoretical limits are the F-distribution T2 and Jackson-Mudholkar Q limits.
    """
    model = fit_model(train_x)
    n_rule = "fixed"
    if n_components_list == "kaiser":
        n_rule = "kaiser"
        n_components_list = [int(np.sum(model.eigenvalues > KAISER_EIGENVALUE_THRESHOLD))]

    rows = []
    for n_components in n_components_list:
        train_projection = project(model, train_x, n_components)
        verify_reference_projection(model, train_projection)
        limits = control_limits(model, train_projection, n_components)
        valid_projection = project(model, valid_x, n_components)
        for limit_type, (t2_key, q_key) in LINEAR_LIMIT_KEYS.items():
            rows.append(
                {
                    "model": "linear",
                    "family": "linear",
                    "width": np.nan,
                    "n_rule": n_rule,
                    "n_components": n_components,
                    "limit_type": limit_type,
                    **_false_alarms(valid_projection, limits[t2_key], limits[q_key]),
                }
            )
    return rows


def cross_validate_far(
    x: pd.DataFrame, round_function, offsets: tuple[int, ...] = OFFSETS, **setting
) -> pd.DataFrame:
    """
    Run round_function (kernel_round or linear_round) on every fold of every offset.

    Returns one row per round, number of components and limit type, with the
    offset, the fold, the training and validation sizes, the limits and the FARs.
    """
    rows = []
    for offset in offsets:
        for fold, (train, valid) in enumerate(make_blocked_folds(len(x), offset=offset)):
            for row in round_function(x.iloc[train], x.iloc[valid], **setting):
                rows.append(
                    {
                        "offset": offset,
                        "fold": fold,
                        "n_train": len(train),
                        "n_valid": len(valid),
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def summarise_far(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean, standard error, minimum and maximum FAR of each setting over the rounds.

    The standard error is SD / sqrt(rounds). Each observation is validated in
    one round only, but the training sets overlap, so the rounds are not
    fully independent. With the Kaiser rule the number of components can
    differ between rounds.
    """
    table = results.assign(
        n_setting=np.where(
            results["n_rule"] == "kaiser", "kaiser", results["n_components"].astype(str)
        )
    )
    summary = (
        table.groupby(SETTING_COLUMNS, dropna=False, sort=False)
        .agg(
            rounds=("far_percent", "size"),
            n_components_min=("n_components", "min"),
            n_components_max=("n_components", "max"),
            far_percent_mean=("far_percent", "mean"),
            far_percent_sd=("far_percent", "std"),
            far_percent_min=("far_percent", "min"),
            far_percent_max=("far_percent", "max"),
            far_t2_percent_mean=("far_t2_percent", "mean"),
            far_q_percent_mean=("far_q_percent", "mean"),
        )
        .reset_index()
    )
    summary.insert(
        summary.columns.get_loc("far_percent_sd"),
        "far_percent_se",
        summary["far_percent_sd"] / np.sqrt(summary["rounds"]),
    )
    return summary.drop(columns="far_percent_sd")


def verify_linear_cross_validation(
    x: pd.DataFrame, n_components_list: list[int], offset: int = 0
) -> None:
    """
    Check that the kernel and linear rounds treat the folds identically.

    The linear kernel reproduces linear PCA, so with mean + 3 SD limits
    (which both models compute the same way) every fold must give the same
    limits and the same false alarms.
    """
    common = {"offsets": (offset,), "n_components_list": n_components_list}
    kernel = cross_validate_far(x, kernel_round, family="linear", width=1.0, **common)
    linear = cross_validate_far(x, linear_round, **common)
    columns = ["n_train", "n_valid", "n_components", "t2_limit", "q_limit",
               "far_t2_percent", "far_q_percent", "far_percent"]
    kernel = kernel[kernel["limit_type"] == "mean_3sd"][columns].to_numpy(dtype=float)
    linear = linear[linear["limit_type"] == "mean_3sd"][columns].to_numpy(dtype=float)
    assert np.allclose(kernel, linear, rtol=1e-6)
