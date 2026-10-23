"""Kernel PCA monitoring functions for the healthy-turbine reference model.

Implements additive kernel MSPC: one kernel per sensor,
summed over the sensors with one shared width
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from pca_monitoring import CONFIDENCE, fit_model, project, t2_limit_f, three_sigma_limit

# One-sensor kernels. The linear kernel is only used to
# check the kernel code against linear PCA.
KERNEL_FAMILIES = ["gaussian", "cauchy", "matern1/2", "matern3/2", "matern5/2", "linear"]
LIMIT_TYPES = ["theoretical", "mean_3sd"]


def sensor_kernel(a: np.ndarray, b: np.ndarray, family: str, width: float) -> np.ndarray:
    """
    Kernel between the values a (length m) and b (length n) of one sensor.

    With d = |a_i - b_j| and r = d / width (amplitude 1):
    gaussian exp(-r^2 / 2), cauchy 1 / (1 + r^2), matern1/2 exp(-r),
    matern3/2 (1 + sqrt(3) r) exp(-sqrt(3) r) and
    matern5/2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).
    The linear kernel a_i b_j ignores the width.
    """
    if family == "linear":
        return np.outer(a, b)
    r = np.abs(a[:, None] - b[None, :]) / width
    if family == "gaussian":
        return np.exp(-0.5 * r**2)
    if family == "cauchy":
        return 1 / (1 + r**2)
    if family == "matern1/2":
        return np.exp(-r)
    if family == "matern3/2":
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    if family == "matern5/2":
        return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    raise ValueError(f"Unknown kernel family: {family}")


def sensor_self_kernel(a: np.ndarray, family: str) -> np.ndarray:
    """k(a, a) for each value of one sensor: a^2 for the linear kernel, 1 otherwise."""
    return a**2 if family == "linear" else np.ones_like(a)


@dataclass
class KernelModel:
    """Autoscaling parameters, reference data and kernel PCA of a reference X."""

    variables: list[int]
    mean: pd.Series
    std: pd.Series
    family: str
    width: float
    # Autoscaled reference observations Z (N x K); every projection needs them.
    scaled: np.ndarray
    # Centring statistics of each sensor kernel: the column means (K x N) and
    # the grand means (K) of the reference kernel matrix of that sensor.
    column_means: np.ndarray
    grand_means: np.ndarray
    # Eigenvectors (N x N) of the centred kernel matrix and the variances of
    # the scores of all components, lambda_a = w_a / (N - 1) for eigenvalue
    # w_a. As in PCAModel, the number of retained components is chosen when
    # projecting, not when fitting.
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    n_observations: int


@dataclass
class KernelProjection:
    """Observations projected onto a kernel PCA model with n retained components."""

    scaled: np.ndarray
    # Scores of each sensor's centred kernel piece (partial scores, K x M x n);
    # they add up over the sensors to the scores.
    partial_scores: np.ndarray
    scores: np.ndarray
    t2: np.ndarray
    q: np.ndarray


def fit_kernel_model(x: pd.DataFrame, family: str, width: float = 1.0) -> KernelModel:
    """
    Autoscale the reference X with its own statistics and fit kernel PCA.

    The kernel matrix is the sum of the one-sensor kernels,
    K = sum_l k_l(z_l, z_l), centred in the feature space as
    Kc = (I - 11'/N) K (I - 11'/N) and decomposed as Kc = U W U'.
    """
    assert family in KERNEL_FAMILIES, f"Unknown kernel family: {family}"
    assert family == "linear" or width > 0
    mean = x.mean()
    std = x.std(ddof=1)
    assert (std > 0).all(), "Zero-variance variables cannot be autoscaled."

    z = ((x - mean) / std).to_numpy()
    n_obs, n_vars = z.shape
    kernel = np.zeros((n_obs, n_obs))
    column_means = np.empty((n_vars, n_obs))
    for index in range(n_vars):
        sensor = sensor_kernel(z[:, index], z[:, index], family, width)
        column_means[index] = sensor.mean(axis=0)
        kernel += sensor
    grand_means = column_means.mean(axis=1)

    # The kernel matrix is symmetric, so its row means are its column means.
    column_total = column_means.sum(axis=0)
    centred = kernel - column_total[None, :] - column_total[:, None] + grand_means.sum()
    tolerance = 1e-9 * np.abs(kernel).max()
    assert np.allclose(centred.mean(axis=0), 0, atol=tolerance)
    assert np.allclose(centred.mean(axis=1), 0, atol=tolerance)

    eigenvalues, eigenvectors = np.linalg.eigh(centred)
    order = np.argsort(eigenvalues)[::-1]

    return KernelModel(
        variables=list(x.columns),
        mean=mean,
        std=std,
        family=family,
        width=width,
        scaled=z,
        column_means=column_means,
        grand_means=grand_means,
        eigenvectors=eigenvectors[:, order],
        eigenvalues=eigenvalues[order] / (n_obs - 1),
        n_observations=n_obs,
    )


def _score_directions(model: KernelModel, n_components: int) -> np.ndarray:
    """Retained eigenvectors divided by sqrt(w_a), so that the scores are y_a = kc u_a / sqrt(w_a)."""
    w = model.eigenvalues[:n_components] * (model.n_observations - 1)
    return model.eigenvectors[:, :n_components] / np.sqrt(w)


def project_kernel(model: KernelModel, x: pd.DataFrame, n_components: int) -> KernelProjection:
    """
    Autoscale X with the model statistics and project it onto n kernel components.

    For each sensor l the kernel row k_l(x, Z) is centred with that sensor's
    reference means and projected, giving the partial scores
    y_a^(l) = kc_l u_a / sqrt(w_a); the scores are their sum over the
    sensors. T2 = sum_a y_a^2 / lambda_a and
    Q = kc(x, x) - sum_a y_a^2, where kc(x, x) = k(x, x) - 2 mean_i k(x, z_i)
    + mean(K) is the squared distance of x from the reference mean in the
    feature space. The model is never refitted on the projected observations.
    """
    assert list(x.columns) == model.variables, "X must use the model variables in order."
    assert 0 < n_components < model.n_observations
    assert model.eigenvalues[n_components - 1] > 0, "Retained components must have positive variance."

    z = ((x - model.mean) / model.std).to_numpy()
    directions = _score_directions(model, n_components)
    partial_scores = np.empty((len(model.variables), len(z), n_components))
    self_kernel = np.zeros(len(z))
    for index in range(len(model.variables)):
        sensor = sensor_kernel(z[:, index], model.scaled[:, index], model.family, model.width)
        row_means = sensor.mean(axis=1)
        piece = (
            sensor
            - row_means[:, None]
            - model.column_means[index][None, :]
            + model.grand_means[index]
        )
        partial_scores[index] = piece @ directions
        self_kernel += (
            sensor_self_kernel(z[:, index], model.family)
            - 2 * row_means
            + model.grand_means[index]
        )
    scores = partial_scores.sum(axis=0)

    return KernelProjection(
        scaled=z,
        partial_scores=partial_scores,
        scores=scores,
        t2=np.sum(scores**2 / model.eigenvalues[:n_components], axis=1),
        q=self_kernel - np.sum(scores**2, axis=1),
    )


def verify_partial_scores(model: KernelModel, projection: KernelProjection) -> None:
    """
    Check that the sensor pieces add up to the full centred kernel row.

    The full row is built directly from the summed kernel and the summed
    centring statistics, and its scores must equal the sum of the partial
    scores.
    """
    z = projection.scaled
    kernel = sum(
        sensor_kernel(z[:, index], model.scaled[:, index], model.family, model.width)
        for index in range(len(model.variables))
    )
    centred = (
        kernel
        - kernel.mean(axis=1, keepdims=True)
        - model.column_means.sum(axis=0)[None, :]
        + model.grand_means.sum()
    )
    full_scores = centred @ _score_directions(model, projection.scores.shape[1])
    assert np.allclose(projection.partial_scores.sum(axis=0), full_scores)


def verify_linear_kernel(x_reference: pd.DataFrame, x_new: pd.DataFrame, n_components: int) -> None:
    """
    Check that the linear kernel reproduces linear PCA exactly.

    With k(x, y) = x'y the kernel scores are the linear PCA scores up to the
    arbitrary sign of each component, so T2, Q and the partial scores
    z_l p_la must agree with pca_monitoring.
    """
    kernel_model = fit_kernel_model(x_reference, "linear")
    linear_model = fit_model(x_reference)
    kernel_projection = project_kernel(kernel_model, x_new, n_components)
    linear_projection = project(linear_model, x_new, n_components)

    assert np.allclose(kernel_projection.t2, linear_projection.t2, rtol=1e-6, atol=1e-8)
    assert np.allclose(kernel_projection.q, linear_projection.q, rtol=1e-6, atol=1e-8)
    signs = np.sign(np.sum(kernel_projection.scores * linear_projection.scores, axis=0))
    linear_partial = (
        linear_projection.scaled.T[:, :, None]
        * linear_model.loadings[:, :n_components][:, None, :]
    )
    assert np.allclose(kernel_projection.partial_scores, linear_partial * signs, rtol=1e-6, atol=1e-8)


def kernel_limits(
    model: KernelModel,
    reference_projection: KernelProjection,
    limit_type: str,
    confidence: float = CONFIDENCE,
) -> dict[str, float]:
    """
    T2 and Q limits from the projection of the reference observations.

    theoretical: the F-distribution T2 limit for new observations (as in
    Phase 1) and the scaled chi-square Q limit g chi2_h with g = var / (2 mean)
    and h = 2 mean^2 / var of the reference Q.
    mean_3sd: mean plus three standard deviations of the reference T2 and Q
    """
    n_components = reference_projection.scores.shape[1]
    if limit_type == "theoretical":
        q_mean = reference_projection.q.mean()
        q_var = reference_projection.q.var(ddof=1)
        return {
            "t2": t2_limit_f(model, n_components, confidence),
            "q": q_var / (2 * q_mean) * stats.chi2.ppf(confidence, 2 * q_mean**2 / q_var),
        }
    if limit_type == "mean_3sd":
        return {
            "t2": three_sigma_limit(reference_projection.t2),
            "q": three_sigma_limit(reference_projection.q),
        }
    raise ValueError(f"Unknown limit type: {limit_type}")


def partial_score_contributions(
    model: KernelModel, projection: KernelProjection, normalise: bool = True
) -> np.ndarray:
    """
    Sensor contributions to T2 from the partial scores, M x K.

    r_l = sum_a (y_a^(l))^2 / lambda_a, and the normalised shares
    c_l = r_l / sum_m r_m add up to one for each observation. Partial scores
    of different sensors can partly cancel, so the r_l do not add up to T2.
    """
    n_components = projection.scores.shape[1]
    contributions = np.sum(
        projection.partial_scores**2 / model.eigenvalues[:n_components], axis=2
    ).T
    if not normalise:
        return contributions
    return contributions / contributions.sum(axis=1, keepdims=True)


def save_kernel_settings(
    model: KernelModel,
    n_components: int,
    limit_type: str,
    limits: dict[str, float],
    path: Path,
) -> None:
    """
    Save the settings that define a final kernel model as a one-row CSV.

    The model itself is not saved: refitting it on the same reference
    observations with these settings is deterministic and fast.
    """
    pd.DataFrame(
        [
            {
                "family": model.family,
                "width": model.width,
                "n_components": n_components,
                "limit_type": limit_type,
                "t2_limit": limits["t2"],
                "q_limit": limits["q"],
                "n_observations": model.n_observations,
                "confidence": CONFIDENCE,
            }
        ]
    ).to_csv(path, index=False)


def load_kernel_settings(path: Path) -> dict:
    """Load settings written by save_kernel_settings. Values are parsed without rounding."""
    row = pd.read_csv(path, float_precision="round_trip").iloc[0]
    return {
        "family": str(row["family"]),
        "width": float(row["width"]),
        "n_components": int(row["n_components"]),
        "limit_type": str(row["limit_type"]),
        "limits": {"t2": float(row["t2_limit"]), "q": float(row["q_limit"])},
        "n_observations": int(row["n_observations"]),
        "confidence": float(row["confidence"]),
    }
