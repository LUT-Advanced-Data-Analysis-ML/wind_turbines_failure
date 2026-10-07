"""PCA monitoring functions for the healthy-turbine reference model.

Implements the equations of the modelling plan (report 2, Section 4):
autoscaling and economy-SVD PCA of a reference X, projection of new
observations, Hotelling's T2 and Q (SPEx) statistics, their control limits,
and the variable contributions used for sensor diagnostics.

Note: No data are loaded or fitted here. The phase scripts decide which
observations a model is fitted on and which observations are projected.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# Confidence level of the theoretical control limits.
CONFIDENCE = 0.99


@dataclass
class PCAModel:
    """Autoscaling parameters and the full PCA decomposition of a reference X."""

    variables: list[int]
    mean: pd.Series
    std: pd.Series
    # Loadings P (K x K) and eigenvalues of all components. The number of
    # retained components is chosen when projecting, not when fitting.
    loadings: np.ndarray
    eigenvalues: np.ndarray
    n_observations: int


@dataclass
class Projection:
    """Observations projected onto a PCA model with n retained components."""

    scaled: np.ndarray
    scores: np.ndarray
    residuals: np.ndarray
    t2: np.ndarray
    q: np.ndarray


def fit_model(x: pd.DataFrame) -> PCAModel:
    """
    Autoscale the reference X with its own statistics and fit PCA by SVD.

    The autoscaled matrix Z is decomposed as Z = T P' + E with economy SVD:
    P = V and the eigenvalues are lambda_a = t_a' t_a / (N - 1).
    """
    mean = x.mean()
    std = x.std(ddof=1)
    assert (std > 0).all(), "Zero-variance variables cannot be autoscaled."

    z = ((x - mean) / std).to_numpy()
    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)

    return PCAModel(
        variables=list(x.columns),
        mean=mean,
        std=std,
        loadings=vt.T,
        eigenvalues=singular_values**2 / (len(x) - 1),
        n_observations=len(x),
    )


def project(model: PCAModel, x: pd.DataFrame, n_components: int) -> Projection:
    """
    Autoscale X with the model statistics and project it onto n components.

    t = z P, e = z - t P', T2 = sum_a t_a^2 / lambda_a and Q = sum_j e_j^2.
    The model is never refitted on the projected observations.
    """
    assert list(x.columns) == model.variables, "X must use the model variables in order."
    assert 0 < n_components < len(model.variables)

    z = ((x - model.mean) / model.std).to_numpy()
    p = model.loadings[:, :n_components]
    scores = z @ p
    residuals = z - scores @ p.T

    return Projection(
        scaled=z,
        scores=scores,
        residuals=residuals,
        t2=np.sum(scores**2 / model.eigenvalues[:n_components], axis=1),
        q=np.sum(residuals**2, axis=1),
    )


def verify_reference_projection(model: PCAModel, projection: Projection) -> None:
    """
    Check the projection of the observations the model was fitted on.

    For the reference observations, mean T2 = n (N - 1) / N and mean Q is
    the sum of the excluded eigenvalues multiplied by (N - 1) / N.
    """
    n_components = projection.scores.shape[1]
    factor = (model.n_observations - 1) / model.n_observations
    assert len(projection.t2) == model.n_observations
    assert np.isclose(projection.t2.mean(), n_components * factor)
    assert np.isclose(projection.q.mean(), model.eigenvalues[n_components:].sum() * factor)


def t2_limit_f(model: PCAModel, n_components: int, confidence: float = CONFIDENCE) -> float:
    """T2 limit for new observations from the F-distribution."""
    n_obs = model.n_observations
    factor = n_components * (n_obs**2 - 1) / (n_obs * (n_obs - n_components))
    return factor * stats.f.ppf(confidence, n_components, n_obs - n_components)


def q_limit_jackson_mudholkar(
    model: PCAModel, n_components: int, confidence: float = CONFIDENCE
) -> float:
    """Q limit from the Jackson-Mudholkar approximation."""
    residual_eigenvalues = model.eigenvalues[n_components:]
    theta1, theta2, theta3 = (np.sum(residual_eigenvalues**power) for power in (1, 2, 3))
    h0 = 1 - 2 * theta1 * theta3 / (3 * theta2**2)
    c_alpha = stats.norm.ppf(confidence)
    return theta1 * (
        c_alpha * np.sqrt(2 * theta2 * h0**2) / theta1
        + 1
        + theta2 * h0 * (h0 - 1) / theta1**2
    ) ** (1 / h0)


def three_sigma_limit(reference_values: np.ndarray) -> float:
    """Mean plus three standard deviations of a reference statistic."""
    return reference_values.mean() + 3 * reference_values.std(ddof=1)


def t2_contributions(model: PCAModel, projection: Projection) -> np.ndarray:
    """
    Variable contributions to T2: c_j = sum_a (t_a / lambda_a) p_ja z_j.

    Each row sums to the T2 of that observation. Individual contributions can
    be negative when a variable moves against the direction of the scores.
    """
    n_components = projection.scores.shape[1]
    weights = (projection.scores / model.eigenvalues[:n_components]) @ model.loadings[:, :n_components].T
    contributions = weights * projection.scaled
    assert np.allclose(contributions.sum(axis=1), projection.t2)
    return contributions


def spe_contributions(projection: Projection) -> np.ndarray:
    """Variable contributions to Q (SPEx): e_j^2. Each row sums to Q."""
    return projection.residuals**2
