"""Shared pretreatment of the wind-turbine SCADA data.

Builds the PCA-ready X matrices used by every PCA script: the three
structurally compatible turbines, their common variables 1-27, the single
missing No.14WT value linearly interpolated, and the variables that are
constant in the healthy turbine removed. The reasoning behind these
decisions is recorded in archive/exploratory/pretreatment_investigation_2026-09-24.md.

Note: Centering and scaling are not applied here, because each analysis must
choose which healthy observations its scaling parameters are fitted on.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Define constants for file paths and turbine selection. No.3 is excluded
# because its variable count is incompatible with the other turbines.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "resources" / "wind_turbine_fault_diagnosis_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"
EXPLORATORY_OUTPUT_DIR = PROJECT_DIR / "archive" / "exploratory"
TURBINES = ["No.2WT", "No.14WT", "No.39WT"]
HEALTHY_TURBINE = "No.2WT"
FAULTY_TURBINES = ["No.14WT", "No.39WT"]
COMMON_VARIABLES = list(range(1, 28))
# The only turbine with a missing value (variable 9, observation 358).
INTERPOLATED_TURBINE = "No.14WT"


def load_aligned_data() -> dict[str, pd.DataFrame]:
    """
    Load the three compatible turbines with their common variables 1-27.

    Restricting every sheet to variables 1-27 removes variable 28 from the
    healthy turbine. Rows keep their original observation order.
    """
    workbook = pd.read_excel(DATA_FILE, sheet_name=None, header=0)
    return {
        name: workbook[name].loc[:, COMMON_VARIABLES].copy()
        for name in TURBINES
    }


def interpolate_missing_value(aligned: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Fill the single missing No.14WT value in place by linear interpolation.

    The time-series row is preserved and the missing value is estimated from
    its adjacent observations in observation order.

    Returns
    -------
    dict
        One-based observation number, variable identifier, interpolated value
        and the preserved row count of the filled turbine.
    """
    faulty_x = aligned[INTERPOLATED_TURBINE]
    missing_row, missing_column = np.argwhere(faulty_x.isna().to_numpy())[0]
    original_row_count = len(faulty_x)
    expected_value = (
        faulty_x.iloc[missing_row - 1, missing_column]
        + faulty_x.iloc[missing_row + 1, missing_column]
    ) / 2

    aligned[INTERPOLATED_TURBINE] = faulty_x.interpolate(method="linear", axis=0)
    interpolated_value = aligned[INTERPOLATED_TURBINE].iloc[missing_row, missing_column]

    # Verify that interpolation preserves the sequence and fills the known gap.
    assert len(aligned[INTERPOLATED_TURBINE]) == original_row_count
    assert not aligned[INTERPOLATED_TURBINE].isna().any().any()
    assert np.isclose(interpolated_value, expected_value)

    return {
        "observation": missing_row + 1,
        "variable": faulty_x.columns[missing_column],
        "value": interpolated_value,
        "rows": original_row_count,
    }


def remove_healthy_constant_variables(
    aligned: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], list[int], list[int]]:
    """
    Remove the variables that are constant in the healthy turbine.

    Variables 12 and 15 are constant in the healthy turbine. They cannot be
    autoscaled and contain no variation for the healthy PCA model, so they
    are removed from every turbine.

    Returns
    -------
    tuple
        PCA-ready X matrices, the retained variables and the removed
        constant variables.
    """
    healthy_std = aligned[HEALTHY_TURBINE].std(ddof=1)
    constant_variables = healthy_std.index[healthy_std == 0].tolist()
    pca_variables = [variable for variable in COMMON_VARIABLES if variable not in constant_variables]
    pca_x = {name: data.loc[:, pca_variables] for name, data in aligned.items()}

    # All turbines must contain the same variables in the same order before
    # healthy-model fitting or any later projection.
    expected_columns = pd.Index(pca_variables)
    assert all(data.columns.equals(expected_columns) for data in pca_x.values())

    return pca_x, pca_variables, constant_variables


def load_pca_data() -> tuple[dict[str, pd.DataFrame], list[int]]:
    """Return the PCA-ready X matrices and the variables shared by all turbines."""
    aligned = load_aligned_data()
    interpolate_missing_value(aligned)
    pca_x, pca_variables, _ = remove_healthy_constant_variables(aligned)
    return pca_x, pca_variables
