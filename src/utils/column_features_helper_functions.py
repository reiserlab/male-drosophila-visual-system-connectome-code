"""
Helper functions used in the notebook `src/column_features/column_features_analysis.ipynb`
"""
import pandas as pd
from neuprint import fetch_custom


def find_neuropil_hex_coords(roi_str: str):
    """
    Fetch the hex_id coordinates of the columns present within each of the neuropils.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    col_hex_ids : pd.DataFrame
        `hex1_id`
            hex1_id value of the column
        `hex2_id`
            hex2_id value of the column

    n_cols : int
        total number of columns in the neuropil (roi_str)

    """
    assert roi_str in ["ME(R)", "LO(R)", "LOP(R)"]\
      , f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    cql = f"""
        MATCH (cp:ColumnPin)
        WHERE cp['{roi_str}']
        RETURN DISTINCT cp.olHex1 AS hex1_id, cp.olHex2 AS hex2_id
        ORDER BY hex1_id, hex2_id
    """

    col_hex_ids = fetch_custom(cql)

    n_cols = len(col_hex_ids)

    return col_hex_ids, n_cols

