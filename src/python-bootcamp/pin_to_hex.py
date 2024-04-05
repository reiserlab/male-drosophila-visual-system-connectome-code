# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd

from dotenv import find_dotenv


def convert_pin_pickle_to_csv(roi_str:str, remove_empty:bool=False):
    """
    Convert data from the existing pickle files (wide data format) to a CSV (long data format).

    Parameter
    ---------
    roi_str : str
        either 'ME(R)', 'LO(R)', or 'LOP(R)'

    Returns
    -------
    col_long : pd.DataFrame
        hex1_id : int
            Hex 1 coordinate
        hex2_it : int
            Hex 2 coordinate
        bin_depth : int
            The depth of the pins is divided into bins, this represents the bin it is in.
            
        x,y,z : float
            location of the pin point
        roi : str
            'ME_R', 'LO_R', or 'LOP_R'
    """
    
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
        f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"

    coords = ['x', 'y', 'z']
    data_path = Path(find_dotenv()).parent / 'cache' / 'eyemap'
    col_df = pd.read_pickle(
        data_path / f"{roi_str[:-3]}_col_center_pins.pickle"
    )

    col_tmp = col_df\
        .drop(['N_syn', 'col_id', 'n_syn'], errors='ignore', axis=1)\
        .reset_index(drop=True)\
        .set_index(['hex1_id', 'hex2_id'])\
        .melt(ignore_index=False)


    col_tmp['coordinate'] = col_tmp.apply(lambda x: coords[x['variable']%3], axis=1)
    col_tmp['bin_depth'] = max(col_tmp['variable']//3) - col_tmp['variable']//3
    col_long = col_tmp\
        .reset_index()\
        .set_index(['hex1_id', 'hex2_id', 'bin_depth'])\
        .pivot(columns=['coordinate'], values=['value'])\
        .reset_index(col_level=1)\
        .droplevel(0, axis=1)
    col_long["roi"] = f"{roi_str[:-3]}_R"
    if remove_empty:
        col_long = col_long[col_long['x'].notna() & col_long['y'].notna() & col_long['z'].notna()]
    return col_long

# %%
result_path = Path(find_dotenv()).parent / 'results' / 'eyemap'

for roi in ['LO(R)', 'LOP(R)', 'ME(R)']:
    print(roi)
    roi_cols = convert_pin_pickle_to_csv(roi, remove_empty=True)
    roi_cols.to_csv(result_path / f"{roi[:-3]}_pindata.csv")

# %% [markdown]
#
# Pseudo-code as an idea how to define the PinPoint data type in neuprint
#
# ```cypher
# PinPoint {
#     location: Point([row.x, row.y, row.z]),
#     f"{row.roi}_col_{row.hex1_id:02d}_{row.hex2_id:02d}": True,
#     depth: roi.depth
# }
# ```
#
# Alternatively to the `ME_R_col_H1_H2`, indexed properties for the "primary ROI", hex1, and hex2 would even be better.
#
# Please have an index on the location
#
#
# Note: With the current data, the bin for ME(R) ranges from 0…120, for LO(R) from 0…75, and for LOP(R) from 0…50.
