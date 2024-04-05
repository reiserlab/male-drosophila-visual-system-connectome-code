# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import os, datetime
from pathlib import Path
import sys

import plotly.graph_objects as go

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.ROI_columns import create_center_column_pins, smooth_center_columns_w_median
from utils.ROI_calculus import load_pins
from utils.hex_hex import all_hex

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
#specify which neuropils to make pins in, and how to anchor the pins to the neuropil ROI
roi_pins_dict_list = [
    {'roi': 'LOP(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 0}
  , {'roi': 'LO(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 37}
  , {'roi': 'ME(R)', 'anchor_method': 'separate', 'n_anchor_bottom': 800, 'n_anchor_top': 800}
  , {'roi': 'ME(R)', 'anchor_method': 'combined', 'n_anchor_bottom': 0, 'n_anchor_top': 0}
]

#max number of columns
col_max_count = all_hex().shape[0]

for roi_pins_dict in roi_pins_dict_list:
    roi_str = roi_pins_dict['roi']

    #create columns: gives some output, i.e., if created columns are straight
    create_center_column_pins(
        roi_str=roi_str
      , anchor_method=roi_pins_dict['anchor_method']
      , n_anchor_bottom=roi_pins_dict['n_anchor_bottom']
      , n_anchor_top=roi_pins_dict['n_anchor_top']
      , verbose=True
    )

    #could number of initially created columns
    col_ids, pin_count, pins = load_pins(roi_str=roi_str)
    col_count = col_ids.shape[0]
    print(f"Number of initial {roi_str[:-3]} columns: {col_ids.shape[0]}")

    #smoothen and fill-in columns
    ctr_smooth = 0
    while col_count < col_max_count:
        smooth_center_columns_w_median(roi_str=roi_str)
        col_ids, pin_count, pins = load_pins(roi_str=roi_str)
        ctr_smooth += 1
        if col_ids.shape[0] == col_count:
            break
        else:
            col_count = col_ids.shape[0]

    print(f"Number of smoothing steps: {ctr_smooth}")
    print(f"Number of final {roi_str[:-3]} columns: {col_ids.shape[0]}")

# Expected runtimes:
#   - ME(R): 7 hrs 23 min
#   - LO(R): 2 hrs 53 min
#   - LOP(R): 20 min

# %%
