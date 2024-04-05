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
import sys

import numpy as np
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
    
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from utils.ROI_calculus import load_pins, load_layer_thre, _get_data_path

# %%
conv_to_um = 8/1000

# %%
roi_df = pd.DataFrame()
for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
    col_ids, n_bins, pins = load_pins(roi_str=roi_str)
    n_col = col_ids.shape[0]
    pins = pins\
        .astype(float)\
        .reshape((n_col, n_bins, 3))
    pins_length = \
        conv_to_um * np.sqrt(
            (np.diff(pins,axis=1)**2)\
            .sum(2)
        )\
        .sum(1)
    depth_bdry = load_layer_thre(roi_str=roi_str)
    bin_bdry = n_bins-1 - np.round(depth_bdry*(n_bins-1)).astype(int)
    bin_bdry[-1] = 0
    bin_bdry[0] = n_bins-1
    n_layers = bin_bdry.shape[0]-1
    layer_thickness = np.zeros(n_layers)
    layer_thickness_std = np.zeros(n_layers)
    for idx in range(n_layers):
        pins_layer = pins[:, bin_bdry[idx+1]:bin_bdry[idx]]\
            .reshape((n_col, -1, 3))
        dist_in_um = conv_to_um * np.sqrt(
                (np.diff(pins_layer, axis=1)**2).sum(2)
            )
        layer_thickness[idx] = \
            dist_in_um\
            .sum(1)\
            .mean(0)
        layer_thickness_std[idx] = \
            dist_in_um\
            .sum(1)\
            .std(0)
    layer_str = ", ".join([f"{thk:.0f}" for thk in layer_thickness.round()])
    layer_str = layer_str + f" ± {layer_thickness_std.mean():.0f}"
    tmp = pd.DataFrame({
            'roi': roi_str
          , 'Number of columns': n_col
          , 'Column length (µm)': f"{pins_length.mean():.0f} ± {pins_length.std():.0f}"
          , 'Number of layers': n_layers
          , 'Layer thicknesses (µm)': layer_str
        }
      , index=[idx]
    )
    roi_df = pd.concat([roi_df, tmp])
roi_df = roi_df\
    .reset_index(drop=True)\
    .set_index('roi')

# %%
table_df = roi_df.T
table_df.to_excel(_get_data_path('data') / 'roi_table.xlsx', index=False)

# %%
