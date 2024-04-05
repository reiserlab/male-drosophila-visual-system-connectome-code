# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: ol-connectome
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd

from IPython.display import display, HTML

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.neuroglancer_plotter import image_saver, group_plotter as ng_group_plotter

from utils.ng_view import NG_View

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
res_df = pd.DataFrame(data={'hex1_id':[], 'hex2_id':[]})
for n_type in ['C2', 'C3', 'L1', 'L2', 'L5', 'Mi1', 'Mi4', 'Mi9', 'T1', 'Tm1', 'Tm2', 'Tm20']:
    data_path = PROJECT_ROOT / "results" / "screenshots" / n_type
    dup = pd.read_pickle(PROJECT_ROOT / "results" / "eyemap" / f"cols_{n_type}.pickle")
    neuron_ls = dup[dup[n_type]>0]\
        .dropna()\
        .groupby([n_type, 'hex1_id', 'hex2_id'])\
        .size()\
        .reset_index()\
        .sort_values(['hex1_id'])
    neuron_ls = neuron_ls[['hex1_id', 'hex2_id', n_type]]
    res_df = res_df.merge(neuron_ls,  how='outer', on=['hex1_id', 'hex2_id'])
res_df = res_df\
    .astype('Int64')\
    .sort_values(['hex1_id', 'hex2_id'])\
    .reset_index(drop=True)

# %%
res_df[(res_df['hex1_id']==17) & (res_df['hex2_id']==4)]

# %%
res_df[['hex1_id', 'hex2_id']]

# %%
seven_bids = []
for clms in [(3,8), (2,8), (4,8), (5,8)]:#, (2,8), (4,8), (3,7), (3,9), (4,7), (2,8)]:
    column = res_df[(res_df['hex1_id']==clms[0]) & (res_df['hex2_id']==clms[1])]
    bids = column\
        .loc[:, ~column.columns.isin(['hex1_id', 'hex2_id'])]\
        .values\
        .flatten()\
        .tolist()
    seven_bids.extend(bids)

# %%
# %autoreload 2
from utils.neuroglancer_plotter import image_saver, group_plotter as ng_group_plotter
colors = [
    (0.48, 0.22, 0.28), (0.55, 0.27, 0.20), (0.62, 0.38, 0.17), (0.71, 0.54, 0.22)
  , (0.81, 0.74, 0.40), (0.83, 0.86, 0.61), (0.73, 0.88, 0.76), (0.55, 0.80, 0.81)
  , (0.38, 0.66, 0.78), (0.31, 0.50, 0.71), (0.35, 0.35, 0.58), (0.42, 0.25, 0.42)
]

(scrn, ng_link) = ng_group_plotter(
                    body_ids=seven_bids
                  , colors=colors
                  , camera_distance=1
                  , background_color="#000000"
                  , size=(1920,1920)
                  , view=NG_View.SVD
                )

scrn

# %%
display(HTML(f'<a href="{ng_link}">open neuroglancer</a>'))

# %%

# %%
