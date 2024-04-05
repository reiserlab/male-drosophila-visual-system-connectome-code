# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: ol-connectome
#     language: python
#     name: ol-connectome
# ---

# %%
# %load_ext autoreload
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
# %autoreload 2
from utils.ROI_voxels import voxelize_col_and_lay

# %%
voxelize_col_and_lay(rois=['ME(R)'], columns=False)

#expected runtime for all neuropils, col and lay: 15 min

# %%
