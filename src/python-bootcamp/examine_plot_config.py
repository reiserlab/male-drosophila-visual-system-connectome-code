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
# %load_ext autoreload
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")


# %%
# %autoreload 2
from utils.plot_config import PlotConfig

# %%
data_path = Path(find_dotenv()).parent / "results" / "gallery-descriptions"

# pcfg = PlotConfig(data_path / "Group_HS_cells.json")

pcfg = PlotConfig(data_path / "Optic-Lobe_OLi_C2.json")

# %%
pcfg.name

# %%
pcfg.bids

# %%
pcfg.bid_dict

# %%
pcfg.camera

# %%
pcfg.rois
