# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
from dotenv import find_dotenv

from utils.plot_config import PlotConfig

# %%
data_path = Path(find_dotenv()).parent / "src" / "gallery_generation"

pcfg = PlotConfig(data_path / "Optic-Lobe_OLi_Fig1_LoVC16_darker.json")

# %%
pcfg.name

# %%
pcfg.bids

# %%
pcfg.bid_dict

# %%
pcfg.camera

# %%
for roi in pcfg.rois:
    print(roi.is_visible)

# %%
for neuron in pcfg.neurons:
    for sl in neuron.slicers:
        print(sl.is_named)

# %%
pcfg.text_dict

# %%
pcfg.directory

# %%
pcfg.basename

# %%
pcfg.max_slice

# %%
pcfg.scalebar

# %%
if pcfg.scalebar:
    print("has scalebar")
else:
    print("has no scalebar")
