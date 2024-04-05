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

# %% [markdown]
# ## Fill and save the template

# %%
import sys
import os
import random
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.rend_params import get_one_off_params
from utils.gallery_filler import generate_one_off
from utils.ol_color import OL_COLOR

c = olc_client.connect(verbose=True)

# %%
plots = get_one_off_params()

# %%
iter_counter = 0
for name, values in random.sample(list(plots.items()), len(plots.items())):
    iter_counter += 1
    generate_one_off(
        plot_name=name
      , columnar_list=values['columnar_list']
      , list_bids_to_plot=values['list_bids_to_plot']
      , hex_assign=values['hex_assign']
      , text_placement=values['text_placement']
      , replace=values['replace']
      , directory=values['directory']
      , body_color_list=values['body_color_list']
      , body_color_order=values['body_color_order']
      , neuropil_color=OL_COLOR.OL_NEUROPIL_LAYERS.rgba
      , the_view=values['view']
    )
    # Stop if number of iterations exceed the environment variable `GALLERY_EXAMPLES`
    stop_after = os.environ.get('GALLERY_EXAMPLES')
    if stop_after:
        if stop_after := int(stop_after):
            if stop_after <= iter_counter:
                break
        else:
            break
    

# %%
