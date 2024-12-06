# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from queries.completeness import fetch_ol_types
from utils.neuron_bag import NeuronBag
from utils.movie_maker import generate_movie_json
from utils.ol_types import OLTypes

from utils import olc_client

c = olc_client.connect(verbose=True)

# %%
olt = OLTypes()

oli_list = olt.get_neuron_list()


# %% [markdown]
# ## Fill and save the template

# %%
for idx, row in oli_list.iterrows():
   
    a_bag = NeuronBag(cell_instance=row['instance'], side=None, rois='OL(R)')
    a_bag.sort_by_distance_to_star()

    sorted_body_ids = a_bag.get_body_ids(a_bag.size)

    if row['main_groups']=='OL_intrinsic' or row['main_groups']=='OL_connecting'\
        or row['main_groups']=='VPN' or row['main_groups']=='VCN':
            the_movie_group=row['main_groups']
    else:
        the_movie_group='other'

    generate_movie_json(
        neuron_type=row['type']
      , sorted_body_ids=sorted_body_ids
      , template='movie-descriptions.json.jinja'
      , is_general_template=False
      , movie_group=the_movie_group
    )
    print(f"Json generation done for {row['type']}")
