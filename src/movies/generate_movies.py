# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: OL-Connectome
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
from utils.movie_maker import create_template_filename, generate_movie_json

from utils import olc_client

c = olc_client.connect(verbose=True)

# %%
# pandas dataframe generated from @aljoschanern's list of optic lobe intrinsic neurons

ol_cell_types = fetch_ol_types()
ol_cell_types['Movie_Template'] = None
ol_intrinsic_types = pd.read_csv(PROJECT_ROOT / "params" / "OL_intrinsic_groups_072623_v1.csv")
movie_template_list = ol_cell_types.merge(ol_intrinsic_types, on='type')


# %%
movie_template_list['Movie_Template'] = movie_template_list.apply(create_template_filename, axis=1)

# %% [markdown]
# ## Fill and save the template

# %%
assigned_neurontypes = movie_template_list[movie_template_list['Movie_Template'].notna()]

for idx, row in assigned_neurontypes.iterrows():
    # fetch the bodyIds sorted by distance to a column
    a_bag = NeuronBag(cell_type=row['type'])
    a_bag.sort_by_distance_to_hex(
        neuropil="ME(R)"
      , hex1_id=18
      , hex2_id=18)
    sorted_body_ids = a_bag.get_body_ids(a_bag.size)

    generate_movie_json(
        neuron_type=row['type']
      , sorted_body_ids=sorted_body_ids
      , template = row['Movie_Template']
    )
    print(f"Json generation done for {row['type']}")

# %%
movie_template_list.to_csv(PROJECT_ROOT / "logs" / 'template_list_for_movies.csv', index=False)
