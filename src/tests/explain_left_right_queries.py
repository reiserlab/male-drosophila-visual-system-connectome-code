# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""

from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv()

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from queries.completeness import fetch_ol_types, fetch_ol_types_and_instances

# %%
fetch_ol_types_and_instances(side='both')

# %%
fetch_ol_types(side='R-dominant')

# %%
# types = fetch_ol_types(side='R-dominant') # 732
types = fetch_ol_types(side="both")  # 732
types

# %%
cell_type = "aMe12"

# %% [markdown]
# 'R-dominant' will return the number of cells of the cell type that innervate the right optic lobe and have '_R' instances, unless the cell type does not have '_R' instances in which case it will use the '_L' instance.

# %%
types = fetch_ol_types(side="R-dominant")
# types.head(10)
types[types["type"] == cell_type]

# %% [markdown]
# 'both' will return the number of cells of the cell type that innervate the right optic lobe, regardless of whether they are '_L' or '_R' instances.

# %%
types = fetch_ol_types(side="both")
# types.head(20)
types[types["type"] == cell_type]

# %% [markdown]
# 'left' will return the number of cells of the cell type that innervate the right optic lobe that have '_L' instances.

# %%
types = fetch_ol_types(side="L")
# types.head(20)
types[types["type"] == cell_type]

# %% [markdown]
# 'right' will return the number of cells of the cell type that innervate the right optic lobe that have '_R' instances.

# %%
types = fetch_ol_types(side="R")
# types.head(20)
types[types["type"] == cell_type]

# %% [markdown]
# Output from 'fetch_ol_types_and_instances':

# %%
cell_type = "aMe12"

# %%
df = fetch_ol_types_and_instances(side="L")
df[df["type"] == cell_type]

# %%
df = fetch_ol_types_and_instances(side="R")
df[df["type"] == cell_type]

# %%
df = fetch_ol_types_and_instances(side="R-dominant")
df[df["type"] == cell_type]

# %%
df = fetch_ol_types_and_instances(side="both")
df[df["type"] == cell_type]
# df

# %% [markdown]
# ### Testing neuron_bag.py functions

# %%
from utils.neuron_bag import NeuronBag


# %%
abag = NeuronBag('LPT57', side='L')
abag.get_body_ids(cell_count=20)

# %% [markdown]
# ### OLTypes

# %%
from utils.ol_types import OLTypes

# %%
olt = OLTypes()

# %%
olt.get_star(type_str='LC14b') # This will throw a warning since the bilateral neuron type has two stars. It will return one of them.

# %%
olt.get_star(instance_str='LC14b_R') # This get the star from one side.

# %%
olt.get_neuron_list(primary_classification='OL', side='R-dominant') # 244

# %%
olt.get_neuron_list() # 732

# %%
df = olt.get_neuron_list(side='both')
df[df['type'] == 'LC14b']

# %%
