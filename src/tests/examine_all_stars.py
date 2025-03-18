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
import pandas as pd
from dotenv import find_dotenv
from utils import olc_client
from utils.ol_neuron import OLNeuron

c = olc_client.connect()
PROJECT_ROOT = Path(find_dotenv()).parent
print(f"Project root directory: {PROJECT_ROOT}")

# %%
all_stars = pd.read_excel(PROJECT_ROOT / 'params' / 'all_stars.xlsx')


# %% [markdown]
# ## Find inconsistencies
#
# This little loop finds the real type and instance for a star_neuron in all_stars, warns about the inconsistency, and generates a new column "real_instance" with the actual instance of the neuron.

# %%
def add_in(row):
    oln = OLNeuron(row['star_neuron'])
    instance = oln.instance
    if len(str(row['instance']))>0 and str(row['instance']) != 'nan':
        if not instance == row['instance']:
            print(f"Instance {instance} != {row['instance']}")
    if not row['type'] == oln.get_type():
        print(f"Type {row['type']} != {oln.get_type()}")
    return oln.instance


all_stars['real_instance'] = all_stars.apply(add_in, axis=1)
