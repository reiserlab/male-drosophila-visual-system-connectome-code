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
# %load_ext autoreload
# %autoreload 2
# Setting up environment and accessing database
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

# import functions
from make_spatial_coverage_plots_for_webpages import make_spatial_coverage_plots_for_webpages
# from plotting_to_html import plot_and_save_3D_fig_to_html

from utils import olc_client
from utils.ol_types import OLTypes
from utils.helper import slugify

from patterns import convert_pkl_to_html_with_layers

c = olc_client.connect(verbose=True)

# %%
# Set up paths

# Output path to cache
output_path_cache = Path(PROJECT_ROOT, 'cache', 'html_pages')
output_path_cache.mkdir(parents=True, exist_ok=True)

# Output path to results
output_path_results = Path(PROJECT_ROOT, 'results', 'html_pages')
output_path_results.mkdir(parents=True, exist_ok=True)

# Coverage and completeness
input_path_coverage = Path(PROJECT_ROOT, 'cache', 'complete_metrics')
input_path_coverage.mkdir(parents=True, exist_ok=True)

# %%
# Generate 3D plotly figures and save to html

# Import star neuron bodyIDs from optic-lobe-connectome/params/
olt = OLTypes()
cell_type_list = olt.get_neuron_list(
    side='both'
)

linked_instance = set(cell_type_list['instance'].to_list())

# # DEBUG
# cell_type_list = cell_type_list[cell_type_list['instance']\
#     .isin(['TmY5a_R', 'Mi1_R'])]
# # .isin(['5-HTPMPV03_R', 'LoVP88_R', 'LoVP100_R', 'LoVP24_R', 'LoVP30_R', 'MeVP55_R', 'MeVP58_R'])]
# cell_type_list

# %%
cell_type_list = cell_type_list.sample(frac=1)

# %%
from utils.ol_instance import OLInstance
import multiprocessing as mp
import numpy as np
# Create a reverse lookup dictionary from filename to main group
# Make an html page for each

def generate_pages(df:pd.DataFrame):
    for index, row in df.iterrows():
        oli = OLInstance(row['instance'])

        print(f"Coverage for {row['instance']}")
        make_spatial_coverage_plots_for_webpages(instance=row['instance'])

        print(f"HTML for {row['instance']}")
        success = convert_pkl_to_html_with_layers(
            oli=oli
          , valid_neuron_names=linked_instance
          , template="html-pages-jinja.html.jinja"
          , input_path_coverage=input_path_coverage
          , output_path=output_path_results
        )
        if not success:
            continue  # Skip to the next instance as before

# DEBUG
# generate_pages(cell_type_list)

splitter = mp.cpu_count() -2
data_split = np.array_split(cell_type_list.sample(frac=1), splitter)
pool = mp.Pool(splitter)
pool.map(generate_pages, data_split)

# %%
