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
#     name: python3
# ---

# %% [markdown]
# Completeness is defined as the percentage of all synapses that are connected to identified optic lobe neurons. 
#
# See "..\queries\completeness.py" for the exact definition. 

# %% Project setup
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
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %% Imports

from utils import olc_client
from utils.excel_exporter import ExcelExporter
from queries.completeness import fetch_ol_stats


# %% Create client and fetch list of comp
c = olc_client.connect(verbose=True)

# %% Get the aggregated statistics directly out of the database.
neuron_types_stats = fetch_ol_stats()

# %% Generate output
# Save statistics to Excel file using the ExcelExporter
exporter = ExcelExporter(
    output_path=Path(PROJECT_ROOT, 'results', 'completeness')
  , output_basename="Output Connection Completeness"
)

exporter.export(neuron_types_stats)

# %%
