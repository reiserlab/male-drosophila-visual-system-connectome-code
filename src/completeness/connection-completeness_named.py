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

# %% [markdown]
# Completeness is defined as the percentage of all synapses that are connected to identified optic lobe neurons. 
#
# See "..\queries\completeness.py" for the exact definition. 

# %% Project setup
from pathlib import Path
from utils.excel_exporter import ExcelExporter
from queries.completeness import fetch_ol_stats
from dotenv import find_dotenv
from utils import olc_client
c = olc_client.connect(verbose=True)

PROJECT_ROOT = Path(find_dotenv()).parent

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
