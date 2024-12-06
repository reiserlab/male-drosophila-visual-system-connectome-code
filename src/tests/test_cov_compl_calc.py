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
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from IPython.display import display
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.trim_helper import TrimHelper
from utils.column_features_functions import hex_from_col, cov_compl_calc
from utils.metric_functions import get_completeness_metrics

from queries.completeness import fetch_ol_types_and_instances
from queries.coverage_queries import fetch_cells_synapses_per_col


c = olc_client.connect(verbose=True)

# %%
instance = 'Li37_R'
quant_df = get_completeness_metrics(instance=instance)

display(quant_df)

# %%
instance = 'aMe2_R'
trim_helper = TrimHelper(instance=instance)
named_df = fetch_ol_types_and_instances(side='both')
pop_size = named_df[named_df['instance']==instance]['count']\
    .values\
    .astype(int)
df = fetch_cells_synapses_per_col(
    cell_instance=instance
  , roi_str=["ME(R)", "LO(R)", "LOP(R)", "AME(R)", "LA(R)"]
)
df = hex_from_col(df)
quant_df = cov_compl_calc(
    df
  , trim_df=trim_helper.trim_df
  , size_df=trim_helper.size_df
  , size_df_raw=trim_helper.size_df_raw
  , n_cells=pop_size
  , instance_type=instance
)

display(quant_df)

# %%
