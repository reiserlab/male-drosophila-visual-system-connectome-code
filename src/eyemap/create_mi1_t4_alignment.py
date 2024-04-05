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
#     name: ol-connectome
# ---

# %%
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from utils.align_mi1_t4 import create_alignment

# %%
create_alignment()

# Expected run time: 12 min
