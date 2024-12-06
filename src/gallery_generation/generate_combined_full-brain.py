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

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

# %load_ext autoreload
# %autoreload 2

from utils.pdf_figure_functions import generate_group_pdf, check_for_imgs, check_for_jsons

from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# Generate combined PDFs of VCN / VPN group plots.

# %%
check_for_imgs(plot_type='group')
check_for_jsons(plot_type='group')

# %%
pdf_specs = {
    "pdf_w": 216
  , "pdf_h": 280
  , "pdf_res": 96
  , "font_size": 5
  , "pdf_margin": [0, 0]
}

# %%
for main_group in ["VPN", "VCN"]:
    generate_group_pdf(plot_type=main_group, pdf_specs=pdf_specs)
