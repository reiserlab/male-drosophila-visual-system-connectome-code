# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
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
import numpy as np

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

# %load_ext autoreload
# %autoreload 2

from utils.pdf_figure_functions import generate_gallery_pdf, check_for_imgs, check_for_jsons
from utils.ol_types import OLTypes

from utils import olc_client
c = olc_client.connect(verbose=True)

# %% [markdown]
# Generate combined PDFs of gallery plots for all neuron types

# %%
plot_type = 'gallery'
check_for_imgs(plot_type=plot_type)
check_for_jsons(plot_type=plot_type)

# %%
pdf_specs = {
    "pdf_w": 216
  , "pdf_h": 280
  , "pdf_res": 96
  , "font_size": 7
  , "pdf_margin": [0, 3]
}

rows_cols = [6, 4]
n_per_pdf = rows_cols[0] * rows_cols[1]

# %%
nudge_dict = {
    'Cm1_R': -2, 'Cm3_R': 7, 'Cm5_R': 3, 'Cm7_R': 7, 'Cm8_R': 6, 'Cm18_R': -2, 'Dm8b_R': 9, 'Dm10_R': -2
  , 'Dm11_R': 6, 'Dm14_R': 4, 'Dm18_R': -3, 'HBeyelet_R': -3, 'Mi10_R': -2, 'Mi15_R': -2, 'Mi16_R': 7
  , 'Pm4_R': 4, 'MeTu3b_R': 7, 'T1_R': -2, 'TmY4_R': 3, 'TmY13_R': 7, 'TmY14_R': 7, 'TmY15_R': 7
  , 'TmY16_R': 7, 'TmY36_R': 7, 'TmY37_R': -2, 'Tm6_R': -2, 'Tm12_R': 6, 'Tm1_R': -2, 'Tm2_R': -2
  , 'Tm3_R': -2, 'Tm4_R': -2, 'Lawf1_R': 7, 'Lawf2_R': -4, 'Tm20_R': 7, 'Tm34_R': 7, 'Tm36_R':-2
  , 'Tm37_R':-1, 'Tm38_R':6, 'Tm39_R':7, 'Tm40_R': 7, 'TmY3_R': 7, 'TmY10_R': 7
}

# %%
olt = OLTypes()
main_groups = ["OL_connecting", "OL_intrinsic", "VCN", "VPN", "other"]


page_number = 0

for main_group in main_groups:
    df_type = olt.get_neuron_list(primary_classification=main_group, side="both")
    inst_neurons = df_type["instance"].to_list()

    n_imgs = len(inst_neurons)
    n_pdfs = int(np.ceil(n_imgs / n_per_pdf))

    for idx, instances in enumerate(inst_neurons[i * n_per_pdf : (i + 1) * n_per_pdf] for i in range(n_pdfs)):
        generate_gallery_pdf(
            pdf_specs=pdf_specs
          , page_idx=page_number
          , instances=instances
          , nudge_dict=nudge_dict
          , rows_cols=rows_cols
        )
        page_number += 1

# %%
