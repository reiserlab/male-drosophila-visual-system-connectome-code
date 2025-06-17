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
from madvisc.utils.website_functions import create_all_scatter_html
from madvisc.utils import olc_client
c = olc_client.connect(verbose=True)

# %%
create_all_scatter_html()
