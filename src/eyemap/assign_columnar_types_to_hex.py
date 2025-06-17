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
from dotenv import find_dotenv
from madvisc.queries.hex_assigned import get_assigned_columnar_types
from madvisc.utils import olc_client

c = olc_client.connect(verbose=True)

# %%
output_path = Path(find_dotenv()).parent / "results" / "supp_table" / "Sup-Table-3_Columnar-cell-type-locations.xlsx"

# %%
display(get_assigned_columnar_types(output_path=output_path))

# %%
