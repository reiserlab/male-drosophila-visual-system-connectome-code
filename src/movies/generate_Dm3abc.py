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
from utils.dm3_movie_functions import generate_movie_description, get_body_id_by_hex
from utils import olc_client

c = olc_client.connect(verbose=True)

# %% [markdown]
# Generate the json files to create movies for all 3 Dm3 cell types

# %%
# choose the colors to use for each cell type
ctype_dict = {"Dm3a": "red", "Dm3b": "blue", "Dm3c": "purple"}

# %%
for cell_type in ["Dm3a", "Dm3b", "Dm3c"]:
    # Generate df with ordered bodyIds
    ids_df =  get_body_id_by_hex(cell_type)

    generate_movie_description(
        cell_type=cell_type,
        df=ids_df,
        template="Dm3-template.json.jinja",
        number_of_neighbors=5,
        color=ctype_dict[cell_type],
        stripes=True,
    )

# %%
