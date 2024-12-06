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

# %% [markdown]
# # Access to Neurons

# %%
"""
This cell does the initial project setup for the Jupyter notebook.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
from utils import olc_client
c = olc_client.connect()

# %%
from utils.ol_neuron import OLNeuron

# %% [markdown]
# ## Define a neuron by bodyId

# %%
bid = 100007

# %% [markdown]
# ## Basic information about the Neuron

# %%
example_neuron = OLNeuron(bid)
syn_count = example_neuron.synapse_count()
print(f"Neuron {bid} is a {example_neuron.get_type()} in the {example_neuron.hemisphere} hemisphere and has {syn_count[0]} pre and {syn_count[1]} post synapses.")

# %% [markdown]
# ## List all the innervated ROIs

# %%
example_neuron.innervated_rois()

# %% [markdown]
# ## Retrieve the skeleton

# %%
skel = example_neuron.get_skeleton()

# %% [markdown]
# ## Save the skeleton to a file
#
# Here we create a `1000007.swc` file in the project root that contains the skeleton.

# %%
swc_filename = PROJECT_ROOT / f"{bid}.swc"
skel.to_swc(swc_filename)

# %% [markdown]
# ## Retrieve the Mesh

# %%
mesh = example_neuron.get_mesh()

# %% [markdown]
# ## Save the Mesh
#
# The following example code stores the mesh in a Wavefront `.obj` file (and the `mesh_obj` variable). Here we use the `export_mesh` function from the [Trimesh](https://trimesh.org) library. This will create a `1000007.obj` file in the project root directory.

# %%
from trimesh.exchange.export import export_mesh

tri_mesh = mesh[0].trimesh
obj_filename = PROJECT_ROOT / f"{bid}.obj" 

mesh_obj = export_mesh(tri_mesh, obj_filename)
