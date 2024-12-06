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
# # Local Demo
#
# In this notebook, we show basic functionality for one of our central classes, the `OLNeuron`. The notebook should run without the connection to `neuPrint` and should create an interactive plot for the skeleton and another interactive plot of the mesh for an Mi1 neuron. The Mi1 we use in the example does not exist in our data release, yet it is very similar to the neuron with the bodyId 56564 near the center of the medulla.

# %% [markdown]
# The following Jupyter cell contains the boiler plate code we use at the beginning of all notebooks. It defines the location of the `PROJECT_ROOT` and adds the directory `src` to the path. This is a workaround to having to install our package and runs directly on the source file. 

# %%
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
from utils import olc_client
c = olc_client.connect()

# %% [markdown]
# Import the necessary components to run this notebook. Notably this is the external ["**N**euron **A**nalysis and **Vis**ualization (NAVis)"](https://navis.readthedocs.io/) library and our class "OLNeuron".

# %%
import navis
from utils.ol_neuron import OLNeuron

# %% [markdown]
# The next Jupyter cell instanciates our `OLNeuron` class with the bodyId `5000`. This body ID does not exist in neuPrint, instead we provide a local copy of the relevant files with the code to run this file.
#
# The next two lines load the skeleton and mesh for this specific neuron.
#
# With access to the neuPrint database, any other existing bodyId can be used to instantiate the `OLNeuron` class and you will get access to enhances functionality, for example to find the hemisphere via `oln.hemisphere`, the list of innervated ROIs via `oln.innervated_rois()`, or the number of synapses via `oln.synapse_count()`.

# %%
oln = OLNeuron(5000)

print(f"The body ID of the neuron is {oln.get_body_id()}")

print("Getting skeleton")
skel_5000 = oln.get_skeleton()

print("Getting mesh")
mesh_5000 = oln.get_mesh()

# %% [markdown]
# The next two cells provide a visual representation of the skeleton and the mesh for the fictive Mi1 cell with bodyId 5000.

# %%
print("Attempt to plot skeleton. This works best in Jupyter.")
navis.plot3d(skel_5000)

# %%
print("Attempt to plot mesh. This works best in Jupyter.")
navis.plot3d(mesh_5000)

# %%
