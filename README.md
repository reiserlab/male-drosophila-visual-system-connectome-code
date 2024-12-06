# Optic Lobe Connectome

This repository is a collection of code for analyzing the optic lobe in the new Male Brain dataset ([Janelia FlyEM](https://neuprint-cns.janelia.org/?dataset=cns&qt=findneurons)). At this point it is only intended for internal use -- if you can see the repository, the [Reiser lab](https://www.janelia.org/lab/reiser-lab/) has invited you to contribute to the effort.

## Infrastructure

To allow others in the team to replicate the analysis, the code should be human-readable and run on several systems. 

### Set up a Python environment

In the guide on ["Getting started in Python"](docs/python-getting-started.md), we provide a step-by-step documentation on how to set up a working environment and run you through a first analysis. In essence, we suggest to install [pixi](https://pixi.sh) and create a copy of the `.env-sample` file with the name `.env`. After completion, you can open a Jupyter notebook via `pixi run jupyter lab src/python-bootcamp/local_demo.ipynb`. The first time this command is run, all necessary dependencies are installed automatically, which might take up to two minutes. Running the example notebook itself on local data inside the Jupyter Lab environment will not require a connection to neuPrint and takes less than a second to produce two interactive plots for a neuron. 

Changing the number of the bodyId is enough to run the same analysis on the most recent neuPrint database, but requires additional setup steps from our ["Getting started in Python"](docs/python-getting-started.md) guide. In our guide [Editing Python files](docs/python-editors-getting-started.md) we show two different ways to edit and run Jupyter notebooks.

Our software depends on Blender>3.6, Python>3.10, and a number of python libraries. The comprehensive list of python libraries and their versions are listed in the `pyproject.toml` file `navis`, `neuprint-python`, `kaleido`, `snakemake`, `pymupdf`, `snakemake-executor-plugin-lsf`, `cloud-volume`, `google-cloud-storage`, `fastcluster`, `ipykernel`, `numpy`, `nptyping`, `pynrrd`, `python-dotenv`, `alphashape`, `openpyxl`, `ipywidgets`, `nbformat`, `cmap`, `k3d`, `datrie`, `nbconvert`, `kneed`, `jupyterlab`, `jupytext`, and `pylint`.

Our software has been tested with Python 3.10 to 3.12 and Blender 3.6 to 4.2 on Linux, MacOS, and Windows.