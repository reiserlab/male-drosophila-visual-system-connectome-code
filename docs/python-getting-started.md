# Getting started

Running our code requires Python and a number of Python libraries. We suggest to use [Pixi](https://pixi.sh), which provides a simple way to manage dependencies across platforms. This suggestion differs from our earlier experiments using [conda](https://docs.conda.io/), [mamba](https://mamba.readthedocs.io), and [pip](https://pip.pypa.io).

## Install pixi

The [Pixi website](https://pixi.sh) provides some good guides on how to install Pixi. Please follow those instructions, but for the impatient here are the commands to install it on Linux and Mac: `curl -fsSL https://pixi.sh/install.sh | bash` or Windows inside the PowerShell: `iwr -useb https://pixi.sh/install.ps1 | iex`. Pixi is a command line tool and you confirm a successfull installation by typing `pixi --version`, which should give you a version number.

## Run shell or code

With pixi installed, you should be able to start a Pixi shell with `pixi shell`. The first time you run this command it will take some time since Pixi will download all dependencies. The Pixi shell is very similar to having a terminal inside a virtual environment. All dependencies, as defined in the `pyproject.toml` file, should be available. You should be able to open any of our notebooks, for example via `jupyter lab src/python-bootcamp/local_demo.ipynb` and run code via `python src/python-bootcamp/local_demo.py`.

You can also run any code without entering the Pixi shell by telling Pixi to run that code. Pixi will then make sure that all dependencies are available. For example, you can open the same notebook via `pixi run jupyter lab src/python-bootcamp/local_demo.ipynb` or run code via `pixi run python src/python-bootcamp/local_demo.py`.


## Project configuration

Access to the connectome data via neuprint requires authentication. To get the example script to work, create a copy of the `.env-sample` file with the name `.env` inside your _project root_. To add your own credentials, you will need to go to the account page at [neuPrint](https://neuprint-cns.janelia.org/account) and copy the (very long) AuthToken to the line starting with `NEUPRINT_APPLICATION_CREDENTIALS=`. Double check that the other two variables are set correctly.

Setting the plotting backend for navis via the environment is convenient as well. For example, to use the plotly library, add this:

```sh
# Configuration of a Jupyter backend for 3D plots
NAVIS_JUPYTER_PLOT3D_BACKEND=plotly
```

## Run the analysis

Now you should be able to run the first analysis by executing `pixi run python src/completeness/connection-completeness_named.py` (if you started a pixi shell before, it is sufficient to run `python src/completeness/connection-completeness_named.py`). Fairly quickly you should see some output describing your configuration and database connection. The analysis takes some time, but after a few minutes the whole output might look like this:

```shell
$ pixi run python src/completeness/connection-completeness_named.py

project root directory: $HOME/Projects/male-drosophila-visual-system-connectome-code
Connected to https://neuprint.janelia.org[optic-lobe:v1.1].
Client: neuprint-python v0.1.0
User: loeschef@janelia.hhmi.org [noauth]

successfully exported results/completeness/2024-09-23T02-35-00_output-connection-completeness_named.xlsx
```

You should see the new Excel spreadsheet in the folder `results/completeness/` inside your _project root_.

## File structure

You might have noticed that the folder `$PROJECT_ROOT/src/completeness` contains two files starting with `connection-completeness_named`: one python file with the extension `.py` and one Jupyter notebook file with the extension `.ipynb`. These two files are connected: if the Python file is edited, this change will automatically be pulled into the notebook the next time it is opened. Also, if edits are done inside the notebook file, the changes will be saved to the Python file.

The connections is done through the [JupyText](https://jupytext.readthedocs.io) software which combines the best of at least three worlds: 1) easy tracking of changes in text files, 2) saving input and output in the notebook format, 3) execution of whole scripts without complicated environments. I will get into more details about this in the guide about [Python editors](python-editors-getting-started.md), but here I just wanted to explain why there are two files with the same name and similar extensions in the example folder.

## Final words

If any of the above failed and you can't fix it, then please [get in contact](mailto:loeschef@janelia.hhmi.org).
