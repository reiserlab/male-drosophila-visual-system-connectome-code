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
#     name: python3
# ---

# %% [markdown]
# # How to wrangle with pandas
#
# Questions to: Frank
#
# [Pandas](https://en.wikipedia.org/wiki/Pandas_(software)) is a Python library that provides data structures for manipulating tables (think spreadsheet or database tables for Python). It is built on top of another library [NumPy](https://en.wikipedia.org/wiki/NumPy), a library introducing arrays and matrices to Python.
#
# Pandas offers two main data types: [Series](https://pandas.pydata.org/docs/user_guide/dsintro.html#series), a one-dimensional array for any Python data type and [DataFrame](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe), a two-dimensional structure of several Series. DataFrames are heavily inspired and very similar to the R [data.frame](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/data.frame) or the tidyverse [tibble](https://tibble.tidyverse.org/).
#
# Many modern Python libraries ([plotting](https://matplotlib.org/), [machine learning](https://scikit-learn.org/), other [scientific libraries](https://scipy.org/)) are compatible with pandas. Pandas are [first-class citizens](https://en.wikipedia.org/wiki/First-class_citizen) in [neuprint-python](https://github.com/connectome-neuprint/neuprint-python). Below I show how to navigate pandas DataFrames and some basic operations.

# %% Project setup
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
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %% [markdown]
# ## Download data from Neuprint
#
# The following cell creates a connection to [neuprint-cns](https://neuprint-cns.janelia.org) (and stores the connection in `c`). The following line creates a [NeuronCriteria](https://connectome-neuprint.github.io/neuprint-python/docs/neuroncriteria.html) to define all neurons that are associated with the medula. The criteria is stored in the `medula_criteria` variable. Finally, I download all neurons and associated ROI information to the `neuron_me_df` and `roi_counts_df` variables.
#
# The intention of this cell is to get large data sets which I can then use to demonstrate how to wrangle with pandas.
#
# Running the cell should give you the verbose output from the `olc_client.connect()` function, telling you the database, client, and user credentials you connected to.

# %%
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons
from utils import olc_client


c = olc_client.connect(verbose=True)

medula_criteria = NC(rois="ME(R)")

neuron_me_df, roi_counts_df = fetch_neurons(medula_criteria, client=c)

# %% [markdown]
# The next step is not really necessary since pandas is automatically imported by `neuprint-python`, but I want to map the 6-character name `pandas` to the much shorter alias `pd`. This is almost considered a standard for using pandas…
#
# Also, for all the output generated by pandas, I am not a big fan of the the "scientific output" but rather want to see float numbers printed non-scientifically with two values behind the comma. You can comment that out, if you prefer scientific for your display.

# %%
pd.options.display.float_format = '{:.2f}'.format

# %% [markdown]
# ## Basic information
#
# First, let's confirm the data type of the variables `neuron_me_df` and `roi_counts_df`. These were returned by the `neuprint.fetch_neurons()` function, other `fetch_*` functions will return similar data types.
#
# To get the data type of any Python object you can use the Python built-in [`type()` function](https://docs.python.org/3/library/functions.html#type). The [`f""` string](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) syntax allows you to mix strings and variables, any code within `{}` is executed and the output becomes part of the string (kind of like `sprintf` in other languages.

# %%
print(f"neuron_me_df is of type {type(neuron_me_df)} and roi_counts_df of type {type(roi_counts_df)}")

# %% [markdown]
# Yay, they are actually panda `DataFrame`s! Let's focus on the `neuron_me_df` for now (and I'll use `df` in this documentation to refer to any DataFrame...
#
# How big is our data frame? [`df.shape` is a property of the DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html#pandas.DataFrame.shape) and will return a tuple with the (number of rows, number of columns). Accessing the first and second value of the tuple (note: Python starts indexing at 0) gives you the rows and columns respectively.
#

# %%
print(f"neuron_me_df has {neuron_me_df.shape[0]} rows and {neuron_me_df.shape[1]} columns")

# %% [markdown]
# Since we know it is a DataFrame, we can also use the `df.info()` to get more information about `neuron_me_df`. This will tell us the data type of the variable and some information about the data types – eg that the data type of the column `bodyID` is of `int64` and that others, like `roiInfo` are of type `object` (which is a kind of place holder at this point). It also tells us the memory usage of the whole data structure.

# %%
print(neuron_me_df.info())

# %% [markdown]
# Another (more or less) useful function is `df.describe()` which generates descriptive statistics for all the number columns (either `float64` or `int64` in the previous `df.info()` output). It tells you home many variables you have for each column and their mean, min, max, and quartiles. This might be partially useful for `pre`, `upstream` etc, really useful for `somaRadius`, and to be ignored for `bodyId`.

# %%
neuron_me_df.describe()

# %% [markdown]
# ## Accessing data
#
# The DataFrame is a table with rows and columns. The rows are addressed by the `index`, which is a key to access rows. The `neuprint.fetch_neuron()` function adds a counter to the table, which starts a 0 for the first row and increases by 1 for each row. The properties `df.index` and `df.columns` expose more information about the _row names_ and _columns names_ of the DataFrame.
#
# With the `df.loc` and `df.iloc` properties and the `df.at()` function, you can access cells.
#
# `df.loc` uses Python's the array syntax to access cells via the index and column names. For example, `df.loc[0:2, 'bodyId':'type']` gives you a DataFrame with the index including 0 and 2 (in this case the first three rows), and the columns including 'bodyId' and 'type' (the first three columns). Either row or column parameter can be a single value, a range by specifying start __COLON__ end, an array (eg `['bodyId', 'instance', 'type']`), or everything (wildcard) by writing just a __COLON__. For example, `df.loc[:,'bodyId']` would return all values from the 'bodyId' column. _Note_: Since the index and columns names refer to the discrete the start and end values are included in the selection, which is different from most other Python functions that use ranges.
#
# Instead of the index or column names, `df.iloc` uses the location in the table. To get the first 3 rows and columns, the corresponding code would be `df.iloc[0:3, 0,3]`. This is similar to most other Python functions using ranges and is easier to remember if you think as the x- and y-indices of the tables as continuous variables.
#
# `df.at` is a faster access to a single cell, which might only get relevant if you have performance issues in a function  with many individual cell accesses. There is a `df.iat` property similar ot the `df.iloc`.

# %%
print(f"The index is: {neuron_me_df.index} and columns are {neuron_me_df.columns}")
print(80*"+")
print(neuron_me_df.loc[0:2,'bodyId':'type'])
print(80*"+")
print(neuron_me_df.iloc[0:3,0:3])
print(80*"+")
print(f"The bodyID for the first row is {neuron_me_df.at[0,'bodyId']}")

# %% [markdown]
# You can look at the first few items with the [`df.head()` function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) (by default it gives you the first five items). This is basically the same as calling `df.iloc[0:5]`.

# %%
neuron_me_df.head()
# same output as neuron_me_df.iloc[0:5]

# %% [markdown]
# …or the last three items (by specifying the number of items in the [`df.tail()` function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html)). This produces the same output as `df.iloc[-3:]`.

# %%
neuron_me_df.tail(3)
# same output as neuron_me_df.iloc[-3:]

# %% [markdown]
# If you want to have a subset of a DataFrame, for exmple when developing a function or for faster plotting, getting a random sample instead of the top or bottom entries is the better solution. You can get a subsample of any size with `df.sample()` function. Instead of a fixed number or results (`df.sample(5)` or `df.sample(n=5)`) you can also get a percentage of your original dataset, eg half a permille (`df.sample(frac=0.00005)`, which should be around 8 neurons).  The sample will return a different set each time you run it.

# %%
neuron_me_df.sample(frac=0.00005)

# %% [markdown]
# Since the output generated by any of the above methods is a DataFrame (or panda Series), it's easy to chain several of these operations together. Not that the following example would make much sense, but this is just to demonstrate this feature.
#
# Let's say we want to get a random sample from our data set and then get the `bodyId` and `type` for the three first items from that subsample. You will notice that the number in the row index column (first column) now is a random number. Since we don't know which indices the `df.sample()` function will select, access via `df.loc()` is not easily possible and I decided to use `df.iloc()` instead.
#

# %%
# The following might be easier to read than: 
# neuron_me_df.sample(20).iloc[0:2, [0, 2]]
# (especially if these lines get longer)

neuron_me_df\
    .sample(20)\
    .iloc[0:2, [0, 2]]

# %% [markdown]
# ## Conditional access
#
# In addition to location based access to the DataFrame, access to a certain subset has many applications. To do that, let's take a step back: Instead of a range, `df.loc` also accepts a list of boolean values of the same length as the DataFrame. A value of `True` means the row will be selected, `False` removes that row. For example, to get the 4th item, one could provide an array with all values `False` except the 4th (at index position 3) which is `True`:
#

# %%
select_rows = [False] * neuron_me_df.shape[0]
select_rows[3] = True
neuron_me_df.loc[select_rows]  # same as neuron_me_df[select_rows]

# %% [markdown]
# Since comparisons of a column produce arrays with boolean values, this mechanis can be used for accessing specific rows. For example I could create an array by comparing the column `type` equal to `"T4"`. This will create an array (or technically a pandas Series…) with `True` for the rows where `type=="T4"`. If I now use this list of indices to filter the DataFrame, I end up with a subset of T4 neurons (and I am storing it for future use in `t4_neuron_df`).

# %%
t4_neuron_indices = neuron_me_df['type'] == "T4a"
print(t4_neuron_indices)  # show the list of indices

print(80*"+")

t4_neuron_df = neuron_me_df[t4_neuron_indices]
t4_neuron_df\
    .head() # show the beginning of the DataFrame

# %% [markdown]
# Often you will see the conditional subsetting written in only one line, as the following examples demonstrate:

# %%
neuron_me_df[neuron_me_df['type'] == "T4a"]\
    .head()

# neuron_me_df[neuron_me_df['somaRadius'] < 320]   # Neurons with small soma

# %% [markdown]
# There are also operations that directly work on the DataFrames, such as the comparison for `Null` or `NaN` (`df.isnull()`, `df.isnotnull()` or `df.isna()`, `df.isnotna()`) as well as boolean operations such as `df.all()` or `df.any()`. Furthermore one can write boolean queries using the `df.query()` function.

# %%
# from the DataFrame
# select columns type and instance
# which are not null
# for both (all)
# 
# Then query resulting DataFrame for rows where type and instance differ

neuron_me_df[\
        neuron_me_df[['type', 'instance']]\
             .notnull()\
             .all(axis=1)\
    ]\
    .query('type != instance')

# %% [markdown]
# ## Change the DataFrame
#
# When `neuprint.fetch_neuron()` returns a DataFrame, the index is an increment starting at 0. There is no guaranteed order in which the neurons are returned (although in most cases it is sorted by `bodyId`). Therefore the number used in the index is meaningless outside the fetch function. Instead, it might be better to use the `bodyId` as an index. You can define a column as the index through `df.set_index()` (and return to an increment as index via `df.reset_index()`). In theory, indices don't have to be unique – for example you could define the column `type` as an index – but that will create issues if you use the index to access rows. Since the `bodyId` is unique, this shouldn't be an issue. Using the `bodyId` as an index in this particular DataFrame also gives us a list of neurons whenever we do conditional selections. For example, the output of the comparison for `neuron_bodyid_df` below is immediately useful -- you could, for example, copy&paste a `bodyId` for a `True` value to [neuprint](https://neuprint-cns.janelia.org) to look at an L1 neuron.

# %%
neuron_bodyid_df = neuron_me_df.set_index('bodyId')

(neuron_bodyid_df['type'] == "L1")\
    .sort_values()\
    .tail(3)

# %% [markdown]
# If you want to rename columns, you can use the `df.rename()` function with the `columns` argument and a dictionary, where the key of the dictionary is the old name and the value is the new name. For example, if you want to rename the column `instance`, you could write:

# %%
neuron_me_df\
    .rename(columns={'instance': 'Instance Names'})\
    .head(3)

# %% [markdown]
# ## Aggregation and transformation
#
#

# %%
# Use type and instance as grouping parameters
# then get the synweight column
# then aggregate the column(s) using the 'mean' function
# then reset the index (to remove type and instance from index)


neuron_me_df\
    .groupby(by=["type"])\
        ['synweight']\
        .agg('mean')\
        .reset_index()\
        .sample(10)\
        .sort_index()

# %%