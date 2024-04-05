"""
Plot the hex1/hex2 data frame as a 2D histrogram.

Functions extracted from @hoellerj's Edge_hex_ids.ipynb.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_cushion_plot(
    df:pd.DataFrame
  , hex1_min:int, hex1_max:int
  , hex2_min:int, hex2_max:int
  , verbose:bool=True
) -> plt.Figure:
    """
Create and return a plot based on the cushion data frame

The advantages of returning a plot from a function to the notebook are as follows:

- The layout can be modified in the notebook where it is plotted.
- You can do several different things with the plot, e.g. show it in the notebook and also 
    write it to disc.
- consistent layout changes across different plots in the paper are easier to apply.
- changing the plotting library (if necessary) is easier inside a function with a well defined 
    interface  than having to modify a lot of notebooks.

For example, if you only want to output the result in a file, you could write a cell like this 
(the `capture` juputerlab magic suppresses matplotlbs plotting.

```
%%capture
fig.savefig(data_path / "cushion_ME_1_36_1_39.pdf")
```

Parameters
----------
df : pd.DataFrame
    pandas DataFrame with columns `hex1_id` and `hex2_id`.
hex1_min : int
    min of hex1 range to show in plot
hex1_max : int
    max of hex1 range to show in plot
hex2_min : int
    min of hex2 range to show in plot
hex2_max : int
    max of hex2 range to show in plot

Returns
-------
fig : plt.Figure
    handle to the figure
    """
    fig, ax = plt.subplots(figsize=(8,8), facecolor='white')
    h, xbins, ybins, _ = ax.hist2d(
        x=df['hex1_id']
      , y= df['hex2_id']
      , bins=[
            np.linspace(hex1_min-.5,hex1_max+.5,hex1_max-hex1_min+2),
            np.linspace(hex2_min-.5,hex2_max+.5,hex2_max-hex2_min+2)]
      , edgecolors='k', linewidth=2
      , cmin=0, cmax=3
    )
    ax.set_xlabel('hex1_id')
    ax.set_ylabel('hex2_id')
    ax.set_xlim([hex1_min-.5,hex1_max+.5])

    return fig
