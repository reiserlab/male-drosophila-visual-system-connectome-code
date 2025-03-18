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

# %%
from pathlib import Path
from dotenv import find_dotenv

from utils.instance_summary import InstanceSummary
from utils.summary_plotter import SummaryPlotter
from utils.ROI_layers import load_roi_layer_params
from utils import olc_client

c = olc_client.connect(verbose=True)
PROJECT_ROOT = Path(find_dotenv()).parent
print(f"Project root directory: {PROJECT_ROOT}")


# %%
# directory to save result figure
result_dir = PROJECT_ROOT / 'results' / 'fig_summary'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
#load all instances that were used to define layer boundaries

layer_instances = []
for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
    _, _, cell_types, _, _, _ = load_roi_layer_params(roi_str=roi_str)
    unique_types = list(set(cell_types))
    for cell in unique_types:
        layer_instances.append(cell+'_R')

# %%
in_list = []
for ins in layer_instances:
    print(ins)
    for instance in [ins]:
        in_list.append(
            InstanceSummary(
                instance
              , connection_cutoff=None
              , per_cell_cutoff=1.0
            )
        )


sp = SummaryPlotter(
    instance_list=in_list
  , figure_title='Figure 3c'
)
fig = sp.plot()

file_name = "Layer_def.pdf"
fig.write_image(
    result_dir / file_name
  , width=8.5 * 96 # inch * ppi
  , height=11 * 96
)

# %% [markdown]
# Fig 3c is obtained from this output in
# [`results/fig_summary/Layer_def.pdf`](../../results/fig_summary/Layer_def.pdf) as follows:
#
# - take the relevant synapse distribution within one neuropil
# - rotate by 90 degrees
# - color certain horizontal gray lines in blue (pre) or orange (post)
