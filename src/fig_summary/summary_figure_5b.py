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
from madvisc.utils import olc_client
from madvisc.utils.instance_summary import InstanceSummary
from madvisc.utils.summary_plotter import SummaryPlotter

c = olc_client.connect(verbose=True)


# %%
# directory to save result figure
result_dir = Path(find_dotenv()).parent / 'results' / 'fig_summary'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
in_list = []
for ins in [
    'L2_R', 'L3_R', 'Tm1_R', 'Tm2_R'
  , 'Dm3a_R', 'Dm3b_R', 'Dm3c_R'
  , 'TmY4_R', 'TmY9a_R', 'TmY9b_R'
  , 'Li16_R', 'LC11_R', 'LC15_R']:
    
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
  , figure_title='Figure 5b'
)
fig = sp.plot()

file_name = "Figure-5b.pdf"
fig.write_image(
    result_dir / file_name
  , width=8.5 * 96 # inch * ppi
  , height=11 * 96
)

# %%
