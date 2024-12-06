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
import sys

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils import olc_client
from utils.instance_summary import InstanceSummary
from utils.summary_plotter import SummaryPlotter

c = olc_client.connect(verbose=True)


# %%
# directory to save result figure
result_dir = PROJECT_ROOT / 'results' / 'fig_summary'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
in_list = []

fig_8_list = [
    'LPLC4_R'
  , 'Mi10_R', 'Mi2_R', 'Mi13_R', 'Cm1_R', 'Pm6_R', 'Li14_R', 'LPi3b_R'
  , 'Tm5Y_R', 'Tm5b_R', 'Tm6_R', 'Tm34_R', 'T2a_R', 'MeLo7_R', 'Tlp12_R']

for instance in fig_8_list:
    in_list.append(
        InstanceSummary(
            instance
          , connection_cutoff=None
          , per_cell_cutoff=1.0
        )
    )

sp = SummaryPlotter(
    instance_list=in_list
  , figure_title='Figure 8f'
)
fig = sp.plot()

fig.write_image(
    result_dir / "Figure-8f.pdf"
  , width = 8.5 * 96    # inch * ppi
  , height = 11 * 96
)
