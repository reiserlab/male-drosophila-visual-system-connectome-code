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
import fitz
from madvisc.utils import olc_client
from madvisc.utils.summary_plotter import SummaryPlotter
from madvisc.utils.instance_summary import InstanceSummary

c = olc_client.connect(verbose=True)
PROJECT_ROOT = Path(find_dotenv()).parent
print(f"Project root directory: {PROJECT_ROOT}")

# %%
for ins in [1]:
    print(ins)
    in_list = []
    # for instance in ['LPi12_R', 'LPi14_R']:
    toc = []
    for idx, instance in enumerate(['5-HTPMPV03_R', 'LoVP88_R', 'LoVP100_R', 'LoVP24_R', 'LoVP30_R', 'MeVP55_R', 'MeVP58_R'], start=1):
        in_list.append(InstanceSummary(instance))
        toc.append([1, instance, 1, (11*96/35)*(idx+1)])



    sp = SummaryPlotter(
        instance_list=in_list
      , figure_title='Test output from "examine SummaryPlotter"'
    #   , col_synapses = {
    #             'ME(R)':.15
    #           , 'LO(R)':.075
    #           , 'LOP(R)':.05
    #         }
    )
    fig = sp.plot()

    mdoc = fitz.Document(
        stream=fig.to_image(
            format='pdf'
          , width=8.5*96
          , height=11*96
        )
      , filetype='pdf'
    )
    mdoc.set_toc(toc)
    mdoc.save(PROJECT_ROOT / "cache" / "fig_summary" / "test.pdf")
    
