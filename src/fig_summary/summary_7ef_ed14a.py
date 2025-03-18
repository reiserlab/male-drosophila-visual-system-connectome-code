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
from utils import olc_client
from utils.instance_summary import InstanceSummary
from utils.summary_plotter import SummaryPlotter

c = olc_client.connect(verbose=True)


# %%
# directory to save result figure
result_dir = Path(find_dotenv()).parent / 'results' / 'fig_summary'
result_dir.mkdir(parents=True, exist_ok=True)


# %%
def generate_synapse_pdf(ct_list:list[str], figure_title:str, pdf_fn:str):
    """
    Helper function to generate PDFs with synapse distributions for a list
    of celltypes.

    Parameters
    ----------

    ct_list : list
        List of cell instance names
    figure_title : str
        Name to print on top of the page
    pdf_fn : str
        filename inide the `result_dir`
    """
    in_list = []

    for instance in ct_list:
        in_list.append(
            InstanceSummary(
                instance
              , connection_cutoff=None
              , per_cell_cutoff=1.0
            )
        )

    sp = SummaryPlotter(
        instance_list=in_list
      , figure_title=figure_title
    )
    fig = sp.plot()

    fig.write_image(
        result_dir / pdf_fn
      , width=8.5 * 96    # inch * ppi
      , height=11 * 96
    )


# %%

fig_7_list = [
  'LPLC4_R', 'Mi10_R', 'Pm6_R', 'Tm6_R', 'Tlp12_R'
]

fig_ed14_list = [
    'Mi2_R', 'Mi13_R', 'Cm1_R',  'Li14_R', 'LPi3b_R'
  , 'Tm5Y_R', 'Tm5b_R',  'Tm34_R', 'T2a_R', 'MeLo7_R'
]

generate_synapse_pdf(fig_7_list, "Figure 7ef", "Figure-7ef.pdf")
generate_synapse_pdf(fig_ed14_list, "Figure ED 14a", "Figure-ED14a.pdf")

