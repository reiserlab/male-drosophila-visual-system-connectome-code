"""
`neurontype_summary.py` is a command-line tool for generating summary plots
organized in pages of 24 cell instances each, grouped into five main categories:
"OL intrinsic", "OL connecting", "VPN", "VCN", and "other". Within each group,
instances are ordered alphabetically. Each of the five groups starts on a fresh
page.


Commands
---------

- 'count': Retrieves the total number of pages.
  Example: `python src/fig_summary/neurontype_summary.py count`
- 'get': Displays information such as cell instance name, main group, and group
  counter. Example: `python src/fig_summary/neurontype_summary.py get 30`
- 'plot': Plots the specified page into a PDF file.
  Example: `python src/fig_summary/neurontype_summary.py plot 30`


Options
--------

- '--per-page': Number of plots per page (default is 25).
- '--main-group': Specify main groups to include in count, get, or plot commands.


Background
-----------

Having a tool that generates figures independently enables efficient
parallelization. Using 'snakemake' for scheduling and monitoring individual
tasks, this tool breaks down large tasks into smaller ones, leading to
significant speedups.


Technology
----------
This script uses 'click' for argument parsing, with annotations specifying
function arguments and options.
"""


import sys
from pathlib import Path

import click

import fitz

import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(
    find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.ol_types import OLTypes
from utils.summary_plotter import SummaryPlotter
from utils.instance_summary import InstanceSummary
from utils import olc_client

def __from_notebook() -> bool:
    try:
        _ = get_ipython
        return True
    except NameError:
        return False


def __get_groups(per_page:int, main_group:list):
    assert isinstance(per_page, int), "per_page must be an integer"
    assert per_page > 0, "per_page must be positive"
    assert set(main_group) <= set(['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other'])
    all_types = pd.DataFrame()
    for maint in main_group:
        olt = OLTypes()
        lst = olt\
            .get_neuron_list(
                primary_classification=maint
              , side='both'
            )
        lst['group_counter'] = lst.index // per_page
        all_types = pd.concat([all_types, lst])
    all_types['group_counter_max'] = all_types.groupby(by=['main_groups'])['group_counter'].transform('max')
    all_types['total_counter'] = all_types.groupby(by=['main_groups', 'group_counter']).ngroup()
    return all_types


@click.command()
@click.option(
    '--per-page'
  , type=click.INT
  , default=25
  , help="Number of cell instances per page, default=25"
)
@click.option(
    '--main-group'
  , type=click.Choice(['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other'])
  , multiple=True
  , default=['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
  , help="main groups to include."
)
def count(per_page:int, main_group:list):
    """
    Count the number of pages that are required to generate all plots. This is useful to identify
    all necessary `group` numbers for the get and plot commands and to parallelize the process.
    """
    _ = olc_client.connect()
    grps = __get_groups(per_page=per_page, main_group=main_group)
    ret = grps['total_counter'].nunique()
    if __from_notebook():
        return ret
    click.echo(ret)


@click.command()
@click.option(
    '--per-page'
  , type=click.INT
  , default=25
  , help="Number of cell instances per page, default=25"
)
@click.option(
    '--main-group'
  , type=click.Choice(['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other'])
  , multiple=True
  , default=['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
  , help="main groups to include."
)
@click.argument("group", type=click.INT)
def get(group, per_page, main_group):
    """
    Return a list of instances that belong to the group with the specified number. Note that group
    is a 0-based count.
    """
    _ = olc_client.connect()
    assert group >= 0 ,\
        f"group must be >0, not {group}"
    grps = __get_groups(per_page=per_page, main_group=main_group)
    assert group <= grps['total_counter'].max(),\
        f"There are only {grps['total_counter'].max()} groups, not {group}"
    ret = grps[grps['total_counter']==group]
    if __from_notebook():
        return ret
    click.echo(ret)


@click.command()
@click.option(
    '--per-page'
  , type=click.INT
  , default=25
  , help="Number of cell instances per page, default=25"
)
@click.option(
    '--main-group'
  , type=click.Choice(['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other'])
  , multiple=True
  , default=['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN', 'other']
  , help="main groups to include."
)
@click.argument("group", type=click.INT)
def plot(group, per_page, main_group):
    """
    Plot a figure for the the group with the specified number. This step can take a long time and
    you can see which instance will be included by running the `get` command. Note that the group
    is a 0-based count.
    """
    my_client = olc_client.connect()
    result_dir = PROJECT_ROOT / 'results' / 'fig_summary'
    result_dir.mkdir(parents=True, exist_ok=True)
    assert group >= 0 ,\
        f"group must be >0, not {group}"
    grps = __get_groups(per_page=per_page, main_group=main_group)
    assert group <= grps['total_counter'].max(),\
        f"There are only {grps['total_counter'].max()} groups, not {group}"

    plot_group = grps[grps['total_counter']==group]
    instance_list = []
    toc_list = []
    row_counter = 1
    for _, plt_grp in plot_group.iterrows():
        in_sum = InstanceSummary(
            plt_grp['instance']
          , connection_cutoff=None
          , per_cell_cutoff=1.0
        )
        instance_list.append(in_sum)
        toc_list.append([1, plt_grp['instance'], 1, 33 + 29 * row_counter])
        row_counter += 1
    group_mapper = {
        'OL_connecting': "Optic Lobe Connecting Neurons"
      , 'OL_intrinsic': "Optic Lobe Intrinsic Neurons"
      , 'VCN': "Visual Centrifugal Neurons"
      , 'VPN': "Visual Projection Neurons"
      , 'other': "Other Visual Neurons"
    }

    fig_title = f"<b>{group_mapper[plot_group['main_groups'].head(1).values[0]]}</b> "\
        f"{plot_group['group_counter'].head(1).values[0] + 1}"\
        "&#8239;/&#8239;"\
        f"{plot_group['group_counter_max'].head(1).values[0] + 1}"
    pltr = SummaryPlotter(
        instance_list=instance_list
      , figure_title=fig_title
      , method='mean'
    )

    fig = pltr.plot()
    for file_type in ['pdf']:
        file_name = f"Summary_Group-{group:02d}.{file_type}"
        mdoc = fitz.Document(
            stream=fig.to_image(
                format='pdf'
              , width=8.5*96
              , height=11*96
            )
          , filetype='pdf'
        )
        mdoc.set_toc(toc_list)
        mdoc.save(result_dir / file_name)


@click.group()
def cli():
    """ 
    This script generates the summary plot for the supplements. The summary plot is currently
    split into five main groups (OL intrinsic, OL connecting, VPN, VCN, and other). Each page
    contains results for 25 instances (cell types per hemisphere) and each group starts on a
    new page. The `count` command returns the total number of pages needed for the plots, the
    `get` command lists the instances for each group, and the `plot` command generates the plots.


    Get more information:

    >>> neurontype_summary.py count --help

    or

    >>> neurontype_summary.py get --help

    """


cli.add_command(count)
cli.add_command(get)
cli.add_command(plot)


if __name__ == '__main__':
    cli()
