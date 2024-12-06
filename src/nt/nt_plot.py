# %% [markdown]
# # Make neurotransmitter related plots

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
import re
import sys
import math
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd
import numpy as np
from IPython.display import display

from scipy.optimize import curve_fit

from statsmodels.formula.api import ols

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

from utils.plotting_functions import plot_heatmap
from utils.metric_functions import get_metrics_df
from utils.ol_color import OL_COLOR
from utils.neurotransmitter import get_special_neuron_list, get_nt_for_bid

# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
# set dirs
result_dir = PROJECT_ROOT / 'results' / 'nt'
result_dir.mkdir(parents=True, exist_ok=True)
cache_dir = PROJECT_ROOT / 'cache' / 'nt'

# %%
# nt colors and types

pal_heatmap = OL_COLOR.HEATMAP.hex

nt_types = ['ACh', 'Glu', 'GABA', 'His', 'Dop', 'OA', '5HT']
pal_nt = OL_COLOR.NT.hex

nt_types7 = nt_types + ['unclear']
pal_nt7 = pal_nt + ["#FFFFFF"]

# %%
# set plotting format
fig_format = {
    'fig_width': 3
  , 'fig_height': 3
  , 'fig_margin': 0.01
  , 'export_type': 'svg'
  , 'font_type': 'arial'
  , 'fsize_ticks_pt': 6
  , 'fsize_title_pt': 6
  , 'markersize': 10
  , 'markerlinewidth': 1
  , 'markerlinecolor': 'black'
  , 'ticklen': 3.5
  , 'tickwidth': 1
  , 'axislinewidth': 1.2
  , 'save_path': result_dir
}

canvas_width = (fig_format['fig_width'] - fig_format['fig_margin']) * 96
canvas_height = (fig_format['fig_height'] - fig_format['fig_margin']) * 96
fsize_ticks_px = fig_format['fsize_ticks_pt'] * (1/72) * 96
fsize_title_px = fig_format['fsize_title_pt'] * (1/72) * 96

layout_scatter = go.Layout(
    paper_bgcolor='rgba(255,255,255,1)'
  , plot_bgcolor='rgba(255,255,255,1)'
  , autosize=False
  , width=canvas_width
  , height=canvas_height
  , margin={
        'l': canvas_width//16
      , 'r': canvas_width//16
      , 'b': canvas_height//16
      , 't': canvas_height//16
      , 'pad': 4
    }
  , showlegend=False
)

layout_xaxis_scatter = go.layout.XAxis(
    title_font={
        'size':fsize_title_px
      , 'family':fig_format['font_type']
      , 'color' : 'black'
    }
  , ticks='outside'
  , ticklen=fig_format['ticklen']
  , tickwidth=fig_format['tickwidth']
  , tickfont={
        'family':fig_format['font_type']
      , 'size':fsize_ticks_px
      , 'color' : 'black'
    }
  , showgrid = False
  , showline=True
  , linewidth=fig_format['axislinewidth']
  , linecolor='black'
)

layout_yaxis_scatter = go.layout.YAxis(
    title_font={
        'size':fsize_title_px
      , 'family': fig_format['font_type']
      , 'color' : 'black'
    }
  , ticks='outside'
  , ticklen=fig_format['ticklen']
  , tickwidth=fig_format['tickwidth']
  , tickfont={
        'family':fig_format['font_type']
      , 'size':fsize_ticks_px
      , 'color' : 'black'
    }
  , showgrid = False
  , showline=True
  , linewidth=fig_format['axislinewidth']
  , linecolor='black'
  , scaleanchor="x"
  , scaleratio=1
)

# %% [markdown]
# # load data

# %%
# load data from nt_query_neuprint.ipynb
neuron_df = get_special_neuron_list()

replacer = {
    'dopamine': 'Dop'
  , 'serotonin': '5HT'
  , 'acetylcholine': 'ACh'
  , 'glutamate': 'Glu'
  , 'gaba': 'GABA'
  , 'histamine': 'His'
  , 'octopamine': 'OA'
}

# change name from "dopamine" to "Dop" and "serotonin" to "5HT" and so on
neuron_df['predictedNt'] = neuron_df['predictedNt'].replace(replacer)
neuron_df['consensusNt'] = neuron_df['consensusNt'].replace(replacer)
neuron_df['celltypePredictedNt'] = neuron_df['celltypePredictedNt'].replace(replacer)

# %%
nt_body = neuron_df[[
    'bodyId', 'instance', 'type'
  , 'main_groups'
  , 'pre', 'post', 'downstream', 'upstream'
  , 'consensusNt']].copy()

nt_celltype = nt_body\
    .groupby(['type', 'main_groups'])\
    .agg(
        cell_count=('bodyId', 'nunique')
      , pre_count=('pre', 'sum')
      , post_count=('post', 'sum')
      , nt = ('consensusNt', lambda x: x.iloc[0]) # picking the fist value of the list suffices
    )\
    .reset_index()

# change column name
nt_body = nt_body.rename(columns={'consensusNt':'nt'})

# %% [markdown]
# nt prediction based on Eckstein et al 2024
# (DOI [10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016)).
# See Methods sections "Training and evaluation data for transmitter predictions" and
# "Neurotransmitter prediction" for details.

# %%
# load nt confusion matrix data
confusion_df = pd.read_csv(
    PROJECT_ROOT / 'params' / 'confusion_df.csv'
  , index_col=0
)
confusion_df.columns.name = 'prediction'
gt_count = pd.read_csv(
    PROJECT_ROOT / 'params' / 'gt_count.csv'
  , index_col=0
)

# %% [markdown]
# FISH data

# %%
fish = pd.read_excel(
    PROJECT_ROOT / 'params' / 'Nern-et-al_SuppTable05_Neurotransmitter_validation.xlsx'
)

# change column names Inferred transmitter to transmitter
fish = fish.rename(columns={
    'Inferred transmitter': 'transmitter'
  , 'Cell Type':'cell_type'}
)
# remove training data
fish = fish[fish['Part_of_training_data'] == 'no'][['transmitter', 'cell_type']]
# remove co-trans and unclear
fish = fish[~fish['cell_type']\
    .isin(['[Cm11]', '[Pm2]', 'Mi15', 'CL357',  'l-LNv', 'MeVC27', 'OLVC4', 'T1'])
]

fish.loc[fish['cell_type'].isin(['HBeyelet', 'R8p', 'R8y']), 'transmitter'] = "His"

# remove duplicates
fish = fish.drop_duplicates()

# %% [markdown]
# ### define cell groups

# %%
# define groups
cell_groups = ['OL_intrinsic', 'OL_connecting', 'VPN', 'VCN']

# %% [markdown]
# ### load synapse data

# %%
## Expected runtime on cold cache: 40…50min
syn = get_nt_for_bid(neuron_df)
syn_raw = syn.copy()

# %% [markdown]
# ### override with cell nt

# %%
syn.head()

# %%
syn = pd.merge(
    syn[['bodyId', 'x', 'y', 'z', 'layer_roi']]
  , nt_body[['bodyId', 'type', 'instance',  'main_groups', 'nt']]
  , on='bodyId'
  , how='left'
)

# %% [markdown]
# # TmY neurons

# %%
TmY_type = [
    "TmY3", "TmY4", "TmY5a", "TmY9a", "TmY9b", "TmY10"
  , "TmY13", "TmY14", "TmY15", "TmY16", "TmY17", "TmY18"
  , "TmY19a", "TmY19b", "TmY20", "TmY21"
]

# %%
# nt prediction
pred_tmy = pd.merge(
    syn_raw[['bodyId','nt']]
  , nt_body[['bodyId', 'type', 'instance',  'main_groups']]
  , on='bodyId'
  , how='left'
)
pred_tmy = pred_tmy[pred_tmy['type'].isin(TmY_type)]
pred_tmy.shape

# %%
# compute relative count for each cell type
cell_nt = pred_tmy.replace(['Dop', 'His',  'OA', '5HT'], 'Other')
cell_nt = cell_nt\
    .groupby(['type','bodyId', 'nt'])\
    .size()\
    .reset_index(name='counts')

cell_nt['rel_counts'] = cell_nt\
    .groupby('bodyId')['counts']\
    .transform(lambda x: x/x.sum())

cell_nt['instance'] = pd.Categorical(cell_nt['type'], categories=TmY_type)
cell_nt = cell_nt.sort_values('instance')

# %%
# plot 4 scatter plots one for each pred1, x-axis are cell types, y-axis are rel_counts

fig = go.Figure(layout = layout_scatter)

for color_name, color_value in OL_COLOR.NT_SHORT.map.items():
    xy = cell_nt[cell_nt['nt'] == color_name]
    fig.add_trace(
        go.Box(
            y=xy['rel_counts']
          , x=xy['instance']
          , name=color_name
          , marker_color=color_value
          , marker_size=1
          , line={'width': 1}
          , boxpoints='all' # can also be outliers, or suspectedoutliers, or False
          , jitter=0.3 # add some jitter for a better separation between points
          , pointpos=0 # relative position of points wrt box
          , showwhiskers=False
          , fillcolor="#FFFFFF"
        )
    )

fig.update_xaxes(layout_xaxis_scatter)
fig.update_yaxes(layout_yaxis_scatter)

fig.update_layout(
    showlegend=True
  , yaxis_title='rel counts'
  , boxmode='group' # group together boxes of the different traces for each value of x,
  , width=1000
  , height=300
)

fig.show()
pio.write_image(fig, file=result_dir / 'TmY_nt.svg')

# %% [markdown]
# # Spatial (21 layer rois) distr of syn

# %%
# layer rois
layer_rois = [
    'ME_R_layer_01'
  , 'ME_R_layer_02'
  , 'ME_R_layer_03'
  , 'ME_R_layer_04'
  , 'ME_R_layer_05'
  , 'ME_R_layer_06'
  , 'ME_R_layer_07'
  , 'ME_R_layer_08'
  , 'ME_R_layer_09'
  , 'ME_R_layer_10'
  , 'LO_R_layer_1'
  , 'LO_R_layer_2'
  , 'LO_R_layer_3'
  , 'LO_R_layer_4'
  , 'LO_R_layer_5'
  , 'LO_R_layer_6'
  , 'LO_R_layer_7'
  , 'LOP_R_layer_1'
  , 'LOP_R_layer_2'
  , 'LOP_R_layer_3'
  , 'LOP_R_layer_4'
  , 'AME(R)'
]

# %%
pred_layer = syn.copy()
# find matches for pred_layer['layer_roi'] in layer_rois, assign the match to pred_layer['layer']
pred_layer['layer_roi_idx'] = pred_layer['layer_roi']\
    .apply(lambda x: next((i for i, y in enumerate(layer_rois) if y in x), None))

# %%
display(pred_layer.shape)
# remove not assigned
# pred_layer = pred_layer[pred_layer['layer'] != 'NA']
pred_layer = pred_layer[pred_layer['layer_roi_idx'].notna()]
display(pred_layer.shape)
# remove None
pred_layer = pred_layer[pred_layer['nt'].notna()]
display(pred_layer.shape)

# %%
# make heatmap matrix 7x21
heatmap = pred_layer.groupby(['nt', 'layer_roi_idx']).size().unstack().fillna(0)
# reindex
heatmap = heatmap.reindex(nt_types7, axis='index')
# change column name
heatmap.columns = layer_rois

display(heatmap)

# %%
# sum up the last 5 rows and rename it as "others"
heatmap.loc['others'] = heatmap.iloc[-5:].sum(axis=0)
heatmap = heatmap.drop(nt_types7[-5:])
display(heatmap)


# %%
# # annotation
# convert larger numbers to k or M but always keep 4 digits

def decimals(v):
    return max(0, min(2,2-int(math.log10(abs(v))))) if v else 2

anno = np.array([
    f"{x:.{decimals(x)}f}" if x < 1000
        else f"{x/1000:.{decimals(x/1000)-1}f}k" if x < 1e5
        else f"{x/1e6:.{decimals(x/1000)+2}f}M" if x < 1e6
        else f"{x/1e6:.{decimals(x/1e6)}f}M"
        for x in heatmap.values.flatten()
    ])\
    .reshape(heatmap.shape)

# remove the leading zero in the decimal numbers, such as 0.45k to .45k
anno = np.array([re.sub(r'0\.', '.', x) for x in anno.flatten()]).reshape(heatmap.shape)

# %%
# plot heatmap, normalize color for each column

pal = pal_heatmap[1:] #omit white

# normalize to the sum of each column
heatmap_norm = heatmap.div(heatmap.sum(axis=0), axis=1)

fig = plot_heatmap(
    heatmap_norm
  , anno
  , anno_text_size=6
  , bins= [0.0, 0.75]
  , pal=pal
  , show_colorbar=True
)
fig.update_layout(title='syn distr by NT and layer, column normalized', width=800, height=300)
fig.update_traces(colorbar={'len': 1})

fig.show()

pio.write_image(
    fig
  , file=result_dir / 'fig_4g_distribution-of-neurotransmitters-across-regions.svg'
)

# %% [markdown]
# # Breakdown of syn/cell/type counts by nt type and group

# %%
display(f"no. type = {neuron_df['type'].nunique()}")
display(f"no. instance = {neuron_df['instance'].nunique()}")
display(f"no. neuron = {neuron_df['bodyId'].nunique()}")

# %%
# color palette for 5 nt types
nt_types_5 = ['ACh', 'Glu', 'GABA', 'others', 'unclear']
nt_color_5 = pd.DataFrame({
    'pred1': nt_types_5
  , 'color': np.array(pal_nt7)[[0, 1, 2, 5, 7]]
})

# %%
# counts of each nt type
count_by_nt = pd.concat(
    [
        syn['nt'].value_counts(dropna=False)
      , nt_body['nt'].value_counts(dropna=False)
      , nt_celltype['nt'].value_counts(dropna=False)
    ]
  , axis=1
  , keys=['syn', 'cell', 'celltype'])\
    .reindex(nt_types + ['unclear'], axis='index')\
    .T

# no. of cell types per nt type for each type group
count_by_type = nt_celltype\
    .groupby(['main_groups', 'nt'])\
    .size()\
    .unstack()\
    .fillna(0)

# keep only cell_groups
count_by_type = count_by_type.loc[cell_groups]

# combine count_by_nt and count_by_type
counts = pd.concat([count_by_nt, count_by_type], axis=0)

# combine His Dop OA 5HT columns to "others"
counts['others'] = counts[['His', 'Dop', 'OA', '5HT']].sum(axis=1)
counts = counts.drop(['His', 'Dop', 'OA', '5HT'], axis=1)

# reorder columns
counts = counts[nt_types_5]

# normalize
bar_by = counts.div(counts.sum(axis=1), axis=0)

# stack
bar_by_stack = bar_by.stack().reset_index()
bar_by_stack.columns = ['bar_type', 'pred1', 'fraction']

# %%
# re-order
xtick = ['syn', 'cell', 'celltype', 'OL_intrinsic', 'OL_connecting', 'VPN', 'VCN']

bar_by_stack['bar_type'] = pd.Categorical(
    bar_by_stack['bar_type']
  , categories=xtick, ordered=True
)

# sort by a predefined order
bar_by_stack['pred1'] = pd.Categorical(
    bar_by_stack['pred1']
  , categories=nt_types_5
  , ordered=True
)
bar_by_stack.sort_values(by=['pred1'], inplace=True)

# add color
bar_by_stack = pd.merge(bar_by_stack, nt_color_5, on='pred1', how='left')

# %%
# plot
fig = go.Figure(layout=layout_scatter)
fig.update_xaxes(layout_xaxis_scatter)
fig.update_yaxes(layout_yaxis_scatter)

fig.add_trace(
    go.Bar(
        x=bar_by_stack['bar_type']
      , y=bar_by_stack['fraction']
      , name="counts by nt"
      , marker_color=bar_by_stack['color']
      , marker_line_width=0
    )
)
fig.update_xaxes(categoryorder='array', categoryarray=xtick)

for i in range(7):
    fig.add_annotation(
        text=int(counts.sum(axis=1).values[i])
      , xref='x', yref='paper'
      , x=xtick[i]
      , y=1.1
      , showarrow=False
      , font={
            'size': 8
          , 'family': fig_format['font_type']
        }
    )

# Change the bar mode
fig.update_layout(
    barmode='stack'
  , autosize=False
  , margin={'t': 50}
  , bargap=0.15 # gap between bars of adjacent location coordinates.
  , showlegend=False
  , title_text="by type"
)
fig.update_yaxes(range=[0, 1])

fig.show()

pio.write_image(
    fig
  , file=result_dir / 'fig_4f_summary-by-neurotransmitter.svg'
)

# %% [markdown]
# # Plot confusion matrix for nt prediction

# %%
# round to 2 digits, print all digits
anno = confusion_df.map(lambda x: f"{x:.2f}" if x > 0 else "0.00").values
heatmap_norm = confusion_df

fig = plot_heatmap(
    heatmap_norm
  , anno
  , anno_text_size=6
  , bins=np.linspace(0,1, 7).astype(int)
  , show_colorbar=True
)
fig.update_traces(colorbar={'len': 1})
# bound box
N = 7
fig.add_shape(
    type="rect", xref="x", yref='y'
  ,  x0=-0.5, y0=-0.5, x1=N-0.5, y1=N-0.5
  ,  line={'color': 'black', 'width': 0.5}
)
# add syn count as annotation, print all digits
for i in range(7):
    fig.add_annotation(
        text=gt_count['cell_type_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.05
      , y=confusion_df.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
    fig.add_annotation(
        text=gt_count['bodyId_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.25
      , y=confusion_df.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
    fig.add_annotation(
        text=gt_count['syn_train_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.45
      , y=confusion_df.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
    fig.add_annotation(
        text=gt_count['syn_test_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.65
      , y=confusion_df.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
for i in range(4):
    fig.add_annotation(
        text= ["cell type", "cell", "training", "test"][i]
      , xref='paper', yref='paper'
      , x=1.05 + 0.2*i, y=1.15
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
# with colorbar
fig.update_traces(showscale=False)
# add space between cells
fig.update_traces(xgap=1, ygap=1)
# aspect ratio and margin
fig.update_layout(yaxis_scaleanchor="x", margin={'l':0, 'r':120, 'pad':0})
fig.update_layout(title='confusion matrix all syn')

pio.write_image(
    fig
  , file=result_dir / 'fig_4e_confusion-matrix-all-synapses.svg'
)
fig.show()

# %% [markdown]
# # Confusion matrix from fish data

# %%
df = pd.merge(
    syn_raw[['bodyId','nt']]
  , nt_body[['bodyId', 'type', 'instance',  'main_groups']]
  , on='bodyId'
  , how='left')
display(fish[~fish['cell_type'].isin(df['type'])])

# %%
# add type
df = pd.merge(
    syn_raw[['bodyId','nt']]
  , nt_body[['bodyId', 'type', 'instance',  'main_groups']]
  , on='bodyId'
  , how='left')

df = df[df['type'].isin(fish['cell_type'])]
df = pd.merge(df, fish, left_on='type', right_on='cell_type', how='left')

# combine cell_type R8p and R8y to R8
df.loc[df['cell_type'].isin(['R8p', 'R8y']), 'cell_type'] = 'R8'

confusion_df_fish = df\
    .groupby(['transmitter', 'nt'])\
    .size()\
    .unstack(-1, 0.0)

# Normalize the rows
confusion_df_fish /= confusion_df_fish.sum(axis=1).values[:, None]
# order the confusion matrix columns and rows to match the order of nt_types
confusion_df_fish = confusion_df_fish[nt_types].reindex(nt_types)

display(confusion_df_fish)

# %%
gt_count_fish = df.groupby('transmitter')\
    .agg(
        cell_type_count=('cell_type', 'nunique')
      , bodyId_count=('bodyId', 'nunique')
      , syn_count=('bodyId', 'size'))\
    .reindex(nt_types)\
    .reset_index()
display(gt_count_fish)

# %%
# plot heatmap

# round to 2 digits, print all digits
anno = confusion_df_fish.map(lambda x: f"{x:.2f}" if x > 0 else "0.00").values

heatmap_norm = confusion_df_fish

fig = plot_heatmap(
    heatmap_norm
  , anno
  , anno_text_size=6
  , bins=np.linspace(0,1, 7).astype(int)
  , show_colorbar=True
)
fig.update_traces(colorbar={'len': 1})

# add anno as annotation, print all digits
fig.update_traces(
    text=anno
  , texttemplate="%{text}"
  , textfont_size=6
  , hovertemplate=None
  , textfont_family=fig_format['font_type']
)

for i in range(3):
    fig.add_annotation(
        text= ["cell type", "cell", "syn"][i]
      , xref='paper', yref='paper'
      , x=1.05 + 0.2*i, y=1.15
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )

# add syn count as annotation, print all digits
for i in range(7):
    fig.add_annotation(
        text=gt_count_fish['cell_type_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.05
      , y=heatmap_norm.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
    fig.add_annotation(
        text=gt_count_fish['bodyId_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.25
      , y=heatmap_norm.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )
    fig.add_annotation(
        text=gt_count_fish['syn_count'][i].astype(str)
      , xref='paper', yref='y'
      , x=1.45
      , y=heatmap_norm.index.values[i]
      , showarrow=False
      , font={'size': 6, 'family': fig_format['font_type']}
    )

# with colorbar
fig.update_traces(showscale=False)

# aspect ratio and margin
fig.update_layout(yaxis_scaleanchor="x", margin={'l':0, 'r':120, 'pad':0})

fig.update_layout(title='confusion matrix fish')

fig.show()

pio.write_image(fig, file=result_dir / 'fig_4e_confusion-matrix-all-synapses_fish.svg' )

# %% [markdown]
# # size (column coverage ) \ syn count vs nt

# %%
type_instance = nt_body[['type', 'instance']].drop_duplicates()
display(type_instance.shape)

# %%
# load coverage tables
tb = get_metrics_df()

size_col_ls = []
size_col_ls.append(
    tb.loc[tb['roi'] == "ME(R)"
  , ['instance', 'cell_size_cols']]
)
size_col_ls.append(
    tb.loc[tb['roi'] == "LO(R)"
  , ['instance', 'cell_size_cols']]
)
size_col_ls.append(
    tb.loc[tb['roi'] == "LOP(R)"
  , ['instance', 'cell_size_cols']]
)
# change instance to type by removing the last 2 characters
size_col_ls = [
    x.assign(cell_type = x['instance'].map(lambda x: x[:-2])) for x in size_col_ls
]

# %%
# layer rois
rois = ['ME(R)', 'LO(R)', 'LOP(R)']

# restrict to OL_intrinsic and OL_connecting
df = nt_celltype[nt_celltype['main_groups']\
    .isin(['OL_intrinsic', 'OL_connecting'])]
df = pd.merge(df, type_instance, on='type', how='left')

# iterate over type, check which roi has most tbars, then assign size column
for i in range(df.shape[0]):
    # get the type
    cell_type = df.loc[i, 'type'] # only intrinsic and connecting, so ok
    # get syn
    s = syn[syn['type'] == cell_type]
    # check roi
    nls = [s['layer_roi'].str.startswith('ME_').sum(),
           s['layer_roi'].str.startswith('LO_').sum(),
           s['layer_roi'].str.startswith('LOP_').sum()
           ]
    # assign roi
    df.loc[i, 'main_roi'] = rois[nls.index(max(nls))]
    # assign size
    size_col = size_col_ls[nls.index(max(nls))]
    # not all types are matched
    if size_col[size_col['cell_type'] == cell_type].shape[0] > 0:
        df.loc[i, 'size_col'] = \
            size_col[size_col['cell_type'] == cell_type]['cell_size_cols'].values

# %%
# keep only 'ACh', 'Glu', 'GABA',
type_nt_size = df[df['nt'].isin(['ACh', 'Glu', 'GABA'])].copy()

# add color to type_nt_size such that it matches the order of nt_types
type_nt_size['color'] = type_nt_size['nt'].map(OL_COLOR.NT.map)

# %%
# box plots
type_nt_size['nt'] = pd.Categorical(
    type_nt_size['nt']
  , categories=nt_types
  , ordered=True
)

fig = go.Figure(layout = layout_scatter)

fig.add_trace(
    go.Box(
        x=type_nt_size['nt']
      , y=type_nt_size['size_col']
      , hoverinfo="text"
      , hovertext=type_nt_size['instance']
      , marker_size=3
      , marker_color="black"
      , line={'width': 0}
      , boxpoints='all' # can also be outliers, or suspectedoutliers, or False
      , jitter=1 # add some jitter for a better separation between points
      , pointpos=0 # relative position of points wrt box
      , showwhiskers=False
      , fillcolor="#FFFFFF"
    )
)
fig.update_xaxes(categoryorder='array', categoryarray=nt_types)

fig.update_xaxes(layout_xaxis_scatter)
fig.update_yaxes(layout_yaxis_scatter)

fig.update_layout(
    yaxis={'title': "column-size"}
  , showlegend=True
)

fig.show()

# # save
pio.write_image(fig, file=result_dir / 'fig_4h_size-NT.svg')

# %% [markdown]
# # NT vs multiplicity of polyadic synapses

# %%
df = nt_body\
    .groupby(['main_groups', 'type'], as_index=True)\
    .agg(
        cell_count=('bodyId', 'size')
      , pred_top=('nt', lambda x: x.value_counts().index[0])
      , pre_mean=('pre', 'mean')
      , downstream_mean=('downstream', 'mean'))\
    .reset_index()

df['fanout'] = df['downstream_mean'] / df['pre_mean']

df.head(3)

# %%
# group together cell types of ['Dop', 'His',  'OA', '5HT']
df['pred_top'] = df['pred_top'].replace(['Dop', 'OA', '5HT', 'unclear'], 'Other')


# %%
def fit_line(x, a, b):
    """Fit a line"""
    return a * x + b * 0


# %%
# go.scatter, downstream_mean vs pre_mean, colored by pred_top,

ab_ls = []

fig = go.Figure(layout=layout_scatter)

for i in range(3):
    df_gp = df[df['pred_top'] == ['ACh', 'Glu', 'GABA',  'His', 'Other'][i]]
    fig.add_trace(
        go.Scatter(
            x=df_gp['pre_mean']
          , y=df_gp['downstream_mean']
          , hoverinfo="text"
          , hovertext=df_gp['type']
          , mode='markers'
          , name=['ACh', 'Glu', 'GABA',  'His', 'Other'][i]
          , marker_color=pal_nt[i]
          , marker_size=5
          , line_width=1
          , opacity=0.9
        )
    )

    # quantiles
    qx = df_gp['pre_mean'].quantile([0.25, 0.5, 0.75]).values
    qy = df_gp['downstream_mean'].quantile([0.25, 0.5, 0.75]).values
    fig.add_shape(
        type="line"
      , x0=qx[0], x1=qx[-1], y0=qy[1], y1=qy[1]
      , line={'color': pal_nt[i], 'width': 1}
    )
    fig.add_shape(
        type="line"
      , x0=qx[1], x1=qx[1], y0=qy[0], y1=qy[-1]
      , line={'color': pal_nt[i], 'width': 1}
    )

    # fit a line to the data
    popt, pcov = curve_fit(fit_line, df_gp['pre_mean'], df_gp['downstream_mean'])
    ab_ls.append(popt)

    # plot the line
    x = np.linspace(10, 100000, 1000)
    y = fit_line(x, *popt)

    if i % 2 == 0:
        fig.add_trace(
            go.Scatter(
                x=x
              , y=y
              , mode='lines'
              , name='fit'
              , marker_color=pal_nt[i]
              , line_width=2
              , opacity=0.9
              , showlegend=False
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x
              , y=y
              , mode='lines'
              , line_dash='dash'
              , name='fit'
              , marker_color=pal_nt[i]
              , line_width=2
              , opacity=0.9
              , showlegend=False
            )
        )

# add y=x line
fig.add_shape(
    type="line", x0=10, x1=1e5, y0=10, y1=1e5
  , line={'color': "black", 'width': 1}
)

fig.update_xaxes(layout_xaxis_scatter)
fig.update_yaxes(layout_yaxis_scatter)

fig.update_xaxes(type='log', range=[1.5, 5.3])
fig.update_yaxes(
    type='log', range=[1.5, 5.3]
  , scaleanchor="x", scaleratio=1
)

fig.update_layout(
    xaxis = {
        'tickmode': 'array'
      , 'tickvals': [20, 100, 1e3, 1e4, 1e5, 2e5]
      , 'ticktext': ["20", "100", "1k", "10k", "100k", "200k"]
    }
  , yaxis = {
        'tickmode': 'array'
      , 'tickvals': [20, 100, 1e3, 1e4, 1e5, 2e5]
      , 'ticktext': ["20", "100", "1k", "10k", "100k", "200k"]
    }
)

fig.update_layout(
    xaxis={'title': "pre_mean"}
  , yaxis={'title': "downstream_mean"}
  , width=600
  , height=600
)

fig.show()

# # save
pio.write_image(
    fig
  , file=result_dir / 'fig_4i_syn-multiplicity.svg'
)

# %%
display(ab_ls)

# %% [markdown]
# ### ANCOVA

# %%
df_ols = df.copy()
# combine 5HT, OA, Dop, His into 'Other'
df_ols['pred_top'] = df_ols['pred_top']\
    .replace(['Dop', 'OA', '5HT','His', 'unclear'], 'Other')
# remove 'Other'
df_ols = df_ols[df_ols['pred_top'] != 'Other']

# %%
# use interaction (dummy var) and remove intersection
fanout_lm = ols('downstream_mean ~ pre_mean*pred_top - 1', data= df_ols).fit()

# %%
display(fanout_lm.summary2(float_format='%.2f'))

# %% [markdown]
# Note the slopes are slightly different than ab_ls. And the interaction term is significant for
# GABA, but not for Glu with p=0.827

# %%
display(fanout_lm.pvalues)

# %%
