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
#     name: ol-connectome
# ---

# %%
# %load_ext autoreload

from pathlib import Path
import sys, os
from IPython.display import display

from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from neuprint import fetch_neurons, NeuronCriteria as NC

from cmap import Colormap


# %%
# %autoreload 2
# load some helper functions
from utils.ROI_calculus import load_layer_thre, _get_data_path, roi_layer_parameters
from utils.celltype_conn_by_roi import CelltypeConnByRoi
from utils import olc_client
from utils.summary_plot_preprocessor import SummaryPlotPreprocessor
from utils.plotting_functions import plot_flip_syn_hist

# %%
c = olc_client.connect(verbose=True)

# %%
fig_format = {'fig_width': 1, 'fig_height': 3, 'fig_margin': 0.1, \
                           'fsize_ticks_pt': 9, 'fsize_title_pt': 11}
f_width = (fig_format['fig_width'] - fig_format['fig_margin'])*96
f_height = (fig_format['fig_height'] - fig_format['fig_margin'])*96
fsize_ticks_px = fig_format['fsize_ticks_pt']*(1/72)*96
fsize_title_px = fig_format['fsize_title_pt']*(1/72)*96

# %%
me_lay_tit = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
lo_lay_tit = ['L1', 'L2', 'L3', 'L4', 'L5a','L5b', 'L6']
lop_lay_tit = ['LP1', 'LP2', 'LP3', 'LP4']
lay_tit_dict = {'ME(R)': me_lay_tit, 'LO(R)': lo_lay_tit, 'LOP(R)': lop_lay_tit}
bdry_color_dict = {'pre': 'rgba(7,127,160,1.0)', 'post': 'rgba(153,128,3,1.0)'}

# %%
#method can be 'median' or 'mean'
method = 'mean'

# %%
for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:

    #load cell-types that were used to define layers
    _, _, cell_types, syn_type, _, layer_bdry_pos = roi_layer_parameters(roi_str=roi_str)
    
    #remove duplicated cell-type in ME (used for pre and post)
    if roi_str=='ME(R)':
        cell_types.remove('C2')
    n_types = len(cell_types)
    
    #correct for layer def in LOP (e.g. the boundary between LOP1 and LOP2 is defined by the mean of T4a and T4b)
    if roi_str=='LOP(R)':
        layer_bdry_pos = [[1],[1,2],[2,3],[3]]

    #get data
    spp = SummaryPlotPreprocessor()
    for cell_type in cell_types:
        ct_conn = CelltypeConnByRoi(cell_type, 'ALL')
        ct_conn.set_input_syn_conf_thresh(None)
        ct_conn.set_output_syn_conf_thresh(None)
        spp.add_ct(ct_conn)
    num_df = spp.get_nums_df()
    hist_df = spp.get_hists_df(method=method)

    depth_bdry = load_layer_thre(roi_str)
    lay_bnd_dict = {roi_str: depth_bdry}

    fig = make_subplots(
        rows=1
      , cols=n_types
      , print_grid=False
      # , vertical_spacing=0.005
      # , horizontal_spacing=1
    )
    
    #plotting cell name
    for j in range(n_types):
        tick_txt = f'<b>{cell_types[j]}</b>'
        
        fig.update_xaxes(
            tickvals=np.arange(0, n_types) + 0.5
          , ticktext=[tick_txt]
          , color='black'
          , tickfont={'size':10}
          , row=1, col=j+1
          , anchor='free'
          , position=1
        )
    
    row_count = 0
    ct = 0
    for j in range(n_types):
        temp_syn = hist_df[hist_df['target']==cell_types[j]]
        fig = plot_flip_syn_hist(
            fig
          , hist_celltype_df=temp_syn
          , roi_to_plot=roi_str
          , row_num=1
          , col_num=j+1
          , layer_bound_dict=lay_bnd_dict
          , lay_tit_dict=lay_tit_dict )
        for i in range(len(layer_bdry_pos[ct])):
            xb = depth_bdry[layer_bdry_pos[ct][i]]
            fig.add_trace(
                go.Scatter(
                    y=[xb,xb]
                  , x=[0,temp_syn[['norm_inp_count','norm_out_count']].max().max()]
                  , mode='lines'
                  , line={'color': bdry_color_dict[syn_type[ct]], 'width': 2}
                , showlegend=False)
              , row=1, col=j+1
            )
        #special rule for ME because C2 layers are defined for both pre and post
        if roi_str=='ME(R)':
            if ct==2:
                ct += 1
                for i in range(len(layer_bdry_pos[ct])):
                    xb = depth_bdry[layer_bdry_pos[ct][i]]
                    fig.add_trace(
                        go.Scatter(
                            y=[xb,xb]
                          , x=[0,temp_syn[['norm_inp_count','norm_out_count']].max().max()]
                          , mode='lines'
                          , line={'color': bdry_color_dict[syn_type[ct]], 'width': 2}
                        , showlegend=False)
                      , row=1, col=j+1
                    )
            
        ct = ct + 1
        
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    fig.update_layout(
        autosize=True,
        width=f_width*n_types,
        height=f_height,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white'
    )
    fig.show()
    fig.write_image(_get_data_path('cache') / f"{roi_str[:-3]}_layer_def.pdf", height=f_height, width=f_width*n_types) 
    fig.write_image(_get_data_path('cache') / f"{roi_str[:-3]}_layer_def.svg", height=f_height, width=f_width*n_types) 


# %%

# %%
