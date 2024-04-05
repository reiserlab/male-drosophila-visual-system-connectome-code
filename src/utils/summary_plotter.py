""" 
Part of the summary figure plotting 
"""
from abc import ABC
import warnings
from itertools import pairwise
from bisect import bisect

import numpy as np
import pandas as pd
import scipy

import plotly.graph_objects as go
import plotly.io as pio   

from plotly.subplots import make_subplots

from neuprint import fetch_custom

from utils.ROI_calculus import load_layer_thre
from utils.ol_color import OL_COLOR
from utils.ol_types import OLTypes

class SummaryPlotter(ABC):

    """
    Plot the summary figures.
    """

    def __init__(
        self
      , instance_list:list
      , figure_title:str=None
      , method:str='median'
      , col_synapses:dict=None
      , col_top5:float=.25
      , col_innervation:dict=None
    ):
        """
        Initialize the summary plotter

        Parameters
        ----------
        instance_list : list[InstanceSummary]
            A list of InstanceSummary objects to plot
        figure_title : str, default=None
            Title used for the figure
        method : str, default='median'
            Can either be 'mean' or 'median'
        col_synapses : dict, default=None
            Contains the columns and their width for the Synapses part of the plot. For the default
            of `col_synapses=None`, it is set to
            `{'ME(R)':.15 , 'LO(R)':.075, 'LOP(R)':.05, 'AME(R)':.05}`.
        col_top5 : float, default=0.2
            width of the top5 column
        col_innervation : dict, default=None
            The columns used for the innervation plot, similar to col_synpases but
            without `AME(R)` in the default
        """
        if col_synapses is None:
            col_synapses = {
                'ME(R)':.13
              , 'LO(R)':.07
              , 'LOP(R)':.04
              , 'AME(R)':.05
            }
        if col_innervation is None:
            col_innervation = {
                'ME(R)':.13
              , 'LO(R)':.07
              , 'LOP(R)':.04
            }
        assert method in ['median', 'mean'], \
            f"method can only be 'median' or 'mean', not {method}"
        assert set(col_synapses.keys()) <= set(['ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']),\
            "Unexpected column for synapses"
        assert set(col_innervation.keys()) <= set(['ME(R)', 'LO(R)', 'LOP(R)']), \
            "Unexpected column for innervation"
        self.__method = method
        self.__title = figure_title
        self.__fig = None
        self.__instances = instance_list

        self.__col_syn = col_synapses

        pio.kaleido.scope.mathjax = None

        if col_top5:
            self.__col_top5 = {'top5': col_top5}
        else:
            self.__col_top5 = None
        self.__col_inn = col_innervation

        # configuration for layout
        self.__axis_label_text_size = 7
        self.__axis_label_text_color = "#777777"
        self.__data_text_size = 8


    def plot(self) -> go.Figure:
        """
        Plot the summary and return the figure

        Returns
        -------
        fig : go.Figure
            The summary plot
        """
        self.__create_figure()
        self.__add_row_labels()
        self.__add_syn()
        self.__add_top5()
        self.__add_innervation()
        self.__update_layout()
        return self.__fig


    def __create_figure(self):
        row_counter = len(self.__instances)
        heights = [1/24] * row_counter + [(24-row_counter)/24]
        self.__fig = make_subplots(
            rows=row_counter + 1
          , cols=len(self.__get_col_widths())
          , column_widths = self.__get_col_widths()
          , horizontal_spacing=0.01
          , vertical_spacing=0.005
          , print_grid=False
          , row_heights=heights
        )


    def __add_row_labels(self):
        olt = OLTypes()
        for ind, instance_sum in enumerate(self.__instances):
            col_str = self.__get_group_col(olt.get_main_group(instance_sum.type_name))
            hem_col = "color:#8b008b;" if instance_sum.hemisphere == 'L' else "color:#000;"
            self.__fig.update_yaxes(tickvals=[], row=ind+1, col=1)
            start_row = len(self.__get_col_widths())* ind
            colcnt = start_row + 1
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.add_annotation(
                    text=f'<b style="font-size:106%{col_str}">{self.__get_print_name(instance_sum.type_name)}</b>'
                  , x=0
                  , y=.56
                  , yshift=0
                  , font={'family': 'Arial', 'size':10}
                  , xref=f"x{colcnt} domain"
                  , yref=f"y{colcnt} domain"
                  , yanchor='bottom'
                  , xanchor='right'
                  , showarrow=False
                )
            self.__fig.add_annotation(
                    text=f'<span style="font-size:80%;color:#777">n=</span>{instance_sum.count}&nbsp;&nbsp;&nbsp;<b style="font-size:80%;{hem_col}">({instance_sum.hemisphere})</b>'
                  , x=0
                  , y=.28
                  , yshift=0
                  , font={'family': 'Arial', 'size':10}
                  , xref=f"x{colcnt} domain"
                  , yref=f"y{colcnt} domain"
                  , yanchor='bottom'
                  , xanchor='right'
                  , showarrow=False
                )
            self.__fig.add_annotation(
                    text=instance_sum.consensus_nt
                  , x=0
                  , y=0
                  , yshift=0
                  , font={'family': 'Arial', 'size':10}
                  , xref=f"x{colcnt} domain"
                  , yref=f"y{colcnt} domain"
                  , yanchor='bottom'
                  , xanchor='right'
                  , showarrow=False
                )


    def __get_col_widths(self) -> list[float]:
        width = []
        if self.__col_syn:
            width += list(self.__col_syn.values())
        if self.__col_top5:
            width += list(self.__col_top5.values())
        if self.__col_inn:
            width += list(self.__col_inn.values())
        return width


    def __get_col_width(self) -> float:
        width = sum(self.__col_syn.values()) \
            + sum(self.__col_top5.values()) \
            + sum(self.__col_inn.values())
        return width


    def __get_group_col(self, group) -> str:
        """
        Helper function to get a CSS string defining the color for one of the main groups
        """
        colors = {
            'OL_intrinsic': '#029e73'
          , 'OL_connecting': '#D55E00'
          , 'VPN': '#0173b2'
          , 'VCN': '#de8f05'
          , 'other': '#cc78bc'
        }
        ret = ""
        if group in colors.keys():
            ret = f';color:{colors[group]}'
        return ret


    def __get_print_name(self, ct_name) -> str:
        """
        Helper function to get a name abbreviation (if there is one defined here).
        """
        names = {
            'Ascending_TBD1': 'AN_TBD1'
          , '5thsLNv_LNd6': '5thsLNv'
          , '5-HTPMPV01': '5HTPMPV1'
          , '5-HTPMPV03': '5HTPMPV3'
        }
        ret = ct_name
        if ct_name in names.keys():
            ret = names[ct_name]
        return ret


    def __get_layer_bnd(self, roi_str:str=None):
        rel_roi = list(set(self.__col_syn.keys()) & {'ME(R)', 'LO(R)', 'LOP(R)'})
        lay_bnd = {}
        for roi in rel_roi:
            lay_bnd[roi] = load_layer_thre(roi)
        if roi_str:
            if roi_str in lay_bnd:
                return lay_bnd[roi_str]
            return []
        return lay_bnd


    def __get_full_roi_name(self, roi_str:str):
        full_names = {
            'ME(R)': "Medulla"
          , 'LO(R)': "Lobula"
          , 'LOP(R)': "Lobula Plate"
          , 'AME(R)': "AME"
        }
        if roi_str in full_names:
            return full_names[roi_str]
        return "Unknown"


    def __get_layer_titles(self, roi_str:str=None):
        layer_titles = {
            'ME(R)': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
          , 'LO(R)': ['LO1', 'LO2', 'LO3', 'LO4', 'LO5A','LO5B', 'LO6']
          , 'LOP(R)': ['LOP1', 'LOP2', 'LOP3', 'LOP4']
        }
        rem_roi = list({'ME(R)', 'LO(R)', 'LOP(R)'} - set(self.__col_syn.keys()))
        for roi in rem_roi:
            del layer_titles[roi]
        if roi_str:
            if roi_str in layer_titles:
                return layer_titles[roi_str]
            return []
        return layer_titles


    def __get_ordered_rois(self, keys):
        sort_order = {'ME(R)':1, 'LO(R)':2, 'LOP(R)':3}
        rel_roi = list(set(keys) & set(sort_order.keys()))
        rel_roi = sorted(rel_roi, key=lambda x: sort_order[x])
        return rel_roi


    def __add_syn(self):
        row_count = 0
        rel_roi = self.__get_ordered_rois(self.__col_syn.keys())
        for ct in self.__instances:
            row_count += 1
            col_count = 0
            for roitp in rel_roi:
                col_count += 1
                self.__plot_syn_hist(
                    instance=ct
                  , roi_to_plot=roitp
                  , row_num=row_count
                  , col_num=col_count
                )
            if 'AME(R)' in self.__col_syn.keys():
                col_count += 1
                self.__plot_syn_count(
                    instance=ct
                  , row_num=row_count
                  , col_num=col_count
                )


    def __plot_syn_count(
        self
      , instance
      , row_num:int, col_num:int
    ):
        roi_to_plot = 'AME(R)'
        col_name = self.__get_full_roi_name(roi_str=roi_to_plot)
        y_range = [0, 1]
        x_range = [0, 1]
        input_color = OL_COLOR.OL_SYNAPSES.nrgb['post']
        input_color = f"rgba{input_color[3:-1]}, 0.7)"
        output_color = OL_COLOR.OL_SYNAPSES.nrgb['pre']
        output_color = f"rgba{output_color[3:-1]}, 0.4)"
        min_area = .01
        max_area = .79
        center = [0.5, 0.5]
        norm_max = instance.distribution['syn_dist'].max()
        if len(instance.ame_count) > 0:
            in_value = 0
            out_value = 0

            if 'post' in instance.ame_count['type'].values:
                in_value = instance.ame_count.set_index('type').loc['post', 'perc']
            if 'pre' in instance.ame_count['type'].values:
                out_value = instance.ame_count.set_index('type').loc['pre', 'perc']

            in_scaled_area = (in_value * (max_area - min_area)) + min_area
            out_scaled_area = (out_value * (max_area - min_area)) + min_area

            in_radius = (in_scaled_area/np.pi) ** 0.5
            out_radius = (out_scaled_area/np.pi) ** 0.5

            theta = np.linspace(-np.pi/2+.7, np.pi/2+.7, 100)

            x_in = center[0] + in_radius * np.cos(np.pi - theta)
            y_in = center[1] + in_radius * np.sin(np.pi - theta)

            x_out = center[0] + out_radius * np.cos(-theta)
            y_out = center[1] + out_radius * np.sin(-theta)

            self.__fig.add_trace(
                go.Scatter()
              , row=row_num
              , col=col_num
            )

            if in_value >= 0.001:
                self.__fig.add_trace(
                    go.Scatter(
                        x=x_in
                      , y=y_in
                      , mode='lines'
                      , fill='toself'
                      , fillcolor=input_color
                      , line={'color': 'black', 'width': 0}
                      , name=in_value
                      , showlegend=False
                    )
                  , row=row_num
                  , col=col_num
                )

                self.__fig.add_annotation(
                    x=0.5
                  , y=0.5
                  , text=f"{in_value*100:.2f}%"
                  , showarrow=False
                  , font={'family': 'Arial', 'size':8, 'color':'black'}
                  , yanchor='middle'
                  , xanchor='right'
                  , xshift=7
                  , yshift=7
                  , row=row_num
                  , col=col_num
                )

            if out_value >= 0.001:
                self.__fig.add_trace(
                    go.Scatter(
                        x=x_out
                      , y=y_out
                      , mode='lines'
                      , fill='toself'
                      , fillcolor=output_color
                      , line={'color': 'black', 'width': 0}
                      , name=out_value
                      , showlegend=False
                    )
                  , row=row_num
                  , col=col_num
                )

                self.__fig.add_annotation(
                    x=0.5
                  , y=0.5
                  , text=f"{out_value*100:.2f}%"
                  , showarrow=False
                  , font={'family': 'Arial', 'size':8, 'color':'black'}
                  , yanchor='middle'
                  , xanchor='left'
                  , xshift=-7
                  , yshift=-7
                  , row=row_num
                  , col=col_num
                )

            if row_num < len(self.__instances):
                self.__fig.update_xaxes(
                    range=x_range
                  , showticklabels=False
                  , row=row_num, col=col_num
                )
            else:
                self.__fig.update_xaxes(
                    range=x_range
                  , showticklabels=False
                  , row=row_num, col=col_num
                )
                start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
                colcnt = start_last_row + col_num
                colcnt = colcnt if colcnt>1 else ""
                self.__fig.add_annotation(
                    text=col_name
                  , y=0
                  , yshift=-39
                  , font={'family': 'Arial', 'size':8, 'color':self.__axis_label_text_color}
                  , xref=f"x{colcnt} domain"
                  , yref=f"y{colcnt} domain"
                  , yanchor='bottom'
                  , xanchor='center'
                  , showarrow=False
                )
            start_row = len(self.__get_col_widths())* (row_num-1)
            colcnt = start_row + col_num
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.update_yaxes(
                range=y_range
              , showticklabels=False
              , scaleanchor=f"x{colcnt}"
              , scaleratio=1
              , row=row_num, col=col_num
            )
        else:
            self.__fig.add_trace(
                go.Scatter(
                    x=[]
                  , y=[]
                  , mode='markers'
                  , marker={'size':0}
                )
              , row=row_num, col=col_num
            )

            if row_num < len(self.__instances):
                self.__fig.update_xaxes(
                    range=x_range
                  , showticklabels=False
                  , row=row_num, col=col_num
                )
            else:
                self.__fig.update_xaxes(
                    range=x_range
                  , showticklabels=False
                  , row=row_num, col=col_num
                )
                start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
                colcnt = start_last_row + col_num
                colcnt = colcnt if colcnt>1 else ""
                self.__fig.add_annotation(
                    text=col_name
                  , y=0
                  , yshift=-39
                  , font={'family': 'Arial', 'size':8, 'color':self.__axis_label_text_color}
                  , xref=f"x{colcnt} domain"
                  , yref=f"y{colcnt} domain"
                  , yanchor='bottom'
                  , xanchor='center'
                  , showarrow=False
                )
            start_row = len(self.__get_col_widths())* (row_num-1)
            colcnt = start_row + col_num
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.update_yaxes(
                range=y_range
              , showticklabels=False
              , scaleanchor=f"x{colcnt}"
              , scaleratio=1
              , row=row_num, col=col_num
            )


    def __plot_syn_hist(
        self
      , instance
      , roi_to_plot
      , row_num:int, col_num:int
    ):
        y_range = [0, 1]
        gridline_col = 'gainsboro'
        input_color = OL_COLOR.OL_SYNAPSES.nrgb['post']
        input_color = f"rgba{input_color[3:-1]}, 0.8)"
        output_color = OL_COLOR.OL_SYNAPSES.nrgb['pre']
        output_color = f'rgba{output_color[3:-1]}, 0.4)'

        layer_bound=self.__get_layer_bnd(roi_str=roi_to_plot)
        layer_title=self.__get_layer_titles(roi_str=roi_to_plot)
        col_name=self.__get_full_roi_name(roi_str=roi_to_plot)

        tmp_bnd = layer_bound
        tmp_tit = layer_title

        for boundary in tmp_bnd:
            self.__fig.add_trace(
                go.Scatter(
                    x=[boundary, boundary]
                  , y=y_range
                  , mode='lines'
                  , line={'color': gridline_col, 'width': .75}
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )
        norm_max = instance.distribution['syn_dist'].max()
        rel_hist_df = instance.distribution[(instance.distribution['roi']==roi_to_plot) & (instance.distribution['type']=='post')]
        self.__fig.add_trace(
            go.Scatter(
                x=rel_hist_df['depth']
              , y=scipy.signal.savgol_filter(rel_hist_df['syn_dist']/norm_max, 5, 1, mode='interp')
              , fill='tozeroy'
              , fillcolor=input_color
              , mode='lines'
              , line={'color': 'black', 'width': .5}
              , showlegend=False
            )
          , row=row_num, col=col_num
        )
        rel_hist_df = instance.distribution[(instance.distribution['roi']==roi_to_plot) & (instance.distribution['type']=='pre')]
        self.__fig.add_trace(
            go.Scatter(
                x=rel_hist_df['depth']
              , y=scipy.signal.savgol_filter(rel_hist_df['syn_dist']/norm_max, 5, 1, mode='interp')
              , fill='tozeroy'
              , fillcolor=output_color
              , mode='lines'
              , line={'color': 'black', 'width': .5}
              , showlegend=False)
          , row=row_num, col=col_num
        )
        #add horizontal line at 0
        self.__fig.add_hline(
            y=0
          , line_width=0.5
          , line_color="black"
          , row=row_num, col=col_num
        )

        if row_num < len(self.__instances):
            self.__fig.update_xaxes(showticklabels=False, row=row_num, col=col_num)
        else:
            ave_db = [np.mean([x,y]) for (x,y) in pairwise(tmp_bnd)]
            self.__fig.update_xaxes(
                tickvals=ave_db
              , ticktext=tmp_tit
              , tickangle=-90
              , color='black'
              , tickfont={'family': 'Arial', 'size': self.__axis_label_text_size, 'color': self.__axis_label_text_color}
              , row=row_num, col=col_num
            )
            start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
            colcnt = start_last_row + col_num 
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.add_annotation(
                text=col_name
              , y=0
              , yshift=-39
              , font={'family': 'Arial', 'size': 8, 'color': self.__axis_label_text_color}
              , xref=f"x{colcnt} domain"
              , yref=f"y{colcnt} domain"
              , yanchor='bottom'
              , xanchor='center'
              , showarrow=False
            )
        if roi_to_plot == 'LOP(R)':
            sc_l = [0,0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000]
            maxbar = sc_l[bisect(sc_l, norm_max) -1]
            self.__fig.add_trace(
                go.Scatter(
                    x=[1, 1]
                  , y=[0, maxbar/norm_max]
                  , mode='lines'
                  , line={'color': 'black', 'width': 1}
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )
            start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
            colcnt = start_last_row + col_num 
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.add_annotation(
                x=1
              , y=maxbar/norm_max/2
              # , y=.5
              , text=f"{np.round(maxbar)}"
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__axis_label_text_size, 'color':self.__axis_label_text_color}
              , xref=f"x{colcnt} domain"
              , yref=f"y{colcnt} domain"
              , yanchor='middle'
              , xanchor='left'
              , textangle=-90
              , xshift=-1
              , yshift=2
              , row=row_num, col=col_num
            )
        if col_num > 1:
            self.__fig.update_yaxes(showticklabels=False, row=row_num, col=col_num)


    def __add_top5(self):
        row_count = 0
        for ct in self.__instances:
            row_count += 1
            col_count = len(self.__col_syn)
            
            col_count += 1
            self.__plot_hor_frac_bar(
                instance=ct
              , row_num=row_count, col_num=col_count
            )


    def __plot_hor_frac_bar(
        self
    #   , frac_celltype_df:pd.DataFrame
      , instance
      , row_num:int, col_num:int
    ):
        #Shading
        input_color = OL_COLOR.OL_SYNAPSES.nrgb['post']
        output_color = OL_COLOR.OL_SYNAPSES.nrgb['pre']

        col_i = []
        col_o = []
        for alpha in [.9, .75, .6, .45, .3]:
            col_i.append(f"rgba{input_color[3:-1]}, {alpha})")
            col_o.append(f"rgba{output_color[3:-1]}, {alpha})")
        col_i.append('rgba(243,246,244,1)')
        col_o.append('rgba(243,246,244,1)')

        x_fudge = 0.005
        y_diff = 0.1

        x_1 = 0
        x_count = 0
        label_loc = 0

        for _, row in instance.top5_upstream.iterrows():
            x_2 = row['cum_perc']
            if row['instance'] is None:
                text = "NoName"
            else:
                text = row['instance'][:-2]
            
            styleadd = "color:#000;"
            if row['instance'][-1] == 'L':
                styleadd = "color:#8b008b;"

            self.__fig.add_trace(
                go.Scatter(
                    x=np.array([x_1, x_2 - x_fudge, x_2 - x_fudge, x_1])
                  , y=np.array([0.005, 0.005, y_diff, y_diff])
                  , fill='tozeroy'
                  , mode='none'
                  , fillcolor=col_i[x_count]
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )

            letter_spacing = ""
            if len(text)>7:
                letter_spacing = "letter-spacing:-.08em;font-stretch:60%;"
            elif len(text)>6:
                letter_spacing = "letter-spacing:-.05em;font-stretch:75%;"
            self.__fig.add_annotation(
                x=label_loc
              , y=y_diff
              , text=f'<span style="{letter_spacing}{styleadd}"><b style="font-size:88%;color:{col_i[x_count]};">{x_count+1}</b>&#8239;{text[:6]}<span style="opacity:.7;">{text[6:7]}</span><span style="opacity:0.3">{text[7:8]}</span><span style="opacity:0;">{text[8:]}</span></span>'
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__data_text_size, 'color':'black'}
              , yanchor='bottom'
              , xanchor='left'
              , yshift=-1
              , xshift=-2
              , row=row_num, col=col_num
            )

            label_loc += .2
            x_count += 1
            x_1 = x_2
        if len(instance.top5_upstream) == 0:
            self.__fig.add_annotation(
                x=label_loc
              , y=y_diff
              , text='<span><b style="font-size:88%"></b>&#8239;</span>'
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__data_text_size, 'color':'black'}
              , yanchor='bottom'
              , xanchor='left'
              , yshift=-1
              , xshift=-2
              , row=row_num, col=col_num
            )

        x_2 = 1
        self.__fig.add_trace(
            go.Scatter(
                x=np.array([x_1, x_2 - x_fudge, x_2 - x_fudge, x_1, x_1])
              , y=np.array([0.005, 0.005, y_diff, y_diff, 0.005])
              , fill='toself'
              , mode='none'
              , fillcolor='rgba(200,200,200,0.3)'
              , showlegend=False)
          , row=row_num, col=col_num
        )

        x_1 = 0
        x_count = 0
        label_loc = 0
        for _, row in instance.top5_downstream.iterrows():
            x_2 = row['cum_perc']
            if row['instance'] is None:
                text = "NoName"
            else:
                text = row['instance'][:-2]

            styleadd = "color:#000;"
            if row['instance'][-1] == 'L':
                styleadd = "color:#8b008b;"

            self.__fig.add_trace(
                go.Scatter(
                    x=np.array([x_1, x_2 - x_fudge, x_2 - x_fudge, x_1])
                  , y=np.array([-0.005, -0.005, -y_diff, -y_diff])
                  , fill='tozeroy'
                  , mode='none'
                  , fillcolor=col_o[x_count]
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )

            letter_spacing = ""
            if len(text)>7:
                letter_spacing = "letter-spacing:-.08em;font-stretch:60%;"
            elif len(text)>6:
                letter_spacing = "letter-spacing:-.05em;font-stretch:75%;"
            self.__fig.add_annotation(
                x=label_loc
              , y=-y_diff
              , text=f'<span style="{letter_spacing}{styleadd}"><b style="font-size:88%;color:{col_o[x_count]};">{x_count+1}</b>&#8239;{text[:6]}<span style="color:#999">{text[6:7]}</span><span style="color:#CCC">{text[7:8]}</span><span style="color:#FFF">{text[8:]}</span></span>'
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__data_text_size, 'color': 'black'}
              , yanchor='top'
              , xanchor='left'
              , yshift=1
              , xshift=-2
              , row=row_num, col=col_num
            )

            label_loc += .2
            x_count +=1
            x_1 = x_2
        if len(instance.top5_downstream) == 0:
            self.__fig.add_annotation(
                x=label_loc
              , y=-y_diff
              , text='<span><b style="font-size:88%"></b>&#8239;</span>'
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__data_text_size, 'color':'black'}
              , yanchor='top'
              , xanchor='left'
              , yshift=1
              , xshift=-2
              , row=row_num, col=col_num
            )
        x_2 = 1
        self.__fig.add_trace(
            go.Scatter(
                x=np.array([x_1, x_2 - x_fudge, x_2 - x_fudge, x_1, x_1])
              , y=np.array([-0.005, -0.005, -y_diff, -y_diff, -0.005])
              , fill='toself'
              , mode='none'
              , fillcolor='rgba(200,200,200,0.3)'
              , showlegend=False)
          , row=row_num, col=col_num
        )
        #add x-axis labels
        if row_num < len(self.__instances):
            self.__fig.update_xaxes(
                range=[0,1]
              , showticklabels=False
              , row=row_num
              , col=col_num
            )
        else:
            self.__fig.update_xaxes(
                range=[0, 1]
              , tickvals=[0, .5, 1]
              , ticks='outside'
              , tickcolor='black'
              , tickwidth=.6
              , ticktext=['0&#8239;%', '50&#8239;%', '100&#8239;%']
              , ticklen=3
              , color='black'
              , tickangle=-90
              , tickfont={'family': 'Arial', 'size':self.__axis_label_text_size, 'color':self.__axis_label_text_color}
              , row=row_num, col=col_num
            )
            start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
            colcnt = start_last_row + col_num
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.add_annotation(
                text='percentage of total connections'
              , y=0
              , yshift=-39
              , font={'family': 'Arial', 'size':8, 'color':self.__axis_label_text_color}
              , xref=f"x{colcnt} domain"
              , yref=f"y{colcnt} domain"
              , yanchor='bottom'
              , xanchor='center'
              , showarrow=False
            )
        self.__fig.update_yaxes(
            showticklabels=False
          , row=row_num, col=col_num
        )


    def __add_innervation(self):
        row_count = 0
        rel_roi = self.__get_ordered_rois(self.__col_inn.keys())
        for ct in self.__instances:
            row_count += 1
            col_count = len(self.__col_syn) + len(self.__col_top5)
            plotsb = True
            for roitp in rel_roi:
                col_count += 1
                self.__plot_wid_hist(
                    instance=ct
                  , roi_to_plot=roitp
                  , row_num=row_count, col_num=col_count
                  , scale_flag=plotsb
                )
                plotsb = False


    def __plot_wid_hist(
        self
      , instance
      , roi_to_plot:str
      , row_num:int, col_num:int
      , scale_flag:bool
    ):
        scale_bar_xpos = -.01
        y_range = [0, 1]
        gridline_col = 'gainsboro'
        tmp_bnd = self.__get_layer_bnd(roi_str=roi_to_plot)
        tmp_tit = self.__get_layer_titles(roi_str=roi_to_plot)
        col_name = self.__get_full_roi_name(roi_str=roi_to_plot)

        for boundary in tmp_bnd:
            self.__fig.add_trace(
                go.Scatter(
                    x=[boundary, boundary]
                  , y=y_range
                  , mode='lines'
                  , line={'color': gridline_col, 'width': .6}
                  , showlegend=False)
              , row=row_num, col=col_num)
            self.__fig.update_yaxes(showticklabels=False, row=row_num, col=col_num)

        #add gridlines
        for bound in tmp_bnd:
            self.__fig.add_trace(
                go.Scatter(
                    x=[bound, bound]
                  , y=y_range
                  , mode='lines'
                  , line={'color': gridline_col, 'width': .6}
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )

        if len(instance.innervation) == 0:
            if row_num < len(self.__instances):
                self.__fig.update_xaxes(
                    showticklabels=False
                  , row=row_num, col=col_num
                )
            else:
                ave_db = [np.mean([x,y]) for (x,y) in pairwise(tmp_bnd)]
                self.__fig.update_xaxes(
                    tickvals=ave_db
                  , ticktext=tmp_tit
                  , tickangle=-90
                  , color='black'
                  , tickfont={'family': 'Arial', 'size':8}
                  , row=row_num, col=col_num
                )
                return

        tmp_max_wid = instance.innervation['col_innervation'].max()
        if tmp_max_wid < 2 or np.isnan(tmp_max_wid):
            sc_bar = 1
        elif tmp_max_wid < 4:
            sc_bar = 2
        elif tmp_max_wid < 10:
            sc_bar = 5
        elif tmp_max_wid < 20:
            sc_bar = 10
        elif tmp_max_wid < 40:
            sc_bar = 20
        elif tmp_max_wid < 100:
            sc_bar = 50
        elif tmp_max_wid < 200:
            sc_bar = 100
        elif tmp_max_wid < 400:
            sc_bar = 200
        elif tmp_max_wid < 1000:
            sc_bar = 500
        else:
            sc_bar = 1000
        div_fac = 2 * sc_bar

        # add scale bar
        if scale_flag:
            self.__fig.add_trace(
                go.Scatter(
                    x=[scale_bar_xpos, scale_bar_xpos]
                  , y=np.divide([0, sc_bar], div_fac)
                  , mode='lines'
                  , line={'color': 'black', 'width': 1}
                  , showlegend=False
                )
              , row=row_num, col=col_num
            )
            self.__fig.add_annotation(
                x=scale_bar_xpos
              , y=sc_bar/div_fac/2+.1
              , text=f"{sc_bar}"
              , showarrow=False
              , font={'family': 'Arial', 'size':self.__axis_label_text_size, 'color':self.__axis_label_text_color}
              , yanchor='middle'
              , xanchor='right'
              , textangle=-90
              , xshift=1
              , row=row_num, col=col_num
            )

        temp_df = instance.innervation[instance.innervation['roi']==roi_to_plot]
        if roi_to_plot == 'ME(R)':
            fill_color = OL_COLOR.OL_NEUROPIL.nrgb['ME']
            fill_color = f"rgba{fill_color[3:-1]}, 0.4)"
        elif roi_to_plot == 'LO(R)':
            fill_color = OL_COLOR.OL_NEUROPIL.nrgb['LO']
            fill_color = f"rgba{fill_color[3:-1]}, 0.4)"
        elif roi_to_plot == 'LOP(R)':
            fill_color = OL_COLOR.OL_NEUROPIL.nrgb['LOP']
            fill_color = f"rgba{fill_color[3:-1]}, 0.4)"
        else:
            fill_color = 'rgba(200,200,200,0.7)'

        self.__fig.add_trace(
            go.Scatter(
                x=temp_df['depth']
              , y=scipy.signal.savgol_filter(temp_df['col_innervation']/div_fac, 5, 1, mode='interp')
              , fill='tozeroy'
              , fillcolor=fill_color
              , mode='lines'
              , line={'color': 'black', 'width': .5}
              , showlegend=False
            )
          , row=row_num, col=col_num
        )

        #add horizontal line at 0
        self.__fig.add_trace(
            go.Scatter(
                x=[-0.01,1.01]
              , y=[0, 0]
              , mode='lines'
              , line_width=0.5
              , line_color="black"
              , showlegend=False
            )
          , row=row_num, col=col_num
        )

        if row_num < len(self.__instances):
            self.__fig.update_xaxes(
                showticklabels=False
              , row=row_num, col=col_num
            )
        else:
            ave_db = [np.mean([x,y]) for (x,y) in pairwise(tmp_bnd)]
            self.__fig.update_xaxes(
                tickvals=ave_db
              , ticktext=tmp_tit
              , tickangle=-90
              , color='black'
              , tickfont={'family': 'Arial', 'size':self.__axis_label_text_size, 'color':self.__axis_label_text_color}
              , row=row_num, col=col_num
            )
            start_last_row = len(self.__get_col_widths())*(len(self.__instances)-1)
            colcnt = start_last_row + col_num 
            colcnt = colcnt if colcnt>1 else ""
            self.__fig.add_annotation(
                text=col_name
              , y=0
              , yshift=-39
              , font={'family': 'Arial', 'size':8, 'color':self.__axis_label_text_color}
              , xref=f"x{colcnt} domain"
              , yref=f"y{colcnt} domain"
              , yanchor='bottom'
              , xanchor='center'
              , showarrow=False
            )

        self.__fig.update_yaxes(showticklabels=False, row=row_num, col=col_num)


    def __update_layout(self):
        col_i = OL_COLOR.OL_SYNAPSES.map['post']
        col_o = OL_COLOR.OL_SYNAPSES.map['pre']
        col_pre = OL_COLOR.OL_SYNAPSES.map['pre']
        col_post = OL_COLOR.OL_SYNAPSES.map['post']
        self.__fig.update_xaxes(showgrid=False)
        self.__fig.update_yaxes(showgrid=False)

        self.__fig.update_layout(
            width=self.__get_col_width()*1200
          , height=(50*len(self.__instances))+100
          , margin={'l': 75, 'r': 23, 't': 85, 'b':50}
          , title=self.__title
          , title_font={'family': 'Arial', 'size':12, 'color':'black'}
          , title_x=0.5
          , title_y=0.985
          , title_yanchor='top'
          , title_xanchor='center'
          , plot_bgcolor='white'
          , font={'family': 'Arial'}
        )

        #add column labels
        self.__fig.add_annotation(
            text='<b style="font-size:106%">cell type</b><br>count&nbsp;<b style="font-size:80%">(R|L)</b><br>transmitter'
          , align='right'
          , x=0
          , y=0
          , yshift=45
          , font={'family': 'Arial', 'size':10, 'color':'black'}
          , xref="x domain"
          , yref="y domain"
          , yanchor='bottom'
          , xanchor='right'
          , showarrow=False
        )

        self.__fig.add_annotation(
            text=f'synapse distribution<br>'\
              f'by depth<br>'\
              f'<span style="color:{col_post};">postsynaptic (input to)</span> '\
              f'and <span style="color:{col_pre};">presynaptic (output from)</span>'
          , x=0
          , y=0
          , xshift=10
          , yshift=45
          , font={'family': 'Arial', 'size':10, 'color':'black'}
          , xref="x domain"
          , yref="y domain"
          , xanchor='left'
          , yanchor='bottom'
          , showarrow=False
        )
        self.__fig.add_annotation(
            text=f'top 5<br>'\
                f'<span style="color:{col_i}">inputs to</span> <b style="font-size:106%">cell type</b><br>'\
                f'<span style="color:{col_o}">outputs from</span> <b style="font-size:106%">cell type</b>'
          , x=0.5
          , y=0
          , yshift=45
          , font={'family': 'Arial', 'size':10, 'color':'black'}
          , xref="x5 domain"
          , yref="y domain"
          , xanchor='center'
          , yanchor='bottom'
          , showarrow=False
        )
        self.__fig.add_annotation(
            text="number of innervated columns<br>by depth<br>&nbsp;"
          , x=0
          , y=0
          , yshift=45
          , font={'family': 'Arial', 'size':10, 'color':'black'}
          , xref="x7 domain"
          , yref="y domain"
          , xanchor='center'
          , yanchor='bottom'
          , showarrow=False
        )
