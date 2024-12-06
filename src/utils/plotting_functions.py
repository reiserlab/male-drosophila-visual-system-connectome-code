from pathlib import Path
from dotenv import find_dotenv
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.ROI_calculus import load_depth_bins, load_layer_thre

from utils.ol_color import OL_COLOR


def order_confusion_matrix(
    confusion_matrix
  , types_unique
  , order_types=True
  , order_clusters=True
  , order_frac=0.05
):
    """
    Plots the confusion matrix as a heatmap with 5 discrete categories and with annotation.
    
    Parameters
    ----------
    confusion_matrix : np.array
        confusion matrix of morphology clustering of size (number of cell types) x (number of clusters)
    types_unique : list[str]
        list of cell type names corresponding to the rows of the confusion_matrix
    order_types : bool, default=True
        order cell type names lexicographically; only works for the exemplary 68 ME cell types right now
    order_clusters : bool, default=True
        order clusters by number of cells contained in consecutive cell types or not
    order_frac : float, default=0.05
        only used if order_clusters=True; specifies minimum fraction of cells of a cell type in a cluster in order to be included in cluster ordering of that cell type
        
    Returns
    -------
    confusion_matrix_ordered : np.array
        ordered confusion matrix of size (number of cell types) x (number of clusters)
    types_ordered : list[str]
        lexicographically ordered list of cell type names if order_types=True
    clusters_ordered_inv : np.array
        array of indices that orders the clusters if order_clusters=True
    """
    assert confusion_matrix.shape[0]==len(types_unique),\
        f"Number of rows of confusion matrix should match number of cell types"    

    if order_types:
        types_ordered = np.array(['Cm1', 'Cm2', 'Cm3', 'Cm4', 'Cm5', 'Cm6', 'Cm7', 'Cm8', 'Cm9', 'Cm10', 'Cm11a', 'Cm11b', 'Cm11c', 'Cm12', 'Cm13', 'Cm14', 'Cm15', 'Cm16', 'Cm17', 'Cm18', \
                                  'Cm19', 'Cm20', 'Cm21', 'Cm22', 'Dm1', 'Dm2', 'Dm3a', 'Dm3b', 'Dm3c', 'Dm4', 'Dm6', 'Dm8a', 'Dm8b', 'Dm9', 'Dm10', 'Dm11', 'Dm12', 'Dm13', 'Dm14', 'Dm15', \
                                  'Dm16', 'Dm18', 'Dm19', 'Dm20', 'Dm-DRA1', 'Dm-DRA2', 'Mi1', 'Mi2', 'Mi4', 'Mi9', 'Mi10', 'Mi13', 'Mi14', 'Mi15', 'Mi16', 'Mi17', 'Mi18', 'Pm1', 'Pm2a', \
                                  'Pm2b', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10'])
        types_ordered = types_ordered[np.isin(types_ordered, types_unique)]
        ind_ordered = np.array([np.where(types_unique==types_ordered[i])[0][0] for i in range(len(types_ordered))])
        assert ind_ordered.shape[0]==types_ordered.shape[0],\
            f"Type ordering can currently only be done for subsets of the exemplary 68 ME-intrinsic cell types"    
        confusion_matrix = confusion_matrix[ind_ordered]
    else:
        types_ordered = types_unique

    if order_clusters:
        n_types = confusion_matrix.shape[0]
        n_clusters = confusion_matrix.shape[1]
        n_cells_per_type = confusion_matrix.sum(1)
        
        clusters_ordered = np.array(range(n_clusters))+1
        confusion_matrix_ordered = confusion_matrix.copy()
        l=0
        for i in range(n_types):
            i2 = np.argmax( confusion_matrix_ordered[i,l:] ) + l
            while (confusion_matrix_ordered[i,i2]>order_frac*n_cells_per_type[i])&(l<n_clusters-1): 
                confusion_matrix_ordered[:,[i2,l]] = confusion_matrix_ordered[:,[l,i2]]
                clusters_ordered[[i2,l]] = clusters_ordered[[l,i2]]
                l+=1
                i2 = np.argmax( confusion_matrix_ordered[i,l:] ) + l
        clusters_ordered_inv = np.array([np.where( clusters_ordered==i+1 )[0][0] for i in range(n_clusters)])
    else:
        confusion_matrix_ordered = confusion_matrix.copy()
        clusters_ordered_inv = np.array(range(n_clusters))

    return confusion_matrix_ordered, types_ordered, clusters_ordered_inv


def plot_morpho_feature_vectors(
    bodyId_list
  , roi_str
  , n_clusters
  , order_types=True
  , order_clusters=True
  , order_frac=0.05
) -> None:
    """
   Store plots of feature vectors that were used for morphological clustering as 'morpho_features_bodyId' (pdf and svg) in the results folder morpho_clustering.
   Also show cluster id after cluster ordering (if order_clusters=True)
    
    Parameters
    ----------
    bodyId_list : list
        list of bodyIds
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    n_clusters : int
        number of clusters\
    order_types : bool, default=True
        order cell type names lexicographically; only works for the exemplary 68 ME cell types right now
    order_clusters : bool, default=True
        order clusters by number of cells contained in consecutive cell types or not
    order_frac : float, default=0.05
        only used if order_clusters=True; specifies minimum fraction of cells of a cell type in a cluster in order to be included in cluster ordering of that cell type
        
    Returns
    -------
    None
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"    

    results_path = Path(Path(find_dotenv()).parent, 'results', 'morpho_clustering')
    results_path.mkdir(parents=True, exist_ok=True)
    
    #load morphology data
    data_path = Path(Path(find_dotenv()).parent, 'cache', 'morpho_clustering')
    features0 = pd.read_csv(os.path.join(data_path, 'morpho_features.csv'),header=None).values
    bodyIds = np.squeeze(pd.read_csv(os.path.join(data_path, 'morpho_bodyIds.csv'),header=None).values)
    types = np.squeeze(pd.read_csv(os.path.join(data_path, 'morpho_types.csv'),header=None).values)
    
    #number of depth bins in feature vector
    n_bins = int(features0.shape[1]/4)                                                                     
    #normalize synapse to size distributions
    features = features0.copy()
    ind_pre = np.where( features0[:,:n_bins].sum(1) )[0]
    features[ind_pre,:n_bins] = ( (features0[:,n_bins:2*n_bins]).mean(1)/features0[ind_pre,:n_bins].mean(1) )[:,np.newaxis]*features0[ind_pre,:n_bins]
    ind_post = np.where( features0[:,2*n_bins:3*n_bins].sum(1) )[0]
    features[ind_post,2*n_bins:3*n_bins] = ( features0[:,3*n_bins:4*n_bins].mean(1)/features0[ind_post,2*n_bins:3*n_bins].mean(1) )[:,np.newaxis]*features0[ind_post,2*n_bins:3*n_bins]
    
    #find cluster_ids for ordered clusters
    cluster_ids = np.squeeze(pd.read_csv(os.path.join(data_path, 'morpho_clusters_'+str(n_clusters)+'clu.csv'),header=None).values).astype(int)
    if order_clusters:
        #need confusion matrix for ordering
        data = pd.read_csv(os.path.join(data_path,  'morpho_confusion_mat_'+str(n_clusters)+'clu.csv'))
        confusion_matrix = data.values[:,1:]
        types_unique = data.values[:,0]

        _, _, clusters_ordered_inv = order_confusion_matrix(confusion_matrix, types_unique, order_types=order_types, order_clusters=order_clusters, order_frac=order_frac)
        cluster_ids_ordered = np.array([clusters_ordered_inv[cluster_ids[i]-1]+1 for i in range(cluster_ids.shape[0])])
    else:
        cluster_ids_ordered = cluster_ids
    
    bin_edges, bin_centers = load_depth_bins(roi_str=roi_str)
    depth_bdry = load_layer_thre(roi_str=roi_str)
    linewidth = 2
    
    for bodyId in bodyId_list:
        n0 = np.where( bodyIds==bodyId )[0][0]
        x_max = np.round(1.1*features[n0,2*n_bins:3*n_bins].max())
        
        fig = make_subplots(rows=1, cols=2,
                           horizontal_spacing=0.2)
        
        for y0 in depth_bdry:
            fig.add_trace(go.Scatter(
                x=[0, x_max]
              , y=[y0, y0]    
              , mode='lines'
              , line={'color': 'gainsboro', 'width': .75}
              , showlegend=False
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features[n0,2*n_bins:3*n_bins]
          , y=bin_centers
          , mode='lines', name='post'
          , line={'color': OL_COLOR.OL_SYNAPSES.nrgb['post'], 'width': linewidth}
          , showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=features[n0,:n_bins]
          , y=bin_centers
          , mode='lines', name='pre'
          , line={'color': OL_COLOR.OL_SYNAPSES.nrgb['pre'], 'width': linewidth}
          , showlegend=False
        ), row=1, col=1)
        
        fig.update_xaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
          , range=[0,x_max]
          , tickvals=[0,x_max]
          , tickfont={'family': 'Arial', 'size':10}
          , row=1, col=1
        )
        fig.update_yaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
          , range=[depth_bdry[-1],depth_bdry[0]]
          , tickvals=[depth_bdry[-1],depth_bdry[0]]
          , showticklabels=False
          , row=1, col=1
        )
        
        for y0 in depth_bdry:
            fig.add_trace(go.Scatter(
                x=[0, x_max]
              , y=[y0, y0]    
              , mode='lines'
              , line={'color': 'gainsboro', 'width': .75}
              , showlegend=False
            ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=features[n0,3*n_bins:4*n_bins]
          , y=bin_centers
          , mode='lines', name='post'
          , line={'color': OL_COLOR.OL_SYNAPSES.nrgb['post'], 'width': linewidth}
          , showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=features[n0,n_bins:2*n_bins]
          , y=bin_centers
          , mode='lines', name='pre'
          , line={'color': OL_COLOR.OL_SYNAPSES.nrgb['pre'], 'width': linewidth}
          , showlegend=False
        ), row=1, col=2)
        
        fig.update_xaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
          , range=[0,x_max]
          , tickvals=[0,x_max]
          , tickfont={'family': 'Arial', 'size':9}
            , row=1, col=2
        )
        fig.update_yaxes(
            showline=True
          , linewidth=1
          , linecolor='black'
          , mirror=True
          , range=[depth_bdry[-1],depth_bdry[0]]
          , tickvals=[depth_bdry[-1],depth_bdry[0]]
          , showticklabels=False
          , row=1, col=2
        )
        fig.update_layout(
            height=400
          , width=300
          , paper_bgcolor='rgba(255,255,255,255)'
          , plot_bgcolor='rgba(255,255,255,255)'
          , title_font={'family': 'Arial', 'size':12, 'color':'black'}
          , title_text='bodyId: %d <br> type: %s <br> cluster: %d'%(bodyId,types[n0],cluster_ids_ordered[n0])
          , title_x=0.5
        )
    
        fig.write_image(results_path / f"morpho_features_{bodyId}.pdf")
        fig.write_image(results_path / f"morpho_features_{bodyId}.svg")


def plot_confusion_matrix_w_colors(   
    confusion_matrix
  , types_unique
  , frac_min = 0.1
  , frac_max = 0.8
  , order_types=True
  , order_clusters=True
  , order_frac=0.05
) -> None:
    """
    Plots the confusion matrix as a heatmap with 5 discrete categories and with annotation.
    
    Parameters
    ----------
    confusion_matrix : np.array
        confusion matrix of morphology clustering of size (number of cell types) x (number of clusters)
    types_unique : list[str]
        list of cell type names corresponding to the rows of the confusion_matrix
    frac_min : float, default=0.1
        lower limit for categorization
    frac_max : float, default=0.8
        upper limit for categorization
    order_types : bool, default=True
        order cell type names lexicographically; only works for the exemplary 68 ME cell types right now
    order_clusters : bool, default=True
        order clusters by number of cells contained in consecutive cell types or not
    order_frac : float, default=0.05
        only used if order_clusters=True; specifies minimum fraction of cells of a cell type in a cluster in order to be included in cluster ordering of that cell type
        
    Returns
    -------
    None
    """

    results_path = Path(Path(find_dotenv()).parent, 'results', 'morpho_clustering')
    results_path.mkdir(parents=True, exist_ok=True)
    
    confusion_matrix_ordered, types_ordered, _ = order_confusion_matrix(confusion_matrix, types_unique, order_types=order_types, order_clusters=order_clusters, order_frac=order_frac)
    n_types = confusion_matrix_ordered.shape[0]
    n_clusters = confusion_matrix_ordered.shape[1]
    n_cells_per_type = confusion_matrix_ordered.sum(1)
    n_cells_per_cluster = confusion_matrix_ordered.sum(0)
    
    #annotation
    anno = np.empty((n_types,n_clusters), dtype=object)
    for j in range(n_clusters):
        for i in range(n_types):
            if confusion_matrix_ordered[i,j] > .5:
                anno[i,j] = confusion_matrix_ordered[i,j]
            else:
                anno[i,j] = None
    anno = np.where(anno == None, '', anno)
    
    #fraction per cluster
    p_hom = np.zeros((n_types,n_clusters))
    id_nonzero = np.where(n_cells_per_cluster>0)[0]
    p_hom[:,id_nonzero] = (confusion_matrix_ordered[:,id_nonzero]/n_cells_per_cluster[np.newaxis,id_nonzero]).astype(float)
    #fraction per cell type
    p_com = np.zeros((n_types,n_clusters))
    id_nonzero = np.where(n_cells_per_type>0)[0]
    p_com[id_nonzero] = (confusion_matrix_ordered[id_nonzero]/n_cells_per_type[id_nonzero,np.newaxis]).astype(float)
    
    discrete_labels = np.zeros((n_types,n_clusters),dtype=int)
    discrete_labels[confusion_matrix_ordered>0] = 1
    for i in range(n_types):
        j1 = np.where( p_com[i]>=frac_max )[0]
        j2 = np.where( (p_com[i]<frac_max)&(p_com[i]>=frac_min) )[0]
        if j1.shape[0]>0:
            #1-1 cluster
            if p_hom[i,j1[0]]>=frac_max:
                discrete_labels[i,j1[0]]=2
            #many-1 cluster
            else:
                discrete_labels[i,j1[0]]=3
        #1-many cluster
        elif j2.shape[0]>0:
            k2 = np.where( p_hom[i,j2]>=frac_max )[0]
            if k2.shape[0]>0:
                discrete_labels[i,j2[k2]]=4    
    for j in range(n_clusters):
        i2 = np.where( (p_hom[:,j]<frac_max)&(p_hom[:,j]>=frac_min) )[0]    
        if i2.shape[0]>0:
            k2 = np.where( (p_com[i2,j]<frac_max)&(p_com[i2,j]>=frac_min) )[0]
            #mixed cluster
            if k2.shape[0]>0:
                discrete_labels[i2[k2],j]=5
    
    discrete_colors = ['white','lightgray','firebrick','royalblue','forestgreen','khaki']
    
    #size of plot
    if n_types==68:
        width = 1200*0.98
        height = 1200
    else:
        width = 400
        height = 400
        
    fig = plot_heatmap(
        pd.DataFrame(discrete_labels)
      , pd.DataFrame(anno)
      , binned=True
      , bins=np.linspace(-.5,5.5,7)
      , pal=discrete_colors
      , show_colorbar=False
      , show_grid=True
      , show_bounding_box=False
      , anno_text_size=6
      , gap=False
      , equal_aspect_ratio=False
      , manual_margin=False
    )
    fig.update_layout(xaxis=dict(side="top", tickangle=0, \
                                 tickfont=dict(size=12), tickvals=np.array(range(n_clusters))[9::10], ticktext=np.array(range(n_clusters))[9::10]+1))
    fig.update_layout(yaxis=dict(tickfont=dict(size=12), tickvals=np.array(range(n_types)), ticktext=types_ordered))    
    fig.update_layout(
          width=width, height=height
        , font={"family":'Arial'}
    )
    fig.update_xaxes(
        showline=True #False
      , mirror=True
      , linewidth=1
      , linecolor='black'
      , range=[-.5,n_clusters-.5]
      , showgrid=False
      , showticklabels=True
      , visible=True
    )
    fig.update_yaxes(
        showline=True #False
      , mirror=True
      , linewidth=1
      , linecolor='black'
      , range=[n_types-.5,-.5]
      , showgrid=False
      , showticklabels=True
      , visible=True
    )    

    fig.write_image(results_path / f'morpho_confusion_mat_{n_clusters}clu.pdf')
    fig.write_image(results_path / f'morpho_confusion_mat_{n_clusters}clu.svg')


def plot_flip_syn_hist(
    fig_obj:go.Figure
  , hist_celltype_df:pd.DataFrame
  , roi_to_plot:str
  , row_num:int, col_num:int
  , layer_bound_dict:dict
  , lay_tit_dict:dict
) -> go.Figure:
    """
    Plot the syn histogram part by roi based on `tot_hist_df` with x and y axes flipped.

    it returns the fig object with the traces added in the designated row and column.

    layer_bound_dict is used to draw the lines for layer boundaries. it is generated with
        `ROI_calculus.load_layer_thre()`

    Parameters
    ----------
    fig_obj : go.Figure
        plotly go.Figure to be modified
    hist_celltype_df : pd.DataFrame
        hist_df for an individual target celltype. Generated using `get_hists_df()`
    roi_to_plot : str
        name of ROI for which to extract the relevant target celltype data
    row_num : int
        row position in the subplot
    col_num : int
        column position in the subplot
    layer_bound_dict : dict
        dictionary for the layer boundaries (value) of each ROI (keys)
    lay_tit_dict : dict
        dictionary for the layer names (value) of each ROI (keys)

    Returns
    -------
    go.Figure :
        modified plotly go.Figure `fig_obj`
    """
    y_range = [0, 1]
    gridline_col = 'gainsboro'
    input_color = ['rgb(253, 198, 43)']
    output_color = ['rgb(7, 177, 210)']
    tmp_bnd = layer_bound_dict[roi_to_plot]
    tmp_tit = lay_tit_dict[roi_to_plot]

    for boundary in tmp_bnd:
        fig_obj.add_trace(
            go.Scatter(
                x=y_range
              , y=[boundary, boundary]
              , mode='lines'
              , line={'color': gridline_col, 'width': .75}
              , showlegend=False)
          , row=row_num, col=col_num)

    rel_hist_df = hist_celltype_df[hist_celltype_df['roi'] == roi_to_plot]
    n_max = 0.9*np.max([rel_hist_df['norm_inp_count'].max(), rel_hist_df['norm_out_count'].max()])
    fig_obj.add_trace(
        go.Scatter(
            x=rel_hist_df['norm_inp_count']/n_max
          , y=rel_hist_df['hist_cen']
          , fill='toself'
          , fillcolor=f"rgba{input_color[0][3:-1]}, 0.8)"
          , mode='lines'
          , line={'color': 'black', 'width': .5}
          , showlegend=False
        )
      , row=row_num
      , col=col_num
    )

    fig_obj.add_trace(
        go.Scatter(
            x=rel_hist_df['norm_out_count']/n_max
          , y=rel_hist_df['hist_cen']
          , fill='toself'
          , fillcolor = f'rgba{output_color[0][3:-1]}, 0.4)'
          , mode = 'lines'
          , line={'color': 'black', 'width': .5}
          , showlegend=False)
      , row=row_num, col=col_num)
    
    #add x-axis labels
    if col_num > 0:
        fig_obj.update_yaxes(showticklabels = False, row=row_num, col=col_num)
    else:
        ave_db = [np.mean([y,x]) for (x,y) in pairwise(tmp_bnd)]
        fig_obj.update_yaxes(
            tickvals=ave_db
          , ticktext=tmp_tit
          # , tickangle=-90
          , color='black'
          , tickfont={'size':8}
          , row=row_num
          , col=col_num
        )
    # if col_num > 1:
    #     fig_obj.update_xaxes(showticklabels=False, row=row_num, col=col_num)
    fig_obj.update_yaxes(autorange="reversed", row=row_num, col=col_num)

    return fig_obj


def plot_consistent_connections_by_type(
    pre_post_syn_df:pd.DataFrame, dir_type:str
) -> go.Figure:
    """
    Plots histograms and boxplots to summerize the pre to post connectivity (from one pre to all
      the post of the same type).

    Plots 2 scatter plots summarizing all the connected pre_type connections
    1. fraction of post_cells that have a connection with a specific pre_type vs. median fraction
      of input from that pre-type
    2. sorted fraction of post_cells that have a connection with a specific pre_type (same
      variable as above)

    plots shows the pre-type and the number of cell that actually have this connections
    (since the fraction can be high but for a small number of post-cells)

    Parameters
    ----------
    pre_post_syn_df : pandas.DataFrame
        Data frame generated by the 'fetch_indiv_pretypes_of_posttype_v2' function
    dir_type: str
        if plotting different pretypes to a target post, type should be 'pre'.
        if plotting different posttypes to a target pre, type should be 'post'.

    Returns
    -------
    fig : go.Figure
    """

    assert dir_type in ['pre', 'post'], 'type should be either pre or post'
    conn_type = f'type_{dir_type}'
    frac_type = f'frac_tot_{dir_type}'
    tot_syn_type = f'tot_syn_per_{dir_type}'
    if dir_type == 'pre':
        body_id_type = 'bodyId_post'
        tit1 = 'frac post cells with <br>pre_type'
        tit2 = 'frac of input from <br>pre_type'
    else:
        body_id_type = 'bodyId_pre'
        tit1 = 'frac pre cells with <br>post_type'
        tit2 = 'frac of output to <br>post_type'

    # calculating connection consistency

    all_types = pre_post_syn_df[conn_type]\
        .unique()
    temp = pre_post_syn_df\
        .groupby([body_id_type])[conn_type]\
        .apply(lambda x: set(x))

    type_counter = np.array([temp.apply(lambda x: ntype in x).sum() for ntype in all_types])

    type_pre_count_df = pd.DataFrame(all_types)
    type_pre_count_df['type_counter'] = type_counter
    type_pre_count_df = type_pre_count_df.rename(columns={0: conn_type})
    type_pre_count_df['type_frac'] = type_pre_count_df['type_counter'] / temp.shape[0]

    # reducing it to by type connections (to not over count)
    crit = pre_post_syn_df['rank_dense'] == 1
    rank1_df = pre_post_syn_df[crit]

    med_df = rank1_df\
        .groupby([conn_type])[frac_type]\
        .median()\
        .reset_index()
    med_syn_df = rank1_df\
        .groupby([conn_type])[tot_syn_type]\
        .median()\
        .reset_index()

    type_pre_count_df = type_pre_count_df\
        .sort_values(by='type_frac', ascending=False)\
        .reset_index(drop=True)

    type_pre_count_df = type_pre_count_df.merge(med_df, on=conn_type)
    type_pre_count_df = type_pre_count_df.merge(med_syn_df, on=conn_type)

    # generating the right strings for hovering
    type_str = type_pre_count_df[conn_type].values
    nums = [int(num) for num in type_pre_count_df['type_counter'].values]
    fracs = [num for num in type_pre_count_df['type_frac'].values]

    num_str = [f'{number:.0f}' for number in nums]
    syn_str = [f'{syn_num:.0f}' for syn_num in type_pre_count_df[tot_syn_type].values]
    frac_str = [f'{number:.2f}' for number in fracs]

    point_txt1 = []
    for cell_type, num, syn_num in zip(type_str, num_str, syn_str):
        point_txt1.append(f'{cell_type}<br>Count:{num}<br>Med_syn_num:{syn_num}')

    point_txt2 = []
    for cell_type, num in zip(type_str, frac_str):
        point_txt2.append(f'{cell_type}<br>Frac:{num}')

    # plotting the figure

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15)

    fig.add_trace(
        go.Scatter(
            x=type_pre_count_df['type_frac'], y=type_pre_count_df[frac_type],
            hoverinfo='text', hovertext=point_txt1,
            mode='markers', marker={'color':'Blue', 'size':6},
            showlegend=False),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=type_pre_count_df[conn_type], y=type_pre_count_df['type_frac'],
            hoverinfo='text', hovertext=point_txt2,
            mode='markers', marker={'color':'Blue', 'size':6},
            showlegend=False),
        row=2, col=1)

    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(title_text=tit1, row=1, col=1)
    fig.update_yaxes(title_text=tit2, row=1, col=1)
    fig.update_yaxes(title_text=tit1, row=2, col=1)

    fig.update_layout(
        font = {'size':12},
        width=1000, height=700
    )

    return fig


def plot_heatmap(
    heatmap:pd.DataFrame
  , anno:pd.DataFrame=None
  , binned:bool=False
  , bins:np.ndarray|list=None
  , pal:list=None
  , show_colorbar:bool=True
  , show_grid:bool=False
  , show_bounding_box:bool=True
  , anno_text_size:int=6
  , gap:bool=True
  , equal_aspect_ratio:bool=True
  , manual_margin:bool=True
) -> go.Figure:
    """
    Plots heatmap, binned or continuous
    Parameters
    ----------
    heatmap : pandas.DataFrame
        dataframe to be plotted
    anno : pd.DataFrame, default=None
        annotation dataframe
    binned : bool, default=False
        if True, plot binned heatmap
    bins : np.ndarray or list, default=None
        7 boundary values for 6 bins. For cont. color scale, bins[-1] sets the max value.
    pal : list, default=[]
        palette for heatmap
    show_colorbar : bool, default=True
        show colorbar
    show_grid : bool, default=False
        show all gray gridlines, behind heatmap
    show_bounding_box : bool, default=True
        show bounding box around heatmap
    anno_text_size : int, default=6
        annotation text size
    gap : bool, default=True
        adds a small gap between squares in the heatmap
    equal_aspect_ratio : bool, default=True
        sets the aspect ratio (of initial plot and for interactive scaling) to 1
    manual_margin : bool, default=True
        manually sets the margin of the layout variables 'l', 'r', and 'pad' to 0
    Returns
    -------
    fig : go.Figure
        Plotly figure object containing a heatmap
    """

    # plotting params, cf. Laura
    fig_format = {
        'fig_width': 3
      , 'fig_height': 3
      , 'fig_margin': 0.01
      , 'font_type': 'arial'
      , 'fsize_ticks_pt': 6
      , 'fsize_title_pt': 6
      , 'markersize': 10
      , 'markerlinewidth': 1
      , 'markerlinecolor': 'black'
      , 'ticklen': 3.5
      , 'tickwidth': 1
      , 'axislinewidth': 1.2
    }

    fig_w = (fig_format['fig_width'] - fig_format['fig_margin'])*96
    fig_h = (fig_format['fig_height'] - fig_format['fig_margin'])*96
    fsize_ticks_px = fig_format['fsize_ticks_pt']*(1/72)*96
    fsize_title_px = fig_format['fsize_title_pt']*(1/72)*96

    layout_heatmap = go.Layout(
        paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
      , autosize = False
      , width = fig_w
      , height = fig_h
      , showlegend = False
    )

    layout_xaxis_heatmap = go.layout.XAxis(
        title_font={
            'size': fsize_title_px
          , 'family': fig_format['font_type']
          , 'color' : 'black'
        }
      , ticks=""
      , tickfont={
            'family': fig_format['font_type']
          , 'size': fsize_ticks_px
          , 'color': 'black'
        }
      , tickangle=45
      , side="top"
      , showgrid = False
      , showline= False
    )

    layout_yaxis_heatmap = go.layout.YAxis(
        title_font={
            'size': fsize_title_px
          , 'family': fig_format['font_type']
          , 'color': 'black'
        }
      , ticks=""
      , tickfont={
            'family': fig_format['font_type']
          , 'size': fsize_ticks_px
          , 'color': 'black'
        }
      , showgrid = False
      , showline= False
      , autorange="reversed"
    )

    # if anno is not None, it must have the same shape as heatmap
    if anno is not None:
        assert heatmap.shape == anno.shape, 'heatmap and annotation should have the same shape'

    # if pal provide
    if pal is not None:
        assert len(pal) > 1, 'palette should have at least 2 colors'
    else:
        # default color
        pal = OL_COLOR.HEATMAP.hex

    # initialize figure
    fig = go.Figure(layout = layout_heatmap)
    fig.update_xaxes(layout_xaxis_heatmap)
    fig.update_yaxes(layout_yaxis_heatmap)

    if show_grid:
        xval = heatmap.columns.values
        if (isinstance(xval[0], np.int32)) | (isinstance(xval[0], np.int64)):
            dx = np.diff(xval)
            for j in range(xval.shape[0]-1):
                fig.add_vline(x=xval[j]+dx[j]/2, line_width=.25, line_color="gray")
        yval = heatmap.index.values
        if (isinstance(yval[0], np.int32)) | (isinstance(yval[0], np.int64)):
            dy = np.diff(yval)
            for i in range(yval.shape[0]-1):
                fig.add_hline(y=yval[i]+dy[i]/2, line_width=.25, line_color="gray")

    # plot either binned/discrete or continuous heatmap
    if binned:
        if bins is not None:
            assert len(bins) > 2, 'bins should have at least 3 values, ie. 2 bins'
            bins_text = bins
        else:
            bins = heatmap.values.max()*np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
            bins_text = [f'{b:.1f}' for b in bins]

        heatmap_bin = heatmap.copy()
        heatmap_bin[:] = 1
        for i in range(len(bins)-1):
            heatmap_bin[(heatmap > bins[i]) & (heatmap <= bins[i+1])] = i+1

        # these 2 var are basically constants, keep them here for possible extension
        bvals = np.arange(len(bins))
        nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values

        # tick positions at boundaries
        bpos = bvals

        # create discrete colorscale
        dcolorscale = []
        for idx, val in enumerate(pal):
            dcolorscale.extend([[nvals[idx], val], [nvals[idx+1], val]])

        # add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_bin
              , x=heatmap_bin.columns.values
              , y=heatmap_bin.index.values
              , colorscale=dcolorscale
              , zmax=len(bins)-1
              , zmin=1
              , colorbar={
                    'tickvals': bpos
                  , 'ticktext': bins_text
                }
            )
        )
    # continuous heatmap
    else:
        if bins is not None:
            assert len(bins) >=2, 'at least 2 value'
            bins_text = bins
        else:
            # 6 bins linearly spaced
            bins = np.linspace(0, heatmap.values.max(), 2)
            bins_text = [f'{b:.1f}' for b in bins]

        # tick positions at boundaries
        bpos = bins

        fig.add_trace(
            go.Heatmap(
                z=heatmap
              , x=heatmap.columns.values
              , y=heatmap.index.values
              , colorscale= OL_COLOR.HEATMAP.rgb
              , zmin=0
              , zmax=bins[-1]
              , colorbar={
                    'tickvals': bpos
                  , 'ticktext': bins_text
                }
            )
        )

    # bounding box around the plot, works better with asp=1
    if show_bounding_box:
        n_y, n_x = heatmap.shape
        fig.add_shape(
            type='rect', xref='x', yref='y'
          , x0=-0.5, y0=-0.5, x1=n_x-0.5, y1=n_y-0.5
          , line={'color': 'black', 'width': 0.5}
        )

    # with colorbar
    fig.update_traces(showscale=show_colorbar)

    # add space between cells
    if gap:
        fig.update_traces(xgap=1, ygap=1)

    # fix aspect ratio
    if equal_aspect_ratio:
        fig.update_layout(yaxis_scaleanchor="x")

    #set margin
    if manual_margin:
        fig.update_layout(margin={'l':0, 'r':0, 'pad':0})

    # with colorbar
    fig.update_traces(showscale=show_colorbar)

    # add space between cells
    if gap:
        fig.update_traces(xgap=1, ygap=1)

    # fix aspect ratio
    if equal_aspect_ratio:
        fig.update_layout(yaxis_scaleanchor="x")

    #set margin
    if manual_margin:
        fig.update_layout(margin={'l':0, 'r':0, 'pad':0})

    # annotations
    fig.update_traces(
        text=anno
      , texttemplate="%{text}"
      , textfont_size=anno_text_size
      , hovertemplate=None
      , textfont_family=fig_format['font_type']
    )

    return fig