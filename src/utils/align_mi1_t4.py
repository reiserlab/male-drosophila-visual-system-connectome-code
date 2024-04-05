import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.ROI_calculus import _get_data_path

from neuprint import fetch_neurons, fetch_synapse_connections,\
    merge_neuron_properties, fetch_all_rois
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC


def _get_out_syn_and_com(
    cell_name:str
  , roi_name:str='ME(R)'
) -> pd.DataFrame:
    """
    private function replacing duplicate code parts
    """
    assert cell_name in ['Mi1'], "Wrong cell type given. "
    assert roi_name in ['ME(R)'], "Wrong ROI given."

    neuron_criteria_pre = NC(
        type=cell_name,
        regex=True) # regex to deal with all types of T4 and T5 (T4.* or T4[a-d])
    neuron_df, _ = fetch_neurons(neuron_criteria_pre)

    syn_criteria = SC(rois=roi_name)
    syn_df = fetch_synapse_connections(
        neuron_criteria_pre
      , None
      , syn_criteria)

    nid_post = pd.unique(syn_df['bodyId_post']).astype(int)
    neuron_criteria  = NC(bodyId=nid_post)
    neuron_post_df , _ = fetch_neurons(neuron_criteria)
    comb_df = pd.concat([
            neuron_df[['bodyId', 'type', 'somaLocation']],
            neuron_post_df[['bodyId', 'type', 'somaLocation']]])\
        .drop_duplicates(subset='bodyId')\
        .reset_index(drop=True)
    rel_syn_df = merge_neuron_properties(comb_df, syn_df, ['type', 'somaLocation'])
    
    com_df = rel_syn_df[rel_syn_df['type_post'].isin(['T4a', 'T4b', 'T4c', 'T4d'])]\
        .groupby(['bodyId_pre'])[['x_pre', 'y_pre', 'z_pre']]\
        .mean()
    com_df['type'] = cell_name
    
    return rel_syn_df, com_df
    

def _get_in_com(
    cell_name:str
  , roi_name:str='ME(R)'
) -> pd.DataFrame:
    """
    Another private function replacing duplicate code parts
    """
    assert cell_name in ['T4a', 'T4b', 'T4c', 'T4d'], "Wrong cell type given. "
    assert roi_name in ['ME(R)'], "Wrong ROI given."

    neuron_criteria_post  = NC(
        type=cell_name,
        regex=True) # regex to deal with all types of T4 and T5 (T4.* or T4[a-d])
    neuron_df, _ = fetch_neurons(neuron_criteria_post)

    syn_criteria = SC(rois=roi_name)
    syn_df = fetch_synapse_connections(
        None
      , neuron_criteria_post
      , syn_criteria)

    nid_pre = pd.unique(syn_df['bodyId_pre']).astype(int)
    neuron_criteria  = NC(bodyId=nid_pre)
    neuron_pre_df , _ = fetch_neurons(neuron_criteria)
    comb_df = pd.concat([
            neuron_df[['bodyId', 'type', 'somaLocation']],
            neuron_pre_df[['bodyId', 'type', 'somaLocation']]])\
        .drop_duplicates(subset='bodyId')\
        .reset_index(drop=True)
    rel_syn_df = merge_neuron_properties(comb_df, syn_df, ['type', 'somaLocation'])

    com_df = syn_df\
            .groupby('bodyId_post')[['x_post', 'y_post', 'z_post']]\
            .mean()
    com_df['type'] = cell_name
    
    return com_df


def create_alignment() -> None:
    """
    Create the file `mi1_t4_alignment.xlsx`
    """

    mi1outsyn, mi1_com = _get_out_syn_and_com('Mi1')
    t4a_com = _get_in_com('T4a')
    t4b_com = _get_in_com('T4b')
    t4c_com = _get_in_com('T4c')
    t4d_com = _get_in_com('T4d')

    mi1_t4a_dist = pd.DataFrame(columns=t4a_com.index, index=mi1_com.index)
    mi1_t4b_dist = pd.DataFrame(columns=t4b_com.index, index=mi1_com.index)
    mi1_t4c_dist = pd.DataFrame(columns=t4c_com.index, index=mi1_com.index)
    mi1_t4d_dist = pd.DataFrame(columns=t4d_com.index, index=mi1_com.index)

    for bid_pre in mi1_com.index:
        for bid_post in t4a_com.index:
            mi1_t4a_dist.at[bid_pre, bid_post] = pow(
                pow(mi1_com['x_pre'][bid_pre] - t4a_com['x_post'][bid_post], 2)
              + pow(mi1_com['y_pre'][bid_pre] - t4a_com['y_post'][bid_post], 2)
              + pow(mi1_com['z_pre'][bid_pre] - t4a_com['z_post'][bid_post], 2)
              , 1 / 2)

        for bid_post in t4b_com.index:
            mi1_t4b_dist.at[bid_pre, bid_post] = pow(
                pow(mi1_com['x_pre'][bid_pre] - t4b_com['x_post'][bid_post], 2)
              + pow(mi1_com['y_pre'][bid_pre] - t4b_com['y_post'][bid_post], 2)
              + pow(mi1_com['z_pre'][bid_pre] - t4b_com['z_post'][bid_post], 2)
              , 1 / 2)

        for bid_post in t4c_com.index:
            mi1_t4c_dist.at[bid_pre, bid_post] = pow(
                pow(mi1_com['x_pre'][bid_pre] - t4c_com['x_post'][bid_post], 2)
              + pow(mi1_com['y_pre'][bid_pre] - t4c_com['y_post'][bid_post], 2)
              + pow(mi1_com['z_pre'][bid_pre] - t4c_com['z_post'][bid_post], 2)
              , 1 / 2)

        for bid_post in t4d_com.index:
            mi1_t4d_dist.at[bid_pre, bid_post] = pow(
                pow(mi1_com['x_pre'][bid_pre] - t4d_com['x_post'][bid_post], 2)
              + pow(mi1_com['y_pre'][bid_pre] - t4d_com['y_post'][bid_post], 2)
              + pow(mi1_com['z_pre'][bid_pre] - t4d_com['z_post'][bid_post], 2)
              , 1 / 2)

    mi1_mi1_dist = pd.DataFrame(columns=mi1_com.index, index=mi1_com.index)

    for bid_pre_idx in range(len(mi1_com)):
        for bid_post_idx in range(bid_pre_idx+1, len(mi1_com)):
            mi1_mi1_dist.iat[bid_pre_idx, bid_post_idx] = pow(
                pow(mi1_com['x_pre'].iloc[bid_pre_idx] - mi1_com['x_pre'].iloc[bid_post_idx], 2)
              + pow(mi1_com['y_pre'].iloc[bid_pre_idx] - mi1_com['y_pre'].iloc[bid_post_idx], 2)
              + pow(mi1_com['z_pre'].iloc[bid_pre_idx] - mi1_com['z_pre'].iloc[bid_post_idx], 2)
              , 1 / 2)

    empty_a = np.zeros((mi1_com.shape[0], t4a_com.shape[0]))
    empty_b = np.zeros((mi1_com.shape[0], t4b_com.shape[0]))
    empty_c = np.zeros((mi1_com.shape[0], t4c_com.shape[0]))
    empty_d = np.zeros((mi1_com.shape[0], t4d_com.shape[0]))

    mi1_t4a_syn = mi1outsyn[mi1outsyn['type_post']=='T4a']\
        .groupby(['bodyId_pre', 'bodyId_post'])\
        .size()
    mi1_t4b_syn = mi1outsyn[mi1outsyn['type_post']=='T4b']\
        .groupby(['bodyId_pre', 'bodyId_post'])\
        .size()
    mi1_t4c_syn = mi1outsyn[mi1outsyn['type_post']=='T4c']\
        .groupby(['bodyId_pre', 'bodyId_post'])\
        .size()
    mi1_t4d_syn = mi1outsyn[mi1outsyn['type_post']=='T4d']\
        .groupby(['bodyId_pre', 'bodyId_post'])\
        .size()

    mi1_t4a_syn_wide = pd.DataFrame(data=empty_a, columns=t4a_com.index, index=mi1_com.index)
    mi1_t4b_syn_wide = pd.DataFrame(data=empty_b, columns=t4b_com.index, index=mi1_com.index)
    mi1_t4c_syn_wide = pd.DataFrame(data=empty_c, columns=t4c_com.index, index=mi1_com.index)
    mi1_t4d_syn_wide = pd.DataFrame(data=empty_d, columns=t4d_com.index, index=mi1_com.index)

    for m_idx, c_val in mi1_t4a_syn.items():
        mi1_t4a_syn_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4b_syn.items():
        mi1_t4b_syn_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4c_syn.items():
        mi1_t4c_syn_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4d_syn.items():
        mi1_t4d_syn_wide.at[m_idx[0], m_idx[1]] = c_val

    mi1_t4a_tot = mi1outsyn[mi1outsyn['type_post']=='T4a'].groupby('bodyId_pre').size()
    mi1_t4b_tot = mi1outsyn[mi1outsyn['type_post']=='T4b'].groupby('bodyId_pre').size()
    mi1_t4c_tot = mi1outsyn[mi1outsyn['type_post']=='T4c'].groupby('bodyId_pre').size()
    mi1_t4d_tot = mi1outsyn[mi1outsyn['type_post']=='T4d'].groupby('bodyId_pre').size()

    mi1_t4a_frac = mi1_t4a_syn\
        .reset_index(name='count')\
        .merge(mi1_t4a_tot.reset_index(name='sum'), on='bodyId_pre')
    mi1_t4b_frac = mi1_t4b_syn\
        .reset_index(name='count')\
        .merge(mi1_t4b_tot.reset_index(name='sum'), on='bodyId_pre')
    mi1_t4c_frac = mi1_t4c_syn\
        .reset_index(name='count')\
        .merge(mi1_t4c_tot.reset_index(name='sum'), on='bodyId_pre')
    mi1_t4d_frac = mi1_t4d_syn\
        .reset_index(name='count')\
        .merge(mi1_t4d_tot.reset_index(name='sum'), on='bodyId_pre')

    mi1_t4a_frac['frac'] = mi1_t4a_frac['count'].div(mi1_t4a_frac['sum'])
    mi1_t4b_frac['frac'] = mi1_t4b_frac['count'].div(mi1_t4b_frac['sum'])
    mi1_t4c_frac['frac'] = mi1_t4c_frac['count'].div(mi1_t4c_frac['sum'])
    mi1_t4d_frac['frac'] = mi1_t4d_frac['count'].div(mi1_t4d_frac['sum'])

    mi1_t4a_ser = mi1_t4a_frac.set_index(['bodyId_pre', 'bodyId_post'])['frac']
    mi1_t4b_ser = mi1_t4b_frac.set_index(['bodyId_pre', 'bodyId_post'])['frac']
    mi1_t4c_ser = mi1_t4c_frac.set_index(['bodyId_pre', 'bodyId_post'])['frac']
    mi1_t4d_ser = mi1_t4d_frac.set_index(['bodyId_pre', 'bodyId_post'])['frac']

    empty_a = np.zeros((mi1_com.shape[0], t4a_com.shape[0]))
    empty_b = np.zeros((mi1_com.shape[0], t4b_com.shape[0]))
    empty_c = np.zeros((mi1_com.shape[0], t4c_com.shape[0]))
    empty_d = np.zeros((mi1_com.shape[0], t4d_com.shape[0]))

    mi1_t4a_frac_wide = pd.DataFrame(data=empty_a, columns=t4a_com.index, index=mi1_com.index)
    mi1_t4b_frac_wide = pd.DataFrame(data=empty_b, columns=t4b_com.index, index=mi1_com.index)
    mi1_t4c_frac_wide = pd.DataFrame(data=empty_c, columns=t4c_com.index, index=mi1_com.index)
    mi1_t4d_frac_wide = pd.DataFrame(data=empty_d, columns=t4d_com.index, index=mi1_com.index)

    for m_idx, c_val in mi1_t4a_ser.items():
        mi1_t4a_frac_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4b_ser.items():
        mi1_t4b_frac_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4c_ser.items():
        mi1_t4c_frac_wide.at[m_idx[0], m_idx[1]] = c_val

    for m_idx, c_val in mi1_t4d_ser.items():
        mi1_t4d_frac_wide.at[m_idx[0], m_idx[1]] = c_val

    mi1_t4a_cost = mi1_t4a_frac_wide
    mi1_t4b_cost = mi1_t4b_frac_wide
    mi1_t4c_cost = mi1_t4c_frac_wide
    mi1_t4d_cost = mi1_t4d_frac_wide

    a_r_idx, a_c_idx = linear_sum_assignment(mi1_t4a_cost, maximize=True)
    b_r_idx, b_c_idx = linear_sum_assignment(mi1_t4b_cost, maximize=True)
    c_r_idx, c_c_idx = linear_sum_assignment(mi1_t4c_cost, maximize=True)
    d_r_idx, d_c_idx = linear_sum_assignment(mi1_t4d_cost, maximize=True)

    t4a_dat = pd.DataFrame(
        data={'mi1_bid': mi1_t4a_cost.index[a_r_idx], 't4a_bid': mi1_t4a_cost.columns[a_c_idx]})
    t4b_dat = pd.DataFrame(
        data={'mi1_bid': mi1_t4b_cost.index[b_r_idx], 't4b_bid': mi1_t4b_cost.columns[b_c_idx]})
    t4c_dat = pd.DataFrame(
        data={'mi1_bid': mi1_t4c_cost.index[c_r_idx], 't4c_bid': mi1_t4c_cost.columns[c_c_idx]})
    t4d_dat = pd.DataFrame(
        data={'mi1_bid': mi1_t4d_cost.index[d_r_idx], 't4d_bid': mi1_t4d_cost.columns[d_c_idx]})

    mi1_t4_align_df = t4a_dat\
        .merge(t4b_dat, how='outer', on='mi1_bid')\
        .merge(t4c_dat, how='outer', on='mi1_bid')\
        .merge(t4d_dat, how='outer', on='mi1_bid')

    temp_col = np.empty((mi1_t4_align_df.shape[0], 1))
    temp_col[:] = np.nan

    for pairs in mi1_t4_align_df.iterrows():
        temp_list = []
        mi1_id = pairs[1][0]
        t4a_id = pairs[1][1]
        if ~np.isnan(t4a_id):
            temp_list.append(mi1_t4a_dist.at[mi1_id, t4a_id])

        t4b_id = pairs[1][2]
        if ~np.isnan(t4b_id):
            temp_list.append(mi1_t4b_dist.at[mi1_id, t4b_id])

        t4c_id = pairs[1][3]
        if ~np.isnan(t4c_id):
            temp_list.append(mi1_t4c_dist.at[mi1_id, t4c_id])
        t4d_id = pairs[1][4]
        if ~np.isnan(t4d_id):
            temp_list.append(mi1_t4d_dist.at[mi1_id, t4d_id])

        temp_col[pairs[0]] = np.max(temp_list)

    mi1_t4_align_df['max_dist'] = temp_col
    mi1_t4_align_df['num_t4s'] = mi1_t4_align_df.iloc[:, 1:5].notna().sum(axis=1)

    dist_cutoff = 10**3
    mi1_t4_align_df['valid_group'] = 0
    logical_crit = \
        (mi1_t4_align_df['max_dist'].lt(dist_cutoff)) \
      & (mi1_t4_align_df['num_t4s'].eq(4))

    mi1_t4_align_df.loc[logical_crit, 'valid_group'] = 1

    red_mi1_t4_align = mi1_t4_align_df[
        (mi1_t4_align_df['max_dist'].gt(dist_cutoff))\
      & (mi1_t4_align_df['num_t4s'].eq(4))
    ]
    potential_col_df = pd.DataFrame(
        data=np.zeros([red_mi1_t4_align.shape[0], 4])
      , columns=['t4a', 't4b', 't4c', 't4d']
      , index=red_mi1_t4_align.index
    )

    for indx, pairs in red_mi1_t4_align.iterrows():
        mi1_id = pairs[0]
        t4a_id = pairs[1]
        potential_col_df.at[indx, 't4a'] = mi1_t4a_dist.at[mi1_id, t4a_id]
        t4b_id = pairs[2]
        potential_col_df.at[indx, 't4b'] = mi1_t4b_dist.at[mi1_id, t4b_id]
        t4c_id = pairs[3]
        potential_col_df.at[indx, 't4c'] = mi1_t4c_dist.at[mi1_id, t4c_id]
        t4d_id = pairs[4]
        potential_col_df.at[indx, 't4d'] = mi1_t4d_dist.at[mi1_id, t4d_id]

    for indx, pairs in potential_col_df.iterrows():
        rel_t4s_bool = pairs.lt(dist_cutoff)
        if rel_t4s_bool.sum() == 3:
            temp_t4type = pairs.index[pairs.gt(dist_cutoff)][0]
            mi1_t4_align_df.at[indx, temp_t4type+'_bid'] = np.nan
            mi1_t4_align_df.at[indx, 'num_t4s'] = 3
            mi1_t4_align_df.at[indx, 'max_dist'] = pairs[rel_t4s_bool].max()
            mi1_t4_align_df.at[indx, 'valid_group'] = 1

    data_path = _get_data_path(reason='data')
    table_fn = data_path / "mi1_t4_alignment.xlsx"
    table_df = mi1_t4_align_df.rename(
                columns={
                    'mi1_bid': 'Mi1'
                  , 't4a_bid': 'T4a'
                  , 't4b_bid': 'T4b'
                  , 't4c_bid': 'T4c'
                  , 't4d_bid': 'T4d'}
        )
    table_df.to_excel(table_fn,index=False)
