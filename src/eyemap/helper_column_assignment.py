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
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from neuprint import fetch_synapse_connections
from utils.hex_hex import get_hex_df, hex_to_bids
from utils.ROI_columns import load_hexed_body_ids
from utils.ROI_calculus import find_neuron_hex_ids
from utils.plotter import group_plotter, show_figure
from utils.hex_hex import bid_to_hex
from utils.neuron_bag import NeuronBag

# %%
# Cells to resolve
# L1: [901806]
# C2: [174453]
# Tm2: [53166, 56117, 59291, 60078, 61701, 65036, 69305, 80428, 108648, 112981]
# Double entry Tm2: [104527]

# Resolutions
# L1: [901806] -> [21,36], overwrite current entry which is 43309 (an L3 cell)
# -> L3: [21,36] = 49422, does not exist, overwite with 43309
# C2: [174453] -> [36,33], overwrite current null entry
# Tm2: [53166] -> [14,25]
# Tm2: [56117] -> [27,22]
# Tm2: [59291] -> [30,33]
# Tm2: [60078] -> [28,24]
# Tm2: [61701] -> [22,19]
# Tm2: [65036] -> [25,34]
# Tm2: [69305] -> [17,23]
# Tm2: [80428] -> [21,23]
# Tm2: [108648] -> [15,33]
# Tm2: [112981] -> [32,38]
#
# Double entry Tm2: [104527] was allocated to Tm4 at [9,1] in a second row. 
# This second row is now eliminated, which I hope eliminates the error.

# %%
# Implementing resolutions. See cells below for working.
hex_df = get_hex_df()
updated_hex_df = hex_df.copy()

# L1: [901806] -> [21,36], overwrite current entry which is 43309 (an L3 cell)
# -> L3: [21,36] = 49422, does not exist, overwite with 43309
row_data = hex_df.loc[(hex_df['hex1_id']==21) & (hex_df['hex2_id']==36)].copy()
row_data['L1'] = 901806
row_data['L3'] = 43309
updated_hex_df.loc[(hex_df['hex1_id']==21) & (hex_df['hex2_id']==36)] = row_data

# C2: [174453] -> [36,33], overwrite current null entry
row_data = hex_df.loc[(hex_df['hex1_id']==36) & (hex_df['hex2_id']==33)].copy()
row_data['C2'] = 174453
updated_hex_df.loc[(hex_df['hex1_id']==36) & (hex_df['hex2_id']==33)] = row_data

# Tm2: [53166] -> [14,25]
row_data = hex_df.loc[(hex_df['hex1_id']==14) & (hex_df['hex2_id']==25)].copy()
row_data['Tm2'] = 53166
updated_hex_df.loc[(hex_df['hex1_id']==14) & (hex_df['hex2_id']==25)] = row_data

# Tm2: [56117] -> [27,22]
row_data = hex_df.loc[(hex_df['hex1_id']==27) & (hex_df['hex2_id']==22)].copy()
row_data['Tm2'] = 56117
updated_hex_df.loc[(hex_df['hex1_id']==27) & (hex_df['hex2_id']==22)] = row_data

# Tm2: [59291] -> [30,33]
row_data = hex_df.loc[(hex_df['hex1_id']==30) & (hex_df['hex2_id']==33)].copy()
row_data['Tm2'] = 59291
updated_hex_df.loc[(hex_df['hex1_id']==30) & (hex_df['hex2_id']==33)] = row_data

# Tm2: [60078] -> [28,24]
row_data = hex_df.loc[(hex_df['hex1_id']==28) & (hex_df['hex2_id']==24)].copy()
row_data['Tm2'] = 60078
updated_hex_df.loc[(hex_df['hex1_id']==28) & (hex_df['hex2_id']==24)] = row_data

# Tm2: [61701] -> [22,19]
row_data = hex_df.loc[(hex_df['hex1_id']==22) & (hex_df['hex2_id']==19)].copy()
row_data['Tm2'] = 61701
updated_hex_df.loc[(hex_df['hex1_id']==22) & (hex_df['hex2_id']==19)] = row_data

# Tm2: [65036] -> [25,34]
row_data = hex_df.loc[(hex_df['hex1_id']==25) & (hex_df['hex2_id']==34)].copy()
row_data['Tm2'] = 65036
updated_hex_df.loc[(hex_df['hex1_id']==25) & (hex_df['hex2_id']==34)] = row_data

# Tm2: [69305] -> [17,23]
row_data = hex_df.loc[(hex_df['hex1_id']==17) & (hex_df['hex2_id']==23)].copy()
row_data['Tm2'] = 69305
updated_hex_df.loc[(hex_df['hex1_id']==17) & (hex_df['hex2_id']==23)] = row_data

# Tm2: [80428] -> [21,23]
row_data = hex_df.loc[(hex_df['hex1_id']==21) & (hex_df['hex2_id']==23)].copy()
row_data['Tm2'] = 80428
updated_hex_df.loc[(hex_df['hex1_id']==21) & (hex_df['hex2_id']==23)] = row_data

# Tm2: [108648] -> [15,33]
row_data = hex_df.loc[(hex_df['hex1_id']==15) & (hex_df['hex2_id']==33)].copy()
row_data['Tm2'] = 108648
updated_hex_df.loc[(hex_df['hex1_id']==15) & (hex_df['hex2_id']==33)] = row_data

# Tm2: [112981] -> [32,38]
row_data = hex_df.loc[(hex_df['hex1_id']==32) & (hex_df['hex2_id']==38)].copy()
row_data['Tm2'] = 112981
updated_hex_df.loc[(hex_df['hex1_id']==32) & (hex_df['hex2_id']==38)] = row_data

# Tm4: [99852] -> [9,1]
# In second row, Tm: [104527] which generates Tm2 duplication error
updated_hex_df = updated_hex_df.drop([0,136])
updated_hex_df = updated_hex_df.reset_index(drop=True)
# updated_hex_df.iloc[130:150,:]


# %%
updated_hex_df.to_csv('ME_hex_ids_updated_10Dec2023.csv')

# %%
columnar_types = ['L1', 'L2', 'L3', 'L5', 'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20', 'T1']
no_assignment = []
double_assignment = []

for cell_type in columnar_types:
    bag = NeuronBag(cell_type=cell_type)
    bids =  bag.get_body_ids(cell_count=bag.size)
    for bid in bids:
        hxs = bid_to_hex(bid=bid)
        if isinstance(hxs, list):
            double_assignment.append(bid)
        elif not hxs:
            no_assignment.append(bid)


# %%
# RESOLVING L1
# Get L1 synapses
conns = fetch_synapse_connections(None,NC(bodyId=901806),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 21; hex2 = 36
hexid_df = find_neuron_hex_ids(syn_df)

# L1: 43309 is assigned to [21,36]
hid = hexid_df[['hex1_id','hex2_id']]
hex_to_bids([21,36])

# %%
# Check which columns have L1
hex_to_bids([20,34],['L1'])

# %%
# Plot L1 cells around this column to assign 901806
# No L1 in 
# [21,37]
# [20,36]
fh = group_plotter([901806,
                    hex_to_bids([21,36],['L1'])['L1'], 
                    hex_to_bids([21,35],['L1'])['L1'],
                    hex_to_bids([21,34],['L1'])['L1'],
                    hex_to_bids([20,35],['L1'])['L1'],
                    hex_to_bids([20,34],['L1'])['L1']],
                   [(0,0,0,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                                (.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)


# %%
# RESOLVING L3
# Check which columns have L3
hex_to_bids([20,34],['L3'])
# Yes [21,36],[21,35],[21,34]
#             [20,35],[20,34]
hex_to_bids([20,34],['L3'])['L3']

# %%
# Plot L3 cells around this column to assign 
fh = group_plotter([43309,
                    #hex_to_bids([21,36],['L3'])['L3'], # [49422] does not exist
                    hex_to_bids([21,35],['L3'])['L3'], # [95115]
                    hex_to_bids([21,34],['L3'])['L3']], # [91550]
                    # hex_to_bids([20,35],['L3'])['L3'], # [97316]
                    # hex_to_bids([20,34],['L3'])['L3']], # [45196]
                   [#(0,0,0,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                                (.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)


# %%
# RESOLVING C2
# Get C2 synapses
conns = fetch_synapse_connections(None,NC(bodyId=174453),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 36; hex2 = 33
hexid_df = find_neuron_hex_ids(syn_df)

# C2: 174453 is assigned to [36,33]
# hid = hexid_df[['hex1_id','hex2_id']]
hex_to_bids([36,33])


# %%
# Check which columns have C2
hex_to_bids([36,33],['C2'])
# Yes ?[36,33], [35,33], [34,33], [33,33]
#      [36,32], [35,32], [34,32], [33,32]

# %%
# Plot C2 cells around this column to assign 
fh = group_plotter([174453,
                    hex_to_bids([35,33],['C2'])['C2'], # 
                    hex_to_bids([34,33],['C2'])['C2'], # 
                    hex_to_bids([33,33],['C2'])['C2'], # 
                    hex_to_bids([36,32],['C2'])['C2'], # 
                    hex_to_bids([35,32],['C2'])['C2'], # 
                    hex_to_bids([34,32],['C2'])['C2'], # 
                    hex_to_bids([33,32],['C2'])['C2']], # 
                   [(0,0,0,1),  (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (0,0,0,0.5),(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)

# %%
# Plot C2 cells around this column to assign 
fh = group_plotter([hex_to_bids([36,32],['C2'])['C2'], # 
                    hex_to_bids([35,32],['C2'])['C2'], # 
                    hex_to_bids([34,32],['C2'])['C2'], # 
                    hex_to_bids([33,32],['C2'])['C2']], # 
                   [(0,0,0,0.5),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (0,0,0,0.5),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)

# %%
# Plot C2 cells around this column to assign 
fh = group_plotter([174453,
                    hex_to_bids([35,33],['C2'])['C2'], # 
                    hex_to_bids([34,33],['C2'])['C2'], # 
                    hex_to_bids([33,33],['C2'])['C2']], # 
                   [(0,0,0,0.5),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (0,0,0,0.5),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)

# %%
# RESOLVING Tm2
# Get Tm2 [53166] synapses
conns = fetch_synapse_connections(None,NC(bodyId=53166),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 14; hex2 = 25
hexid_df = find_neuron_hex_ids(syn_df)

# %%
# Check which columns have Tm2
hex_to_bids([15,24],['Tm2'])
# Yes [15,24], [15,25], [15,26]
#     [14,24],          [14,26]
#     [13,24], [13,25], [13,26]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([15,24],['Tm2'])['Tm2'], # 
                    hex_to_bids([15,25],['Tm2'])['Tm2'], # 
                    hex_to_bids([15,26],['Tm2'])['Tm2'],
                    hex_to_bids([14,24],['Tm2'])['Tm2'], # 
                    53166, 
                    hex_to_bids([14,26],['Tm2'])['Tm2'],
                    hex_to_bids([13,24],['Tm2'])['Tm2'], # 
                    hex_to_bids([13,25],['Tm2'])['Tm2'], # 
                    hex_to_bids([13,26],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
#show_figure(fh,800, 600)
# fit fot [14,25] is good

# %%
# Get Tm2 [56117] synapses
conns = fetch_synapse_connections(None,NC(bodyId=56117),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 27; hex2 = 22
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([26,23],['Tm2'])
# Yes [28,21], [28,22], [28,23]
#     [27,21],          [27,23]
#     [26,21], [26,22], [26,23]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([28,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([28,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([28,21],['Tm2'])['Tm2'],
                    hex_to_bids([27,23],['Tm2'])['Tm2'], # 
                    56117, 
                    hex_to_bids([27,21],['Tm2'])['Tm2'],
                    hex_to_bids([26,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([26,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([26,21],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit fot [27,22] is good

# %%
# Get Tm2 [59291] synapses
conns = fetch_synapse_connections(None,NC(bodyId=59291),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 30; hex2 = 33
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([29,34],['Tm2'])
# Yes [31,32], [31,33], [31,34]
#     [30,32],          [30,34]
#     [29,32], [29,33], [29,34]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([31,32],['Tm2'])['Tm2'], # 
                    hex_to_bids([31,33],['Tm2'])['Tm2'], # 
                    hex_to_bids([31,34],['Tm2'])['Tm2'],
                    hex_to_bids([30,32],['Tm2'])['Tm2'], # 
                    59291, 
                    hex_to_bids([30,34],['Tm2'])['Tm2'],
                    hex_to_bids([29,32],['Tm2'])['Tm2'], # 
                    hex_to_bids([29,33],['Tm2'])['Tm2'], # 
                    hex_to_bids([29,34],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [30,33] is good

# %%
# Get Tm2 [60078] synapses
conns = fetch_synapse_connections(None,NC(bodyId=60078),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 28; hex2 = 24
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([27,25],['Tm2'])
# Yes [29,23], [29,24], [29,25]
#     [28,23],          [28,25]
#     [27,23], [27,24], [27,25]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([29,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([29,24],['Tm2'])['Tm2'], # 
                    hex_to_bids([29,25],['Tm2'])['Tm2'],
                    hex_to_bids([28,23],['Tm2'])['Tm2'], # 
                    60078, 
                    hex_to_bids([28,25],['Tm2'])['Tm2'],
                    hex_to_bids([27,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([27,24],['Tm2'])['Tm2'], # 
                    hex_to_bids([27,25],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [28,24] is good

# %%
# Get Tm2 [61701] synapses
conns = fetch_synapse_connections(None,NC(bodyId=61701),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 22; hex2 = 19
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([23,20],['Tm2'])
# Yes [23,18], [23,19], [23,20]
#     [22,18],          [22,20]
#     [21,18], [21,19], [21,20]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([23,18],['Tm2'])['Tm2'], # 
                    hex_to_bids([23,19],['Tm2'])['Tm2'], # 
                    hex_to_bids([23,20],['Tm2'])['Tm2'],
                    hex_to_bids([22,18],['Tm2'])['Tm2'], # 
                    61701, 
                    hex_to_bids([22,20],['Tm2'])['Tm2'],
                    hex_to_bids([21,18],['Tm2'])['Tm2'], # 
                    hex_to_bids([21,19],['Tm2'])['Tm2'], # 
                    hex_to_bids([21,20],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [22,19] is good

# %%
# Get Tm2 [65036] synapses
conns = fetch_synapse_connections(None,NC(bodyId=65036),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 25; hex2 = 34
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([25,34],['Tm2'])

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([26,33],['Tm2'])['Tm2'], # 
                    hex_to_bids([26,34],['Tm2'])['Tm2'], # 
                    hex_to_bids([26,35],['Tm2'])['Tm2'],
                    hex_to_bids([25,33],['Tm2'])['Tm2'], # 
                    65036, 
                    hex_to_bids([25,35],['Tm2'])['Tm2'],
                    hex_to_bids([24,33],['Tm2'])['Tm2'], # 
                    hex_to_bids([24,34],['Tm2'])['Tm2'], # 
                    hex_to_bids([24,35],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [25,34] is good

# %%
# Get Tm2 [69306] synapses
conns = fetch_synapse_connections(None,NC(bodyId=69305),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 17; hex2 = 23
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([17,23],['Tm2'])

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([18,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([18,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([18,24],['Tm2'])['Tm2'],
                    hex_to_bids([17,22],['Tm2'])['Tm2'], # 
                    69305, 
                    hex_to_bids([17,24],['Tm2'])['Tm2'],
                    hex_to_bids([16,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([16,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([16,24],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [17,23] is good

# %%
# Get Tm2 [80428] synapses
conns = fetch_synapse_connections(None,NC(bodyId=80428),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 21; hex2 = 23
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([21,23],['Tm2'])

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([22,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([22,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([22,24],['Tm2'])['Tm2'],
                    hex_to_bids([21,22],['Tm2'])['Tm2'], # 
                    80428, 
                    hex_to_bids([21,24],['Tm2'])['Tm2'],
                    hex_to_bids([20,22],['Tm2'])['Tm2'], # 
                    hex_to_bids([20,23],['Tm2'])['Tm2'], # 
                    hex_to_bids([20,24],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1),
                    (1,.5,.5,1),(0, 0, 0,1),(.5,.5,1,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [21,23] is good

# %%
# Get Tm2 [108648] synapses
conns = fetch_synapse_connections(None,NC(bodyId=108648),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 15; hex2 = 33
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([14,34],['Tm2'])

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([16,32],['Tm2'])['Tm2'], # 
                    hex_to_bids([16,33],['Tm2'])['Tm2'], # 
                    hex_to_bids([15,32],['Tm2'])['Tm2'], # 
                    108648, 
                    hex_to_bids([14,32],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),
                    (1,.5,.5,1),(0, 0, 0,1),
                    (1,.5,.5,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [15,33] is good

# %%
# Get Tm2 [112981] synapses
conns = fetch_synapse_connections(None,NC(bodyId=112981),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 32; hex2 = 38
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([31,37],['Tm2'])
# Yes [33,37], [33,38]
#     [32,37],          
#     [31,37], [31,38], [31,39]

# %%
# Plot Tm2 cells around this column to assign 
fh = group_plotter([hex_to_bids([33,37],['Tm2'])['Tm2'], # 
                    hex_to_bids([33,38],['Tm2'])['Tm2'], # 
                    hex_to_bids([32,37],['Tm2'])['Tm2'], # 
                    112981, 
                    hex_to_bids([31,37],['Tm2'])['Tm2'], # 
                    hex_to_bids([31,38],['Tm2'])['Tm2'], # 
                    hex_to_bids([31,39],['Tm2'])['Tm2']], # 
                   [(1,.5,.5,1),(.5,1,.5,1),
                    (1,.5,.5,1),(0, 0, 0,1),
                    (1,.5,.5,1),(.5,1,.5,1),(.5,.5,1,1)],
                   shadow_rois=[])
# show_figure(fh,800, 600)
# fit for [32,38] is good

# %%
# Get Tm2 [104527] synapses
conns = fetch_synapse_connections(None,NC(bodyId=104527),SC(rois='ME(R)'))
# Synapse dataframe
syn_df = conns[['bodyId_post', 'x_post','y_post','z_post']]
syn_df = syn_df.rename(columns = {'bodyId_post':'bodyId', 'x_post':'x', 'y_post':'y', 'z_post':'z'})
# Predicted hex columns are
# hex1 = 10; hex2 = 1
hexid_df = find_neuron_hex_ids(syn_df)
hexid_df

# %%
# Check which columns have Tm2
hex_to_bids([9,1],['Tm2'])
# Yes [11,1], [11,2]
#     [10,1], [10,2]      
#     [9,1],  [9,2]

# %%
bid_to_hex(104527)

# %%
# Hmmm! Something fishy!
bid_to_hex(104527)
# ...returns [(10, 1), (9, 1)]

# but...
hex_to_bids([9,1],['Tm2'])
# ...returns {'Tm2': [101887]}
hex_to_bids([10,1],['Tm2'])
# ...returns {'Tm2': [104527]}

# %%
hex_df = get_hex_df()

# %%
hex_df.iloc[130:150,:]

# %%
