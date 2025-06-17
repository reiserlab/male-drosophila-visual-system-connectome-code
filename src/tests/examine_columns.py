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

# %% [markdown]
# # Motivation
#
# This notebook aims to help with the coverage / completion analysis. During the meetings in the week Jan/15 to Jan/19, we learned that the current analysis was based on the summation of pre and postsynaptic sites.
#
# To ensure data comparability, I am starting to compare instances of the current data with that stored in the database.

# %%
import pandas as pd

from madvisc.utils.column_features_functions import\
  find_cmax_across_all_neuropils\
  , cache_syn_df

from madvisc.utils import olc_client
c = olc_client.connect(verbose=True)
pd.options.plotting.backend = "plotly"


# %% [markdown]
# ## Code from cov_compl_with_fragments
#
# I am loading data for the (semi-randomly selected) type C2. 

# %%
cell_type = 'C2'

cache_syn_df(cell_type, synapse_type='post', rois=['ME(R)', 'LO(R)', 'LOP(R)'])
max_synapses_per_column, max_cells_per_column = find_cmax_across_all_neuropils(cell_type, thresh_val=0.98)


# %%
display(f"maximum number of synapses per column: {max_synapses_per_column} and maxiumum number of cells per column: {max_cells_per_column}")


# %% [markdown]
# The documentation for `find_cmax_across_all_neuropils()` specifies: "For a particular cell type, find the maximum number of cells and synapses per column in ME(R), LO(R) and LOP(R) and output the maximum of these values to be used to set 'cmax' when plotting the spatial coverage heatmaps."
#
# For celltype C2, the obtained values are "11" for the number of cells and "246" for the number of synapses in a single column.

# %% [markdown]
# # Neurons per column
#
# In this section, I retrieve data directly from the neuprint database.
#
# ## Neuron level data from the database
#
# Here, I directly access the neuron level data in neuprint. The following query finds all neurons (C2 as an example) that innervate a column in one of the three primary OL neuropils.
#
# The method how neurons are assigned to columns is not in our hand and is done by the FlyEM team. I seem to remember that the location of presynaptic sites play a role in that process.
#
# The lines starting with `//` are comments where I try to explain what the query does.

# %%
cql = f"""
// Get the Neuron of the type "cell_type"
UNWIND ['ME_R_col', 'LO_R_col', 'LOP_R_co'] as roi
MATCH (n:Neuron)
WHERE n.type='{cell_type}'
// Find all assignments to a "ME_R_col" (column in medulla)
UNWIND keys(n) AS syn_keys
WITH n, roi,syn_keys, left(syn_keys, 8) in [roi] AS is_in_OLR
WHERE is_in_OLR
// Return the column, an aggregated count of neurons per column and their body IDs
RETURN 
    left(roi,3)
  , right(syn_keys, 5) as column
  , count(distinct n.bodyId) as cells_per_column
  , collect(distinct n.bodyId) as cell_body_ids
ORDER BY cells_per_column DESC
"""
c.fetch_custom(cql)

# %% [markdown]
# Looking at the results, I see plenty of medula columns that have >11 C2s. Based on the `cells_per_column`, the maximum number of C2 per columns is rather 19 than 11.
#
# But maybe the neuron-level data is to coarse and the method of assignment is outside of our code basis and unknown in its details. The next steps utilize the synapse level data.
#
#
# ## Synapse level data from the data base
#
# Here I find the location of all C2 synapses. I then assign the column of a synapse to the neuron the synaptic site is part of. Based on that, I count the neurons per column.

# %%
cql = f"""
// find synapses for neuron "n"
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
// and "n" is of cell_type. Here I look only inside ME(R)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, count of "n" neurons, their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids
order by cells_per_column desc
"""
c.fetch_custom(cql)

# %% [markdown]
# This looks very similar to the neuron-level data, but this time we know how the assignment is done. If necessary, we can look at each individual synapse and the partners to verify the correctness.
#
#
# ### Connection based
#
# The previous query considered all synaptic sites of neurons.
#
# In the next query I find the synapses where C2 receives input from another named neuron. Again, I find the location of that synapse and assign the location of the synapse to the neuron.

# %%
cql = f"""
// Iterate through the 3 main ROIs
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
// Find neurons of cell_type with synapses in a column
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
// find synapses where "n" receives input from "m"
AND EXISTS {{
    (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)-[:SynapsesTo]->(ns)
}}
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, ROI, count of "n" neurons, and their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids

order by cells_per_column desc
"""
con_out = c.fetch_custom(cql)
display(con_out)

# %% [markdown]
# The cells per columns looks very similar to the neuron level data and to the query where I just count the synapses per neuron.
#
# Yet, if you look at the synaptic sites per column (`synapses_per_column`), their number decreased. That makes sense, since not all synaptic sites are postsynaptic.
#
# If I revert the connection only considering presynaptic sites, I get the following result:

# %%
cql = f"""
// Iterate through the 3 main ROIs
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
// Find neurons of cell_type with synapses in a column
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
// find synapses where "n" provides input for "m"
AND EXISTS {{
    (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)<-[:SynapsesTo]-(ns)
}}
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, ROI, count of "n" neurons, and their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids

order by cells_per_column desc
"""
c.fetch_custom(cql)


# %% [markdown]
# Here it looks very different, the maximum number of C2 per column is closer to 7 than to 11. 
#
# On a side note: Interestingly you can also see, that the number of synapses doesn't seem to correlate well with the number of cells.
#
# Let's have a closer look at the synapses.
#
#
# ### Synapses across columns
#
# Let's have a closer look at the post-synaptic sites of C2 (related to the query 2 up), where the results started like this: `0	22_28	ME(R)	19	120	[111853, 92935, 118290, …`
#
# In this query, I am taking one of the neurons that innervate the columns with the most cells and count its postsynaptic sites per column. I also calculate the neurons percentage for the postsynaptic sites per column.

# %%
roi = con_out.loc[0, 'roi']
cell_body_id=con_out.loc[0, 'cell_body_ids'][0]

cql = f"""
MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
WHERE n.bodyId={cell_body_id} AND ns['{roi}'] IS NOT NULL and exists(ns.olHex1) and exists(ns.olHex2)
AND EXISTS {{
    (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)-[:SynapsesTo]->(ns)
}}
with n, ns, toString(ns.olHex1)+'_'+toString(ns.olHex2) as col
WITH {{bid: n.bodyId, col: col, syn: count(distinct ns)}} as tmp_res, n.bodyId as tmpbid, count(distinct ns) as syn_count
WITH collect(tmp_res) as agg_res, sum(syn_count) as total_syn_count
UNWIND agg_res as per_col
RETURN per_col.col as column, per_col.bid as bodyId, per_col.syn as synapse_count, toFloat(per_col.syn)/total_syn_count as synapse_perc
ORDER BY bodyId, synapse_count DESC
"""
interest_neuron = c.fetch_custom(cql)
display(interest_neuron)

# %%
interest_neuron['synapse_perc']\
    .plot(labels={'index': 'column'})\
    .update_layout(xaxis={'tickvals': interest_neuron.index, 'ticktext': interest_neuron['column']})

# %% [markdown]
# For a C2, this shows that the selected neuron 111853 has 70% of its synapses in one column (22, 26), 14% in a second (21, 27), 6% in a third (22, 27) and so on… The column of interest (22, 28) has only 1.1% (=1 synapse) for that neuron.
#
# So maybe a certain percentage of the synapses should be dropped? So let's have a look at the column with the most cells (22, 28).
#
#
# ### Detais for the most innervated column
#
# Now I look in more detail at the column of interest (22, 28). Note that the percentages are for the individual neurons, not the column (they will not add up to 100%).

# %%
cell_body_ids = con_out.loc[0, 'cell_body_ids']
roi = con_out.loc[0, 'roi']

all = pd.DataFrame()
for cell_body_id in con_out.loc[0, 'cell_body_ids']:
    cql = f"""
    MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
    WHERE n.bodyId={cell_body_id} AND ns['{roi}'] IS NOT NULL and exists(ns.olHex1) and exists(ns.olHex2)
    AND EXISTS {{
        (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)-[:SynapsesTo]->(ns)
    }}
    with n, ns, toString(ns.olHex1)+'_'+toString(ns.olHex2) as col

    WITH {{bid: n.bodyId, col: col, syn: count(distinct ns)}} as tmp_res, n.bodyId as tmpbid, count(distinct ns) as syn_count
    WITH collect(tmp_res) as agg_res, sum(syn_count) as total_syn_count
    UNWIND agg_res as per_col

    RETURN per_col.col as column, per_col.bid as bodyId, per_col.syn as synapse_count, toFloat(per_col.syn)/total_syn_count as synapse_perc

    ORDER BY bodyId, synapse_count DESC
    """
    tmp = c.fetch_custom(cql)
    all = pd.concat([all, tmp])
all = all.reset_index(drop=True)
interest_col = all[all['column']==con_out.loc[0, 'column']].sort_values(by='synapse_perc', ascending=False).reset_index(drop=True)
interest_col['cum_synapse_count'] = interest_col['synapse_count'].cumsum()
display(interest_col)

# %% [markdown]
# For the example of C2, one neuron (93875) has about 43% of its synapses in that column, two (99181, 109861) have around 10% of their synapses here, and another eight have at least 2% of their synapses in the column of interest.
#
# If we excluded all neurons that have only a single synaptic site in the column, this column had 12 neurons. For including 95 percentile, we would get a similar number. If we instead looked at the per-neuron percentage and excluded synapses that represent less than 2% of the neurons known synaptic sites, we would end up at 11. 

# %%
interest_col['synapse_perc']\
    .plot(labels={'index': 'body ID', 'value': '% of synapses in column'})\
    .update_layout(xaxis={'tickvals': interest_col.index, 'ticktext': [f"{a}: {i['bodyId']}" for a, i in interest_col.iterrows()]})


# %% [markdown]
# Anyway, at this point I can't even replicate the number of cells per column that is used in the coverage(?) plots.
#
# # Synapses
#
# The following queries are now related to the other number from the initial code, where a maxiumum of 246 synapses were counted per column.
#
# Here I use the same query as before, where C2 was the presynaptic partner to a named neuron. This time I sort the results by the number of synapses, not the cell count.

# %%
cql = f"""
// find synapses for neuron "n"
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
// and "n" is of cell_type. Here I look only inside ME(R)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
    AND EXISTS {{
        (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)<-[:SynapsesTo]-(ns)
    }}
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, count of "n" neurons, their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids
order by synapses_per_column desc
"""
c.fetch_custom(cql)

# %% [markdown]
# That query shows, that the most synapses per column with a count of 67 is far away from the 246 calculated above.
#
# For C2 as the postsynapic partner, the highest number of synapses per column is 230:

# %%
cql = f"""
// find synapses for neuron "n"
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
// and "n" is of cell_type. Here I look only inside ME(R)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
    AND EXISTS {{
        (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)-[:SynapsesTo]->(ns)
    }}
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, count of "n" neurons, their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids
order by synapses_per_column desc
"""
c.fetch_custom(cql)

# %% [markdown]
# Only if I consider all synapses, no matter if they synapse to named neurons or unknown segments, I get to numbers similar or higher than what the original number was.

# %%
cql = f"""
// find synapses for neuron "n"
UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
MATCH 
    (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
// and "n" is of cell_type. Here I look only inside ME(R)
WHERE n.type='{cell_type}' AND ns[roi] IS NOT NULL and ns.olHex1 IS NOT NULL and ns.olHex2 IS NOT NULL 
WITH DISTINCT n,ns, toString(ns.olHex1)+"_"+toString(ns.olHex2) as column, roi
// Then I return the column, count of "n" neurons, their body IDs
RETURN 
    distinct column
  , roi
  , count(distinct n.bodyId) as cells_per_column
  , count(distinct ns) as synapses_per_column
  , collect(distinct n.bodyId) as cell_body_ids
order by synapses_per_column desc
"""
syn_per_col = c.fetch_custom(cql)
display(syn_per_col)

# %% [markdown]
# In this case, up to 10 columns have more synapses than the previously calculated 246.

# %%
syn_per_col[syn_per_col['synapses_per_column']>246]

# %% [markdown]
# When I look at the column with the most synapses, I reach the number of 246 synapses considering all neurons with more than 3.59% of their synapses in that column, ignoring the ones with 3.57% and less in this column.

# %%
cell_body_ids = syn_per_col.loc[0, 'cell_body_ids']
roi = con_out.loc[0, 'roi']

all = pd.DataFrame()
for cell_body_id in syn_per_col.loc[0, 'cell_body_ids']:
    cql = f"""
    MATCH (n:Neuron)-[:Contains]->(nss:SynapseSet)-[:Contains]->(ns:Synapse)
    WHERE n.bodyId={cell_body_id} AND ns['{roi}'] IS NOT NULL and exists(ns.olHex1) and exists(ns.olHex2)
    //AND EXISTS {{
    //    (m:Neuron)-[:Contains]->(mss:SynapseSet)-[:Contains]->(ms:Synapse)-[:SynapsesTo]->(ns)
    //}}
    with n, ns, toString(ns.olHex1)+'_'+toString(ns.olHex2) as col

    WITH {{bid: n.bodyId, col: col, syn: count(distinct ns)}} as tmp_res, n.bodyId as tmpbid, count(distinct ns) as syn_count
    WITH collect(tmp_res) as agg_res, sum(syn_count) as total_syn_count
    UNWIND agg_res as per_col

    RETURN per_col.col as column, per_col.bid as bodyId, per_col.syn as synapse_count, toFloat(per_col.syn)/total_syn_count as synapse_perc

    ORDER BY bodyId, synapse_count DESC
    """
    tmp = c.fetch_custom(cql)
    all = pd.concat([all, tmp])
all = all.reset_index(drop=True)
interest_col = all[all['column']==syn_per_col.loc[0, 'column']].sort_values(by='synapse_perc', ascending=False).reset_index(drop=True)
interest_col['cum_synapse_count'] = interest_col['synapse_count'].cumsum()
display(interest_col)

# %% [markdown]
# ## Temporary observation
#
# Based on the data from neuprint, I cannot replicate the numbers that the `find_cmax_across_all_neuropils()` returns for the example celltype. 
#
# When I access the "number of cells and synapses", the database has 10% more synapses in the most populated column (or 7% too few) and I see a maximum of 72% more cells per column (or 37% less, depending if only considering pre/post sites.)
#
# Switching from the summation of synaptic sites to just using pre or postsynaptic sites will change the plots for reasons that are not related to that switch.
