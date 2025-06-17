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
import pandas as pd
from madvisc.utils import olc_client
from madvisc.utils.neuron_bag import NeuronBag
from madvisc.utils.ol_neuron import OLNeuron

c = olc_client.connect()

# %%
# Tm5a: 16354
oln = OLNeuron(86459)
oln.get_hex_id()

# %%
oln.me_hex_id

# %%
bag = NeuronBag('Tm20')
bag.sort_by_distance_to_hex(neuropil='ME(R)', hex1_id=18, hex2_id=18)

# %%
ret = {}

# cell_types = ['Tm5a', 'Tm5b', 'Tm29', 'L2']
cell_types = ['L1',	'L2', 'L3', 'L5', 'Mi1', 'Mi4', 'Mi9', 'C2', 'C3', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Tm20', 'T1']

for ct in cell_types:
    bag = NeuronBag(ct)
    for bid in bag.get_body_ids(bag.size):
        oln = OLNeuron(bid)
        ret[bid] = []
        for idx, method in enumerate(['synapse_count', 'assigned', 'centroid']):
            for _, res  in oln.get_hex_id(method=method).iterrows():
                ret[bid].append({
                    'method': method
                  , 'roi': res['ROI']
                  , 'type': ct
                  , 'hex1': res['hex1_id']
                  , 'hex2': res['hex2_id']
                })


# %%
all_df = pd.DataFrame()

for k,v in ret.items():
    df = pd.DataFrame(v)
    df['body_id'] = k
    all_df = pd.concat([all_df, df])

# %%
tbl = all_df.pivot(index=['body_id', 'roi', 'type'], columns='method', values=['hex1', 'hex2'])

# %%
# tbl.to_excel('issue_422.xlsx')
tbl.to_excel('columnar.xlsx')

# %%
all2 = all_df[~all_df[['method', 'roi', 'body_id']].duplicated()]

# %%
# centroid: 86459, 132357, 90380, 138036
