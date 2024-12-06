import pandas as pd

from utils.ol_types import OLTypes
from utils.instance_summary import InstanceSummary


def get_depth_df_all() -> pd.DataFrame:
    """
    Return a dataframe with a count of all the synapses in each of the
    depth bins in ME(R), LO(R) and LOP(R) for all cells from all cell
    types.

    Returns
    -------
    depth_all : pd.DataFrame
        roi
            optic lobe region of interest
        depth_bin
            label of depth bin within that region
        depth
            dpeth value of that bin in that region
        type
            type of synapse, 'pre' or 'post'
        syn_sum
            sum of all the synapses within that depth bin
    """
    olt = OLTypes()
    cell_type_list = olt.get_neuron_list(
        side='both'
    )

    for index, row in cell_type_list.iterrows():
        inst_name = row['instance']
        print(inst_name)
        inst_sum = InstanceSummary(inst_name)
        depth_df = inst_sum.distribution_count

        if index == 0:
            depth_all = depth_df
        else:
            depth_all = depth_all\
                .merge(
                    depth_df
                  , on=['roi', 'depth_bin', 'depth', 'type']
                  , how='right'
                )\
                .reset_index(drop=True)
            depth_all['syn_sum'] = depth_all[['syn_sum_x', 'syn_sum_y']]\
                .sum(axis=1)\
                .astype(int)
            depth_all = depth_all.drop(columns=['syn_sum_x', 'syn_sum_y'])

    return depth_all
