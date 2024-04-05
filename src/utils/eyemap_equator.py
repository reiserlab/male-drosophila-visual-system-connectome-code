# utility functions for generating the eyemap equator
#
# These functions were previously in the `Compute_R7R8_subtypedata.ipynb` 
# notebook and created an intermedite pickle file for each type. These functions
# return the same data structures.

import pandas as pd

def _get_preprocess_columns(result_dir):
    # Load dataframe of cell bodyIds in every column
    #

    cc = pd.read_pickle(result_dir / "column_cells_revR7_R8.pickle")
    # Eliminate the empty columns
    # Replace entries with -1,-2 for nan
    cc = cc.replace(-1,float('nan'))
    cc = cc.replace(-2,float('nan'))

    # Remove columns without cells
    cc = cc.loc[cc.isna().sum(axis = 1)<15,:]
    cc = cc.reset_index(drop=True)
    return cc


def get_columns_without_r7r8(result_dir):
    cc = _get_preprocess_columns(result_dir)
    # No R7 or R8
    df = cc.loc[
        (cc['R78'].isna()) & (cc['R7'].isna()) & (cc['R8'].isna()) & (cc['R7p'].isna()) & (cc['R8p'].isna()) & 
        (cc['R7y'].isna()) & (cc['R8y'].isna()) & (cc['R7d'].isna()) & (cc['R8d'].isna()) & (cc['R78_2'].isna())
        ]
    no_r7_r8 = df.iloc[:,0:2]
    no_r7_r8['hex1_id'] = df.iloc[:,0]
    no_r7_r8['hex2_id'] = df.iloc[:,1]
    no_r7_r8['p'] = df.iloc[:,0] - df.iloc[:,1]
    no_r7_r8['q'] = df.iloc[:,0] + df.iloc[:,1]
    return no_r7_r8

def get_columns_with_r7r8(result_dir):
    cc = _get_preprocess_columns(result_dir)
    # R78
    df = cc.loc[(cc['R78'].notna()) | (cc['R7'].notna()) | (cc['R8'].notna()) | (cc['R78_2'].notna())]
    r78 = df.iloc[:,0:2]
    r78['hex1_id'] = df.iloc[:,0]
    r78['hex2_id'] = df.iloc[:,1]
    r78['p'] = df.iloc[:,0] - df.iloc[:,1]
    r78['q'] = df.iloc[:,0] + df.iloc[:,1]
    return r78

def get_columns_with_pale(result_dir):
    cc = _get_preprocess_columns(result_dir)
    # Pale
    df = cc.loc[(cc['R7p'].notna()) | (cc['R8p'].notna())]
    pale = df.iloc[:,0:2]
    pale['hex1_id'] = df.iloc[:,0]
    pale['hex2_id'] = df.iloc[:,1]
    pale['p'] = df.iloc[:,0] - df.iloc[:,1]
    pale['q'] = df.iloc[:,0] + df.iloc[:,1]
    return pale

def get_columns_with_yellow(result_dir):
    cc = _get_preprocess_columns(result_dir)
    # Yellow
    df = cc.loc[(cc['R7y'].notna()) | (cc['R8y'].notna())]
    yellow = df.iloc[:,0:2]
    yellow['hex1_id'] = df.iloc[:,0]
    yellow['hex2_id'] = df.iloc[:,1]
    yellow['p'] = df.iloc[:,0] - df.iloc[:,1]
    yellow['q'] = df.iloc[:,0] + df.iloc[:,1]
    return yellow

def get_columns_with_dra(result_dir):
    cc = _get_preprocess_columns(result_dir)
    # DRA
    df = cc.loc[(cc['R7d'].notna()) | (cc['R8d'].notna())]
    dra = df.iloc[:,0:2]
    dra['hex1_id'] = df.iloc[:,0]
    dra['hex2_id'] = df.iloc[:,1]
    dra['p'] = df.iloc[:,0] - df.iloc[:,1]
    dra['q'] = df.iloc[:,0] + df.iloc[:,1]
    return dra

def _get_columns_with_l2_ids(result_dir, id_list):
    cc = _get_preprocess_columns(result_dir)
    df = cc[cc.isin(id_list)['L2']]
    pr = df.iloc[:,0:2]
    pr['hex1_id'] = df.iloc[:,0]
    pr['hex2_id'] = df.iloc[:,1]
    pr['p'] = df.iloc[:,0] - df.iloc[:,1]
    pr['q'] = df.iloc[:,0] + df.iloc[:,1]
    return pr

def get_columns_with_8pr(result_dir):
    # These body IDs are based on a manual search by Pavi,
    #   see `/results/eyemap/equator_annotation/{back|middle}.png`
    pr8_ids = [ # array with L2 bodyids with 8 PRs, back, from Pavi
        23205, 22304, 23060, 21343, 23669, 22074, 22245
      , 23268, 22539, 23580, 23272, 23803, 22702, 23335, 23466]
    pr8_ids.extend([ # array with L2 bodyids with 8PRs, center, from Pavi
        31354, 29762, 29681, 28572, 31346, 29000, 26067
      , 31620, 29184, 27808, 29331, 25387, 26703])
    return _get_columns_with_l2_ids(result_dir=result_dir, id_list=pr8_ids)

def get_columns_with_7pr(result_dir):
    # These body IDs are based on a manual search by Pavi,
    #   see `/results/eyemap/equator_annotation/{back|middle}.png`
    pr7_ids = [ # array with L2 bodyids with 7 PRs, back, from Pavi
        27413, 17779, 23605, 22542, 23635, 22921, 20299, 21950]
    pr7_ids.extend([ # array with L2 bodyids with 7 PRs, center, from Pavi
        27653, 27768, 27681])
    return _get_columns_with_l2_ids(result_dir=result_dir, id_list=pr7_ids)

def get_columns_with_6pr(result_dir):
    # These body IDs are based on a manual search by Pavi,
    #   see `/results/eyemap/equator_annotation/{back|middle}.png`
    pr6_ids = [ # array with L2 bodyids with 6 PRs, back, from Pavi
        21847, 23838, 23515, 23084, 22845, 21187, 23455
      , 22724, 21524, 22939, 21791]
    pr6_ids.extend([ # array with L2 bodyids with 6 PRs, center, from Pavi
        29695, 30692, 33408, 29114, 539495, 26906, 26298])
    return _get_columns_with_l2_ids(result_dir=result_dir, id_list=pr6_ids)