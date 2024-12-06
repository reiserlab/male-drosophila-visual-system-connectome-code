from pathlib import Path
from textwrap import dedent
from neuprint import fetch_custom
import pandas as pd

def get_assigned_columnar_types(output_path:Path):
    """
    Get dataframe of neuron bodyIds from types that have been manually assigned to hex 
    columns in the ME(R). Saves the resulting dataframe as a csv file in 'output_path'.

    Parameters
    ----------
    output_path : Path
        path to save csv file
    """
    cql = dedent("""
        MATCH (n:Neuron)
        WHERE NOT n.assignedOlHex1 is NULL
        RETURN n.type as type, 
            n.bodyId as body_id, 
            n.assignedOlHex1 as hex1_id, 
            n.assignedOlHex2 as hex2_id,
            "ME_R_col_" + apoc.text.lpad(toString(toInteger(n.assignedOlHex1)), 2, '0') + "_" + 
            apoc.text.lpad(toString(toInteger(n.assignedOlHex2)), 2, '0') as column_roi
        ORDER BY type, hex1_id, hex2_id
        """)
    df = fetch_custom(cql)
    col_df = df[['hex1_id', 'hex2_id','column_roi']].drop_duplicates()
    df_pivoted = df.pivot_table(\
        index=['hex1_id', 'hex2_id']
        , columns='type'
        , values='body_id'
        , aggfunc=lambda x: ', '.join(map(str, x)))\
        .reset_index()
    df_final = pd.merge(df_pivoted, col_df, on=['hex1_id', 'hex2_id'], how='outer')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(output_path, index=False)
    return df_final
