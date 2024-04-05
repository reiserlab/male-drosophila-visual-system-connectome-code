from pathlib import Path
from neuprint import fetch_custom

def get_assigned_columnar_types(output_path:Path):
    """
    Get dataframe of neuron bodyIds from types that have been manually assigned to hex 
    columns in the ME(R). Saves the resulting dataframe as a csv file in 'output_path'.

    Parameters
    ----------
    output_path : Path
        path to save csv file
    """
    cql = """
        MATCH (n:Neuron)
        WHERE NOT n.assignedOlHex1 is NULL
        RETURN n.type as type, 
            n.bodyId as body_id, 
            n.assignedOlHex1 as hex1_id, 
            n.assignedOlHex2 as hex2_id
        ORDER BY type, hex1_id, hex2_id
        """
    df = fetch_custom(cql)
    df = df.pivot_table(index=['hex1_id', 'hex2_id'], columns='type', values='body_id', aggfunc=lambda x: x.iloc[0])\
        .reset_index()
    df.to_excel(output_path, index=False)
