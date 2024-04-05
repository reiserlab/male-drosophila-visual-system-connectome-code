""" Queries for completeness notebooks """

import pandas as pd

from neuprint import Client
from neuprint.client import inject_client


@inject_client
def fetch_ol_types(
    *
  , include_placeholders:bool=False
  , side:str="R-dominant"
  , return_type:str="dataframe"
  , client:Client=None
) -> pd.DataFrame | list[str]:
    """
    Get a list of named types from the optic lobe.

    Parameters
    ----------
    side : str, default = 'R-dominant'
        options include, 'R', 'L', 'R-dominant' or 'both'.
        'R' means all neurons that have their cellbody on the right side, 'L' means that their
        cellbody is on the left side, 'R-dominant' chooses the neurons that have their dominant features
        in the right hemisphere, and 'both' means to get both sides (if available).
        For most analysis that works on one side, the 'R-dominant' is probably the best choice. There will
        be a 'L-dominant' once the other side is proof-read. 'both' returns the types that are present on
        either side and counts their total. If you know what you are doing and there is a reason to diverge,
        you can choose 'R' or 'L'.
    include_placeholders : bool, default=False
        Include cell types that are intended as placeholders in the results (e.g. ME_VPN). Not
        recommended, therefore defaults to false.
    return_type : str, default='dataframe'
        defines what data type the function returns.
    client : neuprint.Client
        Client used for the connection. If no explicit client is provided, then the `defaultclient`
        is used.

    Returns
    -------
    named_df : pandas.DataFrame | list[str]
        if `return_type` is 'list', it returns a list of neuron types. Otherwise the function
        returns a pandas.DataFrame with:
        type : np.object
            Neuron type as defined in neuprint.
        count : np.int64
            Number of neurons of that type.
    """

    assert return_type in [  "dataframe",
        "list",
    ], "Wrong return type, only 'dataframe' and 'list' are allowed"

    named_df = fetch_ol_types_and_instances(
        side=side
      , include_placeholders=include_placeholders
      , client=client
    )
    named_df = named_df\
        .groupby(by='type')\
        .aggregate(func={'count': 'sum'})\
        .reset_index()

    if return_type == "list":
        return named_df.loc[:, "type"].to_list()
    return named_df


@inject_client
def fetch_ol_types_and_instances(
    *
  , include_placeholders:bool=False
  , side:str="R-dominant"
  , return_type:str="dataframe"
  , client:Client=None
) -> pd.DataFrame | list[str]:
    """Get a list of named types and instances from the optic lobe.

    Parameters
    ----------
    side : str, default = 'R-dominant'
        options include, 'R', 'L', 'R-dominant' or 'both'.
        'R' means all neurons that have their cellbody on the right side, 'L' means that their
        cellbody is on the left side, 'R-dominant' chooses the neurons that have their 'dominant features'
        in the right hemisphere, and 'both' means to get both sides (if available).
        For most analysis that works on one side, the 'R-dominant' is probably the best choice. There will
        be a 'L-dominant' once the other side is proof-read. 'both' returns the types that are present on
        either side and counts their total. If you know what you are doing and there is a reason to diverge,
        you can choose 'R' or 'L'.
    include_placeholders : bool, default=False
        Include cell types that are intended as placeholders in the results (e.g. ME_VPN). Not
        recommended, therefore defaults to false.
    return_type : str, default='dataframe'
        defines what data type the function returns.
    client : neuprint.Client
            Client used for the connection. If no explicit client is provided, then the
            `defaultclient` is used.

    Returns
    -------
    named_df : pandas.DataFrame
        type : str
            type name
        instance : str
            instance name
        count : int
            number of neurons in that instance
    """

    assert return_type in ["dataframe", "list"],\
        f"Wrong return type '{type}', only 'dataframe' and 'list' are allowed"

    assert side in ["R", "L", "R-dominant", "both"],\
        f"Unsupported side '{side}', only 'R', 'L', 'R-dominant' or 'both' are allowed"

    str_ignore = ""
    str_side = ""
    str_dominant = ""

    if not include_placeholders:
        str_ignore = f"""
            AND NOT n.type ENDS WITH '_unclear'
            AND NOT n.instance CONTAINS 'unclear'
            AND n.type<>'Pm7_Li28'
        """

    if side in ['R', 'L']:
        str_side = f"AND n.instance ENDS WITH '_{side}'"

    if side in ['R-dominant', 'L-dominant']:
        side_char = side[0]
        str_dominant = f"""
            WITH DISTINCT n.type AS type
              , count(distinct n.instance) AS instance_count
              , collect(n) AS ns
            UNWIND ns AS n
            WITH n, instance_count
            WHERE instance_count <=1 OR n.instance ENDS WITH '_{side_char}'
        """

    cql = f"""
        MATCH(n:Neuron)
        WHERE (
                n.`LA(R)`=True 
                OR n.`ME(R)`=True 
                OR n.`LO(R)`=True
                OR n.`LOP(R)`=True
                OR n.`AME(R)`=True
            )
            AND (n.type IS NOT NULL and n.type <> '')
            AND (n.instance IS NOT NULL and n.instance <> ''
            {str_ignore}
            {str_side})
        {str_dominant}
        RETURN distinct n.type as type, n.instance as instance, count(n.bodyId) as count
        ORDER BY toLower(type), count DESC
    """

    named_df = client.fetch_custom(cql)

    if return_type == "list":
        return named_df.loc[:, "type"].to_list()
    return named_df


@inject_client
def fetch_ol_complete(
    *
  , client:Client=None
) -> pd.DataFrame:
    """Fetch neurons in the optic lobe and their connection weight

    Parameters
    ----------
    client : neuprint.Client, optional
             The client to access neuprint. This defaults to `default_client()`.

    Returns
    -------
    neuron_df : pd.DataFrame
        `type` : str
        `bodyId` : int
        `total_weight` : float
        `traced_weight` : float
        `completeness` : float
    """
    non_empty_named_types = fetch_ol_types(client=client)

    neuron_df = pd.DataFrame()
    for _, row in non_empty_named_types.iterrows():
        type_name = row["type"]
        # instance = row['instance']
        cql = f"""
        MATCH(n:Neuron)
        WHERE (
            n.`ME(R)`=True OR n.`AME(R)`=True
            OR n.`LO(R)`=True OR n.`LOP(R)`=True)
            AND n.type in ['{type_name}']
        MATCH (n)-[e:ConnectsTo]->(m:Segment)
            WITH n, sum(e.weight) as total_weight
        OPTIONAL MATCH (n)-[et:ConnectsTo]->(m2:Segment)
            WHERE (m2.type<>'' AND m2.type IS NOT NULL)
                AND m2.status = 'Anchor'
                AND total_weight <> 0   // avoid division by 0, might skew results
        RETURN n.type as type, n.bodyId as bodyId,
            total_weight, sum(et.weight) as traced_weight,
            1.0*sum(et.weight)/total_weight as completeness

        """
        nm_t = client.fetch_custom(cql)
        if neuron_df.empty is True:
            neuron_df = nm_t
        else:
            neuron_df = pd.concat([neuron_df, nm_t], sort=False)
    neuron_df = (
        neuron_df.assign(lctype=lambda df: df["type"].map(lambda x: x.lower()))
        .sort_values(by="lctype")
        .drop(labels="lctype", axis=1)
        .set_index(keys="type")
    )
    print(f"Pulled data from {len(neuron_df)} neurons")
    return neuron_df


@inject_client
def fetch_ol_stats(
    *
  , only_ol:bool=False
  , only_named_types:bool=True
  , include_lamina:bool=False
  , client:Client=None
) -> pd.DataFrame:
    """Fetch aggregated information per neuron type.

    The same result can be achieved via `fetch_ol_complete` and then doing the aggregation in
        pandas, eg via
    ```python
    stats_df = neurons_df.groupby(by='type').agg(
    neuron_count = ('bodyId', 'count'),
    complete_min = ('completeness', 'min'),
    complete_max = ('completeness', 'max'),
    complete_median = ('completeness', 'median'),
    complete_15pct_count = ('completeness', lambda x: sum(y>.15 for y in x)),
    complete_25pct_count = ('completeness', lambda x: sum(y>.25 for y in x)),
    complete_40pct_count = ('completeness', lambda x: sum(y>.40 for y in x)),
    complete_50pct_count = ('completeness', lambda x: sum(y>.50 for y in x))
    ).sort_values(by='type')
    ```

    Parameters
    ----------
    only_ol : bool, default=False
        Only looks for connections within the Optic lobe.
    only_named_types : bool, default=True
        Only use neurons with named types. Otherwise also include connected neurons with named
        instances.
    include_lamina : bool, default=False
        consider Lamina as part of the optic lobe. The original implementation did not include it,
        hence default to False
    client : neuprint.Client, optional
        The client to access neuprint. This defaults to `default_client()`.

    Returns
    -------
    stat_df : pandas.DataFrame
        type : str
            neuron type
        count : int
            number of neurons
        min : float
            minimal completion
        max : float
            maximum completion
        median : float
            median
        complete_15pct_count : int
            number of neurons that are >15% completed
        complete_25pct_count : int
            number of neurons that are >25% completed
        complete_40pct_count : int
            number of neurons that are >40% completed
        complete_50pct_count : int
            number of neurons that are >50% completed
    """
    ol_constraint1 = ""
    ol_constraint2 = ""
    roi_la = ""
    if include_lamina:
        roi_la = " OR n.`LA(R)`=True"
    if only_ol:
        ol_constraint1 = (
            f"AND (m1.`ME(R)`=True OR m1.`AME(R)`=True OR "
            "m1.`LO(R)`=True OR m1.`LOP(R)`=True {roi_la})"
        )
        ol_constraint2 = (
            f"AND (m2.`ME(R)`=True OR m2.`AME(R)`=True OR "
            "m2.`LO(R)`=True OR m2.`LOP(R)`=True {roi_la})"
        )
    ol_names = "(m2.type<>'' AND m2.type IS NOT NULL and total_weight <> 0)"
    if not only_named_types:
        ol_names = f"({ol_names} OR (m2.instance<>'' AND m2.instance IS NOT NULL))"

    non_empty_named_types = fetch_ol_types(client=client)
    stat_df = pd.DataFrame()
    for _, row in non_empty_named_types.iterrows():

        type_name = row["type"]
        cql = f"""
            MATCH (n:Neuron)-[e:ConnectsTo]->(m1:Segment)
            WHERE
                n.type<>'' and n.type IS NOT NULL {ol_constraint1}
                AND (
                n.`ME(R)`=True OR n.`AME(R)`=True
                OR n.`LO(R)`=True OR n.`LOP(R)`=True {roi_la})
                AND n.type in ['{type_name}']
            WITH n, sum(e.weight) as total_weight
            MATCH (n)-[et:ConnectsTo]->(m2:Segment)
                WHERE {ol_names}
                {ol_constraint2}
                    AND total_weight <> 0   // avoid division by 0, might skew results
            WITH n, 1.0*sum(et.weight)/total_weight as completeness,
            CASE WHEN (1.0*sum(et.weight)/total_weight > 0.15) THEN 1 ELSE 0 END AS pc15,
            CASE WHEN (1.0*sum(et.weight)/total_weight > 0.25) THEN 1 ELSE 0 END AS pc25,
            CASE WHEN (1.0*sum(et.weight)/total_weight > 0.40) THEN 1 ELSE 0 END AS pc40,
            CASE WHEN (1.0*sum(et.weight)/total_weight > 0.50) THEN 1 ELSE 0 END AS pc50
            RETURN n.type as type, count(n.bodyId) as count, min(completeness) as min,
            max(completeness) as max,
            percentileDisc(completeness, 0.5) as median,
            sum(pc15) as complete_15pct_count, sum(pc25) as complete_25pct_count,
            sum(pc40) as complete_40pct_count, sum(pc50) as complete_50pct_count
        """
        nm_t = client.fetch_custom(cql)
        if stat_df.empty is True:
            stat_df = nm_t
        else:
            stat_df = (
                stat_df.copy()
                if nm_t.empty
                else (
                    nm_t.copy()
                    if stat_df.empty
                    else pd.concat([stat_df, nm_t], sort=False)
                )
            )
    # stat_df = stat_df.set_index('type').sort_values(by='type')
    stat_df = (
        stat_df.assign(lctype=lambda df: df["type"].map(lambda x: x.lower()))
        .sort_values(by="lctype")
        .drop(labels="lctype", axis=1)
        .set_index(keys="type")
    )
    return stat_df
