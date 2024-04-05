
from abc import ABC
from pathlib import Path
import warnings

import pandas as pd
from dotenv import find_dotenv
from scipy.spatial import KDTree

from neuprint import fetch_custom
from utils.helper import slugify

class InstanceSummary(ABC):

    """
    Helper class to generate summary plots

    Parameters
    ----------
    instance_name : str
        Instance name from neuprint
    connection_cutoff : int, default=1
        remove any connecting neurons from the top 5 below that count.
    """

    def __init__(
        self
      , instance_name:str
      , connection_cutoff:int=1
      , per_cell_cutoff:int=None
    ):
        self.__instance_name = instance_name
        self.__synapses = None
        self.__ame_syn = None
        self.__columns = pd.DataFrame()
        self.__count = None

        assert connection_cutoff is None or per_cell_cutoff is None,\
            "you can only cut by total connections (connection_cutoff) or per cell, not both."
        if connection_cutoff:
            assert 0 < connection_cutoff,\
                f"connection cutoff must be >0, not {connection_cutoff}"
            assert isinstance(connection_cutoff, int),\
                f"connection cutoff must be an integer, not {type(connection_cutoff)}"
        if per_cell_cutoff:
            assert 0 < per_cell_cutoff,\
                f"per cell cutoff must be >0, not {per_cell_cutoff}"
            assert isinstance(per_cell_cutoff, (int, float)),\
                f"per cell cutoff must be a number, not {type(per_cell_cutoff)}"

        self.__connection_cutoff = connection_cutoff
        self.__per_cell_cutoff = per_cell_cutoff
        self.__cache = Path(find_dotenv()).parent / "cache" / "fig_summary" / "instance_summary"
        self.__cache.mkdir(parents=True, exist_ok=True)


    @property
    def type_name(self) -> str:
        """
        Get the type name for the specified instance.

        Returns
        -------
        type_name : str
            Name of the cell type
        """
        return self.instance_name[:-2]


    @property
    def instance_name(self) -> str:
        """
        Get the instance name (should be the same used in the constructor).

        Returns
        -------
        instance_name : str
            Name of the instance
        """
        return self.__instance_name


    @property
    def hemisphere(self) -> str:
        """
        Hemisphere for the instance.

        Returns
        -------
        hemisphere : str
            should be R | L
        """
        return self.instance_name[-1]


    @property
    def is_bilateral(self) -> bool:
        """
        Is this a bilateral neuron, which technically means that more than one instance
        exists.

        Returns
        -------
        is_bilateral : bool
            True if bilateral (see definition above and in methods)
        """
        cql = f"""
            MATCH (n:Neuron)
            WHERE n.type = '{self.type_name}'
            RETURN count(distinct n.instance) as instance_n
        """
        named_df = fetch_custom(cql)
        if named_df.empty:
            warnings.warn("no instance for type. Weird...")
        elif named_df.loc[0, 'instance_n'] > 1:
            return True
        return False


    @property
    def is_r_dominant(self) -> bool:
        """
        Is this a R-dominant neuron? R-dominant means either that we have only
        once instance in the database, or it is bilateral but is from the R side.

        Note: this uses a different way to determine the r-dominance than OLTypes and NeuronBag.

        Returns
        -------
        is_r_dominant : bool
            true if R-dominant
        """
        if not self.is_bilateral or (self.is_bilateral and self.hemisphere=='R'):
            return True
        return False


    @property
    def count(self) -> int:
        """
        Count the number of cells for the instance.

        Returns
        -------
        count : int
            number of cells that belong to the instance.
        """
        if self.__count is None:
            cql = f"""
                MATCH (n:Neuron)
                WHERE n.instance = '{self.instance_name}'
                RETURN count(distinct n.bodyId) as n
            """
            named_df = fetch_custom(cql)
            self.__count = 0
            if named_df.size == 1:
                self.__count = named_df.loc[0, 'n']
        return self.__count


    @property
    def consensus_neurotransmitter(self) -> str:
        """
        The consensus neurotransmitter for the instance. This is the value straight from the
        database neuprint. Most of the values there are the full names like acetylcholine,
        serotonin, but also gaba.

        Returns
        -------
        ret : str
            name of the consensus neurotransmitter
        """
        cql = f"""
            MATCH (n:Neuron)
            WHERE n.instance = '{self.instance_name}'
            RETURN distinct n.consensusNt as nt, count(distinct n.bodyId) as n
            ORDER BY n DESC
        """
        named_df = fetch_custom(cql)
        ret = None
        if len(named_df) > 0:
            ret = named_df.loc[0, 'nt']
        if len(named_df) > 1:
            warnings.warn(f"More than 1 consensusNt, using {ret}")
        return ret


    @property
    def consensus_nt(self) -> str:
        """
        Get the abbreviated form of the neurortransmitter. glutamate becomes Glu…

        Returns
        -------
        ret : str
            abbreviated neurotransmitter name.
        """
        nt_mapping = {
            'acetylcholine': 'ACh'
          , 'gaba': 'GABA'
          , 'serotonin': '5HT'
          , 'glutamate': 'Glu'
          , 'dopamine': 'Dop'
          , 'histamine': 'His'
          , 'octopamine': 'OA'
          , 'unclear': 'unclear'
          , '5HT': '5HT'
        }
        return nt_mapping[self.consensus_neurotransmitter]


    @property
    def top5_upstream(self) -> pd.DataFrame:
        """
        Get the top 5 upstream neurons

        Returns
        -------
        df : pd.DataFrame
            instance : str
                source instance.
            perc : float
                ratio of synapses from the source instance of overall synapses.
        """
        return self.__get_input_output(
            direction='upstream'
          , limit=5
        )


    @property
    def top5_downstream(self) -> pd.DataFrame:
        """
        Get the top 5 downstream neurons.

        Returns
        -------
        df : pd.DataFrame
            instance : str
                target instance.
            perc : float
                ratio of synapses to the target instance of overall synapses.
        """
        return self.__get_input_output(
            direction='downstream'
          , limit=5
        )


    @property
    def bids(self) -> list:
        """
        Get all body IDs of the instance.

        Returns
        -------
        bid_lst : list
            List of body IDs that belong to the instance.
        """
        bids_df = self.__bids_df
        return bids_df['bid'].to_list()


    @property
    def synapses(self) -> pd.DataFrame:
        """
        Get all synapses for the instance. This is faster than the get_synapses from
        neuprint-python, because it is direclty geared towards our usecase. On the second call,
        it uses a file based cache.

        Returns
        -------
        df : pd.DataFrame
            roi : str
                Brain region the synapse belongs to.
            bid : int
                body ID of the synapse
            hex1 : int
                hex1 location in the brain region
            hex2 : int
                hex2 location in the brain region
            type : str
                type of the synapse (pre / post)
            x : float
                location in space
            y : float
                location in space
            z : flat
                location in space
         # E.G - missing depth_bin
        """
        if self.__synapses is None:
            cache_file = self.__cache / f"{slugify(self.instance_name)}_synapses.pickle"
            if cache_file.is_file():
                self.__synapses = pd.read_pickle(cache_file)
            else:
                all_bodies = self.__bids_df

                all_syns = pd.DataFrame()
                for start in range(0, len(all_bodies), 20):
                    bodyids = all_bodies.loc[start:start+20, 'bid'].to_list()

                    cql = f"""
                        UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
                        MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]-(ns:Synapse)
                        WHERE n.bodyId IN {bodyids} and ns[roi]
                        WITH DISTINCT ns, roi, n
                        RETURN
                            roi as roi
                          , n.bodyId as bid
                          , ns.olHex1 as hex1
                          , ns.olHex2 as hex2
                          , ns.type as type
                          , ns.location.x as x
                          , ns.location.y as y
                          , ns.location.z as z
                    """
                    all_syns = pd.concat([all_syns, fetch_custom(cql)])
                self.__synapses = all_syns
                tmp_syn = pd.DataFrame()
                for roi in ['ME(R)', 'LO(R)', 'LOP(R)']:
                    l_col = self.\
                        columns[self.columns['roi']==roi].\
                        reset_index(drop=True)
                    l_syn = self.\
                        synapses[self.synapses['roi']==roi].\
                        reset_index(drop=True)
                    tree = KDTree(l_col.loc[:, ['x', 'y', 'z']])
                    _, idx = tree.query(l_syn.loc[:,  ['x', 'y', 'z']])
                    l_syn['depth_bin'] = l_col.iloc[idx, 6].reset_index(drop=True)
                    tmp_syn = pd.concat([tmp_syn, l_syn])

                self.__synapses = tmp_syn
                self.__synapses.to_pickle(cache_file)
        return self.__synapses


    @property
    def distribution(self) -> pd.DataFrame:
        """
        Get the values for the synpse distribution by depth plot.

        Returns
        -------
        l_dist : pd.DataFrame
            roi : str
                name of the brain region
            depth_bin : int
                depth (in arbitrary bucket sizes) within the region
            depth : float
                depth in the brain region (0…1)
            type : type
                type of synapses (pre / post)
            syn_dist : float
                synapse distribution at the depth
        """
        all_depths = self.__get_depths(with_syn_type=True)

        # TODO: put in function
        l_dist = pd.DataFrame()
        l_syn = self.synapses # not self.__synapses?
        l_syn['syn_count'] = 1
        l_syn = l_syn.merge(self.__bids_df, on='bid', how='right')

        def f(grp):
            bid = grp['bid'].unique()[0]
            lgrp =  grp.merge(all_depths, on=['roi', 'depth_bin', 'type'], how='right')\
                .fillna(0)\
                .reset_index(drop=True)
            lgrp['bid'] = bid
            return lgrp

        l_dist = l_syn.groupby(['bid']).apply(f).reset_index(drop=True)
        # END TODO

        l_dist = l_dist\
            .groupby(['roi', 'bid', 'depth_bin', 'depth', 'type'])\
            .agg(count=('syn_count', 'sum'))\
            .groupby(['roi', 'depth_bin', 'depth', 'type'])\
            .agg(syn_dist=('count', 'mean'))\
            .reset_index()\
            .sort_values(['roi', 'depth_bin'])

        return l_dist

    @property
    def ame_count(self) -> pd.DataFrame:
        """
        The percentage of synapses for the instance within the AME.

        Returns
        -------
        ame_syn : pd.DataFrame
            type : str
                type of the synapse, either 'post' or 'pre'.
            perc : float
                Percentage of total synapses in the AME
        """
        if self.__ame_syn is None:
            cache_file = self.__cache / f"{slugify(self.instance_name)}_ame.pickle"
            if cache_file.is_file():
                self.__ame_syn = pd.read_pickle(cache_file)
            else:
                cql = f"""
                    UNWIND ['pre', 'post'] AS type
                    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]-(ns:Synapse)
                    WHERE n.instance = '{self.instance_name}' and ns.type=type
                    WITH ns.type as ns_type, count(distinct ns) as abs_count, type
                    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]-(ns:Synapse)
                    WHERE n.instance = '{self.instance_name}' and ns.type=type and ns['AME(R)']
                    RETURN type, toFloat(count(distinct ns))/abs_count AS perc
                """
                self.__ame_syn = fetch_custom(cql)
                self.__ame_syn.to_pickle(cache_file)
        return self.__ame_syn


    @property
    def innervation(self) -> pd.DataFrame:
        """
        Get the numbers for the innervated columns by depth plot.

        Returns
        -------
        l_innervate : pd.DataFrame
            roi : str
                name of the brain region.
            depth_bin
                depth (in arbitrary bucket sizes, e.g. ~120 for ME, 75 for LO, 50 for LOP)
            depth : float
                depth in fraction
            column_innervation : float
                number of innervated columns
        """
        # TODO put in function
        l_syn = self.synapses
        all_depths = self.__get_depths(with_syn_type=False)
        l_innervate = pd.DataFrame()
        l_syn['syn_count'] = 1
        l_syn = l_syn.merge(self.__bids_df, on='bid', how='right')

        def f(grp):
            bid = grp['bid'].unique()[0]
            lgrp =  grp.merge(all_depths, on=['roi', 'depth_bin'], how='right')\
                .fillna(0)\
                .reset_index(drop=True)
            lgrp['bid'] = bid
            return lgrp

        l_innervate = l_syn.groupby(['bid']).apply(f).reset_index(drop=True)

        # END TODO

        l_innervate = l_innervate\
            .groupby(['roi', 'bid', 'depth_bin', 'depth',  'hex1', 'hex2'])\
            .agg(myall=('syn_count', 'max'))\
            .groupby(['roi', 'bid', 'depth_bin', 'depth'])\
            .agg(per_bid=('myall', 'sum'))\
            .groupby(['roi', 'depth_bin', 'depth'])\
            .agg(col_innervation=('per_bid', 'mean'))\
            .reset_index()\
            .sort_values(['roi', 'depth_bin'])
        return l_innervate


    @property
    def columns(self) -> pd.DataFrame:
        """
        Get all column pins from neuprint. A pin consists of a list of points in space
        that form the centroid of the column. Each point is associated with a virtual
        column location through the hex1 / hex2 coordinates and a depth within the column.
        Each point is also located in space through an xyz location.

        Returns
        -------
        columns : pd.DataFrame
            roi : str
                Brain region, currently either ME(R), LO(R), or LOP(R)
            hex1 : int
                hex1_id, the internal coordinate of the column
            hex2 : int
                hex2_id, the coordinate of the column the pin is located in
            x : float
                location in space
            y : float
                location in space
            z : float
                location in space
            depth_bin : int
                depth within the column.
        """
        if len(self.__columns) == 0:
            cache_file = self.__cache / "columns.pickle"
            if cache_file.is_file():
                self.__columns = pd.read_pickle(cache_file)
            else:
                cql = """
                    UNWIND ['ME(R)', 'LO(R)', 'LOP(R)'] as roi
                    MATCH (cp:ColumnPin)
                    WHERE cp[roi]
                    RETURN
                        roi as roi
                      , cp.olHex1 as hex1
                      , cp.olHex2 as hex2
                      , cp.location.x as x
                      , cp.location.y as y
                      , cp.location.z as z
                      , cp.depth as depth_bin
                """
                self.__columns = fetch_custom(cql)
                # Due to a bug in the assignment of pins to ROIs, neuprint had 9 ColumnPin from
                # ME(R) assigned to LO(R). As a result, the deptb_bin for LO(R) went all the way
                # up to 120. Capping at 75 is a workaround that issue until fixed.
                self.__columns = self.__columns\
                    .drop(self.__columns[
                        (self.__columns['roi']=='LO(R)')\
                      & (self.__columns['depth_bin'] > 75)]\
                    .index)
                self.__columns['depth'] = self.__columns['depth_bin']\
                    / self.__columns.groupby('roi')['depth_bin'].transform('max')
                self.__columns.to_pickle(cache_file)
        return self.__columns


    def __get_input_output(
        self
      , direction:str
      , limit:int=None
    ) -> pd.DataFrame:
        assert direction in ['upstream', 'downstream'],\
            f"Direction must either be upstream or downstream, not {direction}"
        assert limit is None or (0 < limit and isinstance(limit, int)),\
            f"If limit is defined, it must be an integer > 0, not {limit} of type {type(limit)}"

        match direction:
            case 'downstream':
                cql_match = "MATCH (n:Neuron)-[e:ConnectsTo]->(m:Neuron)"
            case 'upstream':
                cql_match = "MATCH (n:Neuron)<-[e:ConnectsTo]-(m:Neuron)"

        if self.__connection_cutoff:
            cutoff = f"WHERE wgt >= {self.__connection_cutoff}"
        elif self.__per_cell_cutoff:
            cutoff = f"WHERE wgt/cell_count > {self.__per_cell_cutoff}"

        cql = f"""
            {cql_match}
            WHERE n.instance = '{self.instance_name}'
            AND NOT m.instance IS NULL AND NOT m.type IS NULL
            WITH sum(e.weight) as all_sym, count(distinct n.bodyId) as cell_count 
            {cql_match}
            WHERE n.instance = '{self.instance_name}'
            AND NOT m.instance IS NULL AND NOT m.type IS NULL
            WITH sum(e.weight) as wgt, m.instance as instance, all_sym, cell_count
            {cutoff}
            RETURN instance, toFloat(wgt)/all_sym as perc
            order by wgt DESC, instance ASC
            LIMIT {limit}
        """
        ret = fetch_custom(cql)
        ret['cum_perc'] = ret['perc'].cumsum()
        return ret


    def __get_depths(
        self
      , with_syn_type:bool=False
    ) -> pd.DataFrame:
        """
        Return a dataFrame with one row per ROI and depth. For example, this will currently create
        a DataFrame with ~120 rows for ME(R), 75 for LO(R), and 50 for LOP(R). If `with_syn_type`,
        it returns the table twice, once with 'pre' in the type column, once with 'post' in the
        type column.

        This function is for internal use to have an easy join target and to extend existing
        data frames to have at least one value per bin (if necessary).

        Parameters
        ----------
        with_syn_type : bool
            If true, the table is meant to be joined with the synapse type specific table.
            Otherwise its for the synapse table without type specification.

        Returns
        -------
        df : pd.DataFrame
            roi : str
                name of the ROI
            depth_bin : int
                depth, as defined by the ColumnPins in neuPrint
            depth : float
                normalized depth between 0…1
            type : str, Optional
                either 'pre' or 'post'
        """
        all_depths = self.columns\
            .loc[:, ['roi', 'depth_bin', 'depth']]\
            .drop_duplicates()
        if with_syn_type:
            all_depths_post = self.columns\
                .loc[:, ['roi', 'depth_bin', 'depth']]\
                .drop_duplicates()
            all_depths['type'] = 'pre'    # E.G isn't this extracting the same set of depths?
            all_depths_post['type'] = 'post'
            all_depths = pd.concat([all_depths_post, all_depths])
        return all_depths

    @property
    def __bids_df(self)->pd.DataFrame:
        cql = f"""
            MATCH (n:Neuron)
            WHERE n.instance='{self.instance_name}'
            RETURN n.bodyId as bid
        """
        all_bodies = fetch_custom(cql)
        return all_bodies
