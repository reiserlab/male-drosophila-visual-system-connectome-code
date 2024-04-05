from abc import ABC

import os
import re
import warnings

from pathlib import Path
import pickle
import trimesh
import pandas as pd
import numpy as np
from dotenv import find_dotenv
from trimesh.transformations import \
    quaternion_from_euler, quaternion_multiply\
  , translation_matrix, quaternion_matrix, quaternion_inverse

import navis
import navis.interfaces.neuprint as neu
from navis.transforms.affine import AffineTransform
from neuprint import fetch_neurons, NeuronCriteria as NC

from queries.ol_neuron import \
    fetch_ol_rois_distance, fetch_ol_rois_synapses, fetch_ol_rois_assigned

from utils.ng_view import NG_View

class OLNeuron(ABC):

    def __init__(
        self
      , body_id:int
    ):
        """
        Create a OLNeuron

        Parameters
        ----------
        body_id : int
            body ID of the neuron.
        """
        if isinstance(body_id, np.generic):
            cast_bid = body_id.item()
        else:
            cast_bid = body_id
        assert cast_bid == body_id, "Make sure body_id is an integer"
        assert isinstance(cast_bid, int), "body_id needs to be an integer."
        self.__body_id = cast_bid


    def get_mesh(
        self
      , ignore_cache:bool=False
    ) -> navis.NeuronList:
        """
        Retrieve mesh for neuron.

        For performance reasons, this function first attempts to load the mesh from a local cache.
        If this fails, the mesh is retrieved from neuprint and saved in the cache. The cache is
        located at the PROJECT_PATH/cache/meshes. To force a reload, you can either set the
        ignore_cache flag or delete files from the cache.

        Parameters
        ----------
        ignore_cache : bool, default = False
            When enabled, load the mesh from neuprint and overwrite the currently cached file.

        Returns
        -------
        mesh : NeuronList
            A mesh for the neuron
        """
        cachedir = Path(find_dotenv()).parent / "cache" / "meshes"
        cachedir.mkdir(parents=True, exist_ok=True)
        mesh_fn = cachedir / f"ms_{self.__body_id}.pickle"
        if not ignore_cache and mesh_fn.is_file():
            with mesh_fn.open('rb') as mesh_fh:
                mesh = pickle.load(mesh_fh)
        else:
            mesh = neu.fetch_mesh_neuron(
                NC(bodyId=self.__body_id)
              , seg_source=os.environ['SEGMENTATION_SOURCE']
              , lod=None
            )
            with mesh_fn.open('wb') as mesh_fh:
                pickle.dump(mesh, mesh_fh)
        return mesh


    def get_skeleton(
        self
      , ignore_cache:bool=False
    ) -> navis.core.skeleton.TreeNeuron:
        """
        Get a skeleton for a neuron.

        For performance reasons, this function first attempts to load the skeleton from a
        local cache. If this fails, the skeleton is retrieved from neuprint and saved in
        the cache. The cache is located at the PROJECT_PATH/cache/skeletons. To force a
        reload, you can either set the `ignore_cache` flag or delete files from the cache.

        Parameters
        ----------
        ignore_cache : bool, default = False
            When enabled, load the skeleton from neuprint and overwrite the currently cached file.

        Returns
        -------
        skel : NeuronList
            A skeleton for the neuron of body_id
        """
        cachedir = Path(find_dotenv()).parent / "cache" / "skeletons"
        cachedir.mkdir(parents=True, exist_ok=True)
        skel_fn = cachedir / f"sk_{self.__body_id}.pickle"
        if not ignore_cache and skel_fn.is_file():
            with skel_fn.open('rb') as skel_fh:
                skel = pickle.load(skel_fh)
        else:
            try:
                neuronlist = neu.fetch_skeletons(NC(bodyId=self.__body_id))
            except ValueError:
                skel = None
            if len(neuronlist)==1:
                skel = neuronlist[0]
            elif len(neuronlist)==0:
                warnings.warn(f"Could not find skeleton for body ID {self.__body_id}")
                skel = None
            else:
                warnings.warn(f"Got more than one skeleton for body ID {self.__body_id}")
                skel = None
            if skel:
                with skel_fn.open('wb') as skel_fh:
                    pickle.dump(skel, skel_fh)
        if skel is None:
            skel = navis.TreeNeuron(None)
        return skel


    def get_body_id(
        self
    ) -> int:
        """
        Get body_id of the OL_Neuron

        Returns
        -------
        bid : int
            body_id of the neuron
        """
        return self.__body_id

    @property
    def hemisphere(
        self
    ) -> str:
        """
        Get hemisphere of the OL_Neuron

        Returns
        -------
        hemisphere : str
            either 'L' or 'R'
        """
        neuron_df, _ = fetch_neurons(NC(bodyId=self.__body_id))
        return neuron_df.loc[0, 'instance'][-1:]


    def me_hex_id(self):
        """
        Get the Medulla hex IDs.

        Return the manually assigned column, if there is one. Otherwise use the
        synapse count method. If that fails, use the centroid method 
        (see `get_hex_id()` for details).
        """
        # from correct and fast to guessing and slow:
        for method in ['assigned', 'synapse_count', 'centroid']:
            ret = self.get_hex_id(roi_str='ME(R)', method=method)
            if ret.size:
                return ret
        return None

    @property
    def lo_hex_id(self):
        """
        Get the Lobula hex IDs.

        Return the synapse count method, otherwise fall back to centroid 
        (see `get_hex_id()` for details).
        """
        # from correct and fast to guessing and slow:
        for method in ['synapse_count', 'centroid']:
            ret = self.get_hex_id(roi_str='LO(R)', method=method)
            if ret.size:
                return ret
        return None

    @property
    def lop_hex_id(self):
        """
        Get the Lobula Plate hex IDs.

        Return the synapse count method, otherwise fall back to centroid 
        (see `get_hex_id()` for details).
        """
        # from correct and fast to guessing and slow:
        for method in ['synapse_count', 'centroid']:
            ret = self.get_hex_id(roi_str='LOP(R)', method=method)
            if ret.size:
                return ret
        return None

    def get_roi_hex_id(
        self
      , roi_str:str=None
    ):
        for method in ['assigned', 'synapse_count', 'centroid']:
            ret = self.get_hex_id(roi_str=roi_str, method=method)
            if ret.size:
                return ret
        return None

    def get_hex_id(
        self
      , roi_str:str=None
      , method:str='synapse_count'
    ) -> pd.DataFrame | None:
        """
        Get a list of hex IDs for the neuron.

        Parameters
        ----------
        roi_str : str, default=None
            only return results for the specified ROI. Use the form 'ME(R)'. A wrongly
            formatted ROI leads to an empty result set
        method : {'assigned', 'synapse_count', 'centroid'}
            For method 'assigned', return the manually assigned hex ID (if there is one).
            For 'synapse_count' method, count how many synapses are in which coulumn and
                return the column with the highest count.
            For 'centroid', get all synapses inside the ROI and calculate their centroid.
                Return the column whose center of mass is closest ot the synapse centroid.

        Returns
        -------
        ret : pd.DataFrame
            ROI : str
                brain region, for example 'ME(R)'
            hex1_id : int
                hex 1 ID (HEX coordinate system)
            hex2_id : int
                hex 2 ID (HEX coordinate system)
        """
        assert method is None \
            or method in ['synapse_count', 'assigned', 'centroid'],\
            "no other method implemented yet"
        ret = None
        match method:
            case 'synapse_count':
                ret = self.__get_synapse_hex_id()
            case 'centroid':
                ret = self.__get_centroid_hex_id()
            case 'assigned':
                ret = self.__get_assigned_hex_id()
        if roi_str:
            ret = ret[ret['ROI']==roi_str]
        return ret


    def __get_synapse_hex_id(
        self
    ):
        all_rois = fetch_ol_rois_synapses(body_id=self.get_body_id())
        if all_rois.size:
            all_rois = all_rois\
                .sort_values(['ROI', 'synapse_count'], ascending=False)\
                .groupby(by='ROI')\
                .head(1)\
                .drop('synapse_count', axis=1)\
                .reset_index(drop=True)
            all_rois['hex1_id'] = all_rois['hex1_id'].astype("Int64")
            all_rois['hex2_id'] = all_rois['hex2_id'].astype("Int64")
        return all_rois

    def __get_assigned_hex_id(
        self
    ):
        all_rois = fetch_ol_rois_assigned(body_id=self.get_body_id())
        return all_rois


    def __get_centroid_hex_id(
        self
    ):
        ret = pd.DataFrame(columns=['ROI', 'hex1_id', 'hex2_id'])
        all_rois = fetch_ol_rois_distance(body_id=self.get_body_id())
        if all_rois.size:
            ret[['ROI', 'hex1_id', 'hex2_id']] = all_rois['syn_keys']\
                .apply(lambda x: re.findall(r'^(ME|LO|LOP)_R_col_(\d+)_(\d+)$', x))\
                .apply(pd.Series)[0].apply(pd.Series)
            ret = ret.dropna().reset_index(drop=True)
            ret['ROI'] = ret['ROI'].astype(str) + '(R)'
            ret['hex1_id'] = ret['hex1_id'].astype("Int64")
            ret['hex2_id'] = ret['hex2_id'].astype("Int64")
            # FIXME: This potentially drops a random result. For close inspection
            # have a look at 86459, 132357, 90380, 138036 (which return 2+ values from the query)
            ret = ret\
                .groupby(by='ROI')\
                .head(1)\
                .reset_index(drop=True)
        else:
            warnings.warn(f"I can't estimate a hex_id for {self.get_body_id()}")
        return ret


    def get_type(
        self
    ) -> str:
        """
        Get neuron type of OL_Neuron. 

        TODO: extend to using the instance name if type is empty or non-sensical.

        Returns
        -------
        type : str
            type of the neuron. Currently this is just the `type` column in the database entry.
        """
        neuron_df, _ = fetch_neurons(NC(bodyId=self.__body_id))
        return neuron_df.loc[:, 'type'].values[0]

    @property
    def instance(
        self
    ) -> str:
        """
        Get instance for OL_Neuron

        Returns
        -------
        instance : str
            instance of the neuron
        """
        neuron_df, _ = fetch_neurons(NC(bodyId=self.__body_id))
        return neuron_df.loc[:, 'instance'].values[0]

    def get_slice(
        self
      , view:NG_View
      , thickness:int=200
      , only_show_cutter:bool=False
    ) -> navis.Volume | None:
        """
        The attempt to slice the neuron mesh using an externally defined view. The method is not
        working very well, was only used for the GalleryPlotter (which is being replaced by
        another plotting approach).

        The slicing is done via trimesh, which uses either blender or openscad(?). So you will
        need one of these software packages for this to work.

        Parameters
        ----------
        view : NG_View
            The view that defines the direction of the cut
        thickness : int, default=200
            Thickness of the slice in nm
        only_show_cutter : bool, default=False
            debugging option: if true, `get_slice` attempts to plot the original mesh and the
            cutting box.

        Returns
        -------
        intersect : navis.Volume
            the result of the intersection. If `only_show_cutter`, then nothing is returned.
        """
        mesh = self.get_mesh().trimesh[0]
        xsz = mesh.bounds[1][0] - mesh.bounds[0][0]
        ysz = mesh.bounds[1][1] - mesh.bounds[0][1]
        cutter = trimesh.creation.box(extents=(xsz, ysz, thickness))
        q1 = view.orientation_camera
        q2 = quaternion_from_euler(np.pi/2, 0, 0, 'sxyz')
        q3 = quaternion_multiply(q1, q2)

        rotate = quaternion_matrix(q3)
        center = mesh.centroid
        translate = translation_matrix(center)
        cutter.apply_transform(rotate)
        cutter.apply_transform(translate)
        if only_show_cutter:
            navis.plot3d([mesh, cutter])
        else:
            intersect = trimesh.boolean.intersection([mesh, cutter])
            return navis.Volume(intersect)


    def get_flat_view(
        self
      , view:NG_View
      , center:list[float]=None
      , thickness:int=None
      , continuous:bool=False
      , as_mesh:bool=True
      , keep_soma:bool=False
      , ignore_fixation:bool=False
    ) -> navis.TreeNeuron | trimesh.Trimesh | navis.MeshNeuron:
        """
        The attempt to project the slice of neuron onto a plane.

        This method is more a draft than a well tested piece of code. It worked OK for a few
        cells, but since GalleryPlotter is being replaced by a new approach, this is not
        needed nor tested anymore.

        Parameters
        ----------
        view : NG_View
            The view that defines the cut and the projection.
        center : list[float]
            x,y,z coordinates for the "center" of the plane and view. Usually the
            center of mass of the neuron is used, but for special cases the "center"
            can be defined. The projection plane and the cutter are all defined around 
            the "center".
        thickness : int, default=None
            thickness of the slice
        continuous : bool, default=False
            only keep parts of the neuron that are connected to the main body and remove fragments
            that become disconnected after slicing
        as_mesh : bool, default=True
            make the skeleton into a mesh before returning.
        keep_soma : bool, default=False
            try to keep the axon that connects the soma to the part that is left after slicing
        ignore_fixation : bool, default=False
            if true, release the camera from fixating on the point defined by the view and instead
            use the neuron center of mass
        
        Returns
        -------
        skel : navis.TreeNeuron | trimesh.Trimesh | navis.MeshNeuron
            return a plotable skeleton or mesh that is projected onto a plane
        """
        skel = self.get_skeleton()
        if skel.name is None:
            return skel

        assert not (as_mesh and keep_soma)\
            , "Can't do both: keeping the soma and converting to meshes (known bug)"
        if view.location_camera:
            center = np.array(view.location_camera)
        elif not center:
            center = self.get_center()

        if ignore_fixation:
            center = self.get_center()

        move_to_center = AffineTransform(translation_matrix(-center))
        move_from_center = AffineTransform(translation_matrix(center))

        q1 = view.orientation_camera
        q22 = quaternion_from_euler(0, np.pi/2, 0, 'sxyz')
        rotate_from_center = AffineTransform(quaternion_matrix(quaternion_inverse(q1)))

        rotate_to_center = AffineTransform(quaternion_matrix(q1))

        rotate_at_center = AffineTransform(quaternion_matrix(q22))
        unrotate_at_center = AffineTransform(quaternion_matrix(quaternion_inverse(q22)))

        flatten = AffineTransform([[1,0,0,0], [0,1,0,0], [0,0,0,1]])

        skel = navis.xform(skel, move_to_center)

        skel = navis.xform(skel, rotate_to_center)
        skel = navis.xform(skel, rotate_at_center)
        if thickness:
            skel = self.__slice_skeleton(
                skel, thickness=thickness, continuous=continuous, keep_soma=keep_soma)
        skel = navis.xform(skel, flatten)
        skel = navis.xform(skel, unrotate_at_center)
        skel = navis.xform(skel, rotate_from_center)
        skel = navis.xform(skel, move_from_center)
        if as_mesh:
            skel = navis.mesh(skel)
        return skel


    def __slice_skeleton(self, skel, thickness, continuous, keep_soma):
        bounds = skel.bbox
        bounds[2] = [-thickness/2, thickness/2]
        bounds = bounds.reshape(6).reshape((2,3), order='F')
        cutter = trimesh.creation.box(bounds=bounds)
        cutter_vol = navis.Volume(cutter)
        p_skel = skel.prune_by_volume(
            cutter_vol
          , mode='IN'
          , prevent_fragments=continuous
          , inplace=False
        )
        if keep_soma:
            cbf = skel.cell_body_fiber(reroot_soma=True)
            p_skel = navis.stitch_skeletons([p_skel, cbf])
        return p_skel


    def get_center(
        self
    ):
        """
        Center of mass of the neuron

        Returns
        -------
        center : list[float]
            get x,y,z coordinates of the neuron's center of mass
        """
        mesh = self.get_mesh().trimesh[0]
        return mesh.centroid


    def innervated_rois(
        self
      , column_threshold:float|int=0
      , layer_threshold:float|int=0
    ):
        """ 
        Find the column and layer ROI names which the neuron innervates. The innervation is 
        defined by the absolute number/fraction of synapses surpassing the threshold in 
        any of the ROIs.

        Parameters
        ----------
        column_threshold : int|float, default=0
            threshold absolute number/fraction of synapses that the bodyId should contain 
            in any column roi
        layer_threshold: int|float, default=0
            threshold absolute number/fraction of synapses that the bodyId should contain 
            in any layer roi

        Returns
        -------
        column_rois : list[str]
            column rois names the neuron innervates
        layer_rois : list[str]
            layer rois names the neuron innervates
        """

        assert column_threshold >= 0, "column threshold needs to be positive."
        assert layer_threshold >= 0, "layer threshold needs to be positive."

        _, roi_df = fetch_neurons(NC(bodyId=self.__body_id))
        roi_df['syncount'] = roi_df['pre'] + roi_df['post']
        roi_df['fraction_syn'] = roi_df['syncount'] / roi_df['syncount'].sum()

        # separate the dataframes for column rois and layer rois
        roi_df_column = roi_df.loc[roi_df["roi"].str.contains("_col_")]
        roi_df_layer = roi_df.loc[roi_df["roi"].str.contains("_layer_")]

        # get the column and layer rois with # synapses >= threshold
        if 0 < column_threshold < 1:
            column_rois = roi_df_column\
                .loc[roi_df_column["fraction_syn"] >= column_threshold, 'roi']\
                .tolist()
        else:
            column_rois = roi_df_column\
                .loc[roi_df_column["syncount"] >= column_threshold, 'roi']\
                .tolist()
        if 0 < layer_threshold < 1:
            layer_rois = roi_df_layer\
                .loc[roi_df_layer["fraction_syn"] >= layer_threshold, 'roi']\
                .tolist()
        else:
            layer_rois = roi_df_layer\
                .loc[roi_df_layer["syncount"] >= layer_threshold, 'roi']\
                .tolist()
        return column_rois, layer_rois


    def synapse_count(
        self
    ) -> list[int]:
        """
        Returns the number or pre and post synapses.

        Returns
        -------
        pre : int
            number of pre-synaptic sites (output)
        post : int
            number of post-synaptic sites (input)
        """
        neuron_df, _ = fetch_neurons(NC(bodyId=self.__body_id))
        return neuron_df.loc[:, ['pre', 'post']].values[0]
