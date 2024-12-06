# %%
from pathlib import Path
import sys
import warnings
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")

import navis.interfaces.neuprint as neu

from utils import olc_client
c = olc_client.connect()


# %%
import trimesh
from math import radians
from trimesh.transformations import euler_matrix, translation_matrix
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate neuropil cross-sections.')
parser.add_argument(
    'set_type'
  , choices=['standard', 'connecting']
  , help='The set of cross-sections to generate.'
)
args = parser.parse_args()

keys_standard = {
    'ME': {
        'rotation': [radians(89.733), radians(-40.965), radians(-2.9247)]
      , 'translation': [16922, 35517, 33376]
    }
  , 'LO' : {
        'rotation': [radians(-94.259), radians(49.165), radians(-20.862)]
      , 'translation': [26774, 31174, 36001]
    }
  , 'LOP' : {
        'rotation': [radians(237.98), radians(85.053), radians(309.74)]
      , 'translation': [23763, 34009, 41900]
  }
}

keys_for_connecting = {
    'ME': {
        'rotation': [radians(184.65), radians(50.773), radians(-10.572)]
      , 'translation': [13269, 36333, 29779]
    }
  , 'LO' : {
        'rotation': [radians(184.65), radians(50.773), radians(-10.572)]
      , 'translation': [13269, 36333, 29779]
    }
  , 'LOP' : {
        'rotation': [radians(184.65), radians(50.773), radians(-10.572)]
      , 'translation': [13269, 36333, 29779]
  }
}

# Choose the correct set of keys and output directory based on the command-line argument
if args.set_type == 'standard':
    config = keys_standard
    outdir = PROJECT_ROOT / "cache" / "blender" / "crossections"
else:  # 'connecting'
    config = keys_for_connecting
    outdir = PROJECT_ROOT / "cache" / "blender" / "crossections_connecting"

outdir.mkdir(parents=True, exist_ok=True)


def cut_layer(layer_mesh, neuropil, keys):
    """
    Create a cross section for a mesh. Using either of the predefined slices.

    Parameters
    ----------
    layer_mesh : navis.Volume
        Mesh to create a cross section from
    neuropil : str
        either 'ME(R)', 'LO(R)', or 'LOP(R)'
    keys : dict
        definition of rotation and translation for each ROI

    Returns
    -------
    crossection : Trimesh
        mesh of the cross section
    """

    cutter = trimesh.primitives.Box(extents=[30000, 750, 70000])
    n_name = neuropil[:-3]

    rot = euler_matrix(
        keys[n_name]['rotation'][0]
      , keys[n_name]['rotation'][1]
      , keys[n_name]['rotation'][2]
    )
    trans = translation_matrix(keys[n_name]['translation'])
    cutter.apply_transform(rot)
    cutter.apply_transform(trans)

    try:
        # Attempt the intersection
        result = trimesh.boolean.intersection(
            [layer_mesh, cutter]
          , check_volume=False
          , engine='blender'
        )
    except:
        warnings.warn(f"Intersection not working for {layer_mesh.name}")

    if result is None or result.is_empty:
        print(f"No intersection found or result is None for {neuropil}")
        return None
    return result

for me_id in [f"ME_R_layer_{i:02d}" for i in range(1,11)]:
    me_l = neu.fetch_roi(me_id)
    me_cut = cut_layer(me_l, "ME(R)", config)
    # me_cut.export( outdir / f"{me_id}.obj")
    if me_cut is not None:
        me_cut.export(outdir / f"{me_id}.obj")
    else:
        print(f"Skipping export for {me_id} due to no intersection or empty result.")

for me_id in [f"LO_R_layer_{i}" for i in range(1,8)]:
    me_l = neu.fetch_roi(me_id)
    me_cut = cut_layer(me_l, "LO(R)", config)
    if me_cut is not None:
        me_cut.export(outdir / f"{me_id}.obj")
    else:
        print(f"Skipping export for {me_id} due to no intersection or empty result.")

for me_id in [f"LOP_R_layer_{i}" for i in range(1,5)]:
    me_l = neu.fetch_roi(me_id)
    me_cut = cut_layer(me_l, "LOP(R)", config)
    if me_cut is not None:
        me_cut.export(outdir / f"{me_id}.obj")
    else:
        print(f"Skipping export for {me_id} due to no intersection or empty result.")
