"""
Get all the files necessary to render output.

This means, downloading the meshes for neurons and brain regions based on
either a config file or a body ID.
"""

from pathlib import Path
import json
import sys
import os
import argparse
import warnings
from dotenv import load_dotenv, find_dotenv

import navis
import cloudvolume
from trimesh.exchange.export import export_mesh

load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))

from utils import olc_client
from utils.plotter import get_roi
from utils.ol_neuron import OLNeuron
from utils.helper import slugify
from utils.plot_config import PlotConfig
from utils.plot_roi import PlotRoi


def download_neuron_mesh(
    body_id:int
  , cache_path:Path
) -> None:
    """
    Get the mesh of a neuron identified by body_id and save it as an stl file, ready to load by
    blender. A metadata json file for the same neuron is also stored.

    Parameters
    ----------
    body_id : int
        body ID of neuron
    cache_path : Path
        path to a PlotConfig json configuration file
    """
    neuron_fn = cache_path / f"neuron.{body_id}.obj"
    if not neuron_fn.is_file():
        neuron = OLNeuron(body_id)
        nl = neuron.get_mesh(ignore_cache=True)
        if isinstance(nl, navis.NeuronList) and len(nl)==1:
            mesh = nl[0].trimesh
        else:
            warnings.warn(f"No mesh found for {body_id}")
        export_mesh(mesh, neuron_fn)


def get_full_brain(
    roi:PlotRoi
  , cache_path:Path
) -> None:
    """
    Downloads the full shell of a brain. Kinda hacky…

    This requries the `SHELL_SOURCE` environment variable from the `.env` file.

    Parameters
    ----------
    cache_path : Path
        path to a PlotConfig json configuration file
    """
    assert os.environ.get('SHELL_SOURCE'),\
        "Please set the `SHELL_SOURCE` variable in your environment."
    assert os.environ.get('FULL_SHELL_SOURCE'),\
        "Please set the `FULL_SHELL_SOURCE` variable in your environment."
    roi_fn = cache_path / f"{roi.filename}.obj"
    if roi_fn.is_file():
        return

    vol = cloudvolume.CloudVolume(
        f"precomputed://{os.environ['SHELL_SOURCE']}"
      , use_https=True
      , progress=False
      , mip=[512,512,512]
    )
    for idx, name in enumerate(['CB', 'OL_L_', 'OL_R_'], start=1):
        roi_fn = cache_path / f"roi.{name}.obj"
        if not roi_fn.is_file():
            vol.mesh.save(idx, roi_fn, file_format='obj')

    vol2 = cloudvolume.CloudVolume(
        f"precomputed://{os.environ['FULL_SHELL_SOURCE']}"
      , use_https=True
      , progress=False
      , mip=[512,512,512]
    )
    roi_fn = cache_path / "roi.full_brain.obj"
    if not roi_fn.is_file():
        vol2.mesh.save(1, roi_fn, file_format='obj')


def download_roi_mesh(
    roi:PlotRoi
  , cache_path:Path
) -> None:
    """
    Download the mesh for region of interests. The result is stored in the cache path
    `(cache/gallery/*.stl)`

    Parameters
    ----------
    roi : PlotRoi
        PlotRoi object for the ROI to download
    cache_path : Path
        path to a PlotConfig json configuration file
    """
    if roi.name in ['CB', 'OL(R)', 'OL(L)']:
        get_full_brain(roi=roi, cache_path=cache_path)
        
    roi_fn = cache_path / f"{roi.filename}.obj"
    # if roi_str in ['ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']:
    #     roi_fn = cache_path / f"roi.{roi_str[:-3].lower()}_{roi_str[-2].lower()}.stl"
    if not roi_fn.is_file():
        mesh = get_roi(roi.name)
        export_mesh(mesh, roi_fn)


def run_script(
    args:argparse.Namespace
) -> None:
    """
    The script itself.

    Parameters
    ----------
    args : argparse.Namespace
        get the arguments from the command line interface
    """
    _ = olc_client.connect(verbose=False)

    cache_path = Path(find_dotenv()).parent / "cache" / "gallery"
    cache_path.mkdir(parents=True, exist_ok=True)

    all_bids = [args.body_id]

    if args.config:
        pcfg = PlotConfig(args.config)
        all_bids = pcfg.bids
        rois = pcfg.rois
        print(f"This PlotConfig file contains {len(all_bids)} body IDs.")

    for bid in all_bids:
        download_neuron_mesh(body_id=bid, cache_path=cache_path)

    for roi in rois:
        download_roi_mesh(roi, cache_path=cache_path)


def cli() -> None:
    """
    Parse the command line and execute the script
    """
    parser = argparse.ArgumentParser()
    arg_group = parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument("--body_id", type=int, help="body ID of the neuron to plot")
    arg_group.add_argument("--config", type=Path, help="path to a PlotConfig file")
    args = parser.parse_args()
    run_script(args)


if __name__ == "__main__":
    """ 
    This gets executed when run from the cli 
    """
    cli()
