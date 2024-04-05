"""
Use a PlotConfig file to add labels ot a png and create a PDF from it.
"""

import sys
from pathlib import Path
import argparse
from dotenv import find_dotenv
sys.path.append(str(Path(find_dotenv()).parent.joinpath('src')))
from utils.plot_config import PlotConfig
from utils.pdf_maker import PDFMaker


def set_config(
    args
) -> dict:
    """
    Set config

    Parameters
    ----------
    args :
        Command line parameters

    Returns
    -------
    config : dict
        configurtion dictionary
    """
    config = {}
    pcfg = PlotConfig(args.config)
    config['text'] = pcfg.text_dict
    config['directory'] = pcfg.directory
    config['name'] = pcfg.name
    config['resolution'] = pcfg.camera['resolution']
    config['filename'] = pcfg.basename
    return config


def run_script(args) -> None:
    """
    Read png, add text, and write pdf.

    Parameters
    ----------
    args :
        Command line parameters
    """
    config = set_config(args)
    height = 57  # comes close to 1920×1080 for full brain view (movie)
    width = height * config['resolution'][0]/config['resolution'][1]

    working_path = Path(find_dotenv()).parent / "results" / "gallery"
    mkr = PDFMaker(width, height, 450)
    coords, _, _ = mkr.get_page_layout_rows_cols(1,1)
    mkr.add_image( working_path / config['directory'] / f"{config['filename']}.png", coords[0])
    for _, values in config['text'].items():
        mkr.add_text(
            text=values['text']
          , position=values['pos']
          , color=values['color'][:3]
          , align=values['align'])
    mkr.save(working_path / config['directory'] / f"{config['filename']}.pdf")


def cli(argv) -> None:
    """
    Extract command line parameters and call `run_script()`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('scriptname')
    parser.add_argument("--config", type=Path, help="path to PlotConfig file")
    args = parser.parse_args(argv)
    run_script(args)


if __name__ == "__main__":
    argv = sys.argv
    cli(argv)
