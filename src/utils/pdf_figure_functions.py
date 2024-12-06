"""
Functions to generate the combined pdfs from optic lobe group or gallery png files. 
"""
from pathlib import Path
import os
import glob
import warnings

from dotenv import find_dotenv
from PIL import Image, ImageDraw

from utils.pdf_maker import PDFMaker
from utils.plot_config import PlotConfig
from utils.ol_types import OLTypes

def generate_group_pdf(plot_type:str, pdf_specs:dict):
    """
    Function to generate PDF plot of premade group pngs.

    Parameters
    ----------
    plot_type : str
        either 'VPN' or 'VCN'
    pdf_specs : dict
        pdf_w : int
            pdf width in mm
        pdf_h : int
            pdf height in mm
        pdf_res : int
            pdf resolution in dpi
        pdf_margin : tuple
            paper margin in mm. [margin_x, margin_y]
    """
    assert plot_type in ['VCN', 'VPN'], f"plot type must be VPN or VCN, not {plot_type}"
    json_string = "Full-Brain_Group_"
    PROJECT_ROOT = Path(find_dotenv()).parent
    gallery_dir = PROJECT_ROOT / "results" / "gallery"
    # paths to PNGs
    if plot_type == "VCN":
        img_path = gallery_dir / "vcn_group_plots"
    else: # VPN
        img_path = gallery_dir / "vpn_group_plots"
    # path to JSONs
    json_path = PROJECT_ROOT / "results" / "gallery-descriptions"

    assert plot_type in ["VPN", "VCN"]\
      , f"type of plot has the unexpected value {plot_type} - only 'VPN' and 'VCN' are allowed."

    default_views = {"VPN": ["whole_brain", "half_brain"], "VCN": ["whole_brain"]}

    default_rows_cols = {
        "VPN": {"whole_brain": [4, 4], "half_brain": [7, 5]}
      , "VCN": {"whole_brain": [5, 4]}
    }

    default_apspect_ratio = {
        "VPN": {"whole_brain": 2, "half_brain": 1}
      , "VCN": {"whole_brain": 2, "half_brain": 2}
    }

    ol = OLTypes()
    if plot_type == "VCN":  # include 'other's with 'VCN' plots
        df_group = ol.get_neuron_list(primary_classification=[plot_type, "other"])
        df_group["fb_view"] = "whole_brain"
    else:
        df_group = ol.get_neuron_list(primary_classification=plot_type)

    figure_groups = df_group["figure_group"].unique()
    views_to_use = default_views[plot_type]

    for view in views_to_use:
        # generate the empty page
        doc = PDFMaker(
            width=pdf_specs["pdf_w"]
          , height=pdf_specs["pdf_h"]
          , resolution=pdf_specs["pdf_res"]
          , margin=pdf_specs["pdf_margin"]
        )

        df = df_group[df_group["fb_view"] == view]
        figure_groups = list(df["figure_group"].unique())

        rows_cols = default_rows_cols[plot_type][view]
        aspect_ratio = default_apspect_ratio[plot_type][view]

        if plot_type == "VPN" and view == "half_brain":
            # some groups have neurons that are classified as half_brain but are only shown in the full_brain plot
            incorrect_grouping = [
                "mainly_AME_1"
              , "LO_VPN_LC_2"
              , "LO_SPS_IB"
              , "LO_VPN_contra_2"
            ]
            # remove these groups from the half_brain plot
            for name_to_remove in incorrect_grouping:
                figure_groups.remove(name_to_remove)

        # get the position of each of the individual figs on the page
        coords, img_width, img_height = doc.get_page_layout_rows_cols(
            rows=rows_cols[0], cols=rows_cols[1], aspect_ratio=aspect_ratio
        )
        save_name = f"{plot_type}_{view}_{rows_cols[0]}R_{rows_cols[1]}C_{pdf_specs['pdf_w']}mm_{pdf_specs['pdf_h']}mm_{pdf_specs['pdf_res']}dpi.pdf"

        for idx, group_name in enumerate(figure_groups):
            x_top, y_top = coords[idx][:2]

            # add image to pdf
            doc.add_image(
                f"{img_path}/{json_string[:-7]}.{group_name}.png", coords[idx]
            )

            # get text and position
            json_name = f"{json_string}{group_name}.json"
            config_name = f"{json_path}/{json_name}"
            config = PlotConfig(config_filename=config_name)
            text_dict = config.text_dict

            for _, t in text_dict.items():
                align = t["align"]
                if align == "c":
                    plot_pos_x = 0.5
                    plot_pos_y =0.98
                else:
                    plot_pos_x = t["pos"][0] - 0.02
                    plot_pos_y = 1 - t["pos"][1]
                
                paper_pos_x = (plot_pos_x * img_width) + x_top
                paper_pos_y = (plot_pos_y * img_height) + y_top

                doc.add_text(
                    text=t["text"],
                    position=[paper_pos_x, paper_pos_y],
                    color=t["color"],
                    align=align,
                    font_size=pdf_specs["font_size"],
                )

        doc.save(filename=save_name, directory=gallery_dir)


def generate_gallery_pdf(
    pdf_specs:dict
  , page_idx:int
  , instances:list
  , nudge_dict:dict
  , rows_cols:tuple=None
  , offset:int=45
):
    """
    Function to generate PDF plot of premade gallery pngs.

    Parameters
    ----------
    pdf_specs : dict
        pdf_w : int
            pdf width in mm
        pdf_h : int
            pdf height in mm
        pdf_res : int
            pdf resolution in dpi
        pdf_margin : tuple
            paper margin in mm. [margin_x, margin_y]
    page_idx : int
        page number
    instances : list
        list of neuron type instances used to generate combined pdf
    main_groups : str
        main group the neurons plotted belong to.
    nudge_dict : dict
        dict with int values of how much to move the text in the y direction for the gallery
        images of the keys (neuron types). Positive values in the dict move the text up.
    rows_cols : tuple, default=None
        tuple of the number of rows and columns to use in the pdf i.e. [4, 6] - 4 rows, 6 columns
        If None, then rows_cols is set as [len(instances), 1].
    offset : int, default=45
        value to move all images by for centring. positive values shift all images to the right.
    """
    # paths to find and save the files
    PROJECT_ROOT = Path(find_dotenv()).parent
    output_path = PROJECT_ROOT / "results" / "fig_summary"
    json_path = PROJECT_ROOT / "results" / "gallery-descriptions"
    crop_path = PROJECT_ROOT / "cache" / "gallery" / "crop"
    crop_path.mkdir(parents=True, exist_ok=True)

    # generate the empty page
    doc = PDFMaker(
        width=pdf_specs["pdf_w"],
        height=pdf_specs["pdf_h"],
        resolution=pdf_specs["pdf_res"],
        margin=pdf_specs["pdf_margin"],
    )

    if rows_cols is None:
        rows_cols = [len(instances), 1]

    # get the position of each of the individual figs on the page
    coords, img_width, img_height = doc.get_page_layout_rows_cols(
        rows=rows_cols[0], cols=rows_cols[1], aspect_ratio=1
    )
    save_name = f"Gallery_Group-{page_idx:02d}.pdf"
    idx = 0

    for idx, instance in enumerate(instances):
        # get text and position
        config = PlotConfig(config_filename=json_path / f"Optic-Lobe_Gallery_{instance}.json")
        x_top, y_top = coords[idx][0] + offset, coords[idx][1]

        png_path = PROJECT_ROOT / "results" / "gallery" / config.directory / f"{config.basename}.png"
        png_crop_path = crop_path/ f"{config.basename}_crop.png"

        # if cropped image doesn't exist then make it
        if not os.path.exists(png_crop_path):
            img_raw = Image.open(png_path)
            img_crop = img_raw.crop((350, 200, 2350, 2500))
            # add white rect with gradient to edge
            rect_w = int(img_crop.size[0] / 10)
            rect_h = img_crop.size[1]
            rectangle_size = (rect_w, rect_h)
            pos_rect_x = img_crop.size[0] - rect_w
            pos_rect_y = 0
            rectangle_position = (pos_rect_x, pos_rect_y)
            overlay = Image.new("RGBA", img_crop.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for i in range(256):
                color = (255, 255, 255, i)
                new_rectangle_size = (
                    int(rectangle_size[0] * (1 - i / 255))
                  , rectangle_size[1]
                )
                new_rectangle_position = (
                    rectangle_position[0]
                      + int((rectangle_size[0]
                      - new_rectangle_size[0]))
                  , rectangle_position[1]
                )
                draw.rectangle(
                    [
                        new_rectangle_position,
                        (
                            new_rectangle_position[0] + new_rectangle_size[0],
                            new_rectangle_position[1] + new_rectangle_size[1],
                        ),
                    ],
                    fill=color,
                )
            result = Image.alpha_composite(img_crop.convert("RGBA"), overlay)
            result.save(png_crop_path)

        # add image to pdf
        img_coords = list(coords[idx])
        img_coords[0] += offset
        img_coords[2] += offset
        doc.add_image(png_crop_path, img_coords)

        text_dict = config.text_dict

        if f"{instance}" in nudge_dict.keys():
            nudge_str = f"{instance}"
            nudge_val = nudge_dict[nudge_str]
        else:
            nudge_val = 0

        # add text
        for key, t in text_dict.items():
            align = t["align"]
            plot_pos_x, plot_pos_y = t["pos"]
            paper_pos_x = (plot_pos_x * img_width) + x_top
            paper_pos_y = (((1 - plot_pos_y) * img_height) + y_top) - nudge_val

            doc.add_text(
                text=t["text"],
                position=[paper_pos_x, paper_pos_y],
                color=t["color"],
                align=align,
                font_size=pdf_specs["font_size"],
            )

    doc.save(filename=save_name, directory=output_path)


def check_for_imgs(plot_type:str):
    """
    Check that at least one png image needed for making the combined pdfs exists.

    This function assumes that having one image in the directory means that the image generation function was run successfully.

    Parameters
    ----------
    plot_type : str
        type of plot being generated. Only options are 'gallery' or 'group' plots.
    
    Returns
    -------
    rtn : bool
        false if images are missing, otherwise true
    """
    assert plot_type in ["group", "gallery"]\
      , f" 'plot_type' has the unexpected value {plot_type} - only 'group' or 'gallery' are allowed."
    rtn = True

    plot_dir = Path(find_dotenv()).parent / "results" / "gallery"

    if plot_type == 'group':
        dirs_to_check = [plot_dir / "vcn_group_plots", plot_dir / "vpn_group_plots"]
        search_string = 'Full-Brain*.png'
    else: # gallery
        dirs_to_check = [plot_dir / 'ol_gallery_plots']
        search_string = 'Optic-Lobe*.png'

    for directory in dirs_to_check:
        image_files = glob.glob(os.path.join(directory, search_string))
        if not image_files:
            warnings.warn(f"No images found in {directory}")
            rtn = False
    return rtn


def check_for_jsons(plot_type: str) -> bool:
    """
    Check that at least one json descriptions of images needed for making the combined pdfs exist.

    This function assumes that having one description means that the generators were run successfully.

    Parameters
    ----------
    plot_type : str
        type of plot being generated. Only options are 'gallery' or 'group' plots.

    Returns
    -------
    rtn : bool
        false if descriptions are missing, otherwise true
    """
    assert plot_type in ["group", "gallery"]\
      , f" 'plot_type' has the unexpected value {plot_type} - only 'group' or 'gallery' are allowed."
    rtn = True

    json_dir = Path(find_dotenv()).parent / 'results' / 'gallery-descriptions'
    
    if plot_type == 'group':
        search_string = 'Full-Brain_Group_*.json'
    else: # gallery
        search_string = 'Optic-Lobe_Gallery_*.json'

    json_files = glob.glob(os.path.join(json_dir, f"{search_string}"))
    if not json_files:
        warnings.warn(f"No jsons found in {json_dir}")
        rtn = False
    return rtn
