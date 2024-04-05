import warnings
import os

from functools import partial

import datetime
from datetime import datetime
from pathlib import Path

import plotly.io as pio

import numpy as np
import pandas as pd
import jinja2

from neuprint import NeuronCriteria as NC, fetch_neurons

from utils.helper import num_expand
from utils.ol_types import OLTypes

from utils.ol_color import OL_COLOR
from utils.ol_instance import OLInstance
from utils.helper import slugify

from html_pages.make_spatial_coverage_plots_for_webpages import\
    plot_synapses_per_column, plot_cells_per_column
from html_pages.plotting_to_html import get_dynamic_plot
from html_pages.webpage_functions import get_meta_data

from queries.webpage_queries import\
    get_layer_synapses, get_roi_synapses, get_io_table, consensus_nt_for_instance



def shorten_nt_name(
    nt_name:str
) -> str:
    """
    Convert full neurotransmitter name to its abbreviation.
    Return the shortened name if in mapping, else return original

    Parameters
    ----------
    nt_name : str
        full neurotransmitter name

    Returns
    -------
    abbr : str
        abbreviated neurotransmitter name : str

    """
    nt_mapping = {
        'acetylcholine': 'ACh',
        'gaba': 'GABA',
        'serotonin': '5HT',
        'glutamate': 'Glu',
        'dopamine': 'Dop',
        'histamine': 'His',
        'octopamine': 'OA'
    }
    # Check if nt_name is None or not a string
    if nt_name is None or not isinstance(nt_name, str):
        return nt_name  # 'Unknown' or return nt_name if you want to keep the None

    return nt_mapping.get(nt_name.lower(), nt_name)


def get_layer_stats(
    instance:str
  , roi_str:str
  , replace_zero:bool=True
) -> pd.DataFrame:
    """
    Get layer information for an instance and ROI. Only get the total stats if there are no layer.
    A "special" ROI is 'non-OL', which includes all primary ROIs except the optic lobe ones.

    Parameters
    ----------
    instance : str
        Name of a cell instance, e.g. 'TmY5a_R'
    roi_str : str
        A database ROI. If 'non-OL', then get the 'central brain' statistics. For ME(R), LO(R),
        and LOP(R),  get the layer statistics in addition to the total.
    replace_zero : bool, default=True
        If true, replace non-existing values (NA, None…) and 0 with '-'

    Returns
    -------
    df : pd.DataFrame
        Wide data table. Rows represent the median count of pre and post synapses per cell,
        columns represent the named layers (for ME, LO, LOP), and the total for the ROI.
        pre : int | str
            Median count of pre-synaptic sites per cell.
        post: int | str
            Median count of post-synaptic sites per cell.
    """

    rt = pd.DataFrame()
    if roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
        s_layer = get_layer_synapses(instance=instance, roi_str=roi_str)
        s_layer = s_layer.pivot(index='syn_type', columns='named_layer', values='mean_count')
        s_layer = s_layer.reindex(sorted(s_layer.columns, key=num_expand), axis=1)
        rt = pd.concat([rt, s_layer], axis=1)
    s_roi = get_roi_synapses(instance=instance, roi_str=roi_str)
    if roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
        s_roi['named_layer'] = 'Total'
    elif roi_str in ['non-OL']:
        s_roi['named_layer'] = 'central brain'
    else:
        s_roi['named_layer'] = roi_str[:-3]
    s_roi = s_roi.pivot(index='syn_type', columns='named_layer', values='mean_count')
    rt = pd.concat([rt, s_roi], axis=1)
    rt = rt.rename(index={'pre':'Pre', 'post': 'Post'})
    rt = rt.sort_index(ascending=True)
    rt.index.name = None
    rt.columns.name = None
    if replace_zero:
        rt = rt.replace(0, np.nan).fillna('-')
    return rt


def convert_pkl_to_html_with_layers(
    oli:OLInstance
  , valid_neuron_names:set[str]
  , template:str
  , input_path_coverage:Path
  , output_path:Path
) -> None:
    """
    Create an html file for the given cell type from the pkl data file of its synaptic connections

    Parameters
    ----------
    oli : OLInstance
        Instance for which to generate the HTML output.
    valid_neuron_names : Set[str]
        Set of valid neuron names that have a webpage of their own.
    template : str
        Name of the jinja template.
    input_path_coverage: Path
        Location of the coverage and completeness images.
    output_path : Path
        Location where the HTML file is stored.

    """
    assert os.environ.get('NEUPRINT_BASE_URL'),\
        "Please set the `NEUPRINT_BASE_URL` variable in your environment."

    assert isinstance(oli, OLInstance), "interface changed, please use OLInstance"

    # Setting the ouputs by which html pages can be saved
    output_cell_filename = f"{oli.slug}.html"
    output_filename = output_path.joinpath(output_cell_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    if output_filename.is_file(): # don't overwrite existing html files
        return

    success = True
    # Jinja template
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(''))
    template = environment.get_template(template)

    olt = OLTypes()

    # Get number of cells per cell type
    neurons_df, _ = fetch_neurons(NC(instance=oli.name))
    # Check whether neurons exist with this instance (_R)
    if neurons_df.empty:
        print(f"No neurons found for instance: {oli.name}. Continuing to next neuron.")
        return None

    # TITLE information
    # Add neurotransmitter prediction to the title from database

    nt_prediction = shorten_nt_name(consensus_nt_for_instance(instance=oli.name))

    in_vals = get_io_table(
        instance=oli.name
      , direction='input'
      , connection_cutoff=None
      , per_cell_cutoff=1.0
    )
    in_vals['consensusNT'] = in_vals['consensusNT'].apply(shorten_nt_name)
    out_vals = get_io_table(
        instance=oli.name
      , direction='output'
      , connection_cutoff=None
      , per_cell_cutoff=1.0
    )
    out_vals['consensusNT'] = out_vals['consensusNT'].apply(shorten_nt_name)

    # Check if invals or outvals empty
    if in_vals.empty or out_vals.empty:
        warnings.warn("Missing input or output tables.")
    # Flag to the jinja-template if tables empty
    in_vals_empty = in_vals.empty
    out_vals_empty = out_vals.empty


    # Rename column headers for connectivity table
    in_vals.rename(
        columns={
            'instance': 'instance'
          , 'consensusNT': 'NT'
          , 'total_connections': 'total connections'
          , 'cell_connections': f'connections /#{oli.name_html}'
          , 'percentage': '%'
          , 'perc_cum': '% cumu.'
        }
      , inplace=True
    )

    out_vals.rename(
        columns={
            'instance': 'instance'
          , 'consensusNT': 'NT'
          , 'total_connections': 'total connections'
          , 'cell_connections': f'connections /#{oli.name_html}'
          , 'percentage': '%'
          , 'perc_cum': '% cumu.'
        }
      , inplace=True
    )

    # Define function for the colored links
    def format_func(
        x
      , valid_neuron_names
    ) -> str:

        # Determine the color based on main_group
        t_oli = OLInstance(x)
        main_group = olt.get_main_group(t_oli.type)
        class_txt = f" txt_{main_group}" if main_group else ""

        if x in valid_neuron_names:
            # Use the determined color for the link
            return f'<a href="{t_oli.slug}.html" class="connectivity-link{class_txt}">{t_oli.name_html}</a>'
        # Non-functional links don't need specific group-based coloring; handling remains the same
        return f'<span class="non-functional-link">{t_oli.name_html}</span>'

    # Create a partial function with the filename_to_main_group and other required mappings
    # pre-specified
    format_func_with_context = partial(
        format_func
      , valid_neuron_names=valid_neuron_names
    )

    # Now specify this partial function for the 'type' column in your format dictionary
    format_dict = {
        'instance': format_func_with_context
      , 'total connections': '{:,.0f}'
      , f'connections /#{oli.name_html}': '{:,.1f}'
      , '%': '{:,.1%}'
      , '% cumu.': '{:,.1%}'
    }

    # Apply the formatting as before
    in_styled = in_vals.style.format(format_dict, na_rep="")
    out_styled = out_vals.style.format(format_dict, na_rep="")

    in_styled.set_table_styles({
        "type_pre":[{"selector": "td", "props": [("font-weight", "bold")]}]
      , "type_post":[{"selector": "td", "props": [("font-weight", "bold")]}]
    })
    in_styled.highlight_null(props="background-color:#FFFFFF;")
    out_styled.set_table_styles({
        "type_pre":[{"selector": "td", "props": [("font-weight", "bold")]}]
      , "type_post":[{"selector": "td", "props": [("font-weight", "bold")]}]
    })
    out_styled.highlight_null(props="background-color:#FFFFFF;")

    n_cell_types = f"n={neurons_df['bodyId'].count()} cell(s)"

    # Colors to the connectivity table
    in_styled.bar(color="#fed76a", subset=['% cumu.'], height=95)
    in_styled.bar(color="#fee395", subset=['%'], vmax=1, height=95)
    out_styled.bar(color="#69d0e4", subset=['% cumu.'], height=95)
    out_styled.bar(color="#9bdfed", subset=['%'], vmax=1, height=95)

    # Connectivity table to html
    in_styled_html = in_styled.to_html(index=False)
    out_styled_html = out_styled.to_html(index=False)

    ## LAYER STATISTICS tables
    # Function to turn hex color to RGBA
    def hex_to_rgba(hex_color, alpha=1.0):
        # Assuming hex_color is in the format "#RRGGBB"
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        rgb_n = ', '.join(str(int(hex_color[i:i + lv // 3], 16)) for i in range(0, lv, lv // 3))
        return f"rgba({rgb_n}, {alpha})"


    def style_dataframe(df, precision=1, remove_index=False, neuropil=None):
    # Define a mapping from neuropil names to their corresponding colors using hex values
        color_mapping = {
            'ME': OL_COLOR.OL_NEUROPIL.hex[0]
          , 'LO': OL_COLOR.OL_NEUROPIL.hex[1]
          , 'LOP': OL_COLOR.OL_NEUROPIL.hex[2]
          , 'LA': '#8EBD78'
          , 'AME': '#598FCF'
          , 'CB': '#000000'
        }
        # Get the background color for the current neuropil
        # Default to grey if neuropil not found
        bg_color_rgba = hex_to_rgba(color_mapping.get(neuropil, '#FFFDFD'), 0.3)

        styles = [{         
            "selector": "th"
          , "props": [ ("text-align", "center"), ("background-color", bg_color_rgba)]
        }, {
            "selector": "td"
          , "props": [ ("text-align", "center"), ("background-color", bg_color_rgba)]
        }]

        if remove_index:
            df = df.reset_index(drop=True)

        return df.style.format(precision=precision)\
            .set_table_styles(styles)\
            .to_html(index=False, escape=False)


    # Apply styling and other operations to each DataFrame
    lay_stats_la_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='LA(R)')
      , precision=1
      , neuropil='LA'
    )
    lay_stats_me_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='ME(R)')
      , precision=1
      , remove_index=True
      , neuropil='ME'
    )
    lay_stats_lo_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='LO(R)')
      , precision=1
      , neuropil='LO'
    )
    lay_stats_lop_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='LOP(R)')
      , precision=1
      , neuropil='LOP'
    )
    lay_stats_ame_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='AME(R)')
      , precision=1
      , remove_index=True
      , neuropil='AME'
    )
    lay_stats_cb_html = style_dataframe(
        get_layer_stats(instance=oli.name, roi_str='non-OL')
      , precision=1
      , remove_index=True
      , neuropil='CB'
    )

    image_tags = []

    fig_syn = plot_synapses_per_column(oli.name)
    fig_syn_fn = output_path / 'img' / f"{oli.slug}_syn.png"
    fig_syn_fn.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(
        fig=fig_syn
      , file=fig_syn_fn
      , width=fig_syn['layout']['width']
      , height=fig_syn['layout']['height']
    )
    image_tags.append(
        f'''
        <figure>
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 2934 793" preserveAspectRatio="xMinYMin meet">
            <image width="2834" height="793" xlink:href="img/{fig_syn_fn.name}" />
            <g class="hover_group" opacity="0.0">
                <a xlink:href="scatterplots/scatterplots-LOP.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="2225" cy="395" r="375" />
                </a>
            </g>
            <g class="hover_group" opacity="0.0">
                <a xlink:href="scatterplots/scatterplots-LO.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="1325" cy="395" r="375" />
                </a>
            </g>
            <g class="hover_group" opacity="0.0">
                <a xlink:href="scatterplots/scatterplots-ME.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="395" cy="395" r="375" />
                </a>
            </g>
            </svg>
        </figure>
        '''
    )

    fig_cell = plot_cells_per_column(oli.name)
    fig_cell_fn = output_path / 'img' / f"{oli.slug}_cell.png"
    pio.write_image(
        fig=fig_cell
      , file=fig_cell_fn
      , width=fig_cell['layout']['width']
      , height=fig_syn['layout']['height']
    )
    image_tags.append(
        f'''
        <figure>
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 2934 793" preserveAspectRatio="xMinYMin meet">
            <image width="2834" height="793" xlink:href="img/{fig_cell_fn.name}" />
            <g class="hover_group" opacity="0">
                <a xlink:href="scatterplots/scatterplots-LOP.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="2225" cy="395" r="375" />
                </a>
            </g>
            <g class="hover_group" opacity="0">
                <a xlink:href="scatterplots/scatterplots-LO.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="1325" cy="395" r="375" />
                </a>
            </g>
            <g class="hover_group" opacity="0">
                <a xlink:href="scatterplots/scatterplots-ME.html">
                    <circle opacity="0.2" style="fill:#ffffff" cx="395" cy="395" r="375" />
                </a>
            </g>
        </figure>
        '''
    )

    # COVERAGE CARD DATA
    def load_and_filter_coverage_data(coverage_image_dir, instance_type):
        neuropils = ['ME', 'LO', 'LOP']
        expected_keys = [
            'cell_type', 'n_pre', 'n_post', 'n_output_conn', 'coverage_factor_trim'
          , 'col_completeness', 'area_completeness', 'cell_size_cols'
        ]

        def generate_zero_data(keys):
            return {key: 0 for key in keys}

        def format_data(data):
            # Format 'coverage_factor' to one decimal place
            data['coverage_factor_trim'] = \
                f"{data['coverage_factor_trim']:.1f}" if data['coverage_factor_trim'] else "0"
            # Conditionally format 'col_completeness' and 'area_completeness'
            for key in ['col_completeness', 'area_completeness']:
                value = data.get(key, 0)
                if value == 0:
                    data[key] = "0"  # Keep as string for consistency
                else:
                    data[key] = f"{value:.2f}"
            # Format 'cell_size_cols' as integer
            data['cell_size_cols'] = f"{int(data.get('cell_size_cols', 0))}"
            return data

        def load_and_filter(neuropil):
            file_path = coverage_image_dir / f"{slugify(instance_type)}.pickle"
            try:
                with file_path.open('rb') as metric_fh:
                    coverage_data = pd.read_pickle(metric_fh)
                # Filter for neuropil
                filtered_data = coverage_data[coverage_data['roi'] == f"{neuropil}(R)"]
                # If no matching data, return zero data
                if filtered_data.empty:
                    return format_data(generate_zero_data(expected_keys))
                # Convert the first matching row to a dictionary
                data_dict = filtered_data.iloc[0].to_dict()
                # Select only the expected keys, handling missing keys by setting them to zero
                result = {key: data_dict.get(key, 0) for key in expected_keys}
                return format_data(result)
            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping {neuropil}.")
                return format_data(generate_zero_data(expected_keys))

        # Load, filter, and return coverage data for each neuropil
        coverage_data = {neuropil: load_and_filter(neuropil) for neuropil in neuropils}
        return coverage_data

    coverage_data = load_and_filter_coverage_data(input_path_coverage, oli.name)

    # Make 'filtered_card_data' dictionary where each key ('ME', 'LO', 'LOP') maps to another
    # dictionary. This inner dictionary contains both the human-readable name ('name') and the
    # actual data ('data').
    neuropil_names = {
        'ME': 'Medulla'
      , 'LO': 'Lobula'
      , 'LOP': 'Lobula Plate'
    }

    # Prepare the data for the template
    filtered_card_data = {
        neuropil_key: {
            'name': neuropil_names[neuropil_key],  # Human-readable name
            'data': coverage_data[neuropil_key]    # Data for this neuropil
        }
        for neuropil_key in coverage_data
    }
    # Pass color mapping or neuropils to the card titles
    color_mapping_regions = {
        'Medulla': OL_COLOR.OL_NEUROPIL.hex[0],
        'Lobula': OL_COLOR.OL_NEUROPIL.hex[1],
        'Lobula Plate': OL_COLOR.OL_NEUROPIL.hex[2],
    }

    # NEUPRINT LINK
    # Base URL (up to the point before 'cellType')
    dynamic_param = f"&qr[0][pm][neuron_name]={oli.type}"
    neuprint_url = f"https://{os.environ.get('NEUPRINT_SERVER_URL')}/{os.environ.get('NEUPRINT_BASE_URL')}{dynamic_param}"

    fig_3d = get_dynamic_plot(oli.name, resample_precision=oli.resample_precision)
    fig_3d_fn = output_path / 'img' / 'dynamic' / f'{oli.slug}.html'
    fig_3d_fn.parent.mkdir(parents=True, exist_ok=True)
    plotly_config ={'displayModeBar': False}
    pio.write_html(
        fig_3d
      , fig_3d_fn
      , config=plotly_config
      , include_plotlyjs='directory'
      , full_html=True
      , auto_open=False
    )
    three_d_image_script = f"img/dynamic/{oli.slug}.html"

    # META TO FOOTER
    meta, last_database_edit, formatted_date = get_meta_data()

    # RENDER TEMPLATE
    # Render the template with the data
    rendered_template = template.render(
        oli=oli,
        n_cell_types=n_cell_types,
        nt_prediction = nt_prediction,
        color_mapping_regions = color_mapping_regions,
        lay_stats_ME_html=lay_stats_me_html,
        lay_stats_LO_html=lay_stats_lo_html,
        lay_stats_LOP_html=lay_stats_lop_html,
        lay_stats_LA_html=lay_stats_la_html,
        lay_stats_AME_html=lay_stats_ame_html,
        lay_stats_CB_html=lay_stats_cb_html,
        in_styled_html=in_styled_html,
        out_styled_html=out_styled_html,
        in_vals_empty=in_vals_empty,
        out_vals_empty=out_vals_empty,
        filtered_card_data=filtered_card_data,
        image_tags=image_tags,
        threeD_image_script=three_d_image_script,
        neuprint_deep_link=neuprint_url,
        meta=meta,
        lastDataBaseEdit=last_database_edit,
        formattedDate=formatted_date
    )

    # Save the rendered template to an HTML file
    with open(output_filename, "w", encoding="utf-8") as _file:
        _file.write(rendered_template)

    if success:
        print(f"HTML page generated successfully for {oli.name}.")
    return success
