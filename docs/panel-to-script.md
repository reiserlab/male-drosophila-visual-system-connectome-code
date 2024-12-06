# Panels

Here we provide a quick connection between figure panels and the script that produces this panel. The scripts will generate the data and raw panels, final steps of the layout were done by hand.


## Main figures

| Panel       | script                                                                                                           |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| Fig 1a      |                                                                                                                  |
| Fig 1b      | (Blender manual rendering, details see [^1])                                                                     |
| Fig 1c      | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 1d      | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`                                                                   |
| Fig 1e      | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`, `summary_plots/process_fig1_summary_tables.ipynb`                |
| Fig 1f      | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`, `summary_plots/process_fig1_summary_tables.ipynb`                |
| Fig 1g      | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`                                                                   |
| Fig 2a      | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 2b(1)   | (no script)                                                                                                      |
| Fig 2b(2)   | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 2c      | `clustering/spatial_maps_clustering.ipynb`                                                                       |
| Fig 2d      | `clustering/clustering_examples.ipynb`                                                                           |
| Fig 2e      | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 2f      |                                                                                                                  |
| Fig 2g,h    | `clustering/spatial_maps_clustering.ipynb`                                                                       |
| Fig 2i      | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`                                                                   |
| Fig 3a      |                                                                                                                  |
| Fig 3b      |                                                                                                                  |
| Fig 3c      |                                                                                                                  |
| Fig 3d(1)   | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_Fig3d1.json), [^2]    |
| Fig 3d(2)   | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_Fig3d2.json), [^2]    |
| Fig 3d(3)   | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_Fig3d3.json), [^2]    |
| Fig 3d(4)   | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_Fig3d4.json), [^2]    |
| Fig 3e      | `eyemap/plot_all_syn.ipynb`                                                                                      |
| Fig 4a      |                                                                                                                  |
| Fig 4b      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 4c      | `nt/nt_plot.ipynb`, `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                           |
| Fig 4d      | (no script)                                                                                                      |
| Fig 4e      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 4f      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 4g      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 4h      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 4i      | `nt/nt_plot.ipynb`                                                                                               |
| Fig 5a      | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 5b      | `fig_summary/summary_figure_5b.ipynb`                                                                            |
| Fig 5c      | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 5d      | `column_features/plot_cov_hex_subplots.ipynb`                                                                    |
| Fig 5e(1)   | `column_features/plot_cov_compl_scatter_plots.ipynb`                                                             |
| Fig 5e(2)   | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 5f(1)   | `column_features/plot_cov_compl_scatter_plots.ipynb`                                                             |
| Fig 5f(2,3) | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                               |
| Fig 6       | `gallery_generation/generate_full_brain_groups_non_ol.ipynb`                                                     |
| Fig 7       | `gallery_generation/generate_full_brain_groups_non_ol.ipynb`                                                     |
| Fig 8       | manual                                                                                                           |
| Fig 8e,f    | `fig_summary/summary_figure_8ef.ipynb` [^3]                                                                      |
| Fig 9a      |                                                                                                                  |
| Fig 9b      |                                                                                                                  |

[^1]: for Fig 1b(1): import meshes, Camera anterior to the animal {location: {X:48572, Y: 50608: Z: 24976}, rotation: {X:270°, Y:0°, Z:-2°}}; Fig 1b(2): view from ventral to dorsal along dorsoventral axis; Fig 1b(3): lateral view
[^2]: Neuroglancer states are also stored in the `params` folder as json (you can copy&paste into Neuroglancer's "Edit JSON state").
[^3]: Synapse distribution by depth from the summary file is manually copied to Fig. 8e,f


## Extended Figures

| Panel         | script                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| ED 1a         |                                                                                                                |
| ED 1b         |                                                                                                                |
| ED 1c,d       |                                                                                                                |
| ED 1e,f       | `completeness/generate_completeness_plots.ipynb`                                                               |
| ED 1g,h       | `quality_control/ed_fig_1gh.ipynb`                                                                             |
| ED 2a,b       | `summary_plots/plot_fig1_ed_fig2_panels.ipynb`                                                                 |
| ED 2c         | `summary_plots/chiasm_connections.ipynb`                                                                       |
| ED 2d         | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_FigED2d.json, [^2]) |
| ED 2e         | (screenshot https://neuroglancer-demo.appspot.com/#!gs://flyem-user-links/short/Nern-et-al_FigED2e.json, [^2]) |
| ED 3a         |                                                                                                                |
| ED 3b         | manual                                                                                                         |
| ED 4a…e       | `clustering/spatial_maps_clustering.ipynb`                                                                     |
| ED 5a         |                                                                                                                |
| ED 5b         |                                                                                                                |
| ED 5c         |                                                                                                                |
| ED 5d         | `clustering/spatial_maps_clustering.ipynb`, `clustering/clustering_examples.ipynb`                             |
| ED 5e         | `gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb`                                             |
| ED 5f         |                                                                                                                |
| ED 6a         |                                                                                                                |
| ED 6b         |                                                                                                                |
| ED 6c,d       | `snakemake generate_optic_lobe_custom` [^4]                                                                    |
| ED 7a,b       | `eyemap/plot_pin_assignment.ipynb`                                                                             |
| ED 8a         |                                                                                                                |
| ED 8b         |                                                                                                                |
| ED 8c         |                                                                                                                |
| ED 9          | (manual, website screenshot)                                                                                   |
| ED 10a…d      | `column_features/plot_cov_compl_scatter_plots.ipynb`                                                           |
| ED 10e        | `column_features/make_coverage_factor_hist.ipynb`                                                              |
| ED 11         | `gallery_generation/generate_full_brain_types_dorsal_rim.ipynb`                                                |
| ED 12         | `gallery_generation/generate_full_brain_types_ame.ipynb`                                                       |
| ED 13a(2,4,6) |                                                                                                                |
| ED 14a        |                                                                                                                |
| ED 14b,c      |                                                                                                                |


[^4]: The description JSON files are in `src/gallery_generation`, the `snakemake` target copies the files to `results/gallery-description`, so it gets rendered with all the other optic lobe descriptions (see Methods section).

## Extended Tables

The extended data table is genrerate from this script:

| Table         | script                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| ET 1          |`tables/get_table1_celltypelist.ipynb`                                                                          |


## Supplemental Figures

| Panel         | script                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| Sup Fig 1     |`gallery_generation/generate_optic_lobe_stars.ipynb`                                                            |

## Supplemental Videos

| Video         | script                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| Sup Video 1   | `gallery_generation/generate_full_brain_stars_flipbook.ipynb`                                                  |
| Sup Video 2   |                                                                                                                |
| Sup Video 3   | `movies/generate_tiling_movies.ipynb`                                                                          |



## Supplementary Tables

| Table         | script                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| Sup Table 1   |                                                                                                                |
| Sup Table 2   |                                                                                                                |
| Sup Table 3   | `eyemap/assign_columnar_types_to_hex.ipynb`                                                                    |
| Sup Table 4   |                                                                                                                |
| Sup Table 5   |                                                                                                                |
| Sup Table 6   |                                                                                                                |
| Sup Table 7   |                                                                                                                |



## Additional Resources

Videos on YouTube are produces via this script:

The Website is generated via this script: `html_pages/generate_html-pages_from_scratch.ipynb`
