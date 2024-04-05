# Guidelines for standardized plotting

Below are suggestions for how to use the formatting guidelines when making plots for the 'Optic Lobe Connectome' project. Based on the [graphing library plotly](https://plotly.com/python/) which is used throughout the project, formatting rules can be provided in code, more specifically in python dictionaries. It was decided that standardized dictionaries containing common formatting parameters would be used as input variables for the functions generating all plots to achieve consistent formatting across the final figures.

## Default values 

These three dictionaries include: `style`, `sizing` and `plot_specs`

The reason for three separate dictionaries is as follows:

### Suggested values for `style`

The parameters in `style` are completely fixed across all plots - e.g. font type and color. In theory, once basic styling parameters are agreed upon, this dictionary could be defined at the beginning of each script that generates plots and not changed. 

```python
style = {
    'font_type': 'arial'
  , 'markerlinecolor': 'black'
  , 'linecolor': 'black'
}
```

### Suggested values for `sizing`

The parameters in `sizing` will determine the size of both the plot and a number of features such as the font size and line widths that might need to change according to the overall size of the plot - e.g. tick length might need to be altered if a plot is very small or very big. Again, these parameters should only be to do with sizing so that once a specified size is agreed upon then values would not need to change. 

```python
sizing = {
    'fig_width': 75  # units = mm, max 180
  , 'fig_height': 75 # units = mm, max 170
  , 'fig_margin': 0
  , 'fsize_ticks_pt': 6
  , 'fsize_title_pt': 7
  , 'markersize': 10
  , 'ticklen': 3.5
  , 'tickwidth': 1
  , 'axislinewidth': 1.2
  , 'markerlinewidth': 1
  , 'cbar_len': 1
  , 'cbar_thickness': 7
}
```

### Suggested values for `plot_specs`

The parameters in `plot_specs` are the most plot specific and may contain values that will change the data being presented or will depend on the type of data that is used to generate each plot - e.g. whether to use a log scale for the x or y axis. This dictionary will likely change with each plot and are the least universal. 

```python
plot_specs = {'log_x':True,
    'log_y':True,
    'export_type':'svg',
    'save_path': f'{PROJECT_ROOT}/results/cov_compl/scatterplots/'}
```

## Explanation of parameters

The following paragraphs explain the key and values from the suggestions in more detail.

### Parameters of `style`

__`'font_type': 'arial'`__: The default font used should be 'arial'. 

`'markerlinecolor': 'black'`: Marker outline colours might vary depending on the plot. In some circumstances the markerline might want to be transparent in which case set `markerlinecolor` to `rgba(0, 0, 0, 0)`. 

`'linecolor': 'black'`: All lines should set to black.


### Parameters of `sizing`

#### Figure size

`fig_width` and `fig_height`: The parameters '`fig_width`' and '`fig_height`' allow the user to set the physical size of the figure they wish to generate. The units used in the dictionaries should be in 'mm' and the code below will convert the desired size in 'mm' into the size in pixels used by the plotting function.  

```python
if export_type in ['svg', 'pdf']:
    pixelsperinch = 72 #96 for png, 72 for svg and pdf
else: # png or other
    pixelsperinch = 96
pixelspermm = pixelsperinch / 25.4
fig_w = (sizing['fig_width'] - sizing['fig_margin']) * pixelspermm
fig_h = (sizing['fig_height'] - sizing['fig_margin']) * pixelspermm
```

Note that the value of `pixelsperinch` changes depending on whether the `export_type` is svg, pdf or another format. These values of `fig_w` and `fig_h` can be used to set the size of the figure by updating the layout:

```python
fig.update_layout(
    autosize=False
  , width=w
  , height=h
)
```

and / or when saving the figure:

```python
pio.write_image(
    fig
  , f"{save_path}{roi_str}_{xval}_versus_{yval}_{fsize_title_pt}pt_w{fig_width}_h{fig_height}.{export_type}"
  , width=fig_w
  , height=fig_h
)
```

If you only specify the width and height in `write_image` then it will show at a default size when using `fig.show()`, but it will save it at the required size.


#### Font size

`fsize_ticks_pt` and `fsize_title_pt`: Font size for both tick labels and axis titles can be specified in the `sizing` dictionary in 'pt' units. The following code within the function will convert the size in pt into the size in pixels used by the plotting functions.

```python
if export_type in ['svg', 'pdf']:
    pixelsperinch = 72 #96 for png, 72 for svg
else: # png or other
    pixelsperinch = 96
fsize_ticks_px = sizing['fsize_ticks_pt'] * (1/72) * pixelsperinch
fsize_title_px = sizing['fsize_title_pt'] * (1/72) * pixelsperinch
```

As the font size is specified as a separate parameter in the dictionary, the size of the font will not change when you change the overall size of the plot by altering `fig_width` and `fig_height`.

These parameters can be used within the plotting function `plotly.graph_objects.Scatter.marker.colorbar` as so:

```python
…
    "tickfont": {"size": fsize_ticks_px, "family":font_type, "color": linecolor}
  , "title": {"font":{"family":font_type, "size":fsize_title_px,"color": linecolor}}
…
```

#### Margins

`fig_margin`: The size of the margin to be used in the figure can be set in pixels by setting `fig_margin` to a float or integer value and then applied as below:

```python
fig.update_layout(
    autosize=False
  , height=fig_h
  , width=fig_w
  , margin={
        'l': fig_margin
      , 'r': fig_margin
      , 'b': fig_margin
      , 't': fig_margin
      , 'pad':0
    }
  , paper_bgcolor='rgba(255,255,255,0)'
  , plot_bgcolor='rgba(255,255,255,0)'
)
```

or you can avoid setting `fig_margin` in the dictionary and instead set the margin values to be proportional to the figure's overall size. 

```python
fig.update_layout(
    autosize=False
  , height=h
  , width=w
  , margin={
        'l': fig_w // 15
      , 'r': fig_w // 4
      , 'b': fig_h // 10
      , 't': fig_h // 10
      , 'pad': fig_w // 20
    }
  , paper_bgcolor='rgba(255,255,255,0)'
  , plot_bgcolor='rgba(255,255,255,0)'
)
```

#### Markersize

`markersize`: The value used for markersize will depend on the size of the plot. Similar to setting `margin` size, this value could be set as a value proportional to `w` or `h` within the plotting function. 


#### Tick lenght and width

`ticklen` and `tickwidth`: Tick length and tick width might also change depending on the size of the plot. Similar to setting `margin` size, this value could be set as a value proportional to `fig_w` or `fig_h` within the plotting function. 


#### Line widths

`axislinewidth` and `markerlinewidth`: These parameters set the axis and marker line width. This again might need to change depending on the size of the plot. 

#### Colorbar specs

`cbar_len` and `cbar_thickness`: These parameters can set the length `cbar_len` and thickness `cbar_thickness` of the colorbar if there is one.



### Parameters of `plot_specs`


#### Axis scale

`'log_x' : True` and `'log_y' : True`: These parameters can determine whether to use a log scale for the x or y axis. 


#### Figure types

`export_type' : 'svg'`: Use a string to specify the export type. Interactive plots can be saved when setting `export_type` to `'html'`

#### Files path

`'save_path': f'{PROJECT_ROOT}/results/cov_compl/scatterplots/'`: A string can also be used to specify the path to save the resulting figure.


## General plot settings

### Lines, axes and grid lines

For many plots we want to remove the presentation of any axis lines and grid lines. To remove these formatting features the following code can be used:

```python
fig.update_xaxes(showgrid=False, showline=False, visible=False)
fig.update_yaxes(showgrid=False, showline=False, visible=False)
```


### Hexagonal scatter plots such as eyemaps

Hexagonal 'eyemap' plots are scatterplots with hexagonal markers. To generate these plots within the plotly.graph_objects.Scatter function `mode` must be set to `"markers"` and `marker_symbol` must be set to `15`.

For example: 

```python
symbol_number = 15

fig.add_trace(
    go.Scatter(
        x=hex1_vals
      , y=hex2_vals
      , mode="markers"
      , marker_symbol=symbol_number
      , marker={
            "size": markersize
          , "color": markerlinecolor
          , "line": {
                "width": markerlinewidth
              , "color": "lightgrey"
            }
        }
      , showlegend=False
    )
)
```
