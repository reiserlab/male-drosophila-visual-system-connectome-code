# Configuration

There are some variables you need to configure via the `.env` file. This includes some credentials for neuprint and other services.

To get started, please copy the `.env-sample` into a `.env` file.

## jupyter

Most analysis is in jupyter notebooks and we use plotly as our primary plotting engine. So please set your variable `NAVIS_JUPYTER_PLOT3D_BACKEND` to `plotly` by adding the following line to your `.env` file.

```sh
NAVIS_JUPYTER_PLOT3D_BACKEND=plotly
```

## Neuprint

We require three neuprint related variables to be configured: The server URL, the data set name, and the credentials. The first two will be published together with the data set, for the latter one you will need to request access.

```sh
NEUPRINT_SERVER_URL='<URL>'
NEUPRINT_DATASET_NAME='<DATASET_NAME>'
NEUPRINT_APPLICATION_CREDENTIALS='<TOKEN>'
```

## Data sources

Larger mesh files are pulled from a number of different sources. Please set the following two variables. Examples for the `<GS-URL>`s should be in the `.env-sample` file.

```sh
# The source for OL(R), OL(L) and CB
SHELL_SOURCE='<GS-URL>'

# The source for full_brain
FULL_SHELL_SOURCE='<GS-URL'
```


## Movie and Gallery generation

To generate the movies and galleries, two additional software packages are required. Please follow the description in <movies-getting-started.md> and don't forget to set the path for the blender executable in `BLENDER_PATH` and the path where you installed neuVid in `NEUVID_PATH`.

```sh
BLENDER_PATH='/usr/bin/blender'
NEUVID_PATH='/home/loeschef/srcs/neuVid'
```