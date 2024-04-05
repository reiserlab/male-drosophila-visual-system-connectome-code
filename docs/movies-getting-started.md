# Getting started with movies

The definition of movies is provided in `.json` files and rendered through the [neuVid](https://github.com/connectome-neuprint/neuVid) system. This documentation builds on top of the neuVid documentation and only specifies how to run the movie makeing in our system.

## Prerequisites

All JSON files are expected to be inside the src folder for now and I base the documentation on an example in `src/movies/Dm15.json`.

You will need to install [Blender](https://www.blender.org/) and [neuVid](https://github.com/connectome-neuprint/neuVid). 

Follow the documentation on how to install Blender. 

Clone the neuVid repository to a path of your liking (for inspiration look at [Getting started with Git](git-getting-started.md)).

Make sure you updated [your dependencies](python-getting-started.md#dependency-management) recently.

## Configuration

You need to add two variables to your `.env` file:

`NEUVID_PATH` should contain the base path where you cloned neuVid to. For example, this could be `/Users/floesche/Documents/GitHub/neuVid`. You can test if you have the right directoy by running the command `cat /Users/floesche/Documents/GitHub/neuVid/VERSION` and you should see the version of neuVid you installed. Currently you need at least neuVid-1.34.0 for everything to work well.

You will also need to provide the path to the Blender executable in the `BLENDER_PATH` variable. This could, for example be `/usr/bin/blender` (on Linux) or `/Applications/Blender.app/Contents/MacOS/blender`. You can see if you specified the correct path by calling the following command in the terminal `/Applications/Blender.app/Contents/MacOS/blender --version` and you should see which version of blender you have installed.

After this step, your `.env` file has two additional lines:

```sh
BLENDER_PATH='/usr/bin/blender'
NEUVID_PATH='/home/loeschef/srcs/neuVid'
```

You will also need to make sure, that the software blender can be started via `blender` in your terminal (needed for some `trimesh` transformations). This should be the case for Linux and Windows once you installed the software. On a Mac you might need to add an alias to your shell. You could do this by running `echo 'alias blender="/Applications/Blender.app/Contents/MacOS/Blender"' >> $HOME/.zshrc` (this assumes, that blender is installed at `/Applications/Blender.app/Contents/MacOS/Blender` and that you are using zsh as your shell).

## Principle 

To render the movie file from the JSON description, 5 steps are needed:

1. Download all meshes to the local computer and include them in a blender file
2. add animations to a blender file
3. render the individual frames of the movie
4. add the labels
5. combine frames into a movie

The individual steps are documented in neuVid. For the movies in ol-connectome we created helpers that make each of the steps easier. We implemented a workflow in Snakemake that allows you to specify your required output and Snakemake figures out what steps are (still) necessary.

## Example

In this example I want to create a preview of the movie defined in `src/movies/Dm15.json`. All you need to run is: `snakemake --cores 10 results/movies/Dm15-preview.avi`. This should start the process. After some processing time, you should end up with preview file. 

The number `10` after `--cores` describes how many of your processing cores you want to dedicate to the rendering. The lower the number, the more responsive your computer will be during the process, but the longer it will take.

If you previously interrupted the process, you don't need to start at the beginning, but you might need to say that the incomplete tasks need to rerun. If you see an error message about incompleteness, you could try to rerun the process by specifying `snakemake --rerun-incomplete --cores 10 results/movies/Dm15-preview.avi`. The current setting for preview is 384×216 pixels, with labels and all frames.

You can render movies at an arbitrary resolution by saying you want `snakemake --cores 5 results/movies/Dm15_192x108.avi` for a very low resolution movie.

All intermediate files will end up in the `cache/blender/` folder. There you will find the `Dm15.blend`, `Dm15Anim.blend`, and the individually rendered frames.

The same process will be used on the cluster, so if the preview runs on your machine, the full movie will render on the cluster.

There are special commands to `snakemake allpreviews` and `snakemake allmovies`. This will potentially take a long time, so don't try this at home.