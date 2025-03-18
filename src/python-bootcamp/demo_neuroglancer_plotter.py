# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo Neuroglancer Plotter
#
# Early plotter (`neuroglancer_plotter`) that uses the neuroglancer web application to make 
# essentially high resolution screenshots through the ptyhon API. The interface is similar to 
# `group_plotter`, produces better quality images, but was not used for the paper.

# %%
from pathlib import Path
from dotenv import find_dotenv

from utils import olc_client

c = olc_client.connect(verbose=True)
PROJECT_ROOT = Path(find_dotenv()).parent
print(f"Project root directory: {PROJECT_ROOT}")

# %%
from utils.neuroglancer_plotter import url_plotter, image_saver
#from IPython.display import Image

# %%
# url_plotter takes a neuroglancer URL and creates an image for it
rt = url_plotter(
    'https://clio-ng.janelia.org/#!gs://flyem-user-links/short/optic-lobe-columns-v0.json'
                    # any neuroglancer URL should work here.

  ## Optional parameters:
  # , wait_sec=50   # I can't yet identify if the image has rendered completely. You have to specify
                    #   how long you want to wait. Default is 30 seconds, which works good enough 
                    #   for the default resolution in many cases.
  # , size=(3000,2000) # size of the screenshot in px. I think the maximum is around 4096×4096. The
                       #   bigger the image, the longer you need to set the `wait_sec`. The default
                       #   is 
  # , background_color="#336699"  # background color for the 3D view in HTML hex. The default is
                                  #   white ("#FFFFFF")
  # , set_3d=True  # Usually this function just plots the exact content of the neuroglancer URL.
                   #   If `set_3d` is true, it only plots the 3D view. This is more of a test if
                   #   this could work than anything else.
  ) # the function returns a PIL.Image, which can be viewed:

rt

# %%
data_path = PROJECT_ROOT / "results" / "screenshots"

# this saves the image
image_saver(rt # rt is the PIL image from `url_plotter`
  , 'test_image' # this is the file name for the PNG file
  , data_path    # this is the path where the file will be saved
  ## Optional parameters:
  # , replace=True # if true, overwrites the existing file. If false, it will attach a timestamp.
)   

# %%
## These functions are probably most useful if you use them with a collection of URLs. For example,
#    one could have a CSV file with file name, URLs, and wait times in separate columns that will
#    create this `screenshot` structure. Then the following loop will create all the screenshots
#    at any time, for example the day before publication, based on the latest data.

screenshots = {
   'optic-lobe-columns' : ["https://clio-ng.janelia.org/#!gs://flyem-user-links/short/optic-lobe-columns-v0.json", 40]
 , 'optic-lobe-layers'  : ["https://clio-ng.janelia.org/#!gs://flyem-user-links/short/optic-lobe-layers-v0.json", 20]
}

data_path = PROJECT_ROOT / "results" / "screenshots"

for name, params in screenshots.items():
    scrn = url_plotter(params[0], wait_sec=params[1], background_color='#000000')
    image_saver(scrn, name, data_path)


# %%
