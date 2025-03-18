import os
import urllib
import json
import re
import time
import datetime

from pathlib import Path
from io import BytesIO
from PIL import Image

import pandas as pd
import numpy as np

from dotenv import find_dotenv, load_dotenv

import neuroglancer
import plotly.express as px

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

import scipy
from scipy.spatial.transform import Rotation

from utils.ng_view import NG_View
from utils.plotter import get_skeleton


def _set_ng_view(driver, set_3d=False, background_color='#FFFFFF'):
    qu_url = driver.current_url
    st_url = urllib.parse.unquote(qu_url)
    ptn = re.compile(r"(http.*!)({.*})")
    furl = ptn.search(st_url)
    js_url = json.loads(furl[2])
    if set_3d:
        js_url['layout'] = '3d'
    js_url['projectionBackgroundColor'] =  background_color
    st_url = json.dumps(js_url)
    qu_url = urllib.parse.quote(st_url)
    driver.get(f"{furl[1]}{qu_url}")


def _is_firefox_installed():
    try:
        options = webdriver.FirefoxOptions()
        options.add_argument("-headless")
        driver = webdriver.Firefox(
            service=FirefoxService(GeckoDriverManager().install())
          , options=options
        )
        driver.quit()
        return True
    except:
        return False

def url_plotter(
    url:str
  , size:tuple[int]=(1920,1080)
  , wait_sec:float=30
  , set_3d:bool=False
  , background_color:str='#FFFFFF'):

    assert wait_sec>=3, f"Need to wait at least 3 seconds, you specified {wait_sec}"

    assert _is_firefox_installed(), "Please install Firefox if you want to use this plotter."

    options = webdriver.FirefoxOptions()
    options.add_argument("-headless")
    options.add_argument(f"--width={size[0]}")
    options.add_argument(f"--height={size[1]}")

    driver = webdriver.Firefox(
        options=options
      , service=FirefoxService(GeckoDriverManager().install())
    )
    driver.get(url)
    time.sleep(3)

    _set_ng_view(driver, set_3d=set_3d, background_color=background_color)

    driver.execute_script(
        "document.getElementsByClassName('neuroglancer-layer-panel')[0].style.display='none';")
    driver.execute_script(
        "document.getElementsByClassName('neuroglancer-viewer-top-row')[0].style.display='none';")
    # canvas = document.getElementsByTagName('canvas')[0]
    # const myArr = Array.from(viewer.layerManager.layerSet)
    # myArr[1].layer.renderLayers.map(x=>x.layerChunkProgressInfo)

    time.sleep(wait_sec-3)

    scrn = driver.get_screenshot_as_png()
    driver.quit()
    return Image.open(BytesIO(scrn))


def image_saver(
    image:Image
  , name:str
  , path:Path
  , replace:bool=False
):
    path.mkdir(parents=True, exist_ok=True)
    img_fn = path / f"{name}.png"
    if img_fn.exists() and not replace:
        today_str = datetime.datetime.today().strftime("%Y-%m-%dT%H-%M-%S")
        img_fn = path / f"{name}_{today_str}.png"
    image.save(img_fn)


def format_nglink(ng_server, link_json_settings):
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def group_plotter(
    body_ids:list[int]
  , colors:list[tuple[int]]=None
  , plot_roi:str=None
  , prune_roi:str=None
  , camera_distance:float=1
  , ignore_cache:bool=False
  , size:tuple[int]=(1920,1080)
  , background_color="#FFFFFF"
  , view:NG_View=NG_View.SVD
):

    assert _is_firefox_installed(), "Please install Firefox if you want to use this plotter."

    cachedir = Path(find_dotenv()).parent / "cache" / "webdriver"
    cachedir.mkdir(parents=True, exist_ok=True)
    skels = None
    for bid in body_ids:
        skels = pd.concat([skels, get_skeleton(bid, ignore_cache=ignore_cache).nodes])

    c_map = {}
    if not colors:
        p_len = len(px.colors.qualitative.Light24)
        for idx, bid in enumerate(body_ids):
            c_map[f'{bid}'] = px.colors.qualitative.Light24[idx % p_len]
    else:
        for idx, bid in enumerate(body_ids):
            if bid:
                if len(colors)>idx:
                    c_map[f'{bid}'] = (f"#{int(colors[idx][0]*255):02x}"
                                        f"{int(colors[idx][1]*255):02x}"
                                        f"{int(colors[idx][2]*255):02x}")
                else:
                    c_map[f'{bid}'] = "#808080"

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        load_dotenv()
        seg_src = os.environ.get('SEGMENTATION_SOURCE')
        assert seg_src is not None,\
            "No segmentation source defined. Please set one in your .env file."

        s.layers.append(
            name='segmentation',
            layer=neuroglancer.SegmentationLayer(
                source=(f'precomputed://{seg_src}')
            )
        )
        for bid in body_ids:
            s.layers['segmentation'].segments.add(bid)
        s.layers['segmentation'].source[0].subsources = {
            "meshes": True
          , "skeletons": False
          , "bounds": False
          , "default": False
        }
        s.layers['segmentation'].segment_colors = c_map

        s.layers.append(
            name='fullbrain-roi',
            layer=neuroglancer.SegmentationLayer(
                source=("precomputed://gs://flyem-optic-lobe"
                    "/fullbrain-roi-v3.7"))
        )
        for r_id in [43, 45, 47]:
            s.layers['fullbrain-roi'].segments.add(r_id)
        s.layers['fullbrain-roi'].source[0].subsources = {"meshes": True}

        s.layers['fullbrain-roi'].segment_colors = {
            '42': '#fff4f2' # LO(L)
          , '43': '#ffded8' # LO(R)
          , '44': '#fffaf2' # LOP(L)
          , '45': '#fff1d8' # LOP(R)
          , '46': '#fdfff2' # ME(L)
          , '47': '#faffd8' # ME(R)
        }
        s.layers['fullbrain-roi'].object_alpha = 0.15

    # General layout stuff
    with viewer.txn() as s:
        s.layout.type = '3d'
        s.show_axis_lines = False
        s.show_default_annotations = False
        s.dimensions = neuroglancer.CoordinateSpace(
            names=['x', 'y', 'z']
          , units='nm'
          , scales=[8,8,8])
        s.relative_display_scales = {'x':1,'y':1,'z':1}

    # view center and zoom calculation
    if view is NG_View.SVD:
        s_xyz = skels[['x', 'y', 'z']]
        skels_com = s_xyz.mean().to_list()
        skels_cobb = s_xyz.min() + (s_xyz.max()- s_xyz.min())/2
        centered = s_xyz-skels_com
        centered_smp = centered.sample(min(5000, len(centered)))
        _, _, v = scipy.linalg.svd(centered_smp)
        s_rn = centered @ v
        skel_dimension_v = np.sort(s_rn.max() - s_rn.min())[1]
        with viewer.txn() as state:
            state.voxel_coordinates = skels_cobb
            rot_v = v.T
            if np.linalg.det(rot_v) < 0:
                rot_v[1,:] *= -1 
            view_rotation = Rotation.from_matrix(rot_v)   # TODO: double check this viewing rotation.
            state.projection_orientation = view_rotation.as_quat()
            state.projection_background_color = background_color
            state.projection_scale =  skel_dimension_v*camera_distance + 3000
    else:
        with viewer.txn() as state:
            state.voxel_coordinates = view.location
            state.projection_orientation = view.orientation
            state.projection_scale = view.scale

    options = webdriver.FirefoxOptions()

    options.add_argument("-headless")

    options.add_argument(f"--width={size[0]}")
    options.add_argument(f"--height={size[1]}")

    with webdriver.Firefox(
        options=options
      , service=FirefoxService(GeckoDriverManager().install())
    ) as driver:
        link = neuroglancer.to_url(viewer.state, prefix='https://clio-ng.janelia.org')
        driver.get(viewer.get_viewer_url())
        scr_rpl = viewer.screenshot()
        img = Image.fromarray(scr_rpl.screenshot.image_pixels)

    return (img, link)
