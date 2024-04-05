"""
Render the sliced optic lobe view of a set of neurons either defined by a
body ID or a config file.
"""

# Import modules and libraries

import os
import sys
import argparse
import time
from math import radians
from pathlib import Path

import bpy
import mathutils

sys.path.append(os.getcwd())
from src.utils.plot_config import PlotConfig
from src.utils.plot_roi import PlotRoi

# Setting path
cache_path = Path("cache") / "gallery"
blend_path = cache_path / "blend"
blend_path.mkdir(parents=True, exist_ok=True)
results_path = Path("results") / "gallery"
results_path.mkdir(parents=True, exist_ok=True)


def set_world(
      clip=1_000_000
    , color=(1, 1, 1, 1)
    , transparent=False
) -> None:

    """
    sets the world in blender by defining features such as transparency, color and view settings.
    It also sets a clipping range for the objects within the world to fall within the camera view.
    It also sets the node settings for the 3D view.

    Parameters
    ----------
    clip : int, default=1000000
        clipping distance for camera
    color : tuple, default=(1,1,1,1)
        background color for the world, default to white
    transparent : bool, default=False
        render on transparent background? Defaults to no.
    """
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    s.clip_end = clip
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = color
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_settings.view_transform = 'Raw'

    bpy.context.scene.render.engine = 'CYCLES'

    if transparent:
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.film_transparent = True
    else:
        bpy.context.scene.use_nodes = True
        tree = bpy.data.scenes['Scene'].node_tree
        alphaover = tree.nodes.new(type='CompositorNodeAlphaOver')
        alphaover.premul = 1.0
        render = bpy.data.scenes['Scene'].node_tree.nodes["Render Layers"]
        comp = bpy.data.scenes['Scene'].node_tree.nodes["Composite"]
        _ = tree.links.new(render.outputs[0], alphaover.inputs[2])
        _ = tree.links.new(alphaover.outputs[0], comp.inputs[0])


def remove_old() -> None:
    """
    removes the old objects from the world
    """
    objs = bpy.data.objects
    for default_objects in ["Cube", "Light", "Camera"]:
        objs.remove(objs[default_objects], do_unlink=True)


def set_camera(
    config:dict
) -> None:
    """
    defines the camera and the associated settings such as the location and 
    rotation of the camera

    Parameters
    ----------
    config : dict
        camera : dict
            resolution : list
                [u, v] dimension of the final rendering.
            location : list
                [x, y, z] location of the camera
            rotation : list
                [alpha, beta, gamma] rotation of the camera
    """
    bpy.context.scene.render.resolution_x=config['camera']['resolution'][0]
    bpy.context.scene.render.resolution_y=config['camera']['resolution'][1]
    bpy.ops.object.camera_add(
        enter_editmode=False
      , align='WORLD'
      , location=tuple(config['camera']['location'])
      , rotation=(
            radians(config['camera']['rotation'][0])
          , radians(config['camera']['rotation'][1])
          , radians(config['camera']['rotation'][2])
        )
    )
    camera = bpy.context.selected_objects[0]
    camera.name = 'Snapshot'
    camera.data.name='OrthoCam'
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale=config['camera']['ortho_scale']
    camera.data.clip_end=100000
    camera.data.shift_x=0
    camera.data.shift_y=-0.1
    bpy.context.scene.camera = camera


def add_scalebar(
    config:PlotConfig
):
    if not config.scalebar:
        return
    bar_len = config.scalebar['length']
    bpy.ops.mesh.primitive_cube_add(
        size=1.0
      , enter_editmode=False
      , align='WORLD'
      , location=[config.scalebar['location'][0]
          , config.scalebar['location'][1]
          , config.scalebar['location'][2]
        ]
      , rotation=(
            radians(config.camera['rotation'][0])
          , radians(config.camera['rotation'][1])
          , radians(config.camera['rotation'][2])
        )
      , scale=[bar_len*1000/8, 300, 300]
    )
    bar1 = bpy.context.object
    bland_mat = bpy.data.materials.new(name=f"scalebar1.material")
    bland_mat.use_nodes = True
    bland_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = [.6, .6, .6, 1]
    bland_mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = 1
    bland_mat.blend_method = 'BLEND'
    bland_mat.shadow_method = 'HASHED'
    bar1.data.materials.clear()
    bar1.data.materials.append(bland_mat)
    if config.scalebar['type'] == 'L':
        bpy.ops.mesh.primitive_cube_add(
            size=1.0
          , enter_editmode=False
          , align='WORLD'
          , location=[config.scalebar['location'][0], 
                config.scalebar['location'][1],
                config.scalebar['location'][2]]
          , rotation=(
                radians(config.camera['rotation'][0])
              , radians(config.camera['rotation'][1])
              , radians(config.camera['rotation'][2])
            )
          , scale=[300, bar_len*1000/8, 300]
        )
        bar2 = bpy.context.object
        camera = bpy.data.objects['Snapshot']
        trans_world = camera.matrix_world.to_3x3() @ mathutils.Vector((-bar_len*1000/16, bar_len*1000/16-150, 0))
        bar2.matrix_world.translation += trans_world
        r_mat = bpy.data.materials.new(name=f"scalebar1.new.material")
        r_mat.use_nodes = True
        r_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = [1, 0, 0, 1]
        r_mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = 1
        r_mat.blend_method = 'BLEND'
        r_mat.shadow_method = 'HASHED'
        bar1.data.materials.clear()
        bar1.data.materials.append(r_mat)
        g_mat = bpy.data.materials.new(name=f"scalebar2.material")
        g_mat.use_nodes = True
        g_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = [0, 1, 0, 1]
        g_mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = 1
        g_mat.blend_method = 'BLEND'
        g_mat.shadow_method = 'HASHED'
        bar2.data.materials.clear()
        bar2.data.materials.append(g_mat)

    if config.scalebar.get('text_location'):
        bpy.ops.object.text_add()
        txt = bpy.context.object
        txt.name = "bar text"
        txt.data.body = f"{bar_len} µm"
        txt.location = [
            config.scalebar['text_location'][0]
        , config.scalebar['text_location'][1]
        , config.scalebar['text_location'][2]
        ]
        txt.rotation_euler = (
            radians(config.camera['rotation'][0])
        , radians(config.camera['rotation'][1])
        , radians(config.camera['rotation'][2])
        )
        size = 1000
        txt.scale = (size, size, size)
        txt.data.materials.clear()
        txt.data.materials.append(bland_mat)


def __import_obj(
    filename
  , scale:float=1.0
):
    if bpy.app.version < (4,0,0):
        bpy.ops.import_scene.obj(
            filepath=filename
          , axis_up="Z", axis_forward="Y"
        )
        obj = bpy.context.selected_objects[0]
        obj.scale = (scale, scale, scale)
    else:
        bpy.ops.wm.obj_import(
            filepath=filename
          , up_axis="Z", forward_axis="Y"
          , global_scale=scale
        )
        obj = bpy.context.selected_objects[0]
    return obj


def import_rois(config:PlotConfig) -> None:
    """
    imports the meshes for all the layer and column rois from the set path
    """
    camera = bpy.data.objects['Snapshot']
    trans_world = camera.matrix_world.to_3x3() @ mathutils.Vector((0,0, -40000))

    for roi in config.rois:
        robj = __import_obj(f"{cache_path}/{roi.filename}.obj", roi.scale)
        robj.hide_render = not roi.is_visible
        robj.hide_set(robj.hide_render)
        robj.name = roi.oname
        # Slicing start
        if roi.is_sliced:
            start_cut = time.time()
            bpy.ops.mesh.primitive_cube_add(
                size=1.0
              , enter_editmode=False
              , align='WORLD'
              , location=roi.location
              , rotation=roi.rotation
              , scale=roi.box
            )

            cutter = bpy.context.selected_objects[0]
            cutter.name = f"{roi.oname}.cutter"
            cutter.hide_render = True
            cutter.hide_set(True)

            mod = robj.modifiers.new(name="Intersect", type='BOOLEAN')
            mod.operation='INTERSECT'
            mod.object = cutter
            mod.solver = 'EXACT'
            mod.use_hole_tolerant = True
            mod.use_self = True
            bpy.context.view_layer.objects.active = robj
            bpy.ops.object.modifier_apply(modifier="Intersect")
            end_cut = time.time()
            print(f"Slicing of '{roi.oname}' took {(end_cut-start_cut)*1000.0:.1f} ms")
            # move if it is a slice
            robj.matrix_world.translation += trans_world

        elif roi.is_flat:
            # bpy.ops.mesh.pr
            bpy.context.view_layer.objects.active = robj
            # bpy.ops.object.modifier_apply(modifier="shrink.me")
        
            robj.matrix_world.translation += camera.matrix_world.to_3x3() @ mathutils.Vector((0,0, -50000))

        mat = bpy.data.materials.new(name=f"{roi.oname}.material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        # breakpoint()
        if default_bsdf := nodes.get('Principled BSDF'):
            nodes.remove(default_bsdf)
        n_output = nodes.get('Material Output')
        n_light_path = nodes.new('ShaderNodeLightPath')
        n_emision = nodes.new('ShaderNodeEmission')
        n_emision.inputs['Color'].default_value = roi.color
        n_mixer = nodes.new('ShaderNodeMixShader')
        mat.node_tree.links.new(n_output.inputs[0], n_mixer.outputs[0])
        mat.node_tree.links.new(n_mixer.inputs[2], n_emision.outputs[0])
        mat.node_tree.links.new(n_mixer.inputs[0], n_light_path.outputs[0])
        robj.data.materials.clear()
        robj.data.materials.append(mat)
        
        if roi.has_outline:
            start_col = time.time()
            wobj = robj.copy()
            wobj.data = robj.data.copy()
            wobj.name = f"{roi.oname}.wireframe"
            bpy.context.collection.objects.link(wobj)
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = wobj
            wobj.select_set(True)
            bpy.ops.object.make_single_user(
                object=True
              , obdata=True
              , material=True
              , animation=True
              , obdata_animation=True
            )
            
            wmat = bpy.data.materials.new(name=f"{roi.oname}.material.wireframe")
            wmat.use_nodes = True
            wmat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = roi.outline_color
            wmat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = roi.outline_color[3]
            wmat.blend_method = 'BLEND'
            wmat.shadow_method = 'HASHED'
            wobj.data.materials.clear()
            wobj.data.materials.append(wmat)
            bpy.ops.object.modifier_add(type='DECIMATE')
            bpy.context.object.modifiers["Decimate"].decimate_type = 'DISSOLVE'
            bpy.context.object.modifiers["Decimate"].angle_limit = 0.0872665
            bpy.context.view_layer.objects.active = wobj
            bpy.ops.object.modifier_apply(modifier="Decimate", single_user=True)
            bpy.ops.object.modifier_add(type='WIREFRAME')
            bpy.context.object.modifiers["Wireframe"].use_replace = True
            bpy.context.object.modifiers["Wireframe"].thickness = 30
            bpy.context.object.modifiers["Wireframe"].offset = 0.5
            bpy.context.object.modifiers["Wireframe"].crease_weight = 1
            bpy.context.object.modifiers["Wireframe"].material_offset = 1
            bpy.context.view_layer.objects.active = wobj
            bpy.ops.object.modifier_apply(modifier="Wireframe", single_user=True)
            end_col = time.time()
            print(f"Outlining of '{roi.oname}' took {(end_col-start_col)*1000.0:.1f} ms")


def import_neurons(config:PlotConfig) -> None:
    """
    Imports the mesh for the neuron of interest.

    Parameters
    ----------
    config : dict
        bids : tuple
            body ID / color combination of neurons and their primary color
    """
    for neurons in config.neurons:
        n_objs = []
        for n_idx, bid in enumerate(neurons.bids):
            n_obj = __import_obj(f"{cache_path}/neuron.{bid}.obj")
            n_obj.name = f"neuron.{bid}.{neurons.name}.{n_idx:04d}"
            mat = bpy.data.materials.new(name=f"{n_obj.name}.material")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = neurons.colors[n_idx]
            mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = neurons.colors[n_idx][3]
            mat.blend_method = 'BLEND'
            mat.shadow_method = 'HASHED'
            n_obj.data.materials.clear()
            n_obj.data.materials.append(mat)
            n_objs.append(n_obj)
        
        start_cut = time.time()
        for s_idx, slc in enumerate(neurons.slicers, start=1):
            if slc.is_named:
                if len(slc.roi)>1:
                    comb_name = "_".join(slc.roi)
                    croi = bpy.context.scene.objects.get(comb_name)
                    if not croi:
                        bpy.ops.mesh.primitive_cube_add(size=0, scale=[0, 0, 0])
                        croi = bpy.context.selected_objects[0]
                        croi.name = comb_name
                        for idx, curr_roi_name in enumerate(slc.roi):
                            mod_name = f"{curr_roi_name}_{idx}"
                            cur_mod = croi.modifiers.new(name=mod_name, type='BOOLEAN')
                            cur_mod.operation='UNION'
                            bpy.context.scene.objects.get(comb_name)
                            cur_mod.solver = 'EXACT'
                            cur_mod.use_hole_tolerant = True
                            cur_mod.use_self = True
                            curr_roi = bpy.context.scene.objects.get(curr_roi_name)
                            if curr_roi:
                                cur_mod.object = curr_roi
                            else:
                                print(f"Can't find the named slicer {curr_roi_name}")
                            bpy.context.view_layer.objects.active = croi
                            bpy.ops.object.modifier_apply(modifier=mod_name)
                        croi.hide_render = True
                        croi.hide_set(True)

                else:
                    croi = bpy.data.objects[slc.roi[0]]
                
                for n_idx, nobj in enumerate(n_objs, start=1):
                    mod_name = f"Mod_{slc.keep}_{s_idx}_{n_idx}"
                    mod = nobj.modifiers.new(name=mod_name, type='BOOLEAN')
                    if slc.keep == 'intersection':
                        mod.operation='INTERSECT'
                    else:
                        mod.operation='DIFFERENCE'
                    mod.solver = 'EXACT'
                    mod.use_hole_tolerant = True
                    mod.use_self = True
                    mod.object = croi
                    bpy.context.view_layer.objects.active = nobj
                    if slc.apply:
                        bpy.ops.object.modifier_apply(modifier=mod_name)
                
            else:
                bpy.ops.mesh.primitive_cube_add(
                    size=1.0
                  , enter_editmode=False
                  , align='WORLD'
                  , location=slc.location
                  , rotation=slc.rotation
                  , scale=slc.box
                )

                cutter = bpy.context.selected_objects[0]
                cutter.name = f"{neurons.name}.{s_idx}.cutter"
                cutter.hide_render = True
                cutter.hide_set(True)
                
                for n_idx, nobj in enumerate(n_objs, start=1):
                    mod_name = f"Mod_{slc.keep}_{s_idx}_{n_idx}"
                    mod = nobj.modifiers.new(name=mod_name, type='BOOLEAN')
                    if slc.keep == 'intersection':
                        mod.operation='INTERSECT'
                    else:
                        mod.operation='DIFFERENCE'
                    mod.solver = 'EXACT'
                    mod.use_hole_tolerant = True
                    mod.use_self = True
                    mod.object = cutter
                    bpy.context.view_layer.objects.active = nobj
                    if slc.apply:
                        bpy.ops.object.modifier_apply(modifier=mod_name)
                    
        end_cut = time.time()
        print(f"Slicing of '{neurons.name}' took {(end_cut-start_cut)*1000.0:.1f} ms")


def set_roi_cutter(
    config:dict
) -> None:
    """
    Create a box that acts as a bounding box for a sub-region of the ROIs.
    The box will be used for creating a "slice" of the ROIs.

    Parameters
    ----------
    config : dict
        roi-cutter : dict
            location : list
                [x, y, z] location of the center of the box
            rotation : list
                [alpha, beta, gamma] rotation of the box
            size : int
                width of the box.
    """
    bpy.ops.mesh.primitive_cube_add(
        size=1.0
      , enter_editmode=False
      , align='WORLD'
      , location=tuple(config['roi-cutter']['location'])
      , rotation=(
            radians(config['roi-cutter']['rotation'][0])
          , radians(config['roi-cutter']['rotation'][1])
          , radians(config['roi-cutter']['rotation'][2])
        )
      , scale=(config['roi-cutter']['size'])
    )
    cutter = bpy.context.selected_objects[0]

    z_shift = config['roi-cutter']['size'][2]/2
    trans_world = cutter.matrix_world.to_3x3() @ mathutils.Vector((0,0,z_shift))
    cutter.matrix_world.translation += trans_world
    cutter.name = "ROI_cutter"
    cutter.hide_render = True
    cutter.hide_set(True)


def set_neuron_cutter(
    config:dict
) -> None:
    """
    Create a box that is used for pruing the neuron.

    Parameters
    ----------
    config : dict
        neuron-cutter : dict
            location : list
                [x, y, z] location of the center of the box
            rotation : list
                [alpha, beta, gamma] rotation of the box
            size : int
                width of the box.
    """
    bpy.ops.mesh.primitive_cube_add(
        size=1.0
      , enter_editmode=False
      , align='WORLD'

      , location=tuple(config['neuron-cutter']['location'])
      , rotation=(
            radians(config['neuron-cutter']['rotation'][0])
          , radians(config['neuron-cutter']['rotation'][1])
          , radians(config['neuron-cutter']['rotation'][2])
        )
      , scale=(config['neuron-cutter']['size'])
    )
    cutter = bpy.context.selected_objects[0]
    cutter.name = "Neuron_cutter"
    cutter.hide_render = True
    cutter.hide_set(True)


def add_mat(
    name:str
  , color:tuple
  , alpha:float
) -> None:
    """
    Add a material.

    Parameters
    ----------
    name : str
        name of the new material
    color : tuple
        base color of the new material
    alpha : float
        alpha value of the new material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color
    mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = (alpha)
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'


def create_layer_material(
    color:tuple=(.81,.81,.81,1)
) -> None:
    """
    Create a set of predefined materials, from "even" and "odd" to ROI colors.

    Parameters
    ----------
    color : tuple
        Defines the even / odd material (odd is even +.11)
    """
    add_mat('Layer.Material.even', color, 1)
    add_mat('Layer.Material.odd', tuple([c+.11 for c in color]), 1)
    add_mat('Layer.Material.boundary', (.3, .3, .3, .8), 1)
    add_mat('Layer.Material.me.boundary', (.366, .098, .301, 1), 1)
    add_mat('Layer.Material.me.even', (.366, .098, .301, 1), .1)
    add_mat('Layer.Material.me.odd', (.366, .098, .301, 1), .05)
    add_mat('Layer.Material.lo.boundary', (.73,.292,.371, 1), 1)
    add_mat('Layer.Material.lo.even', (.73,.292,.371, 1), .1)
    add_mat('Layer.Material.lo.odd', (.73,.292,.371, 1), .05)
    add_mat('Layer.Material.lop.boundary', (.939, .445, .397, 1), 1)
    add_mat('Layer.Material.lop.even', (.939, .445, .397, 1), .1)
    add_mat('Layer.Material.lop.odd', (.939, .445, .397, 1), .05)
    add_mat('Material.ame_r', (.033, .066, .099, 1), .05)
    add_mat('Material.ame_r.boundary', (.1, .13, .16, 1), .05)


def crosssection_layer(
    name:str
) -> None:
    """
    Create a cross section (slice) of a mesh.

    Parameters
    ----------
    name : str
        Name of the mesh to create a cross section for.
    """
    mod = bpy.data.objects[name].modifiers.new(name="Intersect", type='BOOLEAN')
    mod.operation='INTERSECT'
    mod.object = bpy.data.objects['ROI_cutter']
    mod.solver = 'EXACT'
    mod.use_hole_tolerant = True
    mod.use_self = True
    if name == "roi.ame_r":
        lcol = bpy.data.materials['Material.ame_r']
    else:
        count = int(''.join(filter(str.isdigit, name)))
        roi_sig = name[4:7].strip("_").lower()
        if count % 2 == 1:
            lcol = bpy.data.materials[f'Layer.Material.{roi_sig}.odd']
        else:
            lcol = bpy.data.materials[f'Layer.Material.{roi_sig}.even']
    obj = bpy.data.objects[name]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Intersect")
    obj.data.materials.clear()
    obj.data.materials.append(lcol)

    wf_n = f"{name}.wireframe"
    wireframe = bpy.data.objects.new(wf_n, obj.data)

    bpy.context.collection.objects.link(wireframe)
    mod_wf = wireframe.modifiers.new(name="wireframe", type='WIREFRAME')
    mod_wf.thickness = 30
    obj = bpy.data.objects[wf_n]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="wireframe", single_user=True)
    if name == "roi.ame_r":
        wf_col = bpy.data.materials['Material.ame_r.boundary']
    else:
        wf_col = bpy.data.materials[f'Layer.Material.{roi_sig}.boundary']
    obj.data.materials.clear()
    obj.data.materials.append(wf_col)

def crosssection_neuron(
    config:dict
) -> None:
    """
    Create a cross section of a neuron

    Parameters
    ----------
    config : dict
        bids : tuple
            (body ID, color) tuple to be sliced
        keep_neuron_mesh : bool
            if true, don't apply the modifier but keep it attached to the object. 
            This increases file size, but allows later modifcation of the blender
            file.
    """
    for bid, _ in config['bids'].items():
        neuron_name = f"neuron.{bid}"
        mod = bpy.data.objects[neuron_name].modifiers.new(name="Intersect", type='BOOLEAN')
        mod.operation='INTERSECT'
        mod.object = bpy.data.objects['Neuron_cutter']
        mod.solver = 'EXACT'
        mod.use_hole_tolerant = True
        mod.use_self = True
        lcol = bpy.data.materials[f'Neuron.Material.{bid}']
        obj = bpy.data.objects[neuron_name]
        bpy.context.view_layer.objects.active = obj
        if not config['keep_neuron_mesh']:
            bpy.ops.object.modifier_apply(modifier="Intersect")
        obj.data.materials.clear()
        obj.data.materials.append(lcol)

        for roi in ['roi.me_r', 'roi.lo_r', 'roi.lop_r']:
            mod = bpy.data\
                .objects[f"{neuron_name}.outside"]\
                .modifiers\
                .new(name="Outersect", type='BOOLEAN')
            mod.operation='DIFFERENCE'
            mod.object = bpy.data.objects[roi]
            mod.solver = 'EXACT'
            mod.use_hole_tolerant = True
            mod.use_self = True
            obj2 = bpy.data.objects[f"{neuron_name}.outside"]
            bpy.context.view_layer.objects.active = obj2
            if not config['keep_neuron_mesh']:
                bpy.ops.object.modifier_apply(modifier="Outersect")
        obj2 = bpy.data.objects[f"{neuron_name}.outside"]
        obj2.data.materials.clear()
        obj2.data.materials.append(lcol)


def hide_neuropil() -> None:
    """
    Make the neuropils invisible
    """
    for neuropil in ['me_r', 'lo_r', 'lop_r']:
        me_r = bpy.data.objects[f'roi.{neuropil}']
        me_r.hide_render = True
        me_r.hide_set(True)


def slide_back(
    config:dict
) -> None:
    """
    Move the (sliced) brain regions back by the width of the neuron-cutter

    Parameters
    ----------
    config : dict
        neuron-cutter : dict
            size : list
                [u, v, w] size of the neuron cutter. Using `w` to move.
    """
    xs_width = config['neuron-cutter']['size'][2]
    cutter = bpy.data.objects['ROI_cutter']
    trans_world = cutter.matrix_world.to_3x3() @ mathutils.Vector((0,0, -xs_width/2))
    for idx in range(1,11):
        lyr = bpy.data.objects[f'roi.me_r_layer_{idx:02d}']
        lyr.matrix_world.translation += trans_world
        lyr = bpy.data.objects[f'roi.me_r_layer_{idx:02d}.wireframe']
        lyr.matrix_world.translation += trans_world
    for idx in range(1,8):
        lyr = bpy.data.objects[f'roi.lo_r_layer_{idx}']
        lyr.matrix_world.translation += trans_world
        lyr = bpy.data.objects[f'roi.lo_r_layer_{idx}.wireframe']
        lyr.matrix_world.translation += trans_world
    for idx in range(1,5):
        lyr = bpy.data.objects[f'roi.lop_r_layer_{idx}']
        lyr.matrix_world.translation += trans_world
        lyr = bpy.data.objects[f'roi.lop_r_layer_{idx}.wireframe']
        lyr.matrix_world.translation += trans_world
    lyr = bpy.data.objects['roi.ame_r']
    lyr.matrix_world.translation += trans_world
    lyr = bpy.data.objects['roi.ame_r.wireframe']
    lyr.matrix_world.translation += trans_world


def activate_optix() -> None:
    """
    activate optix rendering in blender
    """
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = "OPTIX"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # might be needed for getting devices
    for device_type in prefs.get_device_types(bpy.context):
        prefs.get_devices_for_type(device_type[0])

    devs = prefs.get_devices()
    # enable all devices
    if devs:
        for group in devs:
            print(group)
            for device in group:
                device.use = True


def set_config(
    args
) -> dict:
    """
    Takes the command line arguments and returns a set of config parameters
    specific for render_optic-lobe

    Parameters
    ----------
    args :
        command line arguments

    Returns
    -------
    config : dict
        configuration used in the other functions
    """
    config = {}
    config['camera'] = {
        'rotation' : [91, -223, 162]
      , 'location' : [25000, 30500, 0]
      , 'scale': 55000
    }
    config['neuron-cutter'] = {
        'location' : [16682, 34887, 33158]
      , 'rotation' : [91, -223, 162]
      , 'size': [80000, 80000, 6000]
      , 'active': True
    }
    config['roi-cutter'] = {
        #'location' : [16682, 34887, 33158]
        #'location' : [17247, 36557, 33180]
        # 'location' : [17000, 36000, 33180]
        #'location' : [16800, 35300, 33180]
        #'location' : [16900, 35600, 33180]
        # 'location' : [16900, 35800, 33180]
        # 'location' : [16950, 35900, 33180]
        'location' : [16950, 35900, 33180]
      , 'rotation' : [91, -223, 162]
      , 'size' : [40000, 40000, 20]
    }

    config['keep_neuron_mesh'] = args.keep_neuron_mesh

    if args.body_id:
        config['bids'] = {args.body_id: "#000000"}
        if args.rotation:
            config['camera']['rotation'] = args.rotation
            config['neuron-cutter']['rotation'] = args.rotation
        if args.slice_width:
            tmpsz = config['neuron-cutter']['size']
            config['neuron-cutter']['size'] = [tmpsz[0], tmpsz[1], args.slice_width]
        if args.keep_neuron_mesh:
            config['neuron-cutter']['active'] = False
        config['name'] = args.body_id
        config['filename'] = f"neuron.{args.body_id}"
    elif args.config:
        pcfg = PlotConfig(args.config)
        config['bids'] = pcfg.bid_dict
        config['name'] = pcfg.name
        config['directory'] = pcfg.directory
        config['filename'] = pcfg.basename
        config['rois'] = pcfg.rois
        cam = pcfg.camera
        config['neuron-cutter']['size'][2] = pcfg.max_slice
        if cam:
            config['camera'] = cam
    config['keep-blend'] = bool(args.keep_blend)
    return config


def run_script(args):
    """
    Runs the functions needed to produce a blender / png file.

    Parameters
    ----------
    args :
        command line arguments
    """
    config = set_config(args)

    n_me_layers = 10 #number of layers in the medulla
    n_lo_layers = 7  #number of layers in the lobula
    n_lop_layers = 4 #number of layers in the lobula plate

    if args.optix:
        activate_optix()

    set_world(transparent=args.background_transparent)
    remove_old()
    set_camera(config=config)
    add_scalebar(config=PlotConfig(args.config))
    import_rois(config=PlotConfig(args.config))
    import_neurons(config=PlotConfig(args.config))
    # set_roi_cutter(config=config)
    # set_neuron_cutter(config=config)

    # create_layer_material()

    # hide_neuropil()

    # for idx in range(1, n_me_layers + 1):
    #     crosssection_layer(f'roi.me_r_layer_{idx:02d}')
    # for idx in range(1, n_lo_layers + 1):
    #     crosssection_layer(f'roi.lo_r_layer_{idx}')
    # for idx in range(1, n_lop_layers + 1):
    #     crosssection_layer(f'roi.lop_r_layer_{idx}')
    # crosssection_layer("roi.ame_r")

    # slide_back(config=config)

    # crosssection_neuron(config=config)

    img_dir = results_path / config['directory']
    img_dir.mkdir(exist_ok=True)

    start_render = time.time()
    bpy.context.scene.render.filepath = f"{img_dir}/{config['filename']}.png"
    bpy.ops.render.render(write_still = True)
    end_render = time.time()
    print(f"Rendering took {(end_render-start_render)*1000.0:.1f} ms")

    if config['keep-blend']:
        bpy.ops.wm.save_as_mainfile(
            filepath=f"{blend_path}/{config['filename']}.blend"
          , check_existing=False
        )


def cli(argv):
    """
    Parses the CLI arguments. Calls `run_script()`
    """
    parser = argparse.ArgumentParser()
    arg_grp = parser.add_mutually_exclusive_group(required=True)
    arg_grp.add_argument(
        "--body-id"
      , type=int
      , help="Body ID of the neuron to plot"
    )
    arg_grp.add_argument(
        "--config"
      , type=Path
      , help="path to PlotConfig file"
    )

    parser.add_argument(
        "--slice-width"
      , type=int
      , default=12000 #6000
      , help="thickness of the neuron slicer"
    )

    parser.add_argument(
        "--rotation"
      , nargs=3
      , default=[91, -223, 162]
      , help="Main rotation"
    )

    parser.add_argument(
        "--background-transparent"
      , action='store_true'
      , help="Render on a transparent background"
    )
    parser.add_argument(
        "--keep-neuron-mesh"
      , action='store_true'
      , help="Keep the full neuron mesh in the blend file."
    )
    parser.add_argument(
        "--keep-blend"
      , action='store_true'
      , help="keep the blender file after rendering."
    )
    parser.add_argument(
        "--optix"
      , action='store_true'
      , help="activate optix rendering"
    )
    args = parser.parse_args(argv)
    run_script(args)

if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    cli(argv)
