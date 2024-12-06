rule optic_lobe_view:
    input:
        mesh="cache/gallery/neuron.{body_id}.stl"
    output:
        flag=touch("cache/gallery/done/{body_id}-ol.done")
    threads: 4
    params:
        blender_path = os.environ['BLENDER_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    run:
        shell("{params.blender_path} --background --python src/gallery_generation/render_optic-lobe.py -- --body-id {wildcards.body_id} --background-transparent {params.optix}")

rule full_brain_view:
    input:
        mesh="cache/gallery/neuron.{body_id}.stl"
    output:
        flag=touch("cache/gallery/done/{body_id}-fb.done")
    threads: 4
    params:
        blender_path = os.environ['BLENDER_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    run:
        shell("{params.blender_path} --background --python src/gallery_generation/render_optic_lobe.py -- --body-id {wildcards.body_id} {params.optix}")

rule meshdownload:
    input:
        json="results/gallery-descriptions/{configname}.json"
    output:
        flag=touch("cache/gallery/done/preparation/{configname}-prepc.done")
    run:
        shell("python src/gallery_generation/prepare_blend.py --config {input.json}")

rule render_galleries:
    input:
        json="results/gallery-descriptions/{configname}.json",
        flag="cache/gallery/done/preparation/{configname}-prepc.done"
    output:
        flag=touch("cache/gallery/done/render/{configname}-gal.done")
    params:
        blender_path = os.environ['BLENDER_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    run:
        shell("{params.blender_path} --background --python src/gallery_generation/render_optic-lobe.py -- --config {input.json}  --background-transparent {params.optix}"),
        shell("python src/gallery_generation/render_text.py --config {input.json}")

rule render_galleries_blend:
    input:
        json="results/gallery-descriptions/{configname}.json",
        flag="cache/gallery/done/preparation/{configname}-prepc.done"
    output:
        flag=touch("cache/gallery/done/render/{configname}-galb.done")
    params:
        blender_path = os.environ['BLENDER_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    run:
        shell("{params.blender_path} --background --python src/gallery_generation/render_optic-lobe.py -- --config {input.json} --keep-blend --background-transparent {params.optix}"),
        shell("python src/gallery_generation/render_text.py --config {input.json}")


rule pdf_from_png:
    input:
        json="results/gallery-descriptions/{configname}.json",
    output:
        flag=touch("cache/gallery/done/text/{configname}.done")
    run:
        shell("python src/gallery_generation/render_text.py --config {input.json}")


rule create_full_brain_galleries:
    input:
        prevs=get_all_full_brain_configs

rule create_full_brain_galleries_blender:
    input:
        prevs=get_all_full_brain_configs_blend

rule create_optic_lobe_galleries:
    input:
        prevs=get_all_optic_lobe_configs

rule create_optic_lobe_galleries_blender:
    input:
        prevs=get_all_optic_lobe_configs_blend

rule create_galleries_text:
    input:
        prevs=get_all_texts

rule create_galleries_blender:
    input:
        prevs_ol=get_all_optic_lobe_configs_blend,
        prevs_fb=get_all_full_brain_configs_blend


rule one_description_per_group:
    run:
        import shutil
        import glob
        import random
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_stars.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_stars_flipbook.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_groups_non_ol.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_ame.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_dorsal_rim.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_non_ol.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb")
        shell("GALLERY_EXAMPLES=1 jupyter execute --kernel_name='python3' src/gallery_generation/generate_optic_lobe_stars.ipynb")
        json_fn = random.sample(glob.glob("src/gallery_generation/*json"), 1)[0]
        shutil.copy2(json_fn, "results/gallery-descriptions/")

rule generate_optic_lobe_stars:
    output:
        flag=touch("cache/gallery/done/generation/ol_stars.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_optic_lobe_stars.ipynb")

rule generate_optic_lobe_interests:
    output:
        flag=touch("cache/gallery/done/generation/ol_interest.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_optic_lobe_neurons_of_interest.ipynb")

rule generate_optic_lobe_custom:
    output:
        flag=touch("cache/gallery/done/generation/ol_custom.done")
    run:
        import shutil
        import glob
        for json_fn in glob.glob("src/gallery_generation/*json"):
            shutil.copy2(json_fn, "results/gallery-descriptions/")

rule generate_optic_lobe_descriptions:
    output:
        flag=touch("cache/gallery/done/generation/ol_descriptions.done")
    input:
        rules.generate_optic_lobe_stars.output, rules.generate_optic_lobe_interests.output, rules.generate_optic_lobe_custom.output

rule generate_full_brain_stars:
    output:
        flag=touch("cache/gallery/done/generation/fb_stars.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_stars.ipynb")

rule generate_full_brain_groups_non_ol:
    output:
        flag=touch("cache/gallery/done/generation/fb_non_ol.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_groups_non_ol.ipynb")

rule generate_full_brain_types_ame:
    output:
        flag=touch("cache/gallery/done/generation/fb_types_ame.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_ame.ipynb")

rule generate_full_brain_types_dorsal_rim:
    output:
        flag=touch("cache/gallery/done/generation/fb_dorsal_rim.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_dorsal_rim.ipynb")

rule generate_full_brain_types_non_ol:
    output:
        flag=touch("cache/gallery/done/generation/fb_types_non_ol.done")
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_types_non_ol.ipynb")

rule generate_full_brain_group_descriptions:
    output:
        flag=touch("cache/gallery/done/generation/fb_groups.done")
    input:
        rules.generate_full_brain_groups_non_ol.output, rules.generate_full_brain_types_ame.output,
        rules.generate_full_brain_types_dorsal_rim.output, rules.generate_full_brain_types_non_ol.output

rule generate_full_brain_descriptions:
    output:
        flag=touch("cache/gallery/done/generation/fb_descriptions.done")
    input:
        rules.generate_full_brain_stars.output, rules.generate_full_brain_groups_non_ol.output,
        rules.generate_full_brain_types_ame.output, rules.generate_full_brain_types_dorsal_rim.output,
        rules.generate_full_brain_types_non_ol.output
