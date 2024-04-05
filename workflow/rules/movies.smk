from dotenv import load_dotenv
load_dotenv()

envvars:
    "BLENDER_PATH",
    "NEUVID_PATH"

include: "../scripts/get-mem.py"
include: "../scripts/movie-paths.py"

rule crossection:
    output:
        directory("cache/blender/crossections/"),
        expand("cache/blender/crossections/ME_R_layer_{mel:02d}.obj", mel=[i for i in range(1,11)]),
        expand("cache/blender/crossections/LO_R_layer_{lol}.obj", lol=[i for i in range(1,8)]),
        expand("cache/blender/crossections/LOP_R_layer_{lopl}.obj", lopl=[i for i in range(1,5)])
    log:
        stdout="logs/crossection.out.log",
        stderr="logs/crossection.err.log"
    priority: 1
    shell:
        """
        python src/movies/neuropil-crossections.py > {log.stdout} 2> {log.stderr}
        """

rule blendfile:
    input:
        json="results/movie-descriptions/{movie}.json",
        cs="cache/blender/crossections/"
    output:
        blender="cache/blender/{movie}.blend"
    resources: mem_mb=get_mem_mb
    threads: 3
    priority: 2
    params:
        cache="cache/blender/{movie}-object-cache/",
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH']
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/importMeshes.py -- --cachedir {params.cache} --skipExisting --strict -i {input.json} --output {output.blender}
        """

rule blendanim:
    input:
        json="results/movie-descriptions/{movie}.json",
        blender="cache/blender/{movie}.blend"
    output: "cache/blender/{movie}Anim.blend"
    threads: 3
    resources: mem_mb=get_mem_mb
    priority: 3
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH']
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/addAnimation.py -- --inputJson {input.json} --inputBlender {input.blender} --output {output}
        """

rule renderframes:
    input:
        json="results/movie-descriptions/{movie}.json",
        blenderAnim="cache/blender/{movie}Anim.blend"
    output:
        path = directory("cache/blender/{movie}_{width}x{height}-frames/"),
        flag = touch("cache/blender/.render_{movie}_{width}x{height}.done")
    threads: 4
    priority: 4
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else "",
        render_threads = 36
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/render.py -- --skipExisting --persist --threads {params.render_threads} --resX {wildcards.width} --resY {wildcards.height} -i {input.json} {params.optix} --inputBlender {input.blenderAnim} --output {output.path}
        """

rule addtext:
    input:
        json="results/movie-descriptions/{movie}.json",
        path="cache/blender/{movie}_{width}x{height}-frames/",
        flag="cache/blender/.render_{movie}_{width}x{height}.done"
    threads: 10
    resources: mem_mb=get_mem_mb
    priority: 5
    output:
        path=directory("cache/blender/{movie}_{width}x{height}-labeled/"),
        flag=touch("cache/blender/.label_{movie}_{width}x{height}.done")
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/compLabels.py -- --threads {threads} {params.optix} --input {input.json} --inputFrames {input.path} --output {output.path}
        """

rule assembleframes:
    input:
        json="results/movie-descriptions/{movie}.json",
        path="cache/blender/{movie}_{width}x{height}-labeled/",
        flag="cache/blender/.label_{movie}_{width}x{height}.done"
    resources: mem_mb=get_mem_mb
    priority: 6
    output:
        path=directory("cache/blender/{movie}_{width}x{height}-assembled/"),
        movie="results/movies/{movie}_{width}x{height}.avi"
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH']
    run:
        shell("{params.blender_path} --background --python {params.neuvid_path}/neuVid/assembleFrames.py -- --width {wildcards.width} --height {wildcards.height} -i {input.path} -o {output.path}")
        import shutil, glob
        movie_files = glob.glob(f"{output.path}/*avi")
        shutil.move(movie_files[0], output.movie)

rule movie4k:
    input:
        json="results/movie-descriptions/{movie}.json",
        avi="results/movies/{movie}_3840x2160.avi"
    output:
        "results/movies/{movie}-4k.avi"
    priority: 7
    run:
        import shutil
        shutil.copy2(input.avi, output[0])

rule movie8k:
    input:
        json="results/movie-descriptions/{movie}.json",
        avi="results/movies/{movie}_7680x4320.avi"
    output:
        "results/movies/{movie}-8k.avi"
    priority: 7
    run:
        import shutil
        shutil.copy2(input.avi, output[0])

rule fullmovie:
    input:
        json="results/movie-descriptions/{movie}.json",
        avi="results/movies/{movie}_1920x1080.avi"
    output:
        "results/movies/{movie}.avi"
    priority: 7
    run:
        import shutil
        shutil.copy2(input.avi, output[0])

rule previewmovie:
    input:
        json="results/movie-descriptions/{movie}.json",
        avi="results/movies/{movie}_384x216.avi"
    output:
        "results/movies/{movie}-preview.avi"
    priority: 8
    run:
        import shutil
        shutil.copy2(input.avi, output[0])


rule copydescriptions:
    output:
        json=touch("results/movie-descriptions/copy.done")
    run:
        import shutil
        import glob
        for json_fn in glob.glob("src/movies/*json"):
            shutil.copy2(json_fn, "results/movie-descriptions/")


rule generatedescriptions:
    output:
        json=touch("results/movie-descriptions/generation.done")
    run:
        shell("jupyter execute --kernel_name='.venv' src/movies/generate_movies.ipynb")


rule alldescriptions:
    input:
        "results/movie-descriptions/copy.done",
        "results/movie-descriptions/generation.done"



rule allpreviews:
    input:
        prevs=get_all_movie_previews

rule allmovies:
    input:
        get_all_movies


rule generate_flipbook_images:
    run:
        shell("jupyter execute --kernel_name='.venv' src/gallery_generation/generate_full_brain_stars_flipbook.ipynb")


# Required GNU parallel and imagemagick
rule generate_flipbook:
    input:
        inpdir="results/gallery/{directory}"
    output:
        outdir=directory("cache/movies/{directory}")
    threads: 8
    run:
        import shutil
        ignore_pdf = shutil.ignore_patterns("*.png")
        shutil.copytree(input.inpdir, output.outdir, ignore=ignore_pdf)
        rename_cache_movie_files(output.outdir)
        shell("cd {output.outdir}; parallel mogrify -density 450 -background white -extent 1920x1080 -format png ::: *.pdf")


# requires ffmpeg
rule generate_flipbook2:
    input:
        olin="cache/movies/flipbook_OL_intrinsic",
        olcn="cache/movies/flipbook_OL_connecting",
        olvpn="cache/movies/flipbook_VPN",
        olvcn="cache/movies/flipbook_VCN",
        oloth="cache/movies/flipbook_other"
    output:
        movie_raw="results/movies/Flipbook_uncompressed.mov",
        movie_hq="results/movies/Flipbook_HQ.mov",
        movie_mq="results/movies/Flipbook_MQ.mov",
        movie_lq="results/movies/Flipbook_LQ.mov"
    params:
        desc="cache/movies/flipbook.txt"
    run:
        import shutil
        shutil.copy2('src/movies/flipbook_title.png', 'cache/movies/flipbook_title.png')
        generate_flipbook_title()
        generate_movie_description(input.olin)
        generate_movie_description(input.olcn)
        generate_movie_description(input.olvpn)
        generate_movie_description(input.olvcn)
        generate_movie_description(input.oloth)
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 10 {output.movie_hq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 22 {output.movie_mq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 30 {output.movie_lq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -preset veryslow -tune stillimage -crf 0 {output.movie_raw}")
