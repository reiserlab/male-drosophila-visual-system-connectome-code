from dotenv import load_dotenv
load_dotenv()

envvars:
    "BLENDER_PATH",
    "NEUVID_PATH"

include: "../scripts/get-mem.py"
include: "../scripts/movie-paths.py"

rule crossection_standard:
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
        python src/movies/neuropil-crossections.py standard > {log.stdout} 2> {log.stderr}
        """

rule crossection_connecting:
    output:
        directory("cache/blender/crossections_connecting/"),
        expand("cache/blender/crossections_connecting/ME_R_layer_{mel:02d}.obj", mel=[i for i in range(1,11)]),
        expand("cache/blender/crossections_connecting/LO_R_layer_{lol}.obj", lol=[i for i in range(1,8)]),
        expand("cache/blender/crossections_connecting/LOP_R_layer_{lopl}.obj", lopl=[i for i in range(1,5)])
    log: 
        stdout="logs/crossection_connecting.out.log", 
        stderr="logs/crossection_connecting.err.log"
    priority: 1
    shell:
        """
        python src/movies/neuropil-crossections.py connecting > {log.stdout} 2> {log.stderr}
        """

rule crossection:
    input:
        rules.crossection_standard.output,
        rules.crossection_connecting.output

rule blendfile:
    input:
        json="results/movie-descriptions/{movie}.json",
        cs="cache/blender/crossections/",
        csc="cache/blender/crossections_connecting/"
    output: 
        blender="cache/blender/{movie}.blend"
    threads: 3
    priority: 2
    params:
        cache= "/nrs/reiser/neuvid/{movie}-object-cache/" if os.environ.get('LSB_JOBID') else "cache/blender/{movie}-object-cache/",
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
        path = directory("/nrs/reiser/neuvid/{movie}_{width}x{height}-frames/") if os.environ.get('LSB_JOBID') else directory("cache/blender/{movie}_{width}x{height}-frames/"),
        flag = touch("cache/blender/.render_{movie}_{width}x{height}.done")
    threads: 5
    resources: mem_mb=get_mem_mb
    priority: 4
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else "",
        render_threads = 36
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/render.py -- --skipExisting --persist --threads {params.render_threads} --resX {wildcards.width} --resY {wildcards.height} -i {input.json} {params.optix} --inputBlender {input.blenderAnim} --white --output {output.path}
        """

rule addtext:
    input:
        json="results/movie-descriptions/{movie}.json",
        path="/nrs/reiser/neuvid/{movie}_{width}x{height}-frames/" if os.environ.get('LSB_JOBID') else "cache/blender/{movie}_{width}x{height}-frames/",
        flag="cache/blender/.render_{movie}_{width}x{height}.done"
    threads: 5
    resources: mem_mb=get_mem_mb
    priority: 5
    output:
        path= directory("/nrs/reiser/neuvid/{movie}_{width}x{height}-labeled/") if os.environ.get('LSB_JOBID') else directory("cache/blender/{movie}_{width}x{height}-labeled/"),
        flag=touch("cache/blender/.label_{movie}_{width}x{height}.done")
    params:
        blender_path = os.environ['BLENDER_PATH'],
        neuvid_path = os.environ['NEUVID_PATH'],
        optix = "--optix" if os.environ.get('LSB_JOBID') else ""
    shell:
        """
        {params.blender_path} --background --python {params.neuvid_path}/neuVid/compLabels.py -- --threads {threads} --input {input.json} --inputFrames {input.path} --output {output.path}
        """

rule assembleframes:
    input:
        json="results/movie-descriptions/{movie}.json",
        path="/nrs/reiser/neuvid/{movie}_{width}x{height}-labeled/" if os.environ.get('LSB_JOBID') else "cache/blender/{movie}_{width}x{height}-labeled/",
        flag="cache/blender/.label_{movie}_{width}x{height}.done"
    resources: mem_mb=get_mem_mb
    priority: 5
    output:
        path=directory("/nrs/reiser/neuvid/{movie}_{width}x{height}-assembled/") if os.environ.get('LSB_JOBID') else directory("cache/blender/{movie}_{width}x{height}-assembled/"),
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
        shell("jupyter execute --kernel_name='python3' src/movies/generate_movies.ipynb")


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
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_full_brain_stars_flipbook.ipynb")


# Required GNU parallel and imagemagick
rule generate_flipbook_cache:
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
rule generate_flipbook_all:
    input:
        olin="cache/movies/flipbook_OL_intrinsic",
        olcn="cache/movies/flipbook_OL_connecting",
        olvpn="cache/movies/flipbook_VPN",
        olvcn="cache/movies/flipbook_VCN",
        oloth="cache/movies/flipbook_other"
    output:
        movie_raw="results/movies/Flipbook_uncompressed.mp4",
        movie_hq="results/movies/Flipbook_HQ.mp4",
        movie_mq="results/movies/Flipbook_MQ.mp4",
        movie_lq="results/movies/Flipbook_LQ.mp4"
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
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 26 {output.movie_mq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 30 {output.movie_lq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -preset veryslow -tune stillimage -crf 0 {output.movie_raw}")


# requires ffmpeg
rule generate_flipbook_traditional:
    input:
        olin="cache/movies/flipbook_OL_intrinsic",
        olcn="cache/movies/flipbook_OL_connecting",
        olvpn="cache/movies/flipbook_VPN",
        olvcn="cache/movies/flipbook_VCN",
        oloth="cache/movies/flipbook_other"
    output:
        movie_raw="results/movies/Flipbook_uncompressed_traditional.mp4",
        movie_hq="results/movies/Flipbook_HQ_traditional.mp4",
        movie_mq="results/movies/Flipbook_MQ_traditional.mp4",
        movie_lq="results/movies/Flipbook_LQ_traditional.mp4"
    params:
        desc="cache/movies/flipbook_trad.txt"
    run:
        import shutil
        shutil.copy2('src/movies/flipbook_title.png', 'cache/movies/flipbook_title.png')
        generate_traditional_flipbook_title()
        generate_traditional_movie_description(input.olin)
        generate_traditional_movie_description(input.olcn)
        generate_traditional_movie_description(input.olvpn)
        generate_traditional_movie_description(input.olvcn)
        generate_traditional_movie_description(input.oloth)
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 10 {output.movie_hq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 22 {output.movie_mq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -pix_fmt yuv420p -preset veryslow -tune stillimage -profile:v high422 -crf 30 {output.movie_lq}")
        shell("ffmpeg -f concat -i {params.desc} -vcodec libx264 -preset veryslow -tune stillimage -crf 0 {output.movie_raw}")