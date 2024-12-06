def get_all_movie_previews(wildcards):
    import glob
    jsons = glob.glob("results/movie-descriptions/*.json")
    return [f"results/movies/{Path(fn).stem}-preview.avi" for fn in jsons]

def get_all_movies(wildcards):
    import glob
    jsons = glob.glob("results/movie-descriptions/*.json")
    return [f"results/movies/{Path(fn).stem}.avi" for fn in jsons]

def get_all_full_brain_configs(wildcards):
    import glob
    jsons = glob.glob("results/gallery-descriptions/Full-Brain*json")
    return [f"cache/gallery/done/render/{Path(fn).stem}-gal.done" for fn in jsons]

def get_all_full_brain_configs_blend(wildcards):
    import glob
    jsons = glob.glob("results/gallery-descriptions/Full-Brain*json")
    return [f"cache/gallery/done/render/{Path(fn).stem}-galb.done" for fn in jsons]

def get_all_optic_lobe_configs(wildcards):
    import glob
    jsons = glob.glob("results/gallery-descriptions/Optic-Lobe*json")
    return [f"cache/gallery/done/render/{Path(fn).stem}-gal.done" for fn in jsons]

def get_all_optic_lobe_configs_blend(wildcards):
    import glob
    jsons = glob.glob("results/gallery-descriptions/Optic-Lobe*json")
    return [f"cache/gallery/done/render/{Path(fn).stem}-galb.done" for fn in jsons]

def get_all_texts(wildcards):
    import glob
    jsons = glob.glob("results/gallery-descriptions/*json")
    return [f"cache/gallery/done/text/{Path(fn).stem}.done" for fn in jsons]

def rename_cache_movie_files(directory):
    import glob
    import shutil
    for fn in Path(directory).iterdir():
        basename = str(fn.name).lower()
        newname = re.sub(r'(\d+)', lambda x: f'{x.group(1).zfill(5)}', basename)
        shutil.move(fn, fn.parent / newname)

def generate_flipbook_title():
    outfile = Path("cache/movies/flipbook.txt")
    with open(outfile, "w") as outfh:
        outfh.write("file 'flipbook_title.png'\n")
        outfh.write("duration 7\n")

def generate_traditional_flipbook_title():
    outfile = Path("cache/movies/flipbook.txt")
    with open(outfile, "w") as outfh:
        outfh.write("file 'flipbook_title.png'\n")
        outfh.write("duration 7\n")

def generate_movie_description(directory):
    outfile = Path("cache/movies/flipbook.txt")
    duration = 0.4
    myline = ""
    mfiles = [f for f in Path(directory).glob('*.png')]
    mfiles.sort()
    for fn in mfiles:
        myline += f"file '{fn.relative_to(outfile.parent)}'\nduration {duration}\n"
    with open(outfile, "a") as outfh:
        outfh.write(myline)

def generate_traditional_movie_description(directory):
    outfile = Path("cache/movies/flipbook_trad.txt")
    duration = 0.4
    myline = ""
    mfiles = [f for f in Path(directory).glob('*.png')]
    mfiles = [f for f in mfiles if "_all" not in str(f)]
    mfiles.sort()
    for fn in mfiles:
        myline += f"file '{fn.relative_to(outfile.parent)}'\nduration {duration}\n"
    with open(outfile, "a") as outfh:
        outfh.write(myline)
