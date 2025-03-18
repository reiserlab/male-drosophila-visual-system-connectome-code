include: "../scripts/website.py"

rule generate_website_cover:
    output:
        touch("cache/html_pages/done/cover.done"),
        "results/html_pages/index.html",
        "results/html_pages/cell_types.html",
        "results/html_pages/webpages_glossary.html"
    shell:
        """
        jupyter execute --kernel_name='python3' src/html_pages/make_index_and_cover_page.ipynb
        """

rule generate_website_scatter:
    output:
        touch("cache/html_pages/done/scatter.done")
    shell:
        """
        jupyter execute --kernel_name='python3' src/html_pages/make_scatterplot_html_pages.ipynb
        """

rule generate_website:
    input:
        cover = "cache/html_pages/done/cover.done",
        scatter = "cache/html_pages/done/scatter.done",
        website = get_all_neurons

rule get_one_page:
    input:
        scatter = "cache/html_pages/done/scatter.done"
    output:
        html="results/html_pages/{instance}.html"
    run:
        shell("python src/html_pages/make_single_page.py {wildcards.instance}")
