rule generate_website_cover:
    output:
        touch("cache/html_pages/done/cover.done"),
        "results/html_pages/index.html",
        "results/html_pages/cell_types.html",
        "results/html_pages/webpages_glossary.html"
    shell:
        """
        jupyter execute --kernel_name='.venv' src/html_pages/make_index_and_cover_page.ipynb
        """

rule generate_website_scatter:
    output:
        touch("cache/html_pages/done/scatter.done")
    shell:
        """
        jupyter execute --kernel_name='.venv' src/html_pages/make_scatterplot_html_pages.ipynb
        """

rule generate_website_individual:
    output:
        touch("cache/html_pages/done/websites.done")
    threads: 48
    shell: 
        """
        jupyter execute --kernel_name='.venv' src/html_pages/generate_html-pages_from_scratch.ipynb
        """

rule generate_website:
    input:
        "cache/html_pages/done/cover.done",
        "cache/html_pages/done/scatter.done",
        "cache/html_pages/done/websites.done"

