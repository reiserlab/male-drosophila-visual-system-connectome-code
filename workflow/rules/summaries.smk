rule neurontype_summary:
    threads: 4
    output:
        pdf="results/fig_summary/Summary_Group-{counter}.pdf"
    run:
        shell("python src/fig_summary/neurontype_summary.py plot {wildcards.counter} --per-page 24")


rule all_summaries:
    input:
        expand("results/fig_summary/Summary_Group-{counter:02d}.pdf", counter=[idx for idx in range(0, 36)])


rule all_gallery_groups:
    run:
        shell("jupyter execute --kernel_name='python3' src/gallery_generation/generate_combined_optic-lobe.ipynb")

rule cell_type_catalog:
    run:
        shell("jupyter execute --kernel_name='python3' src/fig_summary/collate_summary_pages.ipynb")
