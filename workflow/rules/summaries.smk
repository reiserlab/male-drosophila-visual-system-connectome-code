rule neurontype_summary:
    threads: 4
    resources: mem_mb=get_mem_mb
    output:
        pdf="results/fig_summary/Summary_Group-{counter}.pdf"
    run:
        shell("python src/fig_summary/neurontype_summary.py plot {wildcards.counter} --per-page 24")


rule all_summaries:
    input:
        expand("results/fig_summary/Summary_Group-{counter:02d}.pdf", counter=[idx for idx in range(0, 35)])