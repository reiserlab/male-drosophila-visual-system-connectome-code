roi_hex_files = $(addprefix results/eyemap/,LO1_hex.pickle LO5_hex.pickle LOP1_hex.pickle LOP4_hex.pickle ME10_hex.pickle ME2_hex.pickle)

# -------------------------------------

.PHONY: update-dependencies install-dependencies eyemap clean completeness patterns weekly

reinstall-venv:
	@rm -rf .venv
	@python -m venv .venv

update-dependencies:
	@PYDEVD_DISABLE_FILE_VALIDATION=1
	@pip install --upgrade pip
	@cat requirements.txt | cut -d"=" -f1 | cut -d"@" -f2 | xargs pip install -U
# This ones gets only the packages that need updating, but runs into dependency issues:
#	@pip list --outdated --format=columns | grep -v "^--\|^Package" | cut -d' ' -f1  | xargs -n1 pip install -U

install-dependencies:
	@PYDEVD_DISABLE_FILE_VALIDATION=1
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@python -m ipykernel install --user --name .venv --display-name "ol-connectome"

freeze-dependencies:
	@pip freeze | grep -v "posix-ipc" > requirements.txt

show-dependencies:
	@pip freeze | cut -d "=" -f1 | xargs pip show | grep -i "^name\|^version\|^requires"

$(roi_hex_files):
	@PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --execute --to notebook --inplace src/eyemap/from_connections_to_grids.ipynb

eyemap: $(roi_hex_files)
	echo "Creating eyemap"

clean:
	rm -f $(roi_hex_files)
	rm -rf cache

completeness:
	@PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --execute --to notebook --inplace src/completeness/connection-completeness_named.ipynb

screenshots:
	@PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --execute --to notebook --inplace src/patterns/make_hex_images.ipynb
#	@jupyter nbconvert --execute --to notebook --inplace src/patterns/make_gallery.ipynb
#	ffmpeg -f image2 -framerate 1 -i Mi1/combined_Mi1-hex2_%02d.png -vf framerate=fps=30 Mi1-hex2.mov

patterns:
	@rm -rf results/patterns/*html
	@PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --execute --to notebook --inplace src/patterns/make_in_out_html_pages.ipynb
	@jupyter nbconvert --clear-output --inplace src/patterns/make_in_out_html_pages.ipynb
	@python src/utils/upload.py results/patterns

weekly: completeness patterns


.PHONY: cluster-previews
cluster-previews:
	source .venv/bin/activate
	bsub -n1 -q local -J previews 'snakemake --profile lsf -k --rerun-incomplete allpreviews'

.PHONY: cluster-previews
cluster-movies:
	source .venv/bin/activate
	bsub -n1 -q local -J previews 'snakemake --profile lsf -k --rerun-incomplete allmovies'


.PHONY: clean-all
clean-all: clean-cache clean-results

## Clean the caches
.PHONY: clean-cache
clean-cache: clean-cache-meshes clean-cache-skeleton clean-cache-rois clean-cache-clustering clean-cache-html clean-coverage clean-summary
	@rm -rf cache/gallery/
	@rm -rf cache/fig_summary_preprocess

.PHONY: clean-cache-skeleton
clean-cache-skeleton:
	@rm -rf cache/skeletons

.PHONY: clean-cache-meshes
clean-cache-meshes:
	@rm -rf cache/meshes

.PHONY: clean-cache-rois
clean-cache-rois:
	@rm -rf cache/rois

.PHONY: clean-cache-clustering
clean-cache-clustering:
	@rm -rf cache/clustering

.PHONY: clean-cache-html
clean-cache-html:
	@rm -rf cache/html_pages

.PHONY: clean-coverage
clean-coverage:
	@rm -rf cache/cov_compl
	@rm -rf cache/coverage_synapses
	@rm -rf cache/coverage_cells
	@rm -rf cache/complete_metrics
	@rm -rf cache/columns

.PHONY: clean-summary
clean-summary:
	@rm -rf cache/fig_summary

.PHONY: clean-results
clean-results: clean-results-html
	@rm -rf results/gallery-descriptions/*
	@rm -rf results/movie-descriptions/*

.PHONY: clean-results-html
clean-results-html:
	@rm -f results/html_pages/*html
	@rm -rf results/html_pages/img
	@rm -rf results/html_pages/scatterplots/

