roi_hex_files = $(addprefix results/eyemap/,LO1_hex.pickle LO5_hex.pickle LOP1_hex.pickle LOP4_hex.pickle ME10_hex.pickle ME2_hex.pickle)

# -------------------------------------

.PHONY: update-dependencies install-dependencies eyemap clean completeness patterns weekly

reinstall-venv:
	@rm -rf .venv
	@python -m venv .venv

show-dependencies:
	@pip freeze | cut -d "=" -f1 | xargs pip show | grep -i "^name\|^version\|^requires"

$(roi_hex_files):
	@PYDEVD_DISABLE_FILE_VALIDATION=1 jupyter nbconvert --execute --to notebook --inplace src/eyemap/from_connections_to_grids.ipynb

eyemap: $(roi_hex_files)
	echo "Creating eyemap"

clean:
	rm -f $(roi_hex_files)
	rm -rf cache

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

