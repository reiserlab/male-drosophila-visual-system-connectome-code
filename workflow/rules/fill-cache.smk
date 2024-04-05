rule layer_boundaries:
    output:
        me="cache/eyemap/ME_layer_bdry.csv",
        lo="cache/eyemap/LO_layer_bdry.csv",
        lop="cache/eyemap/LOP_layer_bdry.csv"
    run:
        shell("jupyter execute --kernel_name='.venv' src/eyemap/create_layer_boundaries.ipynb")

rule mi1_t4_alignment:
    output:
        mi1="results/eyemap/mi1_t4_alignment.xlsx"
    run:
        shell("jupyter execute --kernel_name='.venv' src/eyemap/create_mi1_t4_alignment.ipynb")

rule pin_creation:
    output:
        me="cache/eyemap/ME_col_center_pins.pickle",
        lo="cache/eyemap/LO_col_center_pins.pickle",
        lop="cache/eyemap/LOP_col_center_pins.pickle"
    run:
        shell("jupyter execute --kernel_name='.venv' src/eyemap/create_column_pins.ipynb")

rule populate_cache:
    input:
        mi1="results/eyemap/mi1_t4_alignment.xlsx",
        me="cache/eyemap/ME_layer_bdry.csv",
        lo="cache/eyemap/LO_layer_bdry.csv",
        lop="cache/eyemap/LOP_layer_bdry.csv",
        mepi="cache/eyemap/ME_col_center_pins.pickle",
        lopi="cache/eyemap/LO_col_center_pins.pickle",
        loppi="cache/eyemap/LOP_col_center_pins.pickle"