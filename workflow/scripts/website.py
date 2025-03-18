from utils.ol_types import OLTypes

def get_all_neurons(wildcards):
    from utils import olc_client
    c = olc_client.connect(verbose=False)
    olt = OLTypes()
    cell_type_list = olt.get_neuron_list(side='both')
    ci_list = cell_type_list['instance'].to_list()
    file_list = [f"results/html_pages/{f}.html" for f in ci_list]
    return file_list