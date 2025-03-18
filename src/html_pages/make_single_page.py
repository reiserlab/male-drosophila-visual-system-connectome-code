import click

from html_pages.patterns import generate_single_page

from utils.ol_types import OLTypes
from utils import olc_client


@click.command()
@click.argument('instance')
def cli(instance:str=""):
    """
    Command line tool to generate the html website and dynamic images for a single instance.

    To generate the html website for Mi1 on the right hemisphere (for example), call 
    `make_single_page.py Mi1_R`.

    This command line tool allows an easy distribution across many nodes in a cluster environment.
    To generate all html websites, call `snakemake generate_website`, which will use this script
    to generate the websites for all instances.
    """
    c = olc_client.connect(verbose=True)
    olt = OLTypes()
    cell_type_list = olt.get_neuron_list(side='both')
    linked_instance = set(cell_type_list['instance'].to_list())

    available_tags = []
    for _, row in cell_type_list.iterrows():
        link_to_instance = row['instance']
        filename = f"{row['type']} ({link_to_instance[-1]})"
        tag = {"value": filename, "url": f"{link_to_instance}.html"}
        if tag not in available_tags:
            available_tags.append(tag)
    generate_single_page(
        instance_name=instance
      , available_tags=available_tags
      , linked_instance=linked_instance)


if __name__ == '__main__':
    cli()
