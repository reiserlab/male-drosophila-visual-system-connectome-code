""" OLC Client module """
import warnings
import os
from neuprint import Client


def connect(
    verbose: bool = False,
    instance: str = None,
    dataset: str = None
) -> Client:
    """Return a neuprint.Client using the environments extracted from .env file and set it as
    the default client.

    Parameters
    ----------
    verbose :   bool, default=False
                The function prints some verbose output after a successful connection.
    instance :  str, default=None
                Provide the URL to a neuprint instance. If no instance is given (the default), the content of the environment variable NEUPRINT_SERVER_URL is used.
    dataset :   str, default=None
                If an instance URL is given as a parameter, the `dataset` is considered as the name of the neuprint dataset. If no dataset is give, the content of the environment variable NEUPRINT_DATASET_NAME is used.

    Returns
    -------
    neuprint.Client
                A client that connects to the configured server.

    """
    client = None
    if instance:
        np_url = instance
    else:
        np_url = os.environ['NEUPRINT_SERVER_URL']
        if dataset:
            warnings.warn(f"You are using the default instance '{np_url}' from the environment variable, but specified an alternative data set '{dataset}'. Make sure this is what you want.")

    if dataset:
        np_dta = dataset
    else:
        np_dta = os.environ['NEUPRINT_DATASET_NAME']
        if instance: 
            warnings.warn(f"You specified an alternative instance '{instance}' but use the default dataset '{np_dta}' from the environment variable. That can lead to unexpected results and you might need to specify a dataset, too.")

    try:
        client = Client(
            np_url,
            dataset=np_dta)
    except RuntimeError:
        warnings.warn(f"""
            The connection for instance {np_url} and dataset {np_dta} doesn't work. Please follow the documentation at
            https://connectome-neuprint.github.io/neuprint-python/docs/index.html
            to set up your client.
            """)
        return False
    if verbose and isinstance(client, Client):
        np_version = client.fetch_version()
        np_profile = client.fetch_profile()
        print(f"Connected to https://{np_url}[{np_dta}].")
        print(f"Client: neuprint-python v{np_version}")
        print(f"User: {np_profile['Email']} [{np_profile['AuthLevel']}]\n")
    return client
