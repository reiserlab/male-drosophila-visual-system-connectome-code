import dropbox
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import sys

import click

class TransferData:
    def __init__(self, access_token, target_path):
        self.access_token = access_token
        self.target_path = target_path

    def upload_file(self, file_from=None, file_to=None):

        dbx = dropbox.Dropbox(self.access_token)
        #files_upload(f, path, mode=WriteMode('add', None), autorename=False, client_modified=None, mute=False)

        with open(file_from, 'rb') as f:
            dbx.files_upload(f.read(), f"{self.target_path}{file_to}")
    
    def delete_folder(self, folder_to_delete=None):
        dbx = dropbox.Dropbox(self.access_token)
        dbx.files_delete(f"{self.target_path}{folder_to_delete}")

def check_results_path(rspath):
    """
    Stop exectution if anything but a results folder is uploaded

    This is a simple sanity check.
    """
    rootpath = Path(find_dotenv()).parent
    if not rspath.match(str(rootpath / 'results' / '*')):
        click.echo("Sorry, I can only upload files from result directory")
        sys.exit()

def get_delete_folder(rspath):
    """
    Get the name of the folder to be deleted from dropbox.
    """
    return rspath.relative_to(rspath.parent)


@click.command()
@click.option(
    "-C"
  , "--clean-only"
  , is_flag=True
  , help="Only clean the dropbox content, no upload."
)
@click.argument(
    "rspath",
    nargs=1,
    type=click.Path(
        exists=True
      , file_okay=False
      , dir_okay=True
      , resolve_path=True
      , readable=True
      , path_type=Path
    )
)
def cli(rspath, clean_only):
    load_dotenv()
    access_token = os.environ['DROPBOX_TOKEN']
    target_path = os.environ['DROPBOX_PROJECT_ROOT']
    transferData = TransferData(access_token, target_path)
    check_results_path(rspath)
    try:
        delpath = get_delete_folder(rspath)
        transferData.delete_folder(delpath)
        click.echo(f"Deleted folder '{target_path}{delpath}' from dropbox")
    except dropbox.exceptions.ApiError:
        click.echo("Dropbox folder did not exist.")
    if not clean_only:
        for fname in rspath.glob('**/*'):
            if fname.is_dir():
                continue
            target = fname.relative_to(rspath.parent)
            transferData.upload_file(fname, target)
            click.echo(f"Uploaded: {fname}")

if __name__ == '__main__':
    cli()