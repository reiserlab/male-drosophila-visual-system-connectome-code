# Getting started with Dropbox Upload

[Git](git-getting-started.md) is not meant to host result files, especially binary results (like pdf or png files) are outside of what git can reliably handle. During the development of the project, we had access to several other services that are more geared towards this type of files. Yet, for convenience reasons we kept storing results from the `results` folder in git and uploaded them to GitHub. With the `src/utils/upload.py` script we now have a first draft that allows sciptable upload to at least one of the services: Dropbox.

## How it should work

The script `upload.py` takes an existing path inside the directory `results` and mirrors it to a Dropbox folder. 

For example, running `python src/utils/upload.py results/mytestfolder/` should create or replace the existing folder `<DROPBOX>/OL_CONNECTOME/results/mytestfolder/` and copy all content of the local folder to the Dropbox folder.

> Note: To avoid sharing the same folder on GitHub as well, please add that folder to your `.gitignore` file, too.





## Configuration

You will need to set two variables in your `.env` files: `DROPBOX_PROJECT_ROOT` and `DROPBOX_TOKEN`. 

The first defines the target folder where the results folder will be stored. It is recommended to not use the main OL Connectome folder, but a subfolder here. In our case we have a folder called `OL_CONNECTOME` on dropbox, which is the first part of the path, and I specify the results folder `results` inside this folder as a target to upload the files to.

```sh
DROPBOX_PROJECT_ROOT='/OL_CONNECTOME/results/'
```

Using the above `DROPBOX_PROJECT_ROOT` and specifying `python src/utils/upload.py results/mytestfolder` will create a folder `OL_CONNECTOME/results/mytestfolder` on Dropbox.

Note: the location of your dropbox folder may vary.

The other variable holds your OAuth2 access token. Try to follow the [Dropbox documentation](https://developers.dropbox.com/oauth-guide) on how to create that token.

