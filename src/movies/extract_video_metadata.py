"""
This script is based on https://github.com/davidrazmadzeExtra/YouTube_Python3_Upload_Video. 

All credits go to @davidrazmadzeExtra, all mistakes are mine.
"""

import httplib2
import os
import random
import click
import sys
import warnings
import time
import datetime
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


from apiclient.discovery import build
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

from dotenv import load_dotenv, find_dotenv

load_dotenv()
PROJECT_ROOT = Path(
    find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.ol_types import OLTypes
from utils.instance_summary import InstanceSummary
from utils import olc_client

def get_authenticated_service(args):
    httplib2.RETRIES = 1
    secret_file = Path(find_dotenv()).parent / os.environ.get('YT_CREDENTIALS')
    flow = flow_from_clientsecrets(
        secret_file
      , scope="https://www.googleapis.com/auth/youtube.upload"
      , message="Missing authentication. Please check https://console.cloud.google.com/")

    storage = Storage("%s-oauth2.json" % sys.argv[0])
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

    return build("youtube", "v3",
                 http=credentials.authorize(httplib2.Http()))


def initialize_upload(youtube, options) -> str:
    tags = None
    if options.get('keywords'):
        tags = options.get('keywords').split(",")

    body = dict(
        snippet=dict(
            title=options.get('title'),
            description=options.get('description'),
            tags=tags,
            categoryId=options.get('category')
        ),
        status=dict(
            privacyStatus=options.get('privacyStatus')
        )
    )

    # Call the API's videos.insert method to create and upload the video.
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        # The chunksize parameter specifies the size of each chunk of data, in
        # bytes, that will be uploaded at a time. Set a higher value for
        # reliable connections as fewer chunks lead to faster uploads. Set a lower
        # value for better recovery on less reliable connections.
        #
        # Setting "chunksize" equal to -1 in the code below means that the entire
        # file will be uploaded in a single HTTP request. (If the upload fails,
        # it will still be retried where it left off.) This is usually a best
        # practice, but if you're using Python older than 2.6 or if you're
        # running on App Engine, you should set the chunksize to something like
        # 1024 * 1024 (1 megabyte).
        media_body=MediaFileUpload(options.get('file'), chunksize=-1, resumable=True)
    )

    return resumable_upload(insert_request)

# This method implements an exponential backoff strategy to resume a
# failed upload.


def resumable_upload(insert_request) -> str:
    MAX_RETRIES = 10
    response = None
    error = None
    retry = 0
    ret = None
    while response is None:
        try:
            print("Uploading file...")
            status, response = insert_request.next_chunk()
            if response is not None:
                if 'id' in response:
                    ret = response['id']
                else:
                    warnings.warn(f"The upload failed with an unexpected response: {response}")
        except HttpError as e:
            if e.resp.status in [500, 502, 503, 504]:
                error = "A retriable HTTP error %d occurred:\n%s" % (e.resp.status,
                                                                     e.content)
            else:
                raise
        except (httplib2.HttpLib2Error, IOError) as e:
            error = "A retriable error occurred: %s" % e

        if error is not None:
            warnings.warn(error)
            retry += 1
            if retry > MAX_RETRIES:
                return ret

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            warnings.warn(f"Sleeping {sleep_seconds} seconds and then retrying...")
            time.sleep(sleep_seconds)
    return ret


@click.command()
@click.argument(
    "video_fn"
  , type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True))
def file(
    video_fn
):
    options = {}
    video_f = Path(video_fn)
    if video_f.suffix != '.avi':
        warnings.warn("video format not supported")
        exit(10)
    c = olc_client.connect()
    olt = OLTypes()
    nlist = olt.get_neuron_list(side='both')
    prts = video_f.stem.split("_")
    # prts = [video_f.stem]

    desc = ""
    tags = []
    tlist = nlist[nlist['type']==prts[0]]\
        .sort_values(by='instance', ascending=False)\
        .reset_index()
    if len(tlist)>0:
        ins = []
        total_count = 0
        ntstr = ""
        for _, row in tlist.iterrows():
            cins = InstanceSummary(row['instance'])
            total_count += cins.count
            if cins.consensus_nt != 'unclear':
                ntstr = f"We identified '{cins.consensus_neurotransmitter}' as the consensus neurotransmitter for {row['type']}. "
            ins.append(cins)

        tags.append(tlist.loc[0,'type'])
        tags.append('white background')
        if ntstr != "":
            tags.append(ins[0].consensus_neurotransmitter)
            tags.append(ins[0].consensus_nt)

        abbrvs = {
            "OL_intrinsic": "OLIN"
          , "OL_connecting": "OLCN"
          , "VPN": "VPN"
          , "VCN": "VCN"
          , "other": ""
        }
        longs = {
            "OL_intrinsic": "Optic Lobe Intrinsic Neuron"
          , "OL_connecting": "Optic Lobe Connecting Neuron"
          , "VPN": "Visual Projection Neuron"
          , "VCN": "Visual Centrifugal Neuron"
          , "other": "other"
        }
        tags.append(abbrvs[tlist.loc[0,'main_groups']])
        tags.append(longs[tlist.loc[0,'main_groups']])
        tags.append("Drosophila")
        tags.append("visual system")
        tags.append("FIB-SEM")

        
        desc += f"The cell type '{tlist.loc[0,'type']}' has {total_count} cells in the optic-lobe:v1.0 data release. "
        # desc += ntstr
        desc += f"We assigned '{tlist.loc[0,'type']}' to the group '{longs[tlist.loc[0,'main_groups']]} ({abbrvs[tlist.loc[0,'main_groups']]})'."
        if len(ins)==1:
            desc += f"\n\nThe body ID of the star neuron is '{tlist.loc[0, 'star_neuron']}'"
            if len(ins[0].bids)>1:
                desc += f" which was selected from the following list of body IDs: {', '.join([str(x) for x in ins[0].bids])}"
            desc += "."
            
            # if len(ins[0].top5_upstream)>0 or len(ins[0].top5_downstream)>0:
            #     desc += f"\n\n{tlist.loc[0,'type']} connects to "
            #     if len(ins[0].top5_upstream)>0:
            #         upst = []
            #         for rwid, rw in ins[0].top5_upstream.iterrows():
            #             upst.append(f"{rw['instance']} ({rw['perc']:.0%})")
            #         desc += f"{', '.join(upst)} upstream"
            #     if len(ins[0].top5_upstream)>0 and len(ins[0].top5_downstream)>0:
            #         desc += " and "
            #     if len(ins[0].top5_downstream)>0:
            #         downst = []
            #         for rwid, rw in ins[0].top5_upstream.iterrows():
            #             downst.append(f"{rw['instance']} ({rw['perc']:.0%})")
            #         desc += f"{', '.join(downst)} downstream"
            #     desc += "."


        elif len(ins) > 1:
            desc += f"\n\nThe cell type {tlist.loc[0, 'type']} has cells in both hemispheres. "
            desc += f"While we focus our analysis on the right side, we include reconstructions from the left side instance (marked by '_L') for bilateral types."
            for idx, cins in enumerate(ins):
                desc += f"\n\nFor the instance {cins.instance_name} we identified {cins.count} neurons. "
                desc += f"The star neuron is {tlist.loc[idx, 'star_neuron']}"
                if len(cins.bids)>1:
                    desc += f", selected from these body IDs: {', '.join([str(x) for x in cins.bids])}"
                desc += "."

                # if len(cins.top5_upstream)>0 or len(cins.top5_downstream)>0:
                #     desc += f"\n\n{tlist.loc[0,'type']} connects to "
                #     if len(cins.top5_upstream)>0:
                #         upst = []
                #         for rwid, rw in cins.top5_upstream.iterrows():
                #             upst.append(f"{rw['instance']} ({rw['perc']:.0%})")
                #         desc += f"{', '.join(upst)} upstream"
                #     if len(cins.top5_upstream)>0 and len(cins.top5_downstream)>0:
                #         desc += " and "
                #     if len(cins.top5_downstream)>0:
                #         downst = []
                #         for rwid, rw in cins.top5_upstream.iterrows():
                #             downst.append(f"{rw['instance']} ({rw['perc']:.0%})")
                #         desc += f"{', '.join(downst)} downstream"
                #     desc += "."
    

    if len(prts) == 1:
        title = f"{prts[0]}, a Drosophila {longs[tlist.loc[0,'main_groups']]} from optic-lobe:v1.0"
    elif len(prts) == 2 and prts[1] == 'general':
        title = f"{prts[0]}, a Drosophila {longs[tlist.loc[0,'main_groups']]} from optic-lobe:v1.0 (short)"
        tags.append("short")
    else:
        warnings.warn("Video type not supported")

    # options['title'] = title
    # options['description'] = desc
    # options['keywords'] = ", ".join(tags)
    # options['privacyStatus'] = 'private'
    # options['category'] = 28
    # options['file'] = video_fn
    # youtube = get_authenticated_service(options)
    # video_id = initialize_upload(youtube, options)
    # with open("video_log.txt", 'a') as log_fh:
    #     log_fh.write(f"\n{video_fn}, {video_id}, {datetime.datetime.now().isoformat()}")
    print(title)
    print(desc)
    print(", ".join(tags))


@click.group()
def cli():
    """
    """

cli.add_command(file)


if __name__ == '__main__':
    cli()
    # argparser.add_argument("--file", required=True,
    #                        help="Video file to upload")
    # argparser.add_argument("--title", help="Video title", default="Test Title")
    # argparser.add_argument("--description", help="Video description",
    #                        default="Test Description")
    # argparser.add_argument("--category", default="22",
    #                        help="Numeric video category. " +
    #                        "See https://developers.google.com/youtube/v3/docs/videoCategories/list")
    # argparser.add_argument("--keywords", help="Video keywords, comma separated",
    #                        default="")
    # argparser.add_argument("--privacyStatus", choices=("public", "private", "unlisted"),
    #                        default=VALID_PRIVACY_STATUSES[0], help="Video privacy status.")
    # args = argparser.parse_args()
    # breakpoint()
    # if not os.path.exists(args.file):
    #     exit("Please specify a valid file using the --file= parameter.")

    # youtube = get_authenticated_service(args)
    # try:
    #     initialize_upload(youtube, args)
    # except HttpError as e:
    #     print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
