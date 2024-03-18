# -*- coding: utf-8 -*-
"""
mirror_download.py

A tool to download models from hf-mirror.com, catering to situations where huggingface.co is inaccessible.
This script provides functionalities to:
1. Parse all file links and names from an HTML page.
2. Retrieve complete file information in cases where not all items are loaded in a single go, through JSON.
3. Use wget for downloading files, supporting resume for interrupted downloads.

Example usage:
    python mirror_download.py --url https://huggingface.co/openchat/openchat_3.5/tree/main --tgt_folder my_model_folder --update

Requirements:
    requests, bs4, click
"""
import os
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import unquote, quote, quote_plus

ROOT = "https://hf-mirror.com"


def get_next_page_items(soup, url):
    """ For pages with too many items to load all at once, parse the json from another request to get the complete file list.
        ! This implementation assumes that clicking the `load more` button once is sufficient to load all items.
    """
    obj = soup.find_all('div',attrs={'data-target':"ViewerIndexTreeList"})
    data_props = json.loads(obj[0]['data-props'])
    current_items = data_props['entries']

    next_page_url = data_props['nextURL']
    if next_page_url is not None:  # If there's a next page
        data = requests.get(f"{ROOT}{next_page_url}").json()
        all_items = current_items + data
    else:
        all_items = current_items

    download_url = url.replace('tree/main', 'resolve/main')
    url2names = []
    for item in all_items:
        if item['type'] == 'file':
            name = item['path']
            _url = f"{download_url}/{name}?download=true"
            _url = quote(_url, safe=":/?=&")  # Encode spaces in file names with quote
            url2names.append((_url, name))
    return url2names

def save_with_wget(url, file):
    os.system(f"wget -c {url} -O {file}")

def get_url2names(url):
    """
        Fetch download links and filenames from the webpage.
    """

    print("="*50)
    print(f"Downloading {url}".center(50))
    print("="*50)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Parse all file links and names from the HTML
    a_tags = soup.find_all(title="Download file")
    if len(a_tags) == 0:
        print("!"*80)
        print("No files detected! Please check the input URL. Use the URL for `Files and versions` that contains `/tree/main`, not `model card`.")
        print("Exiting ...")
        exit()

    # unquote: filenames do not need URL encoding
    url2names = [
        (ROOT + a_tag['href'], unquote(a_tag['href'].replace('?download=true', '').split('resolve/main/')[-1]))
        for a_tag in a_tags
    ]

    # Retrieve all file information from JSON, including the links parsed from HTML
    all_url2names = get_next_page_items(soup, url)
    for item in url2names:
        assert item in all_url2names, f"{item} not in all_url2names"  # Verify consistency

    # Overwrite original url2names and print
    url2names = all_url2names
    print("="*50)
    for _url, name in url2names:
        print(f"{name:45} | {_url}")
    print("="*50)
    print(f"{len(url2names)} files in total!")
    return url2names

def download_from_mirror_page(URL, tgt_folder=None, update=True):
    """Download models from hf-mirror.com

    Args:
        url: Model link from huggingface.co or hf-mirror.com's resolve/main/ page
        tgt_folder: Save path. Defaults to None.
        update: If the repository is updated, update all files except the weights. Defaults to True. || todo: wget -c seems to already implement this functionality
    """
    # Check and modify URL if necessary
    if not URL.startswith(ROOT):
        assert URL.startswith("https://huggingface.co"), "make sure download from hf-mirror.com or huggingface.com"
        URL = URL.replace("huggingface.co", "hf-mirror.com")

    # Parse all file links and names from the URL
    url2names = get_url2names(URL)

    if tgt_folder is None:
        tgt_folder = URL.replace(ROOT+"/", '').replace('/tree/main', '')
        tgt_folder = os.path.join(".", tgt_folder)

    # Check if the folder exists, try to create
    try:
        if os.path.exists(tgt_folder):
            print(f"Folder {tgt_folder} already exists !!! continue download will overwrite the files in it. ")
        os.makedirs(tgt_folder, exist_ok=True)
    except Exception as e:
        print(e)
        print("Failed to create folder! Check if the model has been downloaded. Exiting ...")
        exit()
    # Confirm continuation of download
    flag = input(f'Saving to {tgt_folder}\n Continue downloading? Y/N\n').strip().lower()
    if flag == 'y':
        pass
    elif flag == 'n':
        print("Canceled. Exiting ...")
        return None
    else:
        raise Exception("Please answer y or n")


    # Download part
    for url, name in url2names:
        if url.endswith(".h5") or url.endswith(".ot") or url.endswith(".msgpack"):
            # Generally, only download .bin and .safetensors
            continue

        tgt_path = os.path.join(tgt_folder, name)
        if update and (url.endswith(".safetensors") or url.endswith(".bin")) and os.path.exists(tgt_path):
            print(f"Skip {name} because it exists")
            continue

        save_with_wget(url, tgt_path)
        # save_file(url, os.path.join(tgt_folder, name))
        # save_file_with_resume(url, os.path.join(tgt_folder, name))
