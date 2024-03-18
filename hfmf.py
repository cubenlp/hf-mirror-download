"""Main module."""
import os
import json
import requests
import click
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import unquote, quote, quote_plus

ROOT = "https://hf-mirror.com"

def get_next_page_items(soup, url):
    obj = soup.find_all('div', attrs={'data-target': "ViewerIndexTreeList"})
    data_props = json.loads(obj[0]['data-props'])
    current_items = data_props['entries']
    next_page_url = data_props['nextURL']
    if next_page_url is not None:
        data = requests.get(f"{ROOT}{next_page_url}").json()
        all_items = current_items + data
    else:
        all_items = current_items

    download_url = url.replace('tree/main', 'resolve/main')
    url2names = [(f"{download_url}/{item['path']}?download=true", item['path']) for item in all_items if item['type'] == 'file']
    return url2names

def get_url2names(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    a_tags = soup.find_all(title="Download file")
    url2names = [(ROOT + a_tag['href'], unquote(a_tag['href'].replace('?download=true', '').split('resolve/main/')[-1])) for a_tag in a_tags]
    all_url2names = get_next_page_items(soup, url)
    for item in url2names:
        assert item in all_url2names
    return all_url2names

def save_with_wget(url, file):
    os.system(f"wget -c {url} -O {file}")

@click.command()
@click.argument('url')
@click.option('--tgt_folder', default=None, help='Target folder to save the files. Defaults to model name.')
@click.option('--update', is_flag=True, help='Update existing files except weights.')
def download_from_mirror_page(url, tgt_folder, update):
    if not url.startswith(ROOT):
        assert url.startswith("https://huggingface.co"), "URL must be from hf-mirror.com or huggingface.co"
        url = url.replace("huggingface.co", "hf-mirror.com")
    url2names = get_url2names(url)
    if tgt_folder is None:
        tgt_folder = os.path.join(".", url.replace(ROOT + "/", '').replace('/tree/main', ''))
    os.makedirs(tgt_folder, exist_ok=True)
    click.echo(f'Saving to {tgt_folder}\n')
    for url, name in url2names:
        if update and os.path.exists(os.path.join(tgt_folder, name)):
            click.echo(f"Skipping {name} as it exists")
            continue
        save_with_wget(url, os.path.join(tgt_folder, name))

if __name__ == '__main__':
    download_from_mirror_page()
