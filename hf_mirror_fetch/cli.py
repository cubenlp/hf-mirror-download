"""Console script for hf_mirror_fetch."""
import sys
import click
from hf_mirror_fetch.mirror_download import get_url2names, download_from_mirror_page


@click.command()
@click.option('-u', '--url', required=True, help='The URL of the model\'s page on hf-mirror.com or huggingface.co.')
@click.option('-f', '--tgt_folder', default=None, type=str, help='The target folder to save the downloaded files.')
@click.option('--update', is_flag=True, help='Update existing files except for weights.')
def main(url, tgt_folder, update):
    """Console script for hf_mirror_fetch."""
    download_from_mirror_page(url, tgt_folder, update)
    return 0
