"""Console script for hf_mirror_fetch."""
import sys
import click
from hf_mirror_fetch.mirror_download import get_url2names, download_from_mirror_page

def process_url(url):
    ROOT = "https://hf-mirror.com"
    HF = "https://huggingface.co"
    # If URL lacks a known prefix, assume it's intended for Hugging Face
    if not url.startswith((ROOT, HF)):
        assert not url.startswith("http"), "Invalid URL. Please provide a valid URL from hf-mirror.com or huggingface.co."
        url = f"{HF}/{url}"  # Prepend with Hugging Face domain
    if url.startswith(HF):
        url = url.replace(HF, ROOT)
    # Ensure URL ends with /tree/main, append if necessary
    if not url.endswith("/tree/main"):
        # Append correctly depending on whether URL already ends with a slash
        url = f"{url.rstrip('/')}/tree/main"
    return url


@click.command()
@click.argument('link', help='The URL of the model\'s page on hf-mirror.com or huggingface.co.')
@click.option('-f', '--tgt_folder', default=None, type=str, help='The target folder to save the downloaded files.')
@click.option('--update', is_flag=True, help='Update existing files except for weights.')
def main(link, tgt_folder, update):
    """Console script for hf_mirror_fetch."""
    assert link, "Please provide a URL."
    url = process_url(link)
    download_from_mirror_page(url, tgt_folder, update)
    return 0
