# -*- coding: utf-8 -*-
# @File    :   mirror_download.py
# @Time    :   2023/12/03 21:56:47
# @Author  :   Qing
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    由于 huggingface.co无法直接访问，所以从 hf-mirror.com 下载模型

    提供 file and version 页面的 url，下载所有文件
    eg: https://huggingface.co/openchat/openchat_3.5/tree/main

    1. 从 html 中解析出所有文件的链接和文件名
    2. 对于一次性没有加载完的情况，从 json 中获取所有文件的信息, 也包括上面从 html 中解析出来的链接
    3. 调用wget -c 下载所有文件，中断后可以继续下载

'''

import os
import json
try:
  import fire
except:
  raise Exception("fire not installed, run `pip install fire`")
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import unquote, quote, quote_plus

ROOT = "https://hf-mirror.com"


def get_next_page_items(soup, url):
    """ 页面 item 数量太多，一次性没有加载完全，通过再次请求解析 json 得到完整文件列表
        ! 当前实现假设只需要点击一次 `load more` 按钮即可加载完全
    """
    obj = soup.find_all('div',attrs={'data-target':"ViewerIndexTreeList"})
    data_props = json.loads(obj[0]['data-props'])
    current_items = data_props['entries']

    next_page_url = data_props['nextURL']
    if next_page_url is not None:  # 有下一页
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
            _url = quote(_url, safe=":/?=&")                    # 文件名中存在空格, 使用 quote 编码
            url2names.append((_url, name))
    return url2names


def save_with_wget(url, file):
    os.system(f"wget -c {url} -O {file}")

def get_url2names(url):
    """
        获取网页中的下载链接和文件名
    """

    print("="*50)
    print(f"Downloading {url}".center(50))
    print("="*50)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 从 html 中解析出所有文件的链接和文件名
    a_tags = soup.find_all(title="Download file")
    if len(a_tags) == 0 :
        print("!"*80)
        print("No files detected! please check the input url. NOT `model card`, use url for `Files and versions` that contains `/tree/main`.")
        print("Exiting ...")
        exit()

    # unquote: 文件名不需要url encoding
    url2names = [
        (ROOT + a_tag['href'], unquote(a_tag['href'].replace('?download=true', '').split('resolve/main/')[-1]))
        for a_tag in a_tags
    ]

    # 从 json 中获取所有文件的信息, 包括上面从 html 中解析出来的链接
    all_url2names = get_next_page_items(soup, url)
    for item in url2names:
        assert item in all_url2names, f"{item} not in all_url2names" # 验证一下两者的一致性


    # 覆盖原来的 url2names 并打印出来
    url2names = all_url2names
    print("="*50)
    for _url, name in url2names:
        print(f"{name:45} | {_url}")
    print("="*50)
    print(f"{len(url2names)} files in total!")
    return url2names



def download_from_mirror_page(URL, tgt_folder=None, update=True):
    """从hf-mirror.com下载模型

    Args:
        url: 模型链接 huggingface.co 或者 hf-mirror.com 的 resolve/main/ 页面
        tgt_folder: 保存路径. Defaults to None.
        update: 仓库有更新的话，更新所有权重以外的文件. Defaults to True. || todo wget -c 好像已经实现了这个功能
    """
    # 检查 url 是否正确 并修改
    if not URL.startswith(ROOT):
        assert URL.startswith("https://huggingface.co"), "make sure download from hf-mirror.com or huggingface.com"
        URL = URL.replace("huggingface.co", "hf-mirror.com")

    # 从 url 中解析出所有文件的链接和文件名
    url2names = get_url2names(URL)

    # import pdb;pdb.set_trace();
    if tgt_folder is None:
        tgt_folder = URL.replace(ROOT+"/", '').replace('/tree/main', '')

        tgt_folder = os.path.join(".", tgt_folder)

    # 判断文件夹是否存在，尝试创建
    try:
        if os.path.exists(tgt_folder):
            print(f"Folder {tgt_folder} already exists !!! continue download will overwrite the files in it. ")
        os.makedirs(tgt_folder, exist_ok=True)
    except Exception as e:
        print(e)
        print("Failed to create folder! Check if the model has been downloaded. Exiting ...")
        exit()
    # 是否继续下载
    flag = input(f'saving to {tgt_folder}\n Continue downloading? Y/N\n').strip().lower()
    if flag == 'y':
        pass
    elif flag == 'n':
        print("Canceled. Exiting ...")
        exit()
    else:
        raise Exception("y or n")


    # download part
    for url, name in url2names:
        if url.endswith(".h5") or url.endswith(".ot") or url.endswith(".msgpack"):
            # 一般只下载 .bin 和 .safetensors
            continue

        tgt_path = os.path.join(tgt_folder, name)
        if update and (url.endswith(".safetensors") or url.endswith(".bin")) and os.path.exists(tgt_path):
            print(f"Skip {name} because it exists")
            continue

        save_with_wget(url, tgt_path)
        # save_file(url, os.path.join(tgt_folder, name))
        # save_file_with_resume(url, os.path.join(tgt_folder, name))

if __name__ == '__main__':
    fire.Fire(download_from_mirror_page)

    # get_url2names('https://huggingface.co/openchat/openchat_3.5/tree/main') # no next page
    # get_url2names('https://huggingface.co/Qwen/Qwen-72B-Chat/tree/main')       # 99 items
    # download_from_mirror_page("https://huggingface.co/microsoft/phi-1_5/tree/main")
