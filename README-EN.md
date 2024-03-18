# hf-mirror-fetch

[English](README-EN.md) | [简体中文](README.md)

`hf-mirror-fetch` is a command-line tool designed to streamline the process of downloading machine learning models and related files from the Hugging Face model hub mirror site. This tool is particularly useful in environments where access to the primary Hugging Face repository is restricted or where connectivity to the main site is slow.

## Installation

Ensure you have Python 3.6 or higher installed. You can install `hf-mirror-fetch` using the following command:

```bash
pip install hf-mirror-fetch
```

## Usage

Using `hf-mirror-fetch` to download models is straightforward; you only need to provide the URL of the model page. Here is an example of how to use it:

```bash
hfmf -u https://huggingface.co/openchat/openchat_3.5/tree/main
```

### Parameters

- `-u`, `--url`: The URL of the model's page on hf-mirror.com or huggingface.co.
- `-f`, `--tgt_folder`: (Optional) The target folder to save the downloaded files. If not specified, a folder named after the model will be created.
- `--update`: (Optional) Update existing files except for weights.

## Contributing

Contributions of any kind are welcome, whether they are feature suggestions, code improvements, or documentation updates. Please submit your contributions through GitHub Issues or Pull Requests.

## License

This project is licensed under the MIT License. For more details, please check the LICENSE file.