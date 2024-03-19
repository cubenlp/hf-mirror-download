# hf-mirror-fetch

<div align="center">
    <a href="https://pypi.python.org/pypi/hf-mirror-fetch">
        <img src="https://img.shields.io/pypi/v/hf-mirror-fetch.svg" alt="PyPI version" />
    </a>
    <a href="https://github.com/cubenlp/hf-mirror-download/actions/workflows/test.yml">
        <img src="https://github.com/cubenlp/hf-mirror-download/actions/workflows/test.yml/badge.svg" alt="Tests" />
    </a>
    <a href="https://cubenlp.github.io/hf-mirror-download/">
        <img src="https://img.shields.io/badge/docs-github_pages-blue.svg" alt="Documentation Status" />
    </a>
    <a href="https://codecov.io/gh/cubenlp/hf-mirror-download">
        <img src="https://codecov.io/gh/cubenlp/hf-mirror-download/branch/main/graph/badge.svg" alt="Coverage" />
    </a>
</div>

[English](README-EN.md) | [简体中文](README.md)

`hf-mirror-fetch` 是一个命令行工具，使用镜像站 hf-mirror.com 下载 Hugging Face 模型。

## 安装

通过以下命令安装 `hf-mirror-fetch`：

```bash
pip install hf-mirror-fetch
```

## 使用方法

只需要提供模型页面的 URL 即可，示例：

```bash
hfmf https://huggingface.co/openchat/openchat_3.5/tree/main
hfmf https://huggingface.co/openchat/openchat_3.5
hfmf openchat/openchat_3.5
hfmf openchat/openchat_3.5 --tgt_folder /sshfs/pretrains/
```

参数说明：

- `-f`, `--tgt_folder`: （可选）保存下载文件的目标文件夹。如果未指定，将使用模型名创建一个文件夹。
- `--update`: （可选）更新现有文件，但不包括权重文件。

## 贡献

欢迎任何形式的贡献，无论是功能建议、代码改进还是文档更新。请通过 GitHub 的 Issue 或 Pull Request 提交您的贡献。

## 版权

该项目采用 MIT 许可证。有关更多信息，请查看 LICENSE 文件。
