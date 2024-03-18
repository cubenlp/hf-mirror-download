# hf-mirror-fetch

[English](README-EN.md) | [简体中文](README.md)

`hf-mirror-fetch` 是一个命令行工具，旨在简化从 Hugging Face 模型镜像站点下载机器学习模型和相关文件的过程。这个工具特别适用于在访问 Hugging Face 主站受限或连接速度较慢的环境中。

## 安装

确保您已安装 Python 3.6 或更高版本。您可以通过以下命令安装 `hf-mirror-fetch`：

```bash
pip install hf-mirror-fetch
```

## 使用方法

使用 `hf-mirror-fetch` 下载模型非常简单，只需要提供模型页面的 URL 即可。以下是一个使用示例：

```bash
hfmf -u https://huggingface.co/openchat/openchat_3.5/tree/main
```

### 参数说明

- `-u`, `--url`: 模型页面在 hf-mirror.com 或 huggingface.co 上的 URL。
- `-f`, `--tgt_folder`: （可选）保存下载文件的目标文件夹。如果未指定，将使用模型名创建一个文件夹。
- `--update`: （可选）更新现有文件，但不包括权重文件。

## 贡献

欢迎任何形式的贡献，无论是功能建议、代码改进还是文档更新。请通过 GitHub 的 Issue 或 Pull Request 提交您的贡献。

## 版权

该项目采用 MIT 许可证。有关更多信息，请查看 LICENSE 文件。
