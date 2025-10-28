#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

VERSION = "0.1.1"

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=8.0',  # 更新到支持 Py3.12 的最低版本
    'requests>=2.25.1',  # 明确最低支持版本
    'beautifulsoup4>=4.11.1'  # 明确最低支持版本
]

test_requirements = ['pytest>=7.0', ]  # 更新测试框架版本

setup(
    author="Qing",
    author_email='aqsz2526@outlook.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',  # 更新开发状态
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',   
        'Programming Language :: Python :: 3.13', 
    ],
    description="A command-line tool designed to streamline the process of downloading machine learning models and related files from the Hugging Face model hub mirror site.",
    entry_points={
        'console_scripts': [
            'hfmf=hf_mirror_fetch.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',  # 添加内容类型
    include_package_data=True,
    keywords='hf_mirror_fetch',
    name='hf_mirror_fetch',
    packages=find_packages(include=['hf_mirror_fetch', 'hf_mirror_fetch.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Qing25/hf_mirror_fetch',
    version=VERSION,
    zip_safe=False,
)
