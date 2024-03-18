#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

VERSION = "0.1.0"

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=7.0',
    'requests',
    'beautifulsoup4'
]

test_requirements = ['pytest>=3', ]

setup(
    author="Qing",
    author_email='aqsz2526@outlook.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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
