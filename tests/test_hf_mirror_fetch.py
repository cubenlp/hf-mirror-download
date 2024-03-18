#!/usr/bin/env python

"""Tests for `hf_mirror_fetch` package."""

import pytest

from click.testing import CliRunner

from hf_mirror_fetch import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the Command Line Interface (CLI)."""
    runner = CliRunner()

    # Mock URL for testing
    mock_url = "https://huggingface.co/openchat/openchat_3.5/tree/main"

    # Testing the main command with required URL argument
    result = runner.invoke(cli.main, ['--url', mock_url], input="N\n")
    print(result.output)  # To help debugging
    assert result.exit_code == 0, f"Command exited with {result.exit_code} and output: {result.output}"

    # Testing the help option to ensure CLI setup is correct
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0, "Help command failed"
