#!/usr/bin/env python

"""Tests for `imkar` package."""

import pytest


from imkar import imkar  # noqa


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/mberz/cookiecutter-pypackage')


def test_content(response):  # noqa
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_imports():
    import imkar
    assert imkar
    import imkar.diffusion as diffusion
    assert diffusion
    import imkar.scattering as scattering
    assert scattering
