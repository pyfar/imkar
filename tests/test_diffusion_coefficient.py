import pytest
import numpy as np

from imkar import diffusion
from imkar.testing import stub_utils


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield(frequencies):

    mics = stub_utils.spherical_coordinates(
        np.arange(0, 360, 10)/180*np.pi,
        np.arange(10, 90, 10)/180*np.pi,
        1)
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 1, frequencies)
    d = diffusion.coefficient.freefield(p_sample, mics)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("radius",  [
    (1), (10)])
@pytest.mark.parametrize("magnitude",  [
    (1), (10), (np.ones((5, 5)))])
def test_freefield_with_theta_0(frequencies, radius, magnitude):
    mics = stub_utils.spherical_coordinates(
        np.arange(0, 360, 10)/180*np.pi,
        np.arange(0, 100, 10)/180*np.pi,
        radius)
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, magnitude, frequencies)
    d = diffusion.coefficient.freefield(p_sample, mics)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("radius",  [
    (1), (10)])
def test_freefield_not_one(frequencies, radius):
    mics = stub_utils.spherical_coordinates(
        np.arange(0, 360, 10)/180*np.pi,
        np.arange(0, 100, 10)/180*np.pi,
        radius)
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 1, frequencies)
    p_sample.freq[1, 0, :] = 2
    d = diffusion.coefficient.freefield(p_sample, mics)
    np.testing.assert_allclose(d.freq, 0.9918, atol=0.003)
