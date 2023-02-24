import pytest
import numpy as np
import pyfar as pf
from imkar import diffusion
from imkar.testing import stub_utils


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield(frequencies):
    mics = pf.samplings.sph_gaussian(42)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 1, frequencies)
    d = diffusion.coefficient.freefield(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("radius",  [
    (1), (10)])
@pytest.mark.parametrize("magnitude",  [
    (1), (10), (np.ones((5, 5)))])
def test_freefield_with_theta_0(frequencies, radius, magnitude):
    mics = pf.samplings.sph_gaussian(42, radius)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, magnitude, frequencies)
    d = diffusion.coefficient.freefield(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("radius",  [
    (1), (10)])
def test_freefield_not_one(frequencies, radius):
    # validate with code from itatoolbox
    mics = pf.samplings.sph_equal_angle(10, radius)
    mics.weights = pf.samplings.calculate_sph_voronoi_weights(mics)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    theta_is_pi = mics.get_sph().T[1] == np.pi/2
    mics.weights[theta_is_pi] /= 2
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 1, frequencies)
    p_sample.freq[1, :] = 2
    d = diffusion.coefficient.freefield(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 0.9918, atol=0.003)
