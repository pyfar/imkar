import pytest
import numpy as np
import pyfar as pf
from imkar import diffusion


def test_freefield(half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(1)
    d = diffusion.freefield(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 1)


@pytest.mark.parametrize("radius",  [
    (1), (10)])
@pytest.mark.parametrize("magnitude",  [
    (1), (10)])
def test_freefield_with_theta_0(
        half_sphere, pressure_data_mics, radius, magnitude):
    mics = half_sphere
    spherical = mics.get_sph().T
    mics.set_sph(spherical[0], spherical[1], radius)
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(magnitude)
    d = diffusion.freefield(p_sample, mics.weights)
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
    data = np.ones(mics.cshape)
    p_sample = pf.FrequencyData(data[..., np.newaxis], [100])
    p_sample.freq[1, :] = 2
    d = diffusion.freefield(p_sample, mics.weights)
    np.testing.assert_allclose(d.freq, 0.9918, atol=0.003)
