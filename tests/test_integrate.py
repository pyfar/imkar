import pytest
import numpy as np

from imkar import integrate
from pyfar import FrequencyData, Coordinates


@pytest.mark.parametrize("radius",  [1, 2, 3, 4, 5])
@pytest.mark.parametrize("delta,rtol",  [(1, 1e-4), (10, 5e-3)])
def test_spherical_radius(radius, delta, rtol):
    data, coords = _create_test_data(
        np.arange(0, 360+delta, delta),
        np.arange(0, 180+delta, delta))
    result = integrate.spherical(data, coords)
    actual = np.real(result.freq[0, 0])
    desired = 4*np.pi
    np.testing.assert_allclose(actual, desired, rtol=rtol)


@pytest.mark.parametrize("radius",  [1, 2, 3, 4, 5])
@pytest.mark.parametrize("delta,rtol",  [(1, 1e-4), (10, 5e-3)])
def test_spherical_limits(radius, delta, rtol):
    data, coords = _create_test_data(
        np.arange(0, 180+delta, delta),
        np.arange(0, 90+delta, delta))
    result = integrate.spherical(data, coords)
    desired = np.pi
    actual = np.real(result.freq[0, 0])
    np.testing.assert_allclose(actual, desired, rtol=rtol)


@pytest.mark.parametrize("radius",  [1, 2, 3, 4, 5])
@pytest.mark.parametrize("delta,rtol",  [(1, 1e-4), (10, 5e-3)])
def test_spherical_data_dimensions(radius, delta, rtol):
    data_raw = np.arange(1, 7).reshape(2, 3)
    data, coords = _create_test_data(
        np.arange(0, 360+delta, delta),
        np.arange(0, 180+delta, delta), data_raw=data_raw)
    result = integrate.spherical(data, coords)
    desired = data_raw*4*np.pi
    actual = np.squeeze(np.real(result.freq))
    np.testing.assert_allclose(actual, desired, rtol=rtol)


def test_spherical_warning_wrong_radius():
    data, coords = _create_test_data(
        np.arange(0, 360+10, 10),
        np.arange(0, 180+10, 10))
    sph = coords.get_sph()
    # manipualte radius
    sph[:, 1, 2] = 2
    coords.set_sph(sph[..., 0], sph[..., 1], sph[..., 2])
    with pytest.warns(Warning, match='radi'):
        integrate.spherical(data, coords)


def _create_test_data(phi_deg, theta_deg, data_raw=None, n_bins=1, radius=1):
    phi, theta = np.meshgrid(phi_deg, theta_deg)
    coords = Coordinates(
        phi, theta, np.ones(phi.shape)*radius, 'sph', 'top_colat', 'deg')
    if data_raw is None:
        data_raw = np.ones((1))
    for dim in phi.shape:
        data_raw = np.repeat(data_raw[..., np.newaxis], dim, axis=-1)
    data_raw = np.repeat(data_raw[..., np.newaxis], n_bins, axis=-1)
    freq_data = np.arange(1, n_bins+1)*100
    data = FrequencyData(data_raw, freq_data)
    return data, coords
