import pytest
import numpy as np
import imkar as ik
import pyfar as pf


@pytest.mark.parametrize(
    "   phi_0,      phi_1,      theta_0,    theta_1,    desired",  [
        (0,         2*np.pi,    0,          np.pi,       4*np.pi),
        (0,         np.pi,      0,          np.pi,       2*np.pi),
        (0,         np.pi,      0,          np.pi/2,     np.pi),
        (np.pi,     2*np.pi,    0,          np.pi/2,     np.pi),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi/2,     np.pi),
        (np.pi,     2*np.pi,    0,          np.pi,       2*np.pi),
    ])
def test_surface_sphere_uniform_data_different_limits(
        phi_0, phi_1, theta_0, theta_1, desired):
    delta = np.deg2rad(10)
    data, coords = _create_test_data(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta))
    result = ik.integrate.surface_sphere(data, coords)
    actual = np.real(result.freq[0, 0])
    np.testing.assert_allclose(actual, desired, rtol=5e-3)


@pytest.mark.parametrize(
    "   phi_0,      phi_1,      theta_0,    theta_1,    desired",  [
        (0,         2*np.pi,    0,          np.pi,       4*np.pi),
        (0,         np.pi,      0,          np.pi,       2*np.pi),
        (0,         np.pi,      0,          np.pi/2,     np.pi),
        (np.pi,     2*np.pi,    0,          np.pi/2,     np.pi),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi/2,     np.pi),
        (np.pi,     2*np.pi,    0,          np.pi,       2*np.pi),
    ])
def test_surface_sphere_data_preserve_shape_with_different_limits(
        phi_0, phi_1, theta_0, theta_1, desired):
    delta = np.deg2rad(10)
    data_raw = np.arange(1, 7).reshape(2, 3)
    data, coords = _create_test_data(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta),
        data_raw=data_raw)
    result = ik.integrate.surface_sphere(data, coords)
    actual = np.squeeze(np.real(result.freq))
    np.testing.assert_allclose(actual, data_raw*desired, rtol=5e-3)


@pytest.mark.parametrize(
    "   phi_0,      phi_1,      theta_0,    theta_1,    desired",  [
        (0,         2*np.pi,    0,          np.pi,      4*np.pi),
        (0,         np.pi,      0,          np.pi,      4*np.pi/2),
        (0,         np.pi,      0,          np.pi/2,    4*np.pi/4),
        (np.pi,     2*np.pi,    0,          np.pi/2,    4*np.pi/4),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi/2,    4*np.pi/4),
        (0,         2*np.pi,    np.pi/2,    np.pi,      4*np.pi/2),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi,      4*np.pi/2),
    ])
def test_surface_sphere_nonuniform_data_different_limits(
        phi_0, phi_1, theta_0, theta_1, desired):
    delta = np.pi/18
    data, coords = _create_test_data(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta))
    result = ik.integrate.surface_sphere(data, coords)
    actual = np.real(result.freq[0, 0])
    np.testing.assert_allclose(actual, desired, rtol=5e-3, atol=0.04)


def test_spherical_warning_wrong_radius():
    delta = np.deg2rad(10)
    data, coords = _create_test_data(
        np.arange(0, 2*np.pi+delta, delta),
        np.arange(0, np.pi+delta, delta))
    sph = coords.get_sph()
    # manipualte radius
    sph[:, 1, 2] = 2
    coords.set_sph(sph[..., 0], sph[..., 1], sph[..., 2])
    with pytest.warns(Warning, match='radi'):
        ik.integrate.surface_sphere(data, coords)


def test_surface_sphere_error_invalid_coordinates_shape():
    delta = np.deg2rad(10)
    data, coords = _create_test_data(
        np.arange(0, 2*np.pi+delta, delta),
        np.arange(0, np.pi+delta, delta))
    sph = coords.get_sph()
    sph = sph[1:, :, :]
    coords.set_sph(sph[..., 0], sph[..., 1], sph[..., 2])
    with pytest.raises(ValueError, match='Coordinates.cshape'):
        ik.integrate.surface_sphere(data, coords)


def _create_test_data(phi_rad, theta_rad, data_raw=None, n_bins=1, radius=1):
    phi, theta = np.meshgrid(phi_rad, theta_rad)
    coords = pf.Coordinates(
        phi, theta, np.ones(phi.shape)*radius, 'sph')
    if data_raw is None:
        data_raw = np.ones((1))
    for dim in phi.shape:
        data_raw = np.repeat(data_raw[..., np.newaxis], dim, axis=-1)
    data_raw = np.repeat(data_raw[..., np.newaxis], n_bins, axis=-1)
    freq_data = np.arange(1, n_bins+1)*100
    data = pf.FrequencyData(data_raw, freq_data)
    return data, coords
