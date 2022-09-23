import pytest
import numpy as np

from imkar import integrate
from imkar.testing import stub_utils


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
    coords = stub_utils.create_coordinates_sph(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta))
    data = stub_utils.create_const_frequencydata_from_shape(
        coords.cshape, 1, 100)
    result = integrate.surface_sphere(data, coords)
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
    coords = stub_utils.create_coordinates_sph(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta))
    data = stub_utils.create_frequencydata_from_shape(
        coords.cshape, data_raw, 100)
    result = integrate.surface_sphere(data, coords)
    actual = np.squeeze(np.real(result.freq))
    np.testing.assert_allclose(actual, data_raw*desired, rtol=5e-3)


@pytest.mark.parametrize(
    "   phi_0,      phi_1,      theta_0,    theta_1",  [
        (0,         2*np.pi,    0,          np.pi),
        (0,         np.pi,      0,          np.pi),
        (0,         np.pi,      0,          np.pi/2),
        (np.pi,     2*np.pi,    0,          np.pi/2),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi/2),
        (0,         2*np.pi,    np.pi/2,    np.pi),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi),
        (np.pi,     2*np.pi,    0,          np.pi),
    ])
def test_surface_sphere_nonuniform_data_different_limits(
        phi_0, phi_1, theta_0, theta_1):
    delta = np.pi/18
    coords = stub_utils.create_coordinates_sph(
        np.arange(phi_0, phi_1+delta, delta),
        np.arange(theta_0, theta_1+delta, delta))
    data = stub_utils.create_const_frequencydata_from_shape(
        coords.cshape, 1, 100)
    theta = coords.get_sph()[..., 1]
    data.freq[..., 0] = np.cos(theta)
    result = integrate.surface_sphere(data, coords)
    actual = np.real(result.freq[0, 0])
    theta_upper = -np.cos(2*theta_1)/4
    theta_lower = -np.cos(2*theta_0)/4
    desired = (theta_upper - theta_lower) * (phi_1 - phi_0)
    np.testing.assert_allclose(actual, desired, rtol=5e-3, atol=0.04)


def test_spherical_warning_wrong_radius(coords_sphere_10_deg):
    coords = coords_sphere_10_deg
    data = stub_utils.create_const_frequencydata_from_shape(
        coords.cshape, 1, 100)
    sph = coords.get_sph()
    # manipualte radius
    sph[:, 1, 2] = 2
    coords.set_sph(sph[..., 0], sph[..., 1], sph[..., 2])
    with pytest.warns(Warning, match='radi'):
        integrate.surface_sphere(data, coords)


def test_surface_sphere_error_invalid_coordinates_shape(coords_sphere_10_deg):
    coords = coords_sphere_10_deg
    data = stub_utils.create_const_frequencydata_from_shape(
        coords.cshape, 1, 100)
    sph = coords.get_sph()
    sph = sph[1:, :, :]
    coords.set_sph(sph[..., 0], sph[..., 1], sph[..., 2])
    with pytest.raises(ValueError, match='Coordinates.cshape'):
        integrate.surface_sphere(data, coords)

