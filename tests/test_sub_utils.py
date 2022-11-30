import pytest
import numpy as np
import numpy.testing as npt

from imkar.testing import stub_utils


@pytest.mark.parametrize(
    "   phi_0,      phi_1,      theta_0,    theta_1,    radius",  [
        (0,         2*np.pi,    0,          np.pi,       1),
        (0,         np.pi,      0,          np.pi,       2),
        (0,         np.pi,      0,          np.pi/2,     0.1),
        (np.pi,     2*np.pi,    0,          np.pi/2,     np.pi),
        (np.pi-.1,  2*np.pi-.1, 0,          np.pi/2,     np.pi),
        (np.pi,     2*np.pi,    0,          np.pi,       2),
    ])
def test_spherical_coordinates(phi_0, phi_1, theta_0, theta_1, radius):
    delta = np.pi/18
    phi = np.arange(phi_0, phi_1+delta, delta)
    theta = np.arange(theta_0, theta_1+delta, delta)
    coords = stub_utils.spherical_coordinates(phi, theta, radius)
    npt.assert_allclose(coords.get_sph()[1, :, 0], phi)
    npt.assert_allclose(coords.get_sph()[:, 1, 1], theta)
    npt.assert_allclose(
        coords.get_sph()[:, :, 2], np.zeros(coords.cshape)+radius)


@pytest.mark.parametrize(
    "shapes",  [
        (3, 2),
        (5, 2),
        (3, 2, 7),
    ])
@pytest.mark.parametrize(
    "data_in",  [
        0.1,
        0,
        np.array([0.1, 1]),
        np.arange(4*5).reshape(4, 5),
    ])
@pytest.mark.parametrize(
    "frequency",  [
        [100],
        [100, 200],
    ])
def test_frequencydata_from_shape_data(shapes, data_in, frequency):
    data = stub_utils.frequencydata_from_shape(shapes, data_in, frequency)
    # npt.assert_allclose(data.freq, data_in)
    if hasattr(data_in, '__len__'):
        for idx in range(len(data_in.shape)):
            assert data.cshape[idx] == data_in.shape[idx]
        for idx in range(len(shapes)):
            assert data.cshape[idx+len(data_in.shape)] == shapes[idx]

    else:
        for idx in range(len(shapes)):
            assert data.cshape[idx] == shapes[idx]
    assert data.n_bins == len(frequency)
