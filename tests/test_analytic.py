import pytest
import numpy.testing as npt
import numpy as np
import pyfar as pf
import imkar as ik


@pytest.mark.parametrize('x',  [-1, 0, 1])
@pytest.mark.parametrize('y',  [-1, 0, 1])
@pytest.mark.parametrize('z',  [0, 1])  # not defined for z<0
def test_cartesian_to_theta_phi(x, y, z):
    if x != 0 and y != 0 and z != 0:
        coords = pf.Coordinates(x, y, z)
        (theta, phi) = ik.analytical.cartesian_to_theta_phi(coords)

        (coords_actual) = ik.analytical.theta_phi_to_cartesian(theta, phi)
        norm = np.sqrt(x**2 + y**2 + z**2)
        coords_desired = pf.Coordinates(x/norm, y/norm, z/norm)
        # npt.assert_almost_equal((x/norm, y/norm, z/norm), (x_a, y_a, z_a))
        npt.assert_almost_equal(
            coords_desired.get_cart(), coords_actual.get_cart())


# @pytest.mark.parametrize('theta',  np.linspace(-np.pi/2, np.pi/2, 5))
# @pytest.mark.parametrize('phi',  np.linspace(0, np.pi, 5))
# def test_theta_phi_to_cartesian(theta, phi):
#     (coords) = ik.analytical.theta_phi_to_cartesian(theta, phi)
#     (theta_actual, phi_actual) = ik.analytical.cartesian_to_theta_phi(coords)
#     npt.assert_almost_equal(theta, theta_actual)
#     if not (theta < np.pi/2 or theta > -np.pi/2):
#         npt.assert_almost_equal(phi, phi_actual)


@pytest.mark.parametrize('Phi_0, Theta_0, desired',  [
    (0, 0, pf.Coordinates(-1, 0, 0)),
    (0, 40, pf.Coordinates(-0.7660444, -0.6427876, 0)),
    (40, 0, pf.Coordinates(-0.7660444, 0, 0.6427876)),
    ])
def test_specific_to_cartesian(Phi_0, Theta_0, desired):
    actual = ik.analytical.theta_phi_to_cartesian(
        Theta_0/180*np.pi, Phi_0/180*np.pi)
    npt.assert_almost_equal(actual.get_cart(), desired.get_cart())


@pytest.mark.parametrize('coords, Phi_0, Theta_0',  [
    (pf.Coordinates(-1, 0, 0), 0, 0),
    (pf.Coordinates(-0.7660444, -0.6427876, 0), 0, 40),
    (pf.Coordinates(-0.7660444, 0, 0.6427876), 40, 0),
    ])
def test_specific_to_phi_theta(coords, Phi_0, Theta_0):
    (theta_actual, phi_actual) = ik.analytical.cartesian_to_theta_phi(coords)
    npt.assert_almost_equal(theta_actual, Theta_0*np.pi/180)
    npt.assert_almost_equal(phi_actual, Phi_0*np.pi/180)


# def test_sine():
#     # test scattering coefficient with results in paper
#     f = [
#         1.029559118236473e+03, 1.097194388777555e+03, 1.886272545090180e+03,
#         2.171843687374749e+03, 3.021042084168336e+03]
#     coordinates = pf.Coordinates(0, 40, 1, 'sph', 'top_elev', 'deg')
#     desired = [
#         0, 0.954209748892172, 0.908419497784343, 0, 0.968980797636632]
#     s = ik.analytical.sinusoid(
#         f, coordinates, speed_of_sound=340.1)
#     npt.assert_allclose(s.freq.flatten(), np.array(desired))
