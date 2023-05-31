import imkar as ik
import numpy as np
import pyfar as pf
from scipy.special import hankel1
import numpy.testing as npt
import pytest


def test_same_result_for_increasing_A():
    H = 0.0255
    k = 20.270143083466690
    K = 35.498222074460940
    alpha_0 = 0.766044443118978
    m = -1
    n = 0
    (alpha_m, m) = ik.scattering.analytical._alpha_n(m, alpha_0, k, K)
    (alpha_n, n) = ik.scattering.analytical._alpha_n(n, alpha_0, k, K)

    A0 = ik.scattering.analytical._calculate_limit_for_A1(
        k, H, alpha_n, alpha_m, n, m)
    for factor in [1, 2, 4, 8, 16]:
        A = A0*factor

        V = ik.scattering.analytical._get_V_after_A1(
            k, H, K, alpha_n, A, m, n)
        W = ik.scattering.analytical._getW(
            k, alpha_n, H, K, A, m, n)

        print(f'{factor} => V+W = {V+W}, V = {V}, W = {W}')

    assert True


def test_sine():
    # test scattering coefficient with results in paper
    
    f = [
        1.029559118236473e+03, 1.097194388777555e+03, 1.886272545090180e+03,
        2.171843687374749e+03, 3.021042084168336e+03]
    coordinates = pf.Coordinates(0, 40, 1, 'sph', 'top_colat', 'deg')
    desired = [
        0, 0.954209748892172, 0.908419497784343, 0, 0.968980797636632]
    s = ik.scattering.analytical.sinusoid(
        f, coordinates, speed_of_sound=340.1)
    npt.assert_allclose(s.freq.flatten(), np.array(desired))


def test__getW():
    Phi_0 = 40
    c = 343
    f = 1000
    Lambda = 0.177
    k = 2*np.pi / (c/f)
    H = 0.0255
    K = 2*np.pi / 0.177
    ldL = (c/f) / Lambda
    m = 0
    n = 1
    alpha_m = np.cos(Phi_0) + m*ldL
    alpha_n = np.cos(Phi_0) + n*ldL
    A = ik.scattering.analytical._calculate_limit_for_A1(
        k, H, alpha_m, alpha_n, m, n)

    W = ik.scattering.analytical._getW(k, alpha_m, H, K, A, m, n)
    assert W == -0.02418445918393081-0.02121370543325804j


def test__alpha_n():
    incident_angle = 45*np.pi/180
    k = 2*np.pi / (343/4000)
    K = 2*np.pi / 0.177
    n_all = np.arange(-5, 6)
    alpha_0 = np.cos(incident_angle)
    (alpha_n, n_valid) = ik.scattering.analytical._alpha_n(
        n_all, alpha_0, k, K)

    npt.assert_allclose(alpha_n.shape, n_valid.shape)
    npt.assert_allclose(alpha_n, np.array(
            [-0.74628305, -0.26181977, 0.2226435, 0.70710678]))
    npt.assert_array_equal(n_valid, np.array([-3, -2, -1, 0]))


@pytest.mark.parametrize("Phi_0",  [0, 30, 60])
@pytest.mark.parametrize("m",  [-1, 0, 1])
@pytest.mark.parametrize("n",  [-1, 0, 1])
def test__calculate_limit_for_A1(Phi_0, m, n):
    # test the limits discussed in the first paragraph of the Appendix
    c = 343
    f = 1000
    Lambda = 0.177
    k = 2*np.pi / (c/f)
    H = 0.0255
    ldL = (c/f) / Lambda
    alpha_m = np.cos(Phi_0) + m*ldL
    alpha_n = np.cos(Phi_0) + n*ldL
    A = ik.scattering.analytical._calculate_limit_for_A1(
        k, H, alpha_m, alpha_n, m, n)

    # 17.3H < A
    npt.assert_array_less(17.3*H, A)
    # 200 k H^2 < A
    npt.assert_array_less(200*k*H**2, A)
    # 40 < kA
    npt.assert_array_less(40, k*A)


def test__phi_hat():
    # compare with matlab results (itaToolbox)
    m = [-1, 0, 1]
    k = 39.494515074304640
    H = 0.0255
    gamma_0 = 0.642787609686539
    p_hat = ik.scattering.analytical._phi_hat(m, k, H, gamma_0)
    desired = np.array([
        -0.000000000000000 - 0.307016844838359j,
        0.897944283825281 + 0.000000000000000j,
        0.000000000000000 - 0.307016844838359j])
    npt.assert_almost_equal(p_hat, desired)


def test__get_V_integral_2_for_ml2():
    # compare with matlab
    tau = -7.509435633700259e-04
    x = [4.844703797163286e-08, 0.022517796477943, 0.067032311870945]
    H = 0.0255
    k = 1.441658700529420e+02
    K = 35.49822207446094

    desired = np.asarray([
        -5.114150908090244,
        -2.551442223000378,
        2.617393390501456])
    actual = ik.scattering.analytical._get_V_integral_2(x, tau, H, k, K)

    assert not any(np.isnan(desired))
    npt.assert_allclose(actual, desired)


def test__get_V_integral_2_for_ml():
    # compare with matlab
    tau = -18.748750873324310
    x = [4.844703797163286e-08, 0.022517796477943, 0.067032311870945]
    H = 0.025500000000000
    k = 1.441658700529420e+02
    K = 35.498222074460940

    actual = np.asarray([
        -1.536121864092959e-04 - 6.047826161877591e-05j,
        -0.668447984135646 - 0.263459949629663j,
        -0.643611876916951 - 0.253474931840364j])
    result = ik.scattering.analytical._get_V_integral_2(x, tau, H, k, K)

    assert not any(np.isnan(result))
    npt.assert_allclose(actual, result)


def test__get_V_integral_2_for_large_tau():
    # todo compare with matlab results (itaToolbox)
    tau = 5
    x = 1
    H = 0.05
    k = 1000/343*(2*np.pi)
    K = 2*np.pi / 0.177

    rho = np.sqrt(tau**2 + (H*np.cos(K*(x+tau)) - H*np.cos(K*x))**2)
    xi = lambda x: H*np.cos(K*(x))
    dxi = lambda x: -K*H*np.sin(K*(x))
    B = xi(x+tau) - xi(x) - tau*dxi(x)
    actual = 1j*k/(2*rho) * hankel1(1, k*rho) * B
    result = ik.scattering.analytical._get_V_integral_2(x, tau, H, k, K)

    assert not np.isnan(result)
    assert actual == result


def test__get_V_integral_2_for_small_tau():
    # todo compare with matlab results (itaToolbox)
    tau = 5e-5
    x = 1
    H = 0.05
    k = 1000/343*(2*np.pi)
    K = 2*np.pi / 0.177

    actual = -(H*K**2)/(np.pi*2) * (
        np.cos(K*x)-K*tau/3 * np.sin(K*x)) / (
        1 + (H*K*np.sin(K*x))**2 + tau*H**2*K**3*np.sin(K*x)*np.cos(K*x)
        )
    result = ik.scattering.analytical._get_V_integral_2(x, tau, H, k, K)

    assert not np.isnan(result)
    assert actual == result


def test__get_V_after_A1():
    # todo compare with matlab results (itaToolbox)
    k = 1.441658700529420e+02
    H = 0.025500000000000
    K = 35.49822207446094
    alpha_m = -0.942186278912206
    A = 18.7487714003851
    m = -5
    n = -4

    result = ik.scattering.analytical._get_V_after_A1(
        k, H, K, alpha_m, A, m, n)

    actual = 0.536526156484156 - 0.635367204656775j
    npt.assert_almost_equal(actual, result, decimal=4)
