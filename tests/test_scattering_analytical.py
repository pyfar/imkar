import imkar as ik
import numpy as np


def test_holford_urusovskii_method():
    h = 0.051/2
    Lambda = 0.177
    f = 2176
    k = 2*np.pi / (343/f)
    Phi_0 = 40 * np.pi/180
    K = 2 * np.pi / Lambda
    xi = lambda x: h*np.cos(K*x)
    dxi = lambda x: -K*h*np.sin(K*x)
    ms = np.arange(-3, 3)
    ns = np.arange(-3, 3)
    U_m_n = ik.scattering.analytical.holford_urusovskii_method(
        ms, ns, Lambda, h, xi, dxi, k, Phi_0)
    assert 1

