import numpy as np
import pyfar as pf
import warnings

from scipy import integrate
from scipy.special import jv, hankel1


def sinusoid(
        frequencies, coordinates, structural_Lambda=0.177,
        structural_Height=0.051, speed_of_sound=343):
    """
    This function computes the scattering coefficient of a sine-shaped surface
    (with infinite extend) for a given set of frequencies and angles.

    The function implements the method by Holford and Urusovskii with the
    implementation given by Embrechts [#]_.

    Parameters
    ----------
    frequency_vector : array, float
        Vector of frequencies the scattering coefficient is computed at.
    coordinates : pf.Coordinates
        Coordinate object for the incident directions. The up-vector is
        in y-direction, the structure towards x-direction. Suggestion is to use
        the default spherical coordinates.
    structural_Lambda : float
        Structural length of sinusoidal sample. Without an input, the length of
        177mm is used (test sample).
    structural_Height : float
        Structural height/depths of the sinusoidal sample (peak-to-peak).
        Without an input, the height of 51mm is used (test sample).
    speed_of_sound : float
        Speed-of-sound used for the computation.

    Returns
    -------
    scattering_coefficient : FrequencyData
        The analytical solution for the scattering coefficient of a
        sinusoidal surface at the given incidences and frequencies.

    References
    ----------
    .. [#]J. Embrechts et al., "Calculation of the Random-Incidence Scattering
    Coefficients of a Sine-Shaped Surface", Acta Acustica united with Acustica,
    Bd. 92, Nr. 4, S. 593-603, April 2006
    """
    # check inputs

    # Initialization and constants
    Lambda = structural_Lambda
    H = structural_Height/2  # half peak-to-peak
    K = 2*np.pi/Lambda  # structural wavenumber
    frequencies = np.asarray(frequencies, dtype=float)
    # vector of corresponding wavelength in air
    lambda_air = speed_of_sound/frequencies
    k = 2*np.pi/lambda_air

    eta0 = 0  # normalized admittance

    # order of maximum reflection
    n_max = np.ceil(2*Lambda*frequencies.max()/speed_of_sound)
    scattering_coefficient = np.zeros((coordinates.csize, frequencies.size))
    n_all = np.arange(-n_max, n_max+1, 1)

    for i_freq in range(0, np.size(frequencies)):
        for i_coord in range(coordinates.csize):
            coord = coordinates[i_coord].get_sph().flatten()
            azimuth = coord[0]
            incident_angle = coord[1]
            k_theta = k*np.sin(azimuth)  # account out-of-plane incidence

            # alpha_0 and gamma_0 after (2)
            alpha_0 = np.cos(incident_angle)
            gamma_0 = np.sin(incident_angle)

            (alpha_n, n_valid) = _alpha_n(n_all, alpha_0, k[i_freq], K)

            if np.size(n_valid) == 0 or (np.size(n_valid) == 1 and n_valid == 0):
                continue
            # gamma_n after (2)
            gamma_n = np.sin(np.arccos(alpha_n))

            nmax2 = np.max(np.abs(n_valid))
            N = 2*nmax2+1
            n_vec = np.arange(-nmax2, nmax2+1, 1)
            U = np.zeros((N, N))

            # Calculate U_m_n after (A1)
            for iM in range(len(n_valid)):
                for iN in range(len(n_valid)):
                    if (n_vec[iM]-n_vec[iN]) % 2:
                        A = _calculate_limit_for_A1(
                            k[i_freq], H, alpha_n[iM], alpha_n[iN],
                            n_valid[iM], n_valid[iN])
                        V = _get_V_after_A1(
                            k[i_freq], H, K, alpha_n[iM], A,
                            n_valid[iM], n_valid[iN])
                        W = _getW(
                            k[i_freq], alpha_n[iM], H, K, A,
                            n_valid[iM], n_valid[iN])
                        U[iM, iN] = V + W

            # Calculate Phi_m after (6)
            I_U = np.identity(N) - U
            phi_hat_m = _phi_hat(n_vec, k[i_freq], H, gamma_0)
            Phi_m = np.linalg.lstsq(I_U, 2*phi_hat_m)[0]

            # Calculate R_m after
            R = np.zeros((N, N))
            for iM in range(np.size(n_valid)):
                for iN in range(np.size(n_valid)):
                    phi_hat = _phi_hat(
                        n_vec[iM]-n_vec[iN], k[i_freq], H, gamma_n[iM])
                    R[iM, iN] = 1/(2*gamma_n[iM])*Phi_m[iN] * phi_hat * (
                        gamma_n[iM]+(n_vec[iM]-n_vec[iN]) *
                        K*alpha_n[iM] / (k[i_freq]*gamma_n[iM]) + eta0
                        )

            R_m = np.sum(R, 1)
            R_m = R_m[n_valid+nmax2+1]

            normTest = np.sum(np.abs(R_m)**2*gamma_n)/gamma_0
            R_0 = R_m[n_valid == 0][0]
            scattering_coefficient[i_coord, i_freq] = 1 - np.min(
                (1, np.abs(R_0)**2))
    return pf.FrequencyData(scattering_coefficient, frequencies)


def _alpha_n(n, alpha_0, k, K):
    """Calculate alpha_n after Formula 2

    Parameters
    ----------
    n : int
        index n
    alpha_0 : float
        cos(azimuth) of incident angle azimuth
    k : float
        Wave number of the sound in air
    K : float
        Structural wave number of the sample 2pi/Lambda

    Returns
    -------
    alpha_n : float
        returns alpha_n
    """
    # K/k = lambda/Lambda
    n = np.asarray(n)
    alpha_n = alpha_0 + n*K/k
    valid_alpha = np.abs(alpha_n) <= 1
    n = np.array(n[valid_alpha], dtype=int)
    alpha_n = alpha_n[valid_alpha]

    return (alpha_n, n)


def _phi_hat(m, k, H, gamma_0):
    """
    Calculates p_hat after A8 with bessel function

    Parameters
    ----------
    m : array, int
        Index m for p_hat
    k : float
        Wave number of the sound in air
    H : float
        Hight of the sample surface structure in meter
    gamma_0 : float
        Describes the sin(Phi_0), where Phi_0 in the
        incident sound direction

    Returns
    -------
    p_hat: array, number
        result of shape of m
    """
    m = np.asarray(m)
    phi_f = (-1j)**m * jv(m, k*H*gamma_0)

    return phi_f


def _calculate_limit_for_A1(k, H, alpha_m, alpha_n, m, n):
    """Calculates the limit A for the integral in (A7)

    Parameters
    ----------
    k : number
        Wave number of the sound in air
    H : number
        Hight of the sample surface structure in meter
    alpha_m : number
        Describes the cos(Phi_0) + m*lambda/Lambda, where Phi_0 in the
        incident sound direction
    alpha_n : number
        Describes the cos(Phi_0) + n*lambda/Lambda, where Phi_0 in the
        incident sound direction
    m : int
        index m
    n : int
        index n

    Returns
    -------
    A : number
        limit for integration
    """
    if n == m+1 or n == m-1:
        if np.abs(alpha_m) == 1 or np.abs(alpha_n) == 1:
            A = np.max((40, 200*(k*H)**2))/k
        else:
            A = np.max((40, 200*(k*H)**2, 50/np.min(np.abs((
                1+alpha_m, 1-alpha_m, 1-alpha_n, 1+alpha_n)))))/k
    else:
        A = np.max((40, 200*(k*H)**2))/k

    return A


def _get_V_integral_2(x, tau, H, k, K):
    """Second part of formula A1

    Parameters
    ----------
    x : number
        integral parameter x
    tau : number
        integral parameter tau
    H : number
        Hight of the sample surface structure in meter
    k : number
        Wave number of the sound in air
    K : number
        Structural wave number of the sample 2pi/Lambda

    Returns
    -------
    result: number
        result for the second
    """
    x = np.asarray(x, dtype=float)
    # for small values of t the Hankel function explodes due to t~rho
    if np.abs(tau) < 0.001:
        # Formula A9 without the exponential term, this is added in the later
        # after that
        return -(H*K**2)/(np.pi*2) * (
            np.cos(K*x)-K*tau/3 * np.sin(K*x)) / (
            1 + (H*K*np.sin(K*x))**2 + tau*H**2*K**3*np.sin(K*x)*np.cos(K*x)
            )
    else:
        rho = np.sqrt(tau**2 + (H*np.cos(K*(x+tau)) - H*np.cos(K*x))**2)
        xi = lambda x: H*np.cos(K*(x))
        dxi = lambda x: -K*H*np.sin(K*(x))
        B = xi(x+tau) - xi(x) - tau*dxi(x)
        return 1j*k/(2*rho) * hankel1(1, k*rho) * B


def _get_V_after_A1(k, H, K, alpha_m, A, m, n):
    """Solves formula A1 with the following first assumptions described
    in the paragraph after it. All parameter names are same as in the paper.

    Parameters
    ----------
    k : number
        Wave number of the sound in air
    H : number
        Hight of the sample surface structure in meter
    K : number
        Structural wave number of the sample 2pi/Lambda
    alpha_m : number
        Describes the cos(Phi_0) + m*lambda/Lambda, where Phi_0 in the
        incident sound direction
    A : number
        Limits of the second integral
    m : int
        index m
    n : int
        index n

    Returns
    -------
    V : number
        _description_
    """
    pi = np.pi
    # implements A1 here with the limits from -A and A, and the 0 to Lambda/2
    # Note that K/pi = 2/Lambda
    A1 = lambda x, tau: K/pi*np.exp(-1j*(m-n)*K*x) * _get_V_integral_2(
        x, tau, H, k, K) * np.exp(-1j*k*alpha_m*tau)
    # Note that pi/K = Lambda/2
    A1_tau = lambda tau: integrate.quad(
        A1, 0, pi/K, args=(tau), complex_func=True, limit=3000)[0]
    V = integrate.quad(A1_tau, -A, A, complex_func=True, limit=3000)
    if np.abs(V[1]) > 2e-8:
        warnings.warn(f'Error is higher than 2e-8, its {np.abs(V[1])}')
    print(f'Error is higher than 2e-8, its {np.abs(V[1])}')
    return V[0]


def _getW(k, alpha_m, H, K, A, m, n):
    """Calculate W after Formula (A6)

    Parameters
    ----------
    k : _type_
        _description_
    alpha_m : _type_
        _description_
    H : _type_
        _description_
    K : _type_
        _description_
    A : _type_
        _description_
    m : _type_
        _description_
    n : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    pi = np.pi
    if n == m+1 or n == m-1:
        if np.abs(alpha_m) == 1:
            W = 0
        else:
            W = 1j*H*K/(1-alpha_m**2) / np.sqrt(2*pi*k*A) * np.exp(
                1j*(k*A-0.75*pi)) * (
                alpha_m*np.cos(k*A*alpha_m)-1j*np.sin(k*A*alpha_m))
            if n == m+1:
                W = -W
    else:
        W = 0

    return W


def holford_urusovskii_method(ms, ns, Lambda, h, xi, dxi, k, Phi_0):

    pi = np.pi
    K = 2 * pi / Lambda
    # integral one in Formula A1
    rho = lambda x, tau: np.sqrt(tau**2 + (xi(x + tau) - xi(x)**2))
    term_fac = lambda x, tau: 1j*k/(2*rho(x, tau))
    term_hankel = lambda x, tau: hankel1(1, k*rho(x, tau))
    term_xi = lambda x, tau: xi(x+tau) - xi(x) + tau*dxi(x)
    A = np.max((200*k*h**2, 40))
    print(A)
    U_m_n = np.zeros((len(ms), len(ns)))
    for i_m in range(len(ms)):
        m = ms[i_m]
        alpha_m = np.cos(Phi_0) + m * K/k
        term_exp2 = lambda x, tau: 2 / Lambda * np.exp(-1j*k*alpha_m*tau)
        W = (1j*h*K)/(1-alpha_m**2) * np.sqrt(1j*(k*A - 0.75*pi)) * (
            alpha_m*np.cos(k*A*alpha_m) - 1j*np.sin(k*A*alpha_m))
        for i_n in range(len(ns)):
            n = ns[i_n]
            term_exp1 = lambda x: np.exp(-1j*(m-n)*K*x)
            V = integrate.dblquad(
                lambda x, tau: term_exp1(x) * term_fac(x, tau) *
                term_hankel(x, tau) * term_xi(x, tau) * term_exp2(x, tau),
                0, Lambda, lambda tau: -A, lambda tau: A)

            print(V+W)
            U_m_n[i_m, i_n] = V+W

    return U_m_n
