import numpy as np
import pyfar as pf

from scipy import integrate
from scipy.special import jv, hankel1


def cartesian_to_theta_phi(coordinates: pf.Coordinates):
    """Converts pf.Coordinates to Theta_0 und Phi_0 used in the analytical
    approach.

    Note that its not defined for z<0.

    Parameters
    ----------
    coordinates : pf.Coordinates
        input coordinates.

    Returns
    -------
    Theta_0: float
        theta angle in radiant.
    Phi_0: float
        phi angle in radiant.
    """
    x = coordinates.get_cart().T[0]
    y = coordinates.get_cart().T[1]
    z = coordinates.get_cart().T[2]
    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm
    theta = np.arcsin(-y_norm)
    phi = np.arcsin(z_norm/np.cos(theta))

    mask = np.isnan(phi)
    phi[mask] = np.arccos(-x_norm[mask]/np.cos(theta[mask]))
    if x_norm > 0:
        theta = np.pi - theta
        phi = - phi
    assert (z >= 0).all()
    return (theta, phi)


def theta_phi_to_cartesian(Theta_0, Phi_0):
    """Convert Theta_0 and Phi_0 in pf.Coordinate object.

    Note that its not defined for z<0.

    Parameters
    ----------
    Theta_0: float
        theta angle in radiant.
    Phi_0: float
        phi angle in radiant.

    Returns
    -------
    coordinates: pf.Coordinates
        output coordinates.
    """
    x_n = -np.cos(Theta_0) * np.cos(Phi_0)
    y_n = -np.sin(Theta_0) + np.zeros_like(Phi_0)
    z_n = np.cos(Theta_0) * np.sin(Phi_0)
    assert (z_n >= 0).all()
    return pf.Coordinates(x_n, y_n, z_n)


def sinusoid(frequencies, coordinates, length=0.177, height=0.051,
             speed_of_sound=343):
    """
    This function computes the scattering coefficient of a sine-shaped surface
    (with infinite extend) for a given set of frequencies and angles.

    The function implements the method by Holford and Urusovskii with the
    implementation given by Embrechts [#]_.

    Parameters
    ----------
    frequency_vector : array, double
        Vector of frequencies the scattering coefficient is computed at.
    coordinates : pf.Coordinates
        incident directions to be calculated.
    length : double
        Structural length of sinusoidal sample in meters. Without an input,
        the length of 177mm is used (test sample).
    height : double
        Structural height/depths of the sinusoidal sample (peak-to-peak) in
        meters. Without an input, the height of 51mm is used (test sample).
    speed_of_sound : double
        Speed of sound [m/s] used for the computation.

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
    theta, phi = cartesian_to_theta_phi(coordinates)
    # Initialization and constants
    incident_angle = phi  # from degree to radiant
    k_structural = 2*np.pi/length  # structural wavenumber
    # vector of corresponding wavelength in air
    lambda_air = speed_of_sound/frequencies
    k = 2*np.pi/lambda_air
    k_theta = k*np.sin(theta)  # account out-of-plane incidence
    eta0 = 0  # normalized admittance
    height = height/2  # half peak-to-peak

    # order of maximum reflection
    n_max = np.ceil(2*length*frequencies.max()/speed_of_sound)
    s_coeff = np.zeros((frequencies.size(), incident_angle.size()))
    normTest = np.zeros((frequencies.size(), incident_angle.size()))
    for iFrequencies in range(0, np.size(frequencies)):
        sPhi = np.zeros(np.size(incident_angle), 1)
        for iPhis in range(0, np.size(incident_angle)):
            alpha_0 = np.cos(incident_angle[iPhis])
            gamma_0 = np.sin(incident_angle[iPhis])

            alpha_nn = _alpha(
                np.arange(-n_max, n_max+1, 1), k[iFrequencies],
                k_theta[iFrequencies], k_structural)
            alpha_n = alpha_nn[:, 0]
            n = alpha_nn[:, 1]

            if np.size(n) == 0 or (np.size(n) == 1 and n == 0):
                continue

            gamma_n = np.sin(np.arccos(alpha_n))

            nmax2 = np.max(np.abs(n))
            N = 2*nmax2+1
            n_vec = np.arange(-nmax2, nmax2+1, 1)
            phi_hat = _phi_hat(n_vec, k[iFrequencies]*height*gamma_0)
            U = np.zeros((N))

            for iM in range(0, np.size(n)):
                for iN in range(0, np.size(n)):
                    if (n_vec[iM]-n_vec[iN]) % 2:
                        A = _getA(
                            k[iFrequencies], k_theta[iFrequencies],
                            height, k_structural, alpha_0, n[iM], n[iN])
                        U[n[iM]+nmax2+2, n[iN]+nmax2+2] = \
                            _getV(
                                k[iFrequencies], k_theta[iFrequencies],
                                height, k_structural, alpha_0, A, n[iM],
                                n[iN]) + _getW(
                                    k[iFrequencies], k_theta[iFrequencies],
                                    height, k_structural, alpha_0, A, n[iM],
                                    n[iN])
            A = np.identity(N) - U
            b = 2*phi_hat

            Phi = np.linalg.lstsq(A, b)
            R = np.zeros((N))
            for iM in range(0, np.size(n)):
                for iN in range(0, np.size(n)):
                    R[n[iM]+nmax2+2, n[iN]+nmax2+2] = \
                        1/(2*gamma_n[iM])*Phi[iN] * _phi_hat(
                            n_vec[iM]-n_vec[iN],
                            k[iFrequencies] * height*gamma_n[iM] * (
                                gamma_n[iM]+(n_vec[iM]-n_vec[iN]) *
                                k_structural*alpha_n[iM] /
                                (k[iFrequencies]*gamma_n[iM])+eta0))

            R = np.sum(R, 2)
            R = R[n+nmax2+2]

            normTest[iFrequencies, iPhis] = np.sum(
                np.abs(R)**2*gamma_n[:])/gamma_0
            R0 = R[n == 0]
            s_coeff[iFrequencies, iPhis] = 1 - np.min(1, np.abs(R0)**2)
    meta = {}
    meta['normTest'] = normTest  # todo, wichtig für tests
    return pf.FrequencyData(np.transpose(s_coeff), frequencies), meta


def _alpha(n, alpha0, k, k_theta, k_structural):
    alpha_f = (k_theta/k)**2 + alpha0 + n*k_structural/k
    valid_alpha = (np.abs(alpha_f) <= 1)
    n = n[np.nonzero(valid_alpha)]
    alpha_f = alpha_f[np.nonzero(valid_alpha)]

    return np.column_stack((alpha_f, n))


def _phi_hat(m, k_h_gamma0):
    phi_f = np.zeros((np.size(m), 1))
    for iM in range(0, np.size(m)):
        phi_f[iM] = (-1)**m[iM]*jv(m[iM], k_h_gamma0)

    return phi_f


def _getA(k, k_theta, height, k_structural, alpha0, m, n):
    if n == m+1 or n == m-1:
        alpha_m = _alpha(m, alpha0, k, k_theta, k_structural)
        alpha_n = _alpha(n, alpha0, k, k_theta, k_structural)
        if np.abs(alpha_m) == 1 or np.abs(alpha_n) == 1:
            A = 1/k*np.max(40, 200*(k*height)**2)
        else:
            A = 1/k*np.max(
                40,
                200*(k*height)**2,
                50/np.min(1+alpha_m, 1-alpha_m, 1-alpha_n, 1+alpha_n))
    else:
        A = 1/k*np.max(40, 200*(k*height)**2)

    return A


def _int2func(s, t, h, k, k_structural):
    # for small values of t the hankel function explodes due to t~rho
    if np.abs(t) < 0.001:
        res = -(h*k_structural**2)/(np.pi*2) *\
                (np.cos(k_structural*s)-k_structural*t/3 *
                np.sin(k_structural*s)) /\
                (1+(h*k_structural*np.sin(k_structural*s))**2+t*h**2 *
                k_structural**3*np.sin(k_structural*s) *
                np.cos(k_structural*s))
    else:
        rho = np.sqrt(t**2+(h*np.cos(k_structural*(s+t)) -
                            h*np.cos(k_structural*s))**2)
        B = h*np.cos(k_structural*(s+t))-h*np.cos(k_structural*s)-t*(-h) *\
            k_structural*np.sin(k_structural*s)
        res = 1j*k/(2*rho)*hankel1(1, k*rho)*B

    return res


def _getV(k, k_theta, height, k_structural, alpha0, A, m, n):
    alpha_m = _alpha(m, alpha0, k, k_theta, k_structural)
    # add code here


def _getW(k, k_theta, height, k_structural, alpha0, A, m, n):
    if n == m+1 or n == m-1:
        alpha_m = _alpha(m, alpha0, k, k_theta, k_structural)
        if abs(alpha_m) == 1:
            W = 0
        else:
            W = 1j*height*k_structural/(1-alpha_m**2) /\
                np.sqrt(2*np.pi*k*A)*np.exp(1j*(k*A-0.75*np.pi)) *\
                (alpha_m*np.cos(k*A*alpha_m)-1j*np.sin(k*A*alpha_m))
            if n == m+1:
                W = -W
    else:
        W = 0

    return W
