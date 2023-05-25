import numpy as np
import pyfar as pf

from scipy import integrate
from scipy.special import jv, hankel1


def sinusoid(frequencies, incident_angle, theta, length=0.177, height=0.051,
             c=343):
    """
    This function computes the scattering coefficient of a sine-shaped surface
    (with infinite extend) for a given set of frequencies and angles.

    The function implements the method by Holford and Urusovskii with the
    implementation given by Embrechts [#]_.

    Parameters
    ----------
    frequency_vector : array, double
        Vector of frequencies the scattering coefficient is computed at.
    incident_angle : array, double
        Vector of incidences phi the scattering coefficient is computed at.
        Values between -90°-90° allowed.
    theta : double
        Out-of-plane incident angle of incoming waves. Values between -90°-90°
        allowed.
    length : double
        Structural length of sinusoidal sample. Without an input, the length of
        177mm is used (test sample).
    height : double
        Structural height/depths of the sinusoidal sample (peak-to-peak).
        Without an input, the height of 51mm is used (test sample).
    c : double
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
    incident_angle = incident_angle*np.pi/180  # from degree to radiant
    k_structural = 2*np.pi/length  # structural wavenumber
    # vector of corresponding wavelength in air
    lambda_air = c/frequencies
    k = 2*np.pi/lambda_air
    k_theta = k*np.sin(theta)  # account out-of-plane incidence
    eta0 = 0  # normalized admittance
    height = height/2  # half peak-to-peak

    # order of maximum reflection
    n_max = np.ceil(2*length*frequencies.max()/c)
    s_coeff = np.zeros((frequencies.size(), incident_angle.size()))

    for iFrequencies in range (0,np.size(frequencies)):
        sPhi = np.zeros(np.size(incident_angle),1)
        for iPhis in range (0,np.size(incident_angle)):
            alpha_0 = np.cos(incident_angle[iPhis])
            gamma_0 = np.sin(incident_angle[iPhis])

            alpha_nn = alpha(np.arange(-n_max, n_max+1, 1), k[iFrequencies],
                            k_theta[iFrequencies], k_structural)
            alpha_n = alpha_nn[:,0]
            n = alpha_nn[:,1]

            if np.size(n) == 0 or (np.size(n) == 1 and n == 0):
                continue

            gamma_n = np.sin(np.arccos(alpha_n))

            nmax2 = np.max(np.abs(n))
            N = 2*nmax2+1
            n_vec = np.arange(-nmax2, nmax2+1, 1)
            phi_hat = phi_hat(n_vec, k[iFrequencies]*height*gamma_0)
            U = np.zeros((N))

            for iM in range(0,np.size(n)):
                for iN in range(0, np.size(n)):
                    if (n_vec[iM]-n_vec[iN]) % 2:
                        A = getA(k[iFrequencies], k_theta[iFrequencies],
                                 height, k_structural, alpha_0, n[iM], n[iN])
                        U[n[iM]+nmax2+2,n[iN]+nmax2+2] = \
                            getV(k[iFrequencies], k_theta[iFrequencies],
                                 height, k_structural, alpha_0, A, n[iM],
                                 n[iN]) + getW(k[iFrequencies],
                                               k_theta[iFrequencies], height,
                                               k_structural, alpha_0, A, n[iM],
                                               n[iN])
            A = np.identity(N) - U
            b = 2*phi_hat

            Phi = np.linalg.lstsq(A, b)
            R = np.zeros((N))
            for iM in range(0, np.size(n)):
                for iN in range(0, np.size(n)):
                    R[n[iM]+nmax2+2, n[iN]+nmax2+2] = \
                        1/(2*gamma_n[iM])*Phi[iN] * \
                            phi_hat(n_vec[iM]-n_vec[iN], k[iFrequencies] *
                                    height*gamma_n[iM] *
                                    (gamma_n[iM]+(n_vec[iM]-n_vec[iN]) *
                                     k_structural*alpha_n[iM] /
                                     (k[iFrequencies]*gamma_n[iM])+eta0))

            R = np.sum(R,2)
            R = R[n+nmax2+2]

            normTest = np.sum(np.abs(R)**2*gamma_n[:])/gamma_0
            R0 = R[n==0]
            sPhi[iPhis] = 1-np.min(1,np.abs(R0)**2)
        s_coeff[iPhis,:] = sPhi
    s_coeff = np.column_stack(s_coeff, sPhi)

    # helping functions
    # -------------------------------------------------------------------------
    def alpha(n, alpha0, k, k_theta, k_structural):
        alpha_f = (k_theta/k)**2 + alpha0 + n*k_structural/k
        valid_alpha = (np.abs(alpha_f) <= 1)
        n = n[np.nonzero(valid_alpha)]
        alpha_f = alpha_f[np.nonzero(valid_alpha)]

        return np.column_stack((alpha_f, n))

    def phi_hat(m, k_h_gamma0):
        phi_f = np.zeros((np.size(m), 1))
        for iM in range(0, np.size(m)):
            phi_f[iM] = (-1)**m[iM]*jv(m[iM], k_h_gamma0)

        return phi_f

    def getA(k, k_theta, height, k_structural, alpha0, m, n):
        if n == m+1 or n == m-1:
            alpha_m = alpha(m, alpha0, k, k_theta, k_structural)
            alpha_n = alpha(n, alpha0, k, k_theta, k_structural)
            if np.abs(alpha_m) == 1 or np.abs(alpha_n) == 1:
                A = 1/k*np.max(40, 200*(k*height)**2)
            else:
                A = 1/k*np.max(40, 200*(k*height)**2, 50/np.min
                               (1+alpha_m, 1-alpha_m, 1-alpha_n, 1+alpha_n))
        else:
            A = 1/k*np.max(40, 200*(k*height)**2)

        return A

    def int2func(s, t, h, k, k_structural):
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

    def getV(k, k_theta, height, k_structural, alpha0, A, m, n):
        alpha_m = alpha(m, alpha0, k, k_theta, k_structural)
        # add code here

    def getW(k, k_theta, height, k_structural, alpha0, A, m, n):
        if n == m+1 or n == m-1:
            alpha_m = alpha(m, alpha0, k, k_theta, k_structural)
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

    return pf.FrequencyData(np.transpose(s_coeff), frequencies)


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
