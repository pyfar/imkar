import numpy as np
import pyfar as pf

def sinusoid(frequency_vector, phi_vector, theta):
    """
    This function computes the scattering coefficient of a sine-shaped surface
    (with infinite extend) for a given set of frequencies and angles.

    The function implements the method by Holford and Urusovskii with the
    implementation given by Embrechts [#]_.

    Parameters
    ----------
    frequency_vector : array, double
        Vector of frequencies the scattering coefficient is computed at.
    phi_vector : array, double
        Vector of incidences phi the scattering coefficient is computed at.
        Values between -90°-90° allowed.
    theta : double
        Out-of-plane incident angle of incoming waves. Values between -90°-90°
        allowed.

    Returns
    -------
    s : FrequencyData
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
    c = 343.901  # ITA constant
    phi_vector = phi_vector*np.pi/180  # from degree to radiant
    length = 0.354/2  #structural length
    height = 0.051/2  #structural height
    k_structural = 2*np.pi/length  # structural wavenumber
    # vector of corresponding wavelength in air
    lambda_air = c/frequency_vector[:]
    k = 2*np.pi/lambda_air
    k_theta = k*np.sin(theta)  # account out-of-plane incidence
    eta0 = 0

    # order of maximum reflection
    n_max = np.ceil(2*length*frequency_vector.max()/c)
    s_coeff = np.zeros((frequency_vector.size(),phi_vector.size()))





    return 0
