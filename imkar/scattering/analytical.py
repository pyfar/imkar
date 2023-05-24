import numpy as np
import pyfar as pf

import numbers


def rectangular(frequencies, incident_angles, rectangle_width, gap_width,
                height, speed_of_sound=346.18):
    """
    This function computes the scattering coefficient of periodic rectangular
    profiles, assuming that the extent is infinite.

    This function is based on Ducourneau's method described by Embrechts
    [#]_.

    Parameters
    ----------
    frequencies : array, float
        Vector of frequencies the scattering coefficient is computed at.
    incident_angles : array, float
        Vector or value of incident angles in degree for the scattering
        coefficient calculation. Angles between 0°-90° are allowed.
    rectangle_width : float
        Width of the rectangulars in m.
    gap_width : float
        Width of the gap in m.
    height : float
        Height of the profile structure in m.
    speed_of_sound : float
        Speed of sound in m/s with which the scattering coefficient is
        computed. Default is 346.18 m/s like in Mesh2HRTF.
    Returns
    -------
    scattering_coefficient : FrequencyData
        The analytical solution for the scattering coefficient of a
        rectangular profile at the given incident angles and frequencies.

    References
    ----------
    .. [#]  J. Embrechts and A. Billon, "Theoretical Determination of the
            Random-Incidence Scattering Coefficients of Infinite Rigid
            Surfaces with a Periodic Rectangular Roughness Profile",
            Acta Acustica united with Acustica, Bd. 97, Nr. 4, S. 607-617,
            Juli 2011, doi:10.3813/AAA.918441

    Comment
    -------
    In this solution the parameter r in equation (9) is also limited to
    -N <= r <= N. The width of the well is described by the gap_width
    parameter and replaces the parameter 2A.

    """

    # check inputs
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if not isinstance(frequencies, np.ndarray) or\
            not isinstance(frequencies.item(0), numbers.Real):
        raise TypeError("frequencies has to be an array of real numbers")
    if frequencies.ndim > 1:
        raise ValueError("frequencies has to be 1-dimensional")

    incident_angles = np.atleast_1d(np.asarray(incident_angles, dtype=float))
    if incident_angles.ndim > 1:
        raise ValueError("incident_angles has to be 1-dimensional")
    if (np.any(incident_angles <= 0)) or (np.any(incident_angles >= 90)):
        raise ValueError("incident_angles have to be between 0° and 90°")

    if not isinstance(rectangle_width, numbers.Real) or rectangle_width <= 0:
        raise TypeError("rectangle_width has to be a real number >0")
    if not isinstance(gap_width, numbers.Real) or gap_width <= 0:
        raise TypeError("gap_width has to be a real number >0")
    if not isinstance(height, numbers.Real) or height <= 0:
        raise TypeError("height has to be a real number >0")
    if not isinstance(speed_of_sound, numbers.Real) or speed_of_sound <= 0:
        raise TypeError("speed_of_sound has to be a real number >0")

    # Initialization
    incident_angles = incident_angles*np.pi/180  # from degree to radiant
    # vector of corresponding wavelength in air
    lambda_air = speed_of_sound/frequencies
    k = 2*np.pi/lambda_air  # vector of wavenumbers
    eps = 1E-3  # stop criterion for infinite sum over u
    n_max = 50  # truncation parameter for numbers of outgoing waves
    n_max2 = 2*n_max+1  # counter from -N to N including zero
    length = gap_width + rectangle_width  # length of spatial period

    x_u = np.zeros((1, n_max), dtype=complex)
    n = np.arange(-n_max, n_max+1)[:, np.newaxis]
    previous_frequency = 0

    n_frequency = frequencies.size  # number of computed frequencies
    n_phi = incident_angles.size  # number of computed incident angles
    scattering_coefficient = np.zeros((n_frequency, n_phi))

    # calculation
    for iFrequencies in range(0, n_frequency):
        for iPhi in range(0, n_phi):
            phi_0 = incident_angles[iPhi]
            alpha_n = np.cos(phi_0) + n*lambda_air[iFrequencies]/length
            beta_n = np.conj(-(1j*np.emath.sqrt(alpha_n**2-1)))

            # helping variables
            k_frequency = k[iFrequencies]
            k_alpha_n = (k_frequency*alpha_n[:]).T
            jk_alpha_n_over_width = 1j*k_alpha_n/gap_width

            y_nr = np.zeros((n_max2, n_max2), dtype=complex)

            for iN in range(1, (n_max2+1)):
                u_max = np.zeros((1, n_max2), dtype=int)
                u_max_element = 0
                for iR in range(1, (n_max2+1)):
                    u_max[0, iR-1] = 0
                    previous_temp_U = 0
                    err = 1
                    while err > eps:
                        u_max[0, iR-1] = u_max[0, iR-1] + n_max2-1

                        if iR == 1:
                            u_un = np.zeros((u_max.item((iR-1))+1, n_max2),
                                            dtype=complex)
                        elif u_max[0, iR-1] > u_max_element:
                            u_un = np.zeros((u_max.item((iR-1))+1, n_max2),
                                            dtype=complex)

                        if ((iR == 1) or (u_max[0, iR-1] > u_max_element) or
                           ((iFrequencies+1) > previous_frequency)):

                            iU = np.arange(0., round(u_max.item((iR-1)))+1,
                                           dtype=int)[:, np.newaxis]
                            uVals = np.tile(iU, (1, n_max2))
                            k_x_u = iU*np.pi/gap_width
                            x_u = np.emath.sqrt(k_frequency**2 - k_x_u**2)

                            # Calculation of U_un according to formula(6)

                            case_a = abs((k_alpha_n-k_x_u)) < 1E-3
                            case_b = abs((k_alpha_n+k_x_u)) < 1E-3
                            case_c = np.ones(np.shape(case_a))-(case_a+case_b)
                            if np.sum(case_a) > 0:
                                u_un[np.nonzero(case_a)] = 0.5 *\
                                    np.exp(1j*uVals[np.nonzero(case_a)]
                                           * np.pi/2)
                            if np.sum(case_a) > 0:
                                u_un[np.nonzero(case_b)] = 0.5 * \
                                    np.exp(-1j*uVals[np.nonzero(case_b)]
                                           * np.pi/2)
                            temp_help = \
                                (jk_alpha_n_over_width /
                                 (k_x_u**2-k_alpha_n**2))\
                                * ((np.exp(1j*k_alpha_n*gap_width/2))
                                   - ((-1)**uVals *
                                   (np.exp(-1j*k_alpha_n*gap_width/2))))
                            u_un[np.nonzero(case_c)] = \
                                temp_help[np.nonzero(case_c)]

                        # sum of y_nr in equation(9)
                        temp_U = np.sum([(np.sign(iU)+1)*x_u *
                                        np.tanh(1j*x_u*height) *
                                        np.conjugate(u_un[iU, iN-1]) *
                                        u_un[iU, iR-1]],
                                        axis=1)

                        err = np.abs((previous_temp_U-temp_U)/previous_temp_U)
                        previous_temp_U = temp_U
                        previous_frequency = iFrequencies+1

                        if u_max_element < u_max.item((iR-1)):
                            u_max_element = u_max.item((iR-1))

                    # y_nr according to formula(9)
                    y_nr[iN-1, iR-1] = \
                        temp_U*gap_width/(k_frequency*length)

            # solve with least-mean-square for r_n in equation(9) with a*r_n=b
            b = np.sin(phi_0)*(2-(abs(np.sign(n))+1)) + \
                ((y_nr[:, n_max]).reshape(n_max2, 1))
            a = np.diagflat(beta_n) - y_nr
            r_n = np.linalg.lstsq(a, b, 1E-6)

            valid_ids = 1*(np.abs(alpha_n) <= 1)
            checksum = np.sum((np.abs(r_n[0][np.nonzero(valid_ids)])**2)
                              * beta_n[np.nonzero(valid_ids)]/np.sin(phi_0))
            abs_error = np.abs(checksum-1)
            print(f'The absolute error is {abs_error}.')
            # Directional scattering coefficient according to formula(3)
            scattering_coefficient[iFrequencies, iPhi] = \
                1-np.abs(r_n[0][n_max])**2
    return pf.FrequencyData(scattering_coefficient.T, frequencies)
