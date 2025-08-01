"""
This module contains functions for diffuse scattering calculations based on
ISO 17497-1:2004.
"""
import numpy as np
import pyfar as pf


def maximum_sample_absorption_coefficient(frequencies) -> pf.FrequencyData:
    """Maximum absorption coefficient of the test sample.

    Based on section 6.3.4 in ISO 17497-1:2004 [#]_ the absorption coefficient
    of the test sample should not exceed a value of :math:`alpha_s=0.5`.
    However, if sound  absorption is part of the sound-scattering structure,
    this absorption shall also be present in the test sample.

    Parameters
    ----------
    frequencies : numpy.ndarray
        The frequencies at which the absorption coefficient is calculated.

    Returns
    -------
    alpha_s_max : pyfar.FrequencyData
        The maximum sample absorption coefficient.

    References
    ----------
    .. [#] ISO 17497-1:2004, Sound-scattering properties of surfaces. Part 1:
           Measurement of the random-incidence scattering coefficient in a
           reverberation room. Geneva, Switzerland: International Organization
           for Standards, 2004.

    """
    # input checks
    try:
        frequencies = np.asarray(frequencies, dtype=float)
    except ValueError as exc:
        raise TypeError(
            "frequencies must be convertible to a float array.") from exc
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a 1D array.")

    # Calculate the maximum absorption coefficient
    return pf.FrequencyData(
        data=np.ones_like(frequencies) * 0.5,
        frequencies=frequencies,
        comment="Maximum absorption coefficient of the test sample",
    )


def maximum_baseplate_scattering_coefficient(N: int = 1) -> pf.FrequencyData:
    """Maximum scattering coefficient for the base plate alone.

    This is based on Table 1 in ISO 17497-1:2004 [#]_.

    Parameters
    ----------
    N : int
        ratio of any linear dimension in a physical scale model to the
        same linear dimension in full scale (1:N). The default is N=1.

    Returns
    -------
    s_base_max : pyfar.FrequencyData
        The maximum baseplate scattering coefficient.

    References
    ----------
    .. [#] ISO 17497-1:2004, Sound-scattering properties of surfaces. Part 1:
           Measurement of the random-incidence scattering coefficient in a
           reverberation room. Geneva, Switzerland: International Organization
           for Standards, 2004.

    """
    if not isinstance(N, int):
        raise TypeError("N must be a positive integer.")
    if N <= 0:
        raise TypeError("N must be a positive integer.")
    frequencies = [
        100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    ]
    s_base_max = [
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10,
        0.10, 0.10, 0.15, 0.15, 0.15, 0.20, 0.20, 0.20, 0.25,
    ]
    return pf.FrequencyData(
        data=s_base_max,
        frequencies=np.array(frequencies)/N,
        comment="Maximum scattering coefficient of the baseplate",
    )
