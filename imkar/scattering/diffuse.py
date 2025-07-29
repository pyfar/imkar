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


def calculation(
        reverberation_times, speed_of_sound,
        air_attenuation_coefficient,
        volume, surface_sample) -> tuple:
    """
    Calculate the diffuse scattering coefficient after ISO 17497-1:2004.

    Measurement conditions for the four different reverberation times
    and the corresponding speed of sound and air attenuation
    coefficients based on Table 2 in ISO 17497-1:2004 [#]_.

    +------------------------+-------------+--------------+
    | measurement condition  | test sample |  turntable   |
    +------------------------+-------------+--------------+
    |           1            | not present | not rotating |
    +------------------------+-------------+--------------+
    |           2            |   present   | not rotating |
    +------------------------+-------------+--------------+
    |           3            | not present |   rotating   |
    +------------------------+-------------+--------------+
    |           4            |   present   |   rotating   |
    +------------------------+-------------+--------------+

    Parameters
    ----------
    reverberation_times : pf.FrequencyData
        The reverberation times in seconds of the measurement conditions of
        cshape (..., 4).
    speed_of_sound : np.ndarray
        is the speed of sound in air, in metres per second (m/s),
        during the measurement of each measurement conditions of
        shape (..., 4).
    air_attenuation_coefficient : pf.FrequencyData
        the energy attenuation coefficient of air, in reciprocal metres
        (:math:`m^-1`), calculated according to ISO 9613-1,
        using the temperature and relative humidity during the measurement
        of each measurement conditions of cshape (..., 4).
    volume : float
        volume of the reverberation room, in cubic metres (:math:`m^3`).
    surface_sample : float
        is the area of the test sample, in square metres (:math:`m^2`).

    Returns
    -------
    scattering : pf.FrequencyData
        The random-incidence scattering coefficient.
    s_base : pf.FrequencyData
        The base plate scattering coefficient.
    alpha_s : pf.FrequencyData
        The random-incidence absorption coefficient.
    alpha_spec : pf.FrequencyData
        The random-incidence specular absorption coefficient.

    References
    ----------
    .. [#] ISO 17497-1:2004, Sound-scattering properties of surfaces. Part 1:
           Measurement of the random-incidence scattering coefficient in a
           reverberation room. Geneva, Switzerland: International Organization
           for Standards, 2004.
    """
    T_1 = reverberation_times[..., 0]
    T_2 = reverberation_times[..., 1]
    T_3 = reverberation_times[..., 2]
    T_4 = reverberation_times[..., 3]
    c_1 = speed_of_sound[..., 0]
    c_2 = speed_of_sound[..., 1]
    c_3 = speed_of_sound[..., 2]
    c_4 = speed_of_sound[..., 3]
    m_1 = air_attenuation_coefficient[..., 0]
    m_2 = air_attenuation_coefficient[..., 1]
    m_3 = air_attenuation_coefficient[..., 2]
    m_4 = air_attenuation_coefficient[..., 3]
    V = volume
    S = surface_sample

    # random incident absorption coefficient
    alpha_s = 55.3 * V/S * (1/(c_2*T_2) - 1/(c_1*T_1)) - 4*V/S * (m_2-m_1)

    # specular absorption coefficient
    alpha_spec = 55.3 * V/S * (1/(c_4*T_4) - 1/(c_3*T_3)) - 4*V/S * (m_4-m_3)

    # calculate scattering coefficient
    scattering = (alpha_spec - alpha_s) / (1 - alpha_s)

    s_base = 55.3*V/S*(1/(c_3*T_3) - 1/(c_1*T_1)) - 4*V/S*(m_3-m_1)

    return scattering, s_base, alpha_s, alpha_spec
