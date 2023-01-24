import numpy as np
import pyfar as pf
from imkar import integrate


def freefield(sample_pressure, reference_pressure, mic_positions):
    """
    Calculate the free-field scattering coefficient for each incident direction
    using the Mommertz correlation method [#]_. See :py:func:`random_incidence`
    to calculate the random incidence scattering coefficient.


    Parameters
    ----------
    sample_pressure : FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #theta_incident_positions, #phi_incident_positions,
        #angle1, #angle2)
    reference_pressure : FrequencyData
        Reflection Reflection sound pressure or directivity of the test
        reference sample. It has the same shape as `sample_pressure`.
    mic_positions : Coordinates
        A Coordinate object with all microphone positions. Its cshape need to
        be (#angle1, #angle2). In sperical coordinates the radii need to
        be constant. It need to be same for `sample_pressure` and
        `reference_pressure`.

    Returns
    -------
    scattering_coefficients : FrequencyData
        The scattering coefficient for each plane wave direction.


    References
    ----------
    .. [#]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201–203, Juni 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    Examples
    --------
    Calculate freefield scattering coefficients and then the random incidence
    scattering coefficient.

    >>> import imkar
    >>> scattering_coefficients = imkar.scattering.coefficient.freefield(
    >>>     sample_pressure, reference_pressure, mic_positions)
    >>> random_s = imkar.scattering.coefficient.random_incidence(
    >>>     scattering_coefficients, incident_positions)
    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError("sample_pressure has to be FrequencyData")
    if not isinstance(reference_pressure, pf.FrequencyData):
        raise ValueError("reference_pressure has to be FrequencyData")
    if not isinstance(mic_positions, pf.Coordinates):
        raise ValueError("microphone positions have to be Coordinates")
    if not sample_pressure.cshape == reference_pressure.cshape:
        raise ValueError(
            "sample_presure and reference_pressure have to have the "
            "same cshape.")
    if any(sample_pressure.frequencies != reference_pressure.frequencies):
        raise ValueError(
            "sample_presure and reference_pressure have to have the "
            "same frequencies.")

    # calculate according to mommertz correlation method Equation (5)
    p_sample_abs = pf.FrequencyData(
        np.abs(sample_pressure.freq), sample_pressure.frequencies)
    p_reference_abs = pf.FrequencyData(
        np.abs(reference_pressure.freq), reference_pressure.frequencies)
    p_sample_sq = p_sample_abs*p_sample_abs
    p_reference_sq = p_reference_abs*p_reference_abs
    p_reference_conj = pf.FrequencyData(
        np.conj(reference_pressure.freq), reference_pressure.frequencies)
    p_cross = sample_pressure*p_reference_conj

    p_sample_sum = integrate.surface_sphere(p_sample_sq, mic_positions)
    p_ref_sum = integrate.surface_sphere(p_reference_sq, mic_positions)
    p_cross_sum = integrate.surface_sphere(p_cross, mic_positions)
    p_cross_sum_abs = pf.FrequencyData(
        np.abs(p_cross_sum.freq), p_cross_sum.frequencies)

    scattering_coefficients \
        = 1 - ((p_cross_sum_abs**2)/(p_sample_sum*p_ref_sum))
    scattering_coefficients.comment = 'scattering coefficient'

    return scattering_coefficients


def random_incidence(scattering_coefficient, incident_positions):
    """Calculate the random-incidence scattering coefficient
    according to Paris formula. Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    scattering_coefficient : FrequencyData
        The scattering coefficient for each plane wave direction. Its cshape
        need to be (..., #angle1, #angle2)
    incident_positions : Coordinates
        Defines the incidence directions of each `scattering_coefficient` in a
        Coordinates object. Its cshape need to be (#angle1, #angle2). In
        sperical coordinates the radii  need to be constant.

    Returns
    -------
    random_scattering : FrequencyData
        The random-incidence scattering coefficient.
    """
    if not isinstance(scattering_coefficient, pf.FrequencyData):
        raise ValueError("scattering_coefficient has to be FrequencyData")
    if (incident_positions is not None) & \
            ~isinstance(incident_positions, pf.Coordinates):
        raise ValueError("incident_positions have to be None or Coordinates")

    sph = incident_positions.get_sph()
    theta = sph[..., 1]
    weight = np.sin(2*theta)  # sin(2*theta)
    norm = np.sum(weight)
    random_scattering = scattering_coefficient*weight/norm
    random_scattering.freq = np.sum(random_scattering.freq, axis=-2)
    random_scattering.freq = np.sum(random_scattering.freq, axis=-2)
    random_scattering.comment = 'random-incidence scattering coefficient'
    return random_scattering
