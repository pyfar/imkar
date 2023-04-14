import numpy as np
import pyfar as pf


def freefield(sample_pressure, reference_pressure, microphone_weights):
    r"""
    Calculate the free-field scattering coefficient for each incident direction
    using the Mommertz correlation method [1]_:

    .. math::
        s(\vartheta_S,\varphi_S) = 1 -
            \frac{|\sum \underline{p}_{sample}(\vartheta_R,\varphi_R) \cdot
            \underline{p}_{reference}^*(\vartheta_R,\varphi_R) \cdot w|^2}
            {\sum |\underline{p}_{sample}(\vartheta_R,\varphi_R)|^2 \cdot w
            \cdot \sum |\underline{p}_{reference}(\vartheta_R,\varphi_R)|^2
            \cdot w }

    with the ``sample_pressure``, the ``reference_pressure``, and the
    area weights ``weights_microphones``. See
    :py:func:`random_incidence` to calculate the random incidence
    scattering coefficient.

    Parameters
    ----------
    sample_pressure : pyfar.FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #microphones).
    reference_pressure : pyfar.FrequencyData
        Reflected sound pressure or directivity of the
        reference sample. Needs to have the same cshape and frequencies as
        `sample_pressure`.
    microphone_weights : np.ndarray
        Array containing the area weights for the microphone positions.
        Its shape needs to be (#microphones), so it matches the last dimension
        in the cshape of `sample_pressure` and `reference_pressure`.

    Returns
    -------
    scattering_coefficients : pyfar.FrequencyData
        The scattering coefficient for each incident direction.


    References
    ----------
    .. [1]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201-203, June 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError(
            "sample_pressure has to be a pyfar.FrequencyData object")
    if not isinstance(reference_pressure, pf.FrequencyData):
        raise ValueError(
            "reference_pressure has to be a pyfar.FrequencyData object")
    if not isinstance(microphone_weights, np.ndarray):
        raise ValueError("microphone_weights have to be a numpy.array")
    if sample_pressure.cshape != reference_pressure.cshape:
        raise ValueError(
            "sample_pressure and reference_pressure have to have the "
            "same cshape.")
    if microphone_weights.shape[0] != sample_pressure.cshape[-1]:
        raise ValueError(
            "the last dimension of sample_pressure need be same as the "
            "microphone_weights.shape.")
    if not np.allclose(
            sample_pressure.frequencies, reference_pressure.frequencies):
        raise ValueError(
            "sample_pressure and reference_pressure have to have the "
            "same frequencies.")

    # calculate according to mommertz correlation method Equation (5)
    p_sample = np.moveaxis(sample_pressure.freq, -1, 0)
    p_reference = np.moveaxis(reference_pressure.freq, -1, 0)
    p_sample_sq = np.abs(p_sample)**2
    p_reference_sq = np.abs(p_reference)**2
    p_cross = p_sample * np.conj(p_reference)

    p_sample_sum = np.sum(microphone_weights * p_sample_sq, axis=-1)
    p_ref_sum = np.sum(microphone_weights * p_reference_sq, axis=-1)
    p_cross_sum = np.sum(microphone_weights * p_cross, axis=-1)

    data_scattering_coefficient \
        = 1 - ((np.abs(p_cross_sum)**2)/(p_sample_sum*p_ref_sum))

    scattering_coefficients = pf.FrequencyData(
        np.moveaxis(data_scattering_coefficient, 0, -1),
        sample_pressure.frequencies)
    scattering_coefficients.comment = 'scattering coefficient'

    return scattering_coefficients


def random(
        scattering_coefficients, incident_positions):
    r"""Calculate the random-incidence scattering coefficient
    according to Paris formula.

    .. math::
        s_{rand} = \sum s(\vartheta_S,\varphi_S) \cdot cos(\vartheta_S) \cdot w

    with the ``scattering_coefficients``, and the
    area weights ``w`` from the ``incident_positions``.
    Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    scattering_coefficients : pyfar.FrequencyData
        Scattering coefficients for different incident directions. Its cshape
        need to be (..., #source_directions)
    incident_directions : pyfar.Coordinates
        Defines the incidence directions of each `scattering_coefficients` in a
        Coordinates object. Its cshape need to be (#source_directions). In
        sperical coordinates the radii need to be constant. The weights need
        to reflect the area weights.

    Returns
    -------
    random_scattering : pyfar.FrequencyData
        The random-incidence scattering coefficient.
    """
    if not isinstance(scattering_coefficients, pf.FrequencyData):
        raise ValueError("scattering_coefficients has to be FrequencyData")
    if not isinstance(incident_positions, pf.Coordinates):
        raise ValueError("incident_positions have to be None or Coordinates")
    if incident_positions.cshape[0] != scattering_coefficients.cshape[-1]:
        raise ValueError(
            "the last dimension of scattering_coefficients need be same as "
            "the incident_positions.cshape.")

    theta = incident_positions.get_sph().T[1]
    weight = np.cos(theta) * incident_positions.weights
    norm = np.sum(weight)
    coefficients = np.swapaxes(scattering_coefficients.freq, -1, -2)
    random_scattering = pf.FrequencyData(
        np.sum(coefficients*weight/norm, axis=-1),
        scattering_coefficients.frequencies,
        comment='random-incidence scattering coefficient'
    )
    return random_scattering
