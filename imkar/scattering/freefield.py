"""Scattering calculation functions based on free-field data."""
import numpy as np
import pyfar as pf


def correlation_method(
        sample_pressure, reference_pressure, microphone_weights):
    r"""
    Calculate the incident-dependent free-field scattering coefficient.

    This function uses the Mommertz correlation method [#]_ to compute the
    scattering coefficient of the input data:

    .. math::
        s = 1 -
            \frac{|\sum_w \underline{p}_{\text{sample}}(\vartheta,\varphi)
            \cdot \underline{p}_{\text{reference}}^*(\vartheta,\varphi)
            \cdot w(\vartheta,\varphi)|^2}
            {\sum_w |\underline{p}_{\text{sample}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) \cdot \sum_w
            |\underline{p}_{\text{reference}}(\vartheta,\varphi)|^2
            \cdot w(\vartheta,\varphi) }

    where:
        - :math:`\underline{p}_{\text{sample}}` is the reflected sound
          pressure of the sample under investigation.
        - :math:`\underline{p}_{\text{reference}}` is the reflected sound
          pressure from the reference sample.
        - :math:`w` represents the area weights of the sampling, and
          :math:`\vartheta` and :math:`\varphi` are the ``colatitude``
          and ``azimuth`` angles from the
          :py:class:`~pyfar.classes.coordinates.Coordinates` object.

    The test sample is assumed to lie in the x-y-plane.

    Parameters
    ----------
    sample_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the test sample. Its cshape
        must be (..., microphone_weights.size) and broadcastable to the
        cshape of ``reference_pressure``.
    reference_pressure : :py:class:`~pyfar.classes.audio.FrequencyData`
        Reflected sound pressure or directivity of the reference sample. Its
        cshape must be (..., microphone_weights.size) and broadcastable to the
        cshape of ``sample_pressure``.
    microphone_weights : array_like
        1D array containing the area weights for the microphone positions.
        No normalization is required. Its shape must match the last dimension
        in the cshape of ``sample_pressure`` and ``reference_pressure``.

    Returns
    -------
    scattering_coefficients : :py:class:`~pyfar.classes.audio.FrequencyData`
        The scattering coefficient for each incident direction as a function
        of frequency.

    References
    ----------
    .. [#]  E. Mommertz, "Determination of scattering coefficients from the
            reflection directivity of architectural surfaces," Applied
            Acoustics, vol. 60, no. 2, pp. 201-203, June 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check input types
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise TypeError("sample_pressure must be of type pyfar.FrequencyData")
    if not isinstance(reference_pressure, pf.FrequencyData):
        raise TypeError(
            "reference_pressure must be of type pyfar.FrequencyData")
    microphone_weights = np.atleast_1d(
        np.asarray(microphone_weights, dtype=float))

    # check input dimensions
    if sample_pressure.cshape[-1] != microphone_weights.size:
        raise ValueError(
            "The last dimension of sample_pressure must match the size of "
            "microphone_weights")
    if reference_pressure.cshape[-1] != microphone_weights.size:
        raise ValueError(
            "The last dimension of reference_pressure must match the size of "
            "microphone_weights")

    if sample_pressure.cshape[:-1] != reference_pressure.cshape[:-1]:
        raise ValueError(
            "The cshape of sample_pressure and reference_pressure must be "
            "broadcastable except for the last dimension")
    # Test whether the objects are able to perform arithmetic operations.
    # e.g. does the frequency vectors match
    _ = sample_pressure + reference_pressure

    # prepare data
    microphone_weights = microphone_weights[:, np.newaxis]
    p_sample = sample_pressure.freq
    p_reference = reference_pressure.freq

    # calculate according to mommertz correlation method Equation (5)
    p_sample_sum = np.sum(microphone_weights * np.abs(p_sample)**2, axis=-2)
    p_ref_sum = np.sum(microphone_weights * np.abs(p_reference)**2, axis=-2)
    p_cross_sum = np.sum(
            p_sample * np.conj(p_reference) * microphone_weights, axis=-2)

    data_scattering_coefficient \
        = 1 - ((np.abs(p_cross_sum)**2)/(p_sample_sum*p_ref_sum))

    # create pyfar.FrequencyData object
    scattering_coefficients = pf.FrequencyData(
        data_scattering_coefficient,
        sample_pressure.frequencies)

    return scattering_coefficients
