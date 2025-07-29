"""Scattering calculation functions based on free-field data."""
import numpy as np
import pyfar as pf


def correlation_method(
        sample_pressure, reference_pressure, microphone_weights):
    r"""
    Calculate the scattering coefficient from free-field reflection pattern.

    This function uses the Mommertz correlation method [#]_ to compute the
    scattering coefficient by averaging over the microphone positions:

    .. math::
        s = 1 -
            \frac{|\sum_{\Omega_r} \underline{p}_{\text{sample}}(\Omega_r)
            \cdot \underline{p}_{\text{reference}}^*(\Omega_r)
            \cdot w(\Omega_r)|^2}
            {\sum_{\Omega_r} |\underline{p}_{\text{sample}}(\Omega_r)|^2
            \cdot w(\Omega_r) \cdot \sum_{\Omega_r}
            |\underline{p}_{\text{reference}}(\Omega_r)|^2
            \cdot w(\Omega_r) }

    where:
        - :math:`\underline{p}_{\text{sample}}` is the reflected sound
          pressure of the sample under investigation.
        - :math:`\underline{p}_{\text{reference}}` is the reflected sound
          pressure from the reference sample.
        - :math:`\Omega_r` represents the solid angle of the microphone
          positions and :math:`w(\Omega_r)` represents its area weights.

    Parameters
    ----------
    sample_pressure : pyfar.FrequencyData, pyfar.Signal
        Reflected sound pressure or directivity of the test sample. Its cshape
        must be (..., microphone_weights.size) and broadcastable to the
        cshape of ``reference_pressure``. The frequency vectors of both
        ``sample_pressure`` and ``reference_pressure`` must match.
    reference_pressure : pyfar.FrequencyData, pyfar.Signal
        Reflected sound pressure or directivity of the reference sample. Its
        cshape must be (..., microphone_weights.size) and broadcastable to the
        cshape of ``sample_pressure``. The frequency vectors of both
        ``sample_pressure`` and ``reference_pressure`` must match.
    microphone_weights : array_like
        1D array containing the area weights for the microphone positions.
        No normalization is required. Its shape must match the last dimension
        in the cshape of ``sample_pressure`` and ``reference_pressure``.

    Returns
    -------
    scattering_coefficients : pyfar.FrequencyData
        The scattering coefficient of the broadcasted cshape of
        ``sample_pressure`` and ``reference_pressure``, excluding the
        last dimension.

    References
    ----------
    .. [#]  E. Mommertz, "Determination of scattering coefficients from the
            reflection directivity of architectural surfaces," Applied
            Acoustics, vol. 60, no. 2, pp. 201-203, June 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check input types
    if not isinstance(sample_pressure, (pf.FrequencyData, pf.Signal)):
        raise TypeError(
            "sample_pressure must be of type pyfar.FrequencyData or "
            "pyfar.Signal")
    if not isinstance(reference_pressure, (pf.FrequencyData, pf.Signal)):
        raise TypeError(
            "reference_pressure must be of type pyfar.FrequencyData or "
            "pyfar.Signal")
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

    data_scattering_coefficient = 1 - np.abs(p_cross_sum)**2/(
        p_sample_sum*p_ref_sum)

    # create pyfar.FrequencyData object
    scattering_coefficients = pf.FrequencyData(
        data_scattering_coefficient,
        sample_pressure.frequencies)

    return scattering_coefficients
