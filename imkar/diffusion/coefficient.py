import numpy as np
import pyfar as pf
from imkar import integrate


def freefield(sample_pressure, mic_positions):
    """
    Calculate the free-field diffusion coefficient for each incident direction
    using the Mommertz correlation method [1]_. See :py:func:`random_incidence`
    to calculate the random incidence diffusion coefficient.


    Parameters
    ----------
    sample_pressure : FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #theta_incident_positions, #phi_incident_positions,
        #angle1, #angle2)
    mic_positions : Coordinates
        A Coordinate object with all microphone positions. Its cshape need to
        be (#angle1, #angle2). In sperical coordinates the radii need to
        be constant. It need to be same for `sample_pressure`.

    Returns
    -------
    diffusion_coefficients : FrequencyData
        The diffusion coefficient for each plane wave direction.


    References
    ----------
    .. [1]  ISO 17497-2:2012, Sound-scattering properties of surfaces.
            Part 2: Measurement of the directional diffusion coefficient in a
            free field. Geneva, Switzerland: International Organization for
            Standards, 2012.


    Examples
    --------
    Calculate free-field diffusion coefficients and then the random incidence
    diffusion coefficient.

    >>> import imkar as ik
    >>> diffusion_coefficients = ik.diffusion.coefficient.freefield(
    >>>     sample_pressure, mic_positions)
    >>> random_d = ik.diffusion.coefficient.random_incidence(
    >>>     diffusion_coefficients, incident_positions)
    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError("sample_pressure has to be FrequencyData")
    if not isinstance(mic_positions, pf.Coordinates):
        raise ValueError("microphone positions have to be Coordinates")

    # calculate according to Mommertz correlation method Equation (6)
    p_sample_sq = pf.FrequencyData(
        np.abs(sample_pressure.freq)**2, sample_pressure.frequencies)

    p_sample_sum_sq = integrate.surface_sphere(
        p_sample_sq*p_sample_sq, mic_positions)
    p_sample_sq_sum = integrate.surface_sphere(p_sample_sq, mic_positions)**2
    n = 1
    diffusion_coefficients \
        = (p_sample_sum_sq - p_sample_sq_sum) / ((n-1) * p_sample_sq_sum)
    diffusion_coefficients.comment = 'diffusion coefficients'

    return diffusion_coefficients
