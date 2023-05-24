import numpy as np
import pyfar as pf
from imkar import utils


def freefield(sample_pressure, microphone_weights):
    r"""
    Calculate the free-field diffusion coefficient for each incident direction
    after ISO 17497-2:2012 [1]_. See :py:func:`random_incidence`
    to calculate the random incidence diffusion coefficient.

    .. math::
        d(\vartheta_S,\varphi_S) =
            \frac{(\sum |\underline{p}_{sample}(\vartheta_R,\varphi_R)| \cdot
            N_i)^2 - \sum (|\underline{p}_{sample}(\vartheta_R,\varphi_R)|)^2
            \cdot N_i}
            {(\sum N_i - 1) \cdot \sum
            (|\underline{p}_{sample}(\vartheta_R,\varphi_R)|)^2 \cdot N_i}

    with

    .. math::
        N_i = \frac{A_i}{A_{min}}

    and ``A`` being the area weights ``microphone_weights``.

    Parameters
    ----------
    sample_pressure : pyfar.FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #microphones).
    microphone_weights : ndarray
        An array object with all weights for the microphone positions.
        Its cshape need to be (#microphones). Microphone positions need to be
        same for `sample_pressure` and `reference_pressure`.

    Returns
    -------
    diffusion_coefficients : pyfar.FrequencyData
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
    >>>     sample_pressure, mic_positions.weights)
    >>> random_d = ik.scattering.coefficient.random_incidence(
    >>>     diffusion_coefficients, incident_positions)
    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError("sample_pressure has to be FrequencyData")
    if not isinstance(microphone_weights, np.ndarray):
        raise ValueError("weights_microphones have to be a numpy.array")
    if not microphone_weights.shape[0] == sample_pressure.cshape[-1]:
        raise ValueError(
            "the last dimension of sample_pressure need be same as the "
            "weights_microphones.shape.")

    # parse angles
    N_i = microphone_weights / np.min(microphone_weights)

    # calculate according to Mommertz correlation method Equation (6)
    p_sample_abs_sq = np.moveaxis(np.abs(sample_pressure.freq)**2, -1, 0)

    p_sample_sum_sq = np.sum(
        p_sample_abs_sq**2 * N_i, axis=-1)
    p_sample_sq_sum = np.sum(
        p_sample_abs_sq * N_i, axis=-1)**2
    n = np.sum(N_i)
    diffusion_array \
        = (p_sample_sq_sum - p_sample_sum_sq) / ((n-1) * p_sample_sum_sq)
    diffusion_coefficients = pf.FrequencyData(
        np.moveaxis(diffusion_array, 0, -1),
        sample_pressure.frequencies)
    diffusion_coefficients.comment = 'diffusion coefficients'

    return diffusion_coefficients


def random(
        random_diffusions, incident_directions):
    r"""
    Calculate the random-incidence scattering coefficient
    according to Paris formula [2]_.

    .. math::
        d_{rand} = \sum d(\vartheta_S,\varphi_S) \cdot cos(\vartheta_S) \cdot w

    with the ``random_diffusions``, and the
    area weights ``w`` from the ``incident_directions``.
    Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    random_diffusions : pyfar.FrequencyData
        Diffusion coefficients for different incident directions. Its cshape
        needs to be (..., #source_directions)
    incident_directions : pyfar.Coordinates
        Defines the incidence directions of each `random_diffusions` in a
        Coordinates object. Its cshape needs to be (#source_directions). In
        sperical coordinates the radii needs to be constant. The weights need
        to reflect the area weights.

    Returns
    -------
    random_diffusion : pyfar.FrequencyData
        The random-incidence diffusion coefficient.

    References
    ----------
    .. [2]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
            CRC Press/Taylor & Francis Group, 2017.
    """
    random_diffusion = utils.paris_formula(
        random_diffusions, incident_directions)
    random_diffusion.comment = 'random-incidence diffusion coefficient'
    return random_diffusion
