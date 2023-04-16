import numpy as np
import pyfar as pf


def paris_formula(coefficients, incident_directions):
    r"""Calculate the random-incidence coefficient
    according to Paris formula.

    .. math::
        c_{rand} = \sum c(\vartheta_S,\varphi_S) \cdot cos(\vartheta_S) \cdot w

    with the ``coefficients``, and the
    area weights ``w`` from the ``incident_directions``.
    Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    coefficients : pyfar.FrequencyData
        coefficients for different incident directions. Its cshape
        need to be (..., #incident_directions)
    incident_directions : pyfar.Coordinates
        Defines the incidence directions of each `coefficients` in a
        Coordinates object. Its cshape need to be (#incident_directions). In
        sperical coordinates the radii need to be constant. The weights need
        to reflect the area weights.

    Returns
    -------
    random_coefficient : pyfar.FrequencyData
        The random-incidence scattering coefficient.
    """
    if not isinstance(coefficients, pf.FrequencyData):
        raise ValueError("coefficients has to be FrequencyData")
    if not isinstance(incident_directions, pf.Coordinates):
        raise ValueError("incident_directions have to be None or Coordinates")
    if incident_directions.cshape[0] != coefficients.cshape[-1]:
        raise ValueError(
            "the last dimension of coefficients need be same as "
            "the incident_directions.cshape.")

    theta = incident_directions.get_sph().T[1]
    weight = np.cos(theta) * incident_directions.weights
    norm = np.sum(weight)
    coefficients_freq = np.swapaxes(coefficients.freq, -1, -2)
    random_coefficient = pf.FrequencyData(
        np.sum(coefficients_freq*weight/norm, axis=-1),
        coefficients.frequencies,
        comment='random-incidence coefficient'
    )
    return random_coefficient
