"""Utilities for imkar."""
import numpy as np
import pyfar as pf


def paris_formula(coefficients, incident_directions):
    r"""
    Calculate the random-incidence coefficient
    according to Paris formula.

    The implementation follows the Equation 2.53 from [#]_ and is
    discretized as:

    .. math::
        c_{rand} = \sum_{\Omega_S} c(\Omega_S) \cdot |\Omega_S \cdot n| \cdot w

    with the ``coefficients`` :math:`c`, and the
    area weights :math:`w` from the ``incident_directions``.
    :math:`|\Omega_S \cdot n|` represent the cosine of the angle between the
    surface normal and the incident direction.

    .. note::
        The incident directions should be
        equally distributed to get a valid result.

    Parameters
    ----------
    coefficients : pyfar.FrequencyData
        coefficients for different incident directions. Its cshape
        needs to be (..., n_incident_directions)
    incident_directions : pyfar.Coordinates
        Defines the incidence directions of each `coefficients` in a
        Coordinates object. Its cshape needs to be (n_incident_directions). In
        sperical coordinates the radii needs to be constant. The weights need
        to reflect the area weights.

    Returns
    -------
    random_coefficient : pyfar.FrequencyData
        The random-incidence scattering coefficient.

    References
    ----------
    .. [#]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
            CRC Press/Taylor & Francis Group, 2017.
    """
    if not isinstance(coefficients, pf.FrequencyData):
        raise ValueError("coefficients has to be FrequencyData")
    if not isinstance(incident_directions, pf.Coordinates):
        raise ValueError("incident_directions have to be None or Coordinates")
    if incident_directions.cshape[0] != coefficients.cshape[-1]:
        raise ValueError(
            "the last dimension of coefficients needs be same as "
            "the incident_directions.cshape.")

    theta = incident_directions.colatitude
    weight = np.cos(theta) * incident_directions.weights
    norm = np.sum(weight)
    coefficients_freq = np.swapaxes(coefficients.freq, -1, -2)
    random_coefficient = pf.FrequencyData(
        np.sum(coefficients_freq*weight/norm, axis=-1),
        coefficients.frequencies,
        comment='random-incidence coefficient',
    )
    return random_coefficient
