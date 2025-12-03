"""Utilities for imkar."""
import numpy as np
import pyfar as pf


def paris_formula(coefficients, colatitude_rad, area_weights):
    r"""
    Calculate the random-incidence coefficient
    according to the Paris formula.

    The Paris formula computes the random-incidence coefficient based on
    directional coefficients. It is valid for scattering and absorption
    coefficients and requires equally distributed incident directions
    to get a valid result.
    
    The implementation follows the Equation 2.53 from [#]_ and is
    discretized as:

    .. math::
        c_{rand} = \sum_{\Omega_S} c(\Omega_S) \cdot \cos(\theta) \cdot w

    with the `coefficients` :math:`c`, and the
    area weights :math:`w` from the `incident_directions`.
    :math:`\theta` represents the angle between the
    surface normal and the incident direction.

    Parameters
    ----------
    coefficients : pyfar.FrequencyData
        coefficients for different incident directions. Its cshape
        needs to be (..., n_incident_directions).
    colatitude_rad : pyfar.Coordinates
        Defines the angle between the surface normal and the sound
        incidence in radiant. Its cshape
        needs to be (n_incident_directions).
    area_weights : np.ndarray
        Area weights for each incident direction. Its cshape
        needs to be (n_incident_directions).

    Returns
    -------
    random_coefficient : pyfar.FrequencyData
        The random-incidence coefficient.

    References
    ----------
    .. [#]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
            CRC Press/Taylor & Francis Group, 2017.
    """
    if not isinstance(coefficients, pf.FrequencyData):
        raise ValueError("coefficients has to be FrequencyData")
    if colatitude_rad.shape != area_weights.shape:
        raise ValueError(
            "colatitude_rad and area_weights need to have the same shape.")

    theta = colatitude_rad
    weight = np.cos(theta) * area_weights
    weight = weight[..., np.newaxis]
    norm = np.sum(weight)
    random_coefficient = coefficients*weight/norm
    random_coefficient.freq = np.sum(random_coefficient.freq, axis=-2)
    return random_coefficient
