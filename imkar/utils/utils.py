import numpy as np
import pyfar as pf


def paris_formula(coefficients, incident_directions):
    r"""
    Calculate the random-incidence coefficient from free-field
    data for several incident directions.

    Uses the Paris formula [2]_.

    .. math::
        c_{rand} = \sum c(\vartheta,\varphi) \cdot cos(\vartheta) \cdot
        w(\vartheta,\varphi)

    with the coefficients :math:`c(\vartheta,\varphi)`, the area
    weights ``w`` from the `incident_directions.weights`,
    and :math:`\vartheta` and :math:`\varphi` are the incidence
    angle and azimuth angles. Note that the incident directions should be
    equally distributed to get a valid result.

    Parameters
    ----------
    coefficients : pyfar.FrequencyData
        Scattering coefficients for different incident directions. Its cshape
        needs to be (..., `incident_directions.csize`)
    incident_directions : pyfar.Coordinates
        Defines the incidence directions of each `coefficients` in a
        pyfar.Coordinates object. Its cshape needs to match the last dimension
        of coefficients.
        Points contained in `incident_directions` must have the same radii.
        The weights need to reflect the area ``incident_directions.weights``.

    Returns
    -------
    random_coefficient : pyfar.FrequencyData
        The random-incidence coefficient depending on frequency.

    References
    ----------
    .. [2]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
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
