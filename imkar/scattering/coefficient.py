import numpy as np
import pyfar as pf
from imkar import integrate


def freefield(p_sample, p_reference, mics, incident_directions=None):
    """
    This function calculates the free-field scattering coefficient using the
    Mommertz correlation method [#]_.

    Parameters
    ----------
    p_sample : FrequencyData
        Reflection sound pressure or directivity of the test sample. Its cshape
        need to be (..., #theta_incident_directions, #phi_incident_directions,
        #theta_mics, #phi_mics)
    p_reference : FrequencyData
        Reflection Reflection sound pressure or directivity of the test
        reference sample. It has the same shape as p_sample.
    mics : Coordinates
        A Coordinate object with all microphone directions. Its cshape need to
        be (#theta_mics, #phi_mics)
    incident_directions : Coordinates, optinal
        A Coordinate object with all incident directions. Its cshape need to
        be (#theta_incident, #phi_incident). This input is optinal, if it is
        the random-incidence scattering coefficient can be calculated.

    Returns
    -------
    s : FrequencyData
        The scattering coefficient for each plane wave direction.
    s_rand : FrequencyData, optional
        The random-incidence scattering coefficient. Is just retruned if the
        incident_directions are given.

    References
    ----------
    .. [#]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201–203, Juni 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """
    # check inputs
    if not isinstance(p_sample, pf.FrequencyData):
        raise ValueError("p_sample has to be FrequencyData")
    if not isinstance(p_reference, pf.FrequencyData):
        raise ValueError("p_reference has to be FrequencyData")
    if not isinstance(mics, pf.Coordinates):
        raise ValueError("mics have to be Coordinates")
    if (incident_directions is not None) & \
            ~isinstance(incident_directions, pf.Coordinates):
        raise ValueError("incident_directions have to be None or Coordinates")

    # calculate according to mommertz correlation method
    p_sample_sq = p_sample*p_sample
    p_reference_sq = p_reference*p_reference
    p_reference_conj = p_reference.copy()
    p_reference_conj.freq = np.conj(p_reference_conj.freq)
    p_cross = p_sample*p_reference_conj

    p_sample_sum = integrate.surface_sphere(p_sample_sq, mics)
    p_ref_sum = integrate.surface_sphere(p_reference_sq, mics)
    p_cross_sum = integrate.surface_sphere(p_cross, mics)

    s = (1 - ((p_cross_sum*p_cross_sum)/(p_sample_sum*p_ref_sum)))
    s.comment = 'scattering coefficient'

    # calculate random-incidence scattering coefficient
    if incident_directions is not None:
        s_rand = random_incidence(s, incident_directions)
        return s, s_rand
    return s


def random_incidence(s, incident_directions):
    """This function claculates the random-incidence scattering coefficient
    according to Paris formula.

    Parameters
    ----------
    s : FrequencyData
        The scattering coefficient for each plane wave direction. Its cshape
        need to be (..., #theta, #phi)
    incident_directions : Coordinates
        A Coordinate object with all incident directions. Its cshape need to
        be (#theta, #phi).


    Returns
    -------
    s_rand : FrequencyData
        The random-incidence scattering coefficient.
    """
    sph = incident_directions.get_sph()
    theta = sph[..., 1, None]
    weight = np.sin(2*theta)  # sin(2*theta)
    norm = np.sum(weight)
    s_rand = s*weight/norm
    s_rand.freq = np.sum(s_rand.freq, axis=-2)
    s_rand.freq = np.sum(s_rand.freq, axis=-2)
    s_rand.comment = 'random-incidence scattering coefficient'
    return s_rand
