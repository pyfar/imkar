import numpy as np
import pyfar as pf
from imkar import integrate


def freefield(p_sample, p_reference, mics, incident_directions=None):
    """
    This function calculates the free-field scattering coefficient using the
    Mommertz correlation method [#]_. The input objects are the reflection
    directivity of the test sample as well as a reference surface of equal
    dimensions. The third input argument is the microphone array positions.

    If an optional fourth argument is given, the direction of the different
    plane waves are expected. They will be used to determine the
    random-incidence value using Paris' formula.

    Parameters
    ----------
    p_sample : FrequencyData
        Reflection directivity of the test sample.
    p_reference : FrequencyData
        Reflection directivity of the test reference sample.
    mics : _type_
        _description_
    incident_directions : _type_
        _description_

    Returns
    -------
    s : FrequencyData
        The output is the scattering coefficient for each plane wave direction.
        If a second output argument is requested, the random-incidence value is
        also returned.
    s_rand : FrequencyData, optional
        If a second output argument is requested, the random-incidence value is
        also returned.

    References
    ----------
    .. [#]  E. Mommertz, „Determination of scattering coefficients from the
            reflection directivity of architectural surfaces“, Applied
            Acoustics, Bd. 60, Nr. 2, S. 201–203, Juni 2000,
            doi: 10.1016/S0003-682X(99)00057-2.

    """

    if not isinstance(p_sample, pf.FrequencyData):
        raise ValueError("p_sample has to be FrequencyData")
    if not isinstance(p_reference, pf.FrequencyData):
        raise ValueError("p_reference has to be FrequencyData")
    if not isinstance(mics, pf.Coordinates):
        raise ValueError("mics have to be Coordinates")
    if (incident_directions is not None) & \
            ~isinstance(incident_directions, pf.Coordinates):
        raise ValueError("incident_directions have to be None or Coordinates")

    # calculate according to mommertz correlation method (inlcuding weights)
    p_sample_sq = p_sample*p_sample
    p_reference_sq = p_reference*p_reference
    p_reference_conj = p_reference.copy()
    p_reference_conj.freq = np.conj(p_reference_conj.freq)
    p_cross = p_sample*p_reference_conj

    p_sample_sum = integrate.spherical_2d(p_sample_sq, mics)
    p_ref_sum = integrate.spherical_2d(p_reference_sq, mics)
    p_cross_sum = integrate.spherical_2d(p_cross, mics)

    s = (1 - ((p_cross_sum*p_cross_sum)/(p_sample_sum*p_ref_sum)))
    s.comment = 'scattering coefficient'
    # calculate random-incidence scattering coefficient
    if incident_directions is not None:
        sph = incident_directions.get_sph()
        theta = sph[..., 1, None]
        weight = np.sin(2*theta)  # sin(2*theta)
        norm = np.sum(weight)
        s_rand = s*weight/norm
        s_rand.freq = np.sum(s_rand.freq, axis=-2)
        s_rand.freq = np.sum(s_rand.freq, axis=-2)
        s_rand.comment = 'random-incidence scattering coefficient'
        return s, s_rand
    return s
