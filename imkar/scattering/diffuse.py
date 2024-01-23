import pyfar as pf
import numpy as np
import numbers


def diffuse(reverberation_time, speed_of_sound, air_attenuation,
            calculation_method='ISO', scale_factor=5,
            sample_area=0.5026548, baseplate_area=0.58088,
            surface_room=9.05, volume_room=1.67):
    """
    This function calculates the diffuse scattering coefficient after the ISO
    17497-1 method. It uses four reverberation times measured under different
    conditions [#]_.

    Parameters
    ----------
    reverberation_time : pf.FrequencyData
        Frequency object containing the reverbaration times calculated during
        the four different measurements. If the measurement is performed at
        multiple positions, the mean value for every condition of the
        measurements is taken. Its cshape has to be (..., 4).

    speed_of_sound : array, float
        Array containing the speed of sound in m/s for every measurement
        condition calculated by the temperature and air humidity during
        measurement time.

    air_attenuation : array, pf.FrequencyData
        Matrix or FrequencyData containing the air attenuation in 1/m for every
        measurement condition calculated by temperature and air humidity during
        measurement time. Its cshape has to be (..., 4).

    calculation_method : 'ISO','Eyring' or 'Sabine'
        Defines the method by which alpha is calculated.
        For method 'ISO' the baseplate_area and sample_area need to be equal.

    scale_factor : float
        Factor that corrects the frequencies when using a smaller scale
        reverbaration room than defined in the ISO standard. Standard value is
        5, according to the measurement chamber at IHTA.

    sample_area : float
        Surface area of the measurement sample in square meter. Standard value
        according to the test sample.

    baseplate_area : float
        Surface area of the baseplate in square meter. Standard value according
        to the measurement chamber at IHTA.

    surface_room : float
        Surface area of the measurement room in square meter. Standard value
        according to measurement chamber at IHTA.

    volume_room : float
        Volume of the empty measurement room in cubic meter. Standard value
        according to measurement chamber at IHTA.

    Returns
    -------
    scattering_coefficient : pf.FrequencyData
        The calculated solution for the diffuse scattering coefficient after
        International Standard 17497-1:2004(E) and the scattering coefficient
        of the base plate.

    References
    ----------
    ..[#]ISO 17497-1 First edition 2004-05-01
    """
    # check inputs
    if reverberation_time.cshape[-1] != 4:
        raise ValueError("reverberation_time.cshape has to be (..., 4)")

    if (np.asarray(speed_of_sound, dtype=float).ndim == 1):
        speed_of_sound = (np.asarray(speed_of_sound, dtype=float)[:, None]).T
    else:
        speed_of_sound = (np.asarray(speed_of_sound, dtype=float))

    if (speed_of_sound.shape != reverberation_time.cshape):
        raise ValueError("speed_of_sound has to be an array containing the \
                         speed of sound during all 4 measurement conditions. \
                         Its shape has to be (..., 4) and equal to \
                         reverberation_time.cshape.")

    if (np.any(np.asarray(speed_of_sound, dtype=float) <= 0)):
        raise ValueError("speed_of_sound has to be a value bigger than zero")

    if isinstance(air_attenuation, pf.FrequencyData):
        air_attenuation = np.asarray(air_attenuation.freq, dtype=float)
    else:
        air_attenuation = np.asarray(air_attenuation, dtype=float)
    if air_attenuation.shape != (reverberation_time.cshape +
                                 (reverberation_time.n_bins, )):
        raise ValueError("air_attenuation has to be of shape (..., 4,\
                         number_of_frequencies) and equal the shape of \
                         reverberation_time.")

    if (np.any(air_attenuation < 0)):
        raise ValueError("air_attenuation has to be a value bigger or equal \
                         to zero")

    if calculation_method not in ['ISO', 'Eyring', 'Sabine']:
        raise ValueError("Use 'ISO', 'Eyring' or 'Sabine' as calculation \
                         method")

    if (calculation_method == "ISO") & (sample_area != baseplate_area):
        raise ValueError("For calculation method 'ISO' the baseplate_area and \
                         sample_area need to be equal.")

    if not isinstance(scale_factor, numbers.Real) or scale_factor <= 0:
        raise TypeError("scale_factor has to be a real number >0")

    if not isinstance(sample_area, numbers.Real) or sample_area <= 0:
        raise TypeError("sample_area has to be a real number >0")

    if not isinstance(baseplate_area, numbers.Real) or baseplate_area <= 0:
        raise TypeError("baseplate_area has to be a real number >0")

    if not isinstance(surface_room, numbers.Real) or surface_room <= 0:
        raise TypeError("surface_room has to be a real number >0")

    if not isinstance(volume_room, numbers.Real) or volume_room <= 0:
        raise TypeError("volume_room has to be a real number >0")

    # start calculations

    if (calculation_method == "ISO"):
        # random-incidence absorption coefficient
        a_s = (55.3*volume_room/sample_area *
               (1/(speed_of_sound[:, 1]*reverberation_time.freq[:, 1])-1 /
                (speed_of_sound[:, 0]*reverberation_time.freq[:, 0])) -
               4*volume_room/sample_area *
               (air_attenuation[:, 1]-air_attenuation[:, 0]))

        # specular absorption coefficient
        a_spec = (55.3*volume_room/sample_area *
                  (1/(speed_of_sound[:, 3]*reverberation_time.freq[:, 3])
                   - 1/(speed_of_sound[:, 2]*reverberation_time.freq[:, 2])) -
                  4*volume_room/sample_area *
                  (air_attenuation[:, 3]-air_attenuation[:, 2]))

        # scattering coefficient for the base plate
        s_base = (55.3*volume_room/sample_area *
                  (1/(speed_of_sound[:, 2]*reverberation_time.freq[:, 2])
                   - 1/(speed_of_sound[:, 0]*reverberation_time.freq[:, 0])) -
                  4*volume_room/sample_area *
                  (air_attenuation[:, 2]-air_attenuation[:, 0]))

    else:
        # calculate equivalent surface area
        A = np.zeros((reverberation_time.cshape +
                      (reverberation_time.n_bins, )))
        for i in range(0, 4):
            A[:, i] = volume_room * \
                ((24*np.log(10)) /
                 (speed_of_sound[:, i][:, None]*reverberation_time.freq[:, i])
                    - 4*air_attenuation[:, i])

        # calculate alpha based on chosen method
        if (calculation_method == "Eyring"):
            alpha = 1-np.exp(-A/surface_room)
        elif (calculation_method == "Sabine"):
            alpha = A/surface_room

        # random-incidence absorption coefficient
        a_s = surface_room/sample_area*(alpha[:, 1]-alpha[:, 0])+alpha[:, 0]

        # specular absorption coefficient
        a_spec = surface_room/sample_area*(alpha[:, 3]-alpha[:, 2])+alpha[:, 2]

        # scattering coefficient for the base plate
        s_base = surface_room/baseplate_area * \
            (alpha[:, 2]-alpha[:, 0]).clip(min=0)

    # random-incidence scattering coefficient
    scatterring_coefficient = (1-(1-a_spec)/(1-a_s)).clip(min=0)

    return pf.FrequencyData([scatterring_coefficient, s_base],
                            reverberation_time.frequencies/scale_factor)
