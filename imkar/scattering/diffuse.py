import pyfar as pf
import numpy as np


class ReverberationRoom:
    """
    """
    def __init__(self, volume, surface):
        """_summary_.

        Parameters
        ----------
        volume : float
            the volume of the room, in cubic metres.
        surface : _type_
            _description_
        """
        self.volume = volume
        self.surface = surface

    def check_iso(self, I_max, c, T1, m1):
        """_summary_.

        Parameters
        ----------
        I_max : float
            the length of the longest straight line which fits within the
            boundary of the room (e.g. in a rectangular room it is the
            major diagonal), in metres.
        c : float
            the propagation speed of sound in air, in metres per second.
        T1 : float
            the reverberation time, in seconds,
            of the empty reverberation room;
        m1 : float
            is the power attenuation coefficient, in reciprocal metres,
            calculated according to ISO 9613-1 using the climatic conditions
            that have been present in the empty reverberation room during
            the measurement. The value of m can be calculated from the
            attenuation coefficient, alpha, which is used in ISO 9613-1
            according to the formula
        """
        # 6.1.1: V > 150, better V > 200, V < 500
        # 6.1.2: I_max < 1.9 V ** (1/3)
        # 6.1.3 diffusion
        # 6.1.4: A_1 = 55.3 * V / (c*T_1) - (4*V *m_1)


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
    reverberation_time : :py:class:`~pyfar.classes.audio.FrequencyData`
        Frequency object containing the reverberation times calculated during
        the four different measurements. If the measurement is performed at
        multiple positions, the mean value for every condition of the
        measurements is taken. Its cshape has to be (..., 4).
    speed_of_sound : array, float
        Array containing the speed of sound in m/s for every measurement
        condition during the measurement.
    air_attenuation : :py:class:`~pyfar.classes.audio.FrequencyData`
        Matrix or FrequencyData containing the air attenuation in 1/m for every
        measurement condition calculated by temperature and air humidity during
        measurement time. Its cshape has to be (..., 4).
    calculation_method : 'ISO', 'Eyring' or 'Sabine'
        Defines the method by which alpha is calculated.
        For method 'ISO' the baseplate_area and sample_area need to be equal.
    scale_factor : float
        Factor that corrects the frequencies when using a smaller scale
        reverberation room than defined in the ISO standard. Standard value is
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
    scattering_coefficient : :py:class:`~pyfar.classes.audio.FrequencyData`
        The calculated solution for the diffuse scattering coefficient after
        International Standard 17497-1:2004(E) and the scattering coefficient
        of the base plate.

    References
    ----------
    .. [#] ISO 17497-1:2004, Sound-scattering properties of surfaces. Part 1:
           Measurement of the random-incidence scattering coefficient in a
           reverberation room. Geneva, Switzerland: International
           Organization for Standards, 2004.
    """
    # check inputs
    if reverberation_time.cshape[-1] != 4:
        raise ValueError("reverberation_time.cshape has to be (..., 4)")

    speed_of_sound = np.asarray(speed_of_sound, dtype=float)

    if (speed_of_sound.shape != reverberation_time.cshape):
        raise ValueError("speed_of_sound has to be an array containing the \
                         speed of sound during all 4 measurement conditions. \
                         Its shape has to be (..., 4) and equal to \
                         reverberation_time.cshape.")

    if (np.any(np.asarray(speed_of_sound, dtype=float) <= 0)):
        raise ValueError("speed_of_sound has to be a value bigger than zero")

    if not isinstance(air_attenuation, pf.FrequencyData):
        raise ValueError("air_attenuation has to be of type pf.FrequencyData.")

    if air_attenuation.freq.shape != reverberation_time.freq.shape:
        raise ValueError("air_attenuation and reverberation_time has to have \
                         cshape and number number of frequency bins.")

    if (np.any(air_attenuation.freq < 0)):
        raise ValueError("air_attenuation has to be a value bigger or equal \
                         to zero")

    if calculation_method not in ['ISO', 'Eyring', 'Sabine']:
        raise ValueError("Use 'ISO', 'Eyring' or 'Sabine' as calculation \
                         method")

    if (calculation_method == "ISO") & (sample_area != baseplate_area):
        raise ValueError("For calculation method 'ISO' the baseplate_area and \
                         sample_area need to be equal.")

    if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
        raise TypeError("scale_factor has to be a real number >0")

    if not isinstance(sample_area, (int, float)) or sample_area <= 0:
        raise TypeError("sample_area has to be a real number >0")

    if not isinstance(baseplate_area, (int, float)) or baseplate_area <= 0:
        raise TypeError("baseplate_area has to be a real number >0")

    if not isinstance(surface_room, (int, float)) or surface_room <= 0:
        raise TypeError("surface_room has to be a real number >0")

    if not isinstance(volume_room, (int, float)) or volume_room <= 0:
        raise TypeError("volume_room has to be a real number >0")

    # start calculations

    if (calculation_method == "ISO"):
        c_1 = speed_of_sound[..., 0]
        c_2 = speed_of_sound[..., 1]
        c_3 = speed_of_sound[..., 2]
        c_4 = speed_of_sound[..., 3]

        # random-incidence absorption coefficient
        alpha_s = (55.3*volume_room/sample_area *
               (1/(c_2*reverberation_time[..., 2])-1 / (
                   c_1*reverberation_time[..., 0])) -
               4*volume_room/sample_area *
               (air_attenuation[..., 1, :]-air_attenuation[..., 0, :]))

        # specular absorption coefficient
        alpha_spec = (55.3*volume_room/sample_area *
                  (1/(c_4*reverberation_time[..., 3])
                   - 1/(c_2*reverberation_time[..., 2])) -
                  4*volume_room/sample_area *
                  (air_attenuation[:, 3]-air_attenuation[:, 2]))

        # scattering coefficient for the base plate
        scattering_base = (55.3*volume_room/sample_area *
                  (1/(c_3*reverberation_time[..., 2])
                   - 1/(c_1*reverberation_time[..., 0])) -
                  4*volume_room/sample_area *
                  (air_attenuation[:, 2]-air_attenuation[:, 0]))

    else:
        # calculate equivalent surface area
        A = volume_room * (
            (24*np.log(10)) / (speed_of_sound*reverberation_time)
            - 4*air_attenuation)

        # calculate alpha based on chosen method
        if (calculation_method == "Eyring"):
            alpha = pf.FrequencyData(
                1-np.log(A.freq/(-surface_room)), A.frequencies)
        elif (calculation_method == "Sabine"):
            alpha = A/surface_room

        alpha_1 = alpha.freq[..., 0, :]
        alpha_2 = alpha.freq[..., 1, :]
        alpha_3 = alpha.freq[..., 2, :]
        alpha_4 = alpha.freq[..., 3, :]
        # random-incidence absorption coefficient
        alpha_s = surface_room/sample_area*(alpha_2-alpha_1)+alpha_1

        # specular absorption coefficient
        alpha_spec = surface_room/sample_area*(alpha_4-alpha_3)+alpha_3

        # scattering coefficient for the base plate
        scattering_base = surface_room/baseplate_area * \
            (alpha_3-alpha_1).clip(min=0)

    # random-incidence scattering coefficient
    scatterring_coefficient = (1-(1-alpha_spec)/(1-alpha_s)).clip(min=0)

    return pf.FrequencyData([scatterring_coefficient, scattering_base],
                            reverberation_time.frequencies/scale_factor)
