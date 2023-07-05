import pyfar as pf
import numpy as np

def diffuse(reverb_time, speed_of_sound, air_attenuation,
             alpha_method = 'Eyring',scale_factor = 5, sample_area = 0.5027,
                baseplate_area = 0.58088, surface_room = 9.05, 
                    volume_room = 1.67):
    """
    This function calculates the diffuse scattering coefficient after the ISO
    17497-1 method. It uses four reverberation times measured under different
    conditions [#]_.

    Parameters
    ----------
    reverb_time : pf.FrequencyData
        Frequency object containing the reverbaration times calculated during
        the four different measurements. If the measurement is performed at
        multiple positions, the mean value for every condition of the
        measurements is taken.

    speed_of_sound : array, float
        Vector containing the speed of sound for every measurement condition
        calculated by the temperature and air humidity during measurement time.

    air_attenuation : array, pf.FrequencyData
        Vector containing the air attenuation for every measurement condition
        calculated by temperature and air humidity during measurement time.

    alpha_method : 'Eyring' or 'Sabine'
        Defines the method by which alpha is calculated.

    scale_factor : float
        Factor that corrects the frequencies when using a smaller scale
        reverbaration room than defined in the ISO standard. Standard value is
        5, according to the measurement chamber at IHTA.

    sample_area : float
        Surface area of the measurement sample in square meter. Standard value
        according to the test sample at IHTA.

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
        International Standard 17497-1:2004(E).

    References
    ----------
    ..[#]ISO 17497-1 First edition 2004-05-01
    """
    # check inputs


    # Scale frequencies
    reverb_time[:,1] = reverb_time[:,1]/scale_factor

    # random-incidence absorption coefficient
    a_s = 55.3*volume_room/sample_area*(1/(speed_of_sound[1]*reverb_time[1])
            -1/(speed_of_sound[0]*reverb_time[0]))-4*volume_room/sample_area* \
                (air_attenuation[1]-air_attenuation[0])
    
    # specular absorption coefficient
    a_spec = 55.3*volume_room/sample_area*(1/(speed_of_sound[3]*reverb_time[3])
            -1/(speed_of_sound[2]*reverb_time[2]))-4*volume_room/sample_area* \
                (air_attenuation[3]-air_attenuation[2])
    
    # scattering coefficient for the base plate
    s_base = 55.3*volume_room/sample_area*(1/(speed_of_sound[2]*reverb_time[2])
            -1/(speed_of_sound[0]*reverb_time[0]))-4*volume_room/sample_area* \
                (air_attenuation[2]-air_attenuation[0])
    
    # random-incidence scattering coefficient
    scatterring_coefficient = 1-(1-a_spec)/(1-a_s)

    return scatterring_coefficient
    
