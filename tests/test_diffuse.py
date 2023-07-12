import imkar as ik
import numpy as np
import pyfar as pf
import numpy.testing as npt


def test_diffuse_eyring():
    f = [500, 1000, 2000, 4000, 8000, 16000, 25000]
    T60_1 = [0.91, 0.71, 0.72, 0.59, 0.48, 0.28, 0.17] # T1
    T60_2 = [0.7, 0.58, 0.55, 0.46, 0.38, 0.23, 0.15] # T2
    T60_3 = [0.91, 0.71, 0.73, 0.59, 0.48, 0.27, 0.17] # T3
    T60_4 = [0.71, 0.58, 0.50, 0.30, 0.26, 0.18, 0.13] # T4
    reverb_time = pf.FrequencyData([T60_1, T60_2, T60_3, T60_4], f)
    speed_of_sound = np.array([344.5287, 344.5287, 344.5287, 344.5287])
    air_attenuation_s = [0.000652350693755930, 0.00111221984304401,
                       0.00226837553058824, 0.00659994837403191,
                       0.0232464500602580, 0.0815039446237653, 
                       0.167026524991143]
    air_attenuation = np.array([[air_attenuation_s], [air_attenuation_s],
                       [air_attenuation_s], [air_attenuation_s]])
    
    desired_eyring = [0, 0, 0.1359, 0.8021, 0.8808, 0.9331, 0.8963]

    scattering_coefficient = ik.scattering.diffuse(reverb_time, speed_of_sound,
                                                   air_attenuation, "Eyring")

    npt.assert_almost_equal(scattering_coefficient.freq[0],
                             desired_eyring, 2)
    

def test_diffuse_sabine():
    f = [500, 1000, 2000, 4000, 8000, 16000, 25000]
    T60_1 = [0.91, 0.71, 0.72, 0.59, 0.48, 0.28, 0.17] # T1
    T60_2 = [0.7, 0.58, 0.55, 0.46, 0.38, 0.23, 0.15] # T2
    T60_3 = [0.91, 0.71, 0.73, 0.59, 0.48, 0.27, 0.17] # T3
    T60_4 = [0.71, 0.58, 0.50, 0.30, 0.26, 0.18, 0.13] # T4
    reverb_time = pf.FrequencyData([T60_1, T60_2, T60_3, T60_4], f)
    speed_of_sound = np.array([344.5287, 344.5287, 344.5287, 344.5287])
    air_attenuation_s = [0.000652350693755930, 0.00111221984304401,
                       0.00226837553058824, 0.00659994837403191,
                       0.0232464500602580, 0.0815039446237653, 
                       0.167026524991143]
    air_attenuation = np.array([[air_attenuation_s], [air_attenuation_s],
                       [air_attenuation_s], [air_attenuation_s]])
    
    desired_sabine = [0, 0, 0.1455, 0.8834, 0.9758, 1.0672, 1.0289]

    scattering_coefficient = ik.scattering.diffuse(reverb_time, speed_of_sound,
                                                   air_attenuation, "Sabine")

    npt.assert_almost_equal(scattering_coefficient.freq[0],
                             desired_sabine, 2)
