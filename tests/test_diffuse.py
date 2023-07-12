import imkar as ik
import numpy as np
import pyfar as pf


def test_diffuse():
    f = [500, 1000, 2000, 4000, 8000, 16000, 25000]
    T60 = [[0.1, 0.3, 0.5, 0.8, 1.0, 0.9, 0.95], # T1
           [0.05, 0.1, 0.55, 0.7, 1.0, 0.9, 0.95], # T2
           [0.15, 0.2, 0.4, 0.85, 0.75, 0.9, 0.8], # T3
           [0.2, 0.35, 0.45, 0.6, 0.85, 1.0, 0.9]] # T4
    reverb_time = pf.FrequencyData(T60, f)
    c = [343, 342.5, 342.0, 342.5]
    
    pass
