"""
Contains tools to easily generate stubs for the most common pyfar Classes.
Stubs are used instead of pyfar objects for testing functions that have pyfar
objects as input arguments. This makes testing such functions independent from
the pyfar objects themselves and helps to find bugs.
"""
import numpy as np
import pyfar as pf


def spherical_coordinates(phi_rad, theta_rad, r=1):
    phi, theta = np.meshgrid(phi_rad, theta_rad)
    coords = pf.Coordinates(
        phi, theta, np.ones(phi.shape)*r, 'sph')
    return coords


def frequencydata_from_shape(shape, data_raw, frequencies):
    frequencies = np.atleast_1d(frequencies)
    shape_new = np.append(shape, frequencies.shape)
    if hasattr(data_raw, "__len__"):  # is array
        if len(shape) > 0:
            for dim in shape:
                data_raw = np.repeat(data_raw[..., np.newaxis], dim, axis=-1)
        data = np.repeat(data_raw[..., np.newaxis], len(frequencies), axis=-1)
    else:
        data = np.zeros(shape_new) + data_raw
    p_reference = pf.FrequencyData(data, frequencies)
    return p_reference
