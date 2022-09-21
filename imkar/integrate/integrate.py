import warnings
from scipy.integrate import trapezoid
import numpy as np


def spherical(data, coords):
    """Intergrated over a sphere in spherical coordinates.

    .. math::

        S &= \\int(\\int(data(\\phi, \\theta)d\\phi)sin(\\theta)\\dtheta),

    Parameters
    ----------
    data : FrequencyData
        Input data to integrate. Its shape is (..., #phi, #theta)
    coords : Coordinates
        A Coordinate object with the sample points corresponding to the data
        values. It has the shape of (#phi, #theta).

    Returns
    -------
    FrequencyData
        the integrated object. Its shape is (...) and 2 dimentions shorter
        then before.
    """
    ptr = coords.get_sph(convention='top_colat')
    phi = ptr[1, :, 0]
    theta = ptr[:, 0, 1]
    r_mean = np.mean(ptr[:, :, 2])
    if not np.allclose(ptr[:, :, 2], r_mean):
        warnings.warn(r'all radii should be almost same')
    last_close_to_0 = np.isclose(phi[-1], 0, atol=1e-16)
    pi2_consistant = np.isclose(phi[-2]+np.diff(phi)[0], 2*np.pi, atol=1e-16)
    if last_close_to_0 and pi2_consistant:
        phi[-1] = 2*np.pi
    weights = np.reshape(np.sin(theta), (len(theta), 1))
    result_raw = trapezoid(data.freq, x=phi, axis=-2)
    result_raw1 = trapezoid(result_raw*weights, x=theta, axis=-2)
    result = data.copy()
    result.freq = result_raw1
    return result
