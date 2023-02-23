import warnings
from scipy.integrate import trapezoid
import numpy as np
import pyfar as pf


def surface_sphere(data, coords, weights=None):
    r"""Integrate over a set of points sampled on a spherical surface.

    .. math::

        S = \int \int data(\phi, \theta)  sin(\theta) d\phi d\theta

    Parameters
    ----------
    data : FrequencyData, Signal
        Input data to integrate. Its `cshape` needs to be (..., #theta, #phi)
        or (..., #phi, #theta).
    coords : Coordinates
        Coordinate points at which the data is sampled. Its `cshape` needs to
        be (#theta, #phi) or (#phi, #theta), matching the `cshape` of `data`.
    weights : np.array
        weights for the integration over the coordinates. Its `cshape`
        needs to be same as for `coords`.
        By default the its sin(theta).

    Returns
    -------
    FrequencyData
        The integration result. Its dimension is reduced by the last two,
        which are consumed by the integration.

    Examples
    --------
    >>> from imkar import integrate
    >>> from pyfar import FrequencyData, Coordinates
    >>> import numpy as np
    >>> phi, theta = np.meshgrid(
    >>>     np.linspace(0, 2*np.pi, num=361),
    >>>     np.linspace(0, np.pi, num=181))
    >>> coords = Coordinates(
    >>>     phi, theta, np.ones(phi.shape), 'sph')
    >>> data_raw = np.ones(phi.shape)
    >>> data = FrequencyData(data_raw[..., None], 100)
    >>> result = integrate.surface_sphere(data, coords)
    >>> result.freq
    array([[12.56605162+0.j]])
    """

    if coords.cshape != data.cshape[-2:]:
        raise ValueError(
            f'Coordinates.cshape should be same as {data.cshape[-2:]}')

    # parse angles
    coords_spherical = coords.get_sph(convention='top_colat')
    phi = coords_spherical[1, :, 0]
    theta = coords_spherical[:, 1, 1]
    axis_index = -1
    if np.sum(np.diff(phi[1:-2])) < 1e-3:
        phi = coords_spherical[:, 1, 0]
        theta = coords_spherical[1, :, 1]
        axis_index = -2
    radius = coords_spherical[:, :, 2]
    r_mean = np.mean(radius)
    if not np.allclose(radius, r_mean):
        warnings.warn(r'all radii should be almost same')

    # pf.Coordinates turns phi = 2*pi to 0, due to circular
    # for the integration the upper limit cannot be zero but 2pi
    last_phi_is_zero = np.isclose(phi[-1], 0, atol=1e-16)
    increasing_phi_to_2pi = np.isclose(
        phi[-2]+np.diff(phi)[0], 2*np.pi, atol=1e-16)
    if last_phi_is_zero and increasing_phi_to_2pi:
        phi[-1] = 2 * np.pi

    # integrate over angles with wight for spherical integration
    data_int = np.moveaxis(data.freq, -1, 0)
    if weights is None:
        weights = np.sin(coords_spherical[..., 1])
    result_raw = trapezoid(data_int*weights, x=phi, axis=axis_index)
    result_raw1 = trapezoid(result_raw, x=theta, axis=-1)

    return pf.FrequencyData(np.moveaxis(result_raw1, 0, -1), data.frequencies)
