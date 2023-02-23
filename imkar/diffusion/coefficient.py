import numpy as np
import pyfar as pf


def freefield(sample_pressure, mic_positions):
    """
    Calculate the free-field diffusion coefficient for each incident direction
    after ISO 17497-2:2012 [1]_. See :py:func:`random_incidence`
    to calculate the random incidence diffusion coefficient.


    Parameters
    ----------
    sample_pressure : FrequencyData
        Reflected sound pressure or directivity of the test sample. Its cshape
        need to be (..., #theta_incident_positions, #phi_incident_positions,
        #angle1, #angle2)
    mic_positions : Coordinates
        A Coordinate object with all microphone positions. Its cshape need to
        be (#angle1, #angle2). In sperical coordinates the radii need to
        be constant. It need to be same for `sample_pressure`.

    Returns
    -------
    diffusion_coefficients : FrequencyData
        The diffusion coefficient for each plane wave direction.


    References
    ----------
    .. [1]  ISO 17497-2:2012, Sound-scattering properties of surfaces.
            Part 2: Measurement of the directional diffusion coefficient in a
            free field. Geneva, Switzerland: International Organization for
            Standards, 2012.


    Examples
    --------
    Calculate free-field diffusion coefficients and then the random incidence
    diffusion coefficient.

    >>> import imkar as ik
    >>> diffusion_coefficients = ik.diffusion.coefficient.freefield(
    >>>     sample_pressure, mic_positions)
    >>> random_d = ik.scattering.coefficient.random_incidence(
    >>>     diffusion_coefficients, incident_positions)
    """
    # check inputs
    if not isinstance(sample_pressure, pf.FrequencyData):
        raise ValueError("sample_pressure has to be FrequencyData")
    if not isinstance(mic_positions, pf.Coordinates):
        raise ValueError("microphone positions have to be Coordinates")

    # parse angles
    coords_spherical = mic_positions.get_sph(convention='top_colat')
    phi = coords_spherical[1, :, 0]
    theta = coords_spherical[:, 1, 1]
    if np.sum(np.diff(phi[1:-2])) < 1e-3:
        phi = coords_spherical[:, 1, 0]
        theta = coords_spherical[1, :, 1]
    d_theta = np.mean(np.diff(theta))
    d_phi = np.mean(np.diff(phi))

    # calculate weights
    thetas = coords_spherical[..., 1]
    A_i = 2 * np.sin(thetas) * np.sin(d_theta/2)
    A_i[thetas == 0] = 2 * np.sin(d_theta/2)
    A_i[thetas == np.pi/2] = 4 * np.pi / d_phi * (np.sin(d_theta/4)**2)
    N_i = A_i / np.min(A_i)

    # calculate according to Mommertz correlation method Equation (6)
    p_sample_abs_sq = np.moveaxis(np.abs(sample_pressure.freq)**2, -1, 0)

    p_sample_sum_sq = np.sum(np.sum(
        p_sample_abs_sq**2 * N_i, axis=-1), axis=-1)
    p_sample_sq_sum = np.sum(np.sum(
        p_sample_abs_sq * N_i, axis=-1), axis=-1)**2
    n = np.sum(N_i)
    diffusion_array \
        = (p_sample_sq_sum - p_sample_sum_sq) / ((n-1) * p_sample_sum_sq)
    diffusion_coefficients = pf.FrequencyData(
        np.moveaxis(diffusion_array, 0, -1),
        sample_pressure.frequencies)
    diffusion_coefficients.comment = 'diffusion coefficients'

    return diffusion_coefficients
