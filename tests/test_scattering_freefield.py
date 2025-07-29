import pyfar as pf
import numpy as np
import numpy.testing as npt
import pytest
from imkar.scattering import freefield as sff


def plane_wave(amplitude, direction, sampling):
    """Generate a plane wave for a given direction and sampling.

    This function is just used for testing purposes, so the frequency is
    fixed to 5000 Hz.

    Parameters
    ----------
    amplitude : float
        The amplitude of the plane wave.
    direction : pf.Coordinates
        The direction of the plane wave.
    sampling : pf.Sampling
        The sampling grid for the plane wave.

    Returns
    -------
    pf.FrequencyData
        The generated plane wave in the frequency domain.
    """

    f = 5000
    c = 343
    x = sampling
    direction.cartesian = direction.cartesian/direction.radius
    dot_product = direction.x*x.x+direction.y*x.y+direction.z*x.z
    dot_product = dot_product[..., np.newaxis]
    f = np.atleast_1d(f)
    return pf.FrequencyData(
        amplitude*np.exp(-1j*2*np.pi*f/c*dot_product),
        frequencies=f,
    )


def test_correlation_zero_scattering():
    sampling = pf.samplings.sph_equal_area(5000)
    sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
    sampling = sampling[sampling.z>0]
    sample_pressure = plane_wave(1, pf.Coordinates(0, 0, 1), sampling)
    reference_pressure = plane_wave(1, pf.Coordinates(0, 0, 1), sampling)
    s = sff.correlation_method(
        sample_pressure, reference_pressure, sampling.weights,
    )
    npt.assert_almost_equal(s.freq, 0)


@pytest.mark.parametrize("s_scatter", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("Phi_scatter_deg", [30, 60, 90, 120, 150, 42])
def test_correlation_fractional_scattering(s_scatter, Phi_scatter_deg):
    """
    Test analytic scattering coefficient calculation for non 0 or 1.

    for the sample pressure, two plane waves are used. The scattered and the
    specular wave. The reflection factor of both waves is calculated based on
    the scattering coefficient and the scattering angle.
    """
    # calculate the specular and scattered reflection factors
    s_spec = 1-s_scatter
    Phi_spec = 45/180*np.pi
    Phi_scatter = Phi_scatter_deg/180*np.pi
    R_spec = np.sqrt(s_spec)
    R_scatter = np.sqrt(np.abs(s_scatter*np.sin(Phi_spec)/np.sin(Phi_scatter)))

    # define the sampling grid and generate the plane waves
    # for the sample pressure and the reference pressure
    sampling = pf.samplings.sph_equal_area(10000)
    sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
    sampling = sampling[sampling.z>0]
    sample_pressure = plane_wave(
        R_spec,
        pf.Coordinates.from_spherical_front(np.pi/2, Phi_spec, 1), sampling)
    sample_pressure += plane_wave(
        R_scatter,
        pf.Coordinates.from_spherical_front(np.pi/2, Phi_scatter, 1), sampling)

    # Calculate the scattering coefficient with for the specular wave
    # in other words, the reference pressure is the specular wave (R=1)
    reference_pressure = plane_wave(
        1, pf.Coordinates.from_spherical_front(np.pi/2, Phi_spec, 1), sampling)
    sd_spec = 1-sff.correlation_method(
        sample_pressure, reference_pressure, sampling.weights,
    )
    npt.assert_almost_equal(sd_spec.freq, s_spec, 1)

    # Calculate the scattering coefficient with for the scattered wave
    # in other words, the reference pressure is the scattered wave (R=1)
    # to the result will be 1-s_scatter
    reference_pressure = plane_wave(
        1, pf.Coordinates.from_spherical_front(
            np.pi/2, Phi_scatter, 1), sampling)
    sd_scatter = 1-sff.correlation_method(
        sample_pressure, reference_pressure, sampling.weights,
    )
    npt.assert_almost_equal(sd_scatter.freq, s_scatter, 1)

    # Calculate the scattering coefficient with for 5° more then the specular
    # wave, than the scattering coefficient should be 0
    reference_pressure = plane_wave(
        1, pf.Coordinates.from_spherical_front(
            np.pi/2, Phi_spec+5/180*np.pi, 1), sampling)
    sd_scatter_0 = 1-sff.correlation_method(
        sample_pressure, reference_pressure, sampling.weights,
    )
    npt.assert_almost_equal(sd_scatter_0.freq, 0, 1)


def test_correlation_one_scattering():
    sampling = pf.samplings.sph_equal_area(5000)
    sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
    sampling = sampling[sampling.z>0]
    sample_pressure = plane_wave(1, pf.Coordinates(0, 1, 0), sampling)
    reference_pressure = plane_wave(1, pf.Coordinates(0, 0, 1), sampling)
    s = sff.correlation_method(
        sample_pressure, reference_pressure, sampling.weights,
    )
    npt.assert_almost_equal(s.freq, 1, decimal=3)


def test_correlation_method_invalid_sample_pressure_type():
    reference_pressure = pf.FrequencyData(np.array([1, 2, 3]), [100, 200, 300])
    microphone_weights = np.array([0.5, 0.5, 0.5])
    with pytest.raises(
            TypeError, match="sample_pressure must be of type "
            "pyfar.FrequencyData"):
        sff.correlation_method(
            "invalid_type", reference_pressure, microphone_weights)


def test_correlation_method_invalid_reference_pressure_type():
    sample_pressure = pf.FrequencyData(np.array([1, 2, 3]), [100, 200, 300])
    microphone_weights = np.array([0.5, 0.5, 0.5])
    with pytest.raises(
            TypeError, match="reference_pressure must be of type "
            "pyfar.FrequencyData"):
        sff.correlation_method(
            sample_pressure, "invalid_type", microphone_weights)


def test_correlation_method_mismatched_sample_pressure_weights():
    sample_pressure = pf.FrequencyData(np.array([[1, 2, 3]]), [100, 200, 300])
    reference_pressure = pf.FrequencyData(
        np.array([[1, 2, 3]]), [100, 200, 300])
    microphone_weights = np.array([0.5, 0.5])
    with pytest.raises(
            ValueError, match="The last dimension of sample_pressure must "
            "match the size of microphone_weights"):
        sff.correlation_method(
            sample_pressure, reference_pressure, microphone_weights)


def test_correlation_method_mismatched_reference_pressure_weights():
    sample_pressure = pf.FrequencyData(np.array([[1, 2, 3]]), [100, 200, 300])
    reference_pressure = pf.FrequencyData(
        np.array([[1, 2, 3]]), [100, 200, 300])
    microphone_weights = np.array([0.5, 0.5])
    with pytest.raises(
            ValueError, match="The last dimension of sample_pressure must "
            "match the size of microphone_weights"):
        sff.correlation_method(
            sample_pressure, reference_pressure, microphone_weights)


def test_correlation_method_mismatched_sample_reference_shapes():
    sample_pressure = pf.FrequencyData(np.array([[1, 2, 3]]), [100, 200, 300])
    reference_pressure = pf.FrequencyData(np.array([[1, 2]]), [100, 200])
    microphone_weights = np.array([0.5, 0.5, 0.5])
    with pytest.raises(
            ValueError, match="The last dimension of sample_pressure must "
            "match the size of microphone_weights"):
        sff.correlation_method(
            sample_pressure, reference_pressure, microphone_weights)
