import pytest
import numpy as np
import pyfar as pf

from imkar import scattering


def test_freefield_1(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_sample.freq.fill(1)
    p_reference = pressure_data_mics.copy()
    p_sample.freq[5, :] = 0
    p_reference.freq[5, :] = np.sum(p_sample.freq.flatten())/2
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 1)


def test_freefield_wrong_input(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()

    with pytest.raises(ValueError, match='sample_pressure'):
        scattering.freefield(1, p_reference, mics.weights)
    with pytest.raises(ValueError, match='reference_pressure'):
        scattering.freefield(p_sample, 1, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        scattering.freefield(p_sample, p_reference, 1)
    with pytest.raises(ValueError, match='cshape'):
        scattering.freefield(
            p_sample[:-2, ...], p_reference, mics.weights)
    with pytest.raises(ValueError, match='microphone_weights'):
        scattering.freefield(
            p_sample, p_reference, mics.weights[:10])
    with pytest.raises(ValueError, match='same frequencies'):
        p_sample.frequencies[0] = 1
        scattering.freefield(p_sample, p_reference, mics.weights)


def test_freefield_05(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_sample.freq[7, :] = 1
    p_sample.freq[28, :] = 1
    p_reference.freq[7, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0.5)


def test_freefield_0(
        half_sphere, pressure_data_mics):
    mics = half_sphere
    p_sample = pressure_data_mics.copy()
    p_reference = pressure_data_mics.copy()
    p_reference.freq[5, :] = 1
    p_sample.freq[5, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins


def test_freefield_0_with_incident(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 2, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0)
    np.testing.assert_allclose(s_rand.freq, 0)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins


def test_freefield_1_with_incidence(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 3, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 1)
    np.testing.assert_allclose(s_rand.freq, 1)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins


def test_freefield_05_with_inci(
        half_sphere, quarter_half_sphere,
        pressure_data_mics_incident_directions):
    mics = half_sphere
    incident_directions = quarter_half_sphere
    p_sample = pressure_data_mics_incident_directions.copy()
    p_reference = pressure_data_mics_incident_directions.copy()
    p_sample.freq[:, 7, :] = 1
    p_sample.freq[:, 28, :] = 1
    p_reference.freq[:, 7, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0.5)
    np.testing.assert_allclose(s_rand.freq, 0.5)
    assert s.freq.shape[-1] == p_sample.n_bins
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == p_sample.n_bins


@pytest.mark.parametrize("s_value",  [
    (0), (0.2), (0.5), (0.8), (1)])
@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_constant_s(
        s_value, frequencies, half_sphere):
    incident_directions = half_sphere
    shape = np.append(half_sphere.cshape, len(frequencies))
    s = pf.FrequencyData(np.zeros(shape)+s_value, frequencies)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s_rand.freq, s_value)


def test_random_non_constant_s():
    data = pf.samplings.sph_gaussian(10)
    incident_directions = data[data.get_sph().T[1] <= np.pi/2]
    incident_cshape = incident_directions.cshape
    s_value = np.arange(
        incident_cshape[0]).reshape(incident_cshape) / incident_cshape[0]
    theta = incident_directions.get_sph().T[1]
    actual_weight = np.cos(theta) * incident_directions.weights
    actual_weight /= np.sum(actual_weight)
    s = pf.FrequencyData(s_value.reshape((50, 1)), [100])
    s_rand = scattering.random(s, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(s_rand.freq, desired)
