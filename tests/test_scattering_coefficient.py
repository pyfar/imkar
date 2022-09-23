import pytest
import numpy as np

from imkar import scattering
from pyfar import FrequencyData, Coordinates


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    p_sample = _create_frequencydata(mics.cshape, 1, frequencies)
    p_reference = _create_frequencydata(mics.cshape, 0, frequencies)
    p_sample.freq[5, 0, :] = 0
    p_reference.freq[5, 0, :] = np.sum(p_sample.freq.flatten())/2
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_wrong_input(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    p_sample = _create_frequencydata(mics.cshape, 1, frequencies)
    p_reference = _create_frequencydata(mics.cshape, 0, frequencies)
    p_sample.freq[5, 0, :] = 0
    p_reference.freq[5, 0, :] = np.sum(p_sample.freq.flatten())/2
    with pytest.raises(ValueError, match='p_sample'):
        scattering.coefficient.freefield(1, p_reference, mics)
    with pytest.raises(ValueError, match='p_reference'):
        scattering.coefficient.freefield(p_sample, 1, mics, mics)
    with pytest.raises(ValueError, match='mics'):
        scattering.coefficient.freefield(p_sample, p_reference, 1, mics)
    with pytest.raises(ValueError, match='incident_directions'):
        scattering.coefficient.freefield(p_sample, p_reference, mics, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_05(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    p_sample = _create_frequencydata(mics.cshape, 0, frequencies)
    p_reference = _create_frequencydata(mics.cshape, 0, frequencies)
    p_sample.freq[5, 7, :] = 1
    p_sample.freq[5, 5, :] = 1
    p_reference.freq[5, 5, :] = 1
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 0.5)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    p_sample = _create_frequencydata(mics.cshape, 0, frequencies)
    p_reference = _create_frequencydata(mics.cshape, 0, frequencies)
    p_reference.freq[5, 0, :] = 1
    p_sample.freq[5, 0, :] = 1
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0_with_inci(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    incident_directions = _create_coordinates(
        np.linspace(0, 2*np.pi, num=8), np.linspace(0, np.pi/2, num=4))
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = _create_frequencydata(data_shape, 0, frequencies)
    p_reference = _create_frequencydata(data_shape, 0, frequencies)
    p_reference.freq[:, :, 1, 2, :] = 1
    p_sample.freq[:, :, 1, 2, :] = 1
    s, s_rand = scattering.coefficient.freefield(
        p_sample, p_reference, mics, incident_directions=incident_directions)
    np.testing.assert_allclose(s.freq, 0)
    np.testing.assert_allclose(s_rand.freq, 0)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1_with_inci(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=19))
    incident_directions = _create_coordinates(
        np.linspace(0, 2*np.pi, num=8), np.linspace(0, np.pi/2, num=4))
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = _create_frequencydata(data_shape, 0, frequencies)
    p_reference = _create_frequencydata(data_shape, 0, frequencies)
    p_reference.freq[:, :, 1, 2, :] = 1
    p_sample.freq[:, :, 2, 3, :] = 1
    s, s_rand = scattering.coefficient.freefield(
        p_sample, p_reference, mics, incident_directions=incident_directions)
    np.testing.assert_allclose(s.freq, 1)
    np.testing.assert_allclose(s_rand.freq, 1)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_05_with_inci(frequencies):
    mics = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi, num=10))
    incident_directions = _create_coordinates(
        np.linspace(0, 2*np.pi, num=8), np.linspace(0, np.pi/2, num=4))
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = _create_frequencydata(data_shape, 0, frequencies)
    p_reference = _create_frequencydata(data_shape, 0, frequencies)
    p_sample.freq[:, :, 5, 7, :] = 1
    p_sample.freq[:, :, 5, 5, :] = 1
    p_reference.freq[:, :, 5, 5, :] = 1
    s, s_rand = scattering.coefficient.freefield(
        p_sample, p_reference, mics, incident_directions=incident_directions)
    np.testing.assert_allclose(s.freq, 0.5)
    np.testing.assert_allclose(s_rand.freq, 0.5)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("s_value",  [
    (0), (0.2), (0.5), (0.8), (1)])
@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_incidence_constant_s(s_value, frequencies):
    incident_directions = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi/2, num=10))
    s = _create_frequencydata(incident_directions.cshape, s_value, frequencies)
    s_rand = scattering.coefficient.random_incidence(s, incident_directions)
    np.testing.assert_allclose(s_rand.freq, s_value)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_incidence_non_constant_s(frequencies):
    incident_directions = _create_coordinates(
        np.linspace(0, 2*np.pi, num=37),
        np.linspace(0, np.pi/2, num=10))
    s_value = np.arange(37*10).reshape(incident_directions.cshape) / 370
    sph = incident_directions.get_sph()
    actual_weight = np.sin(2*sph[..., 1])   # sin(2*theta)
    actual_weight /= np.sum(actual_weight)
    s = _create_frequencydata(incident_directions.cshape, s_value, frequencies)
    s_rand = scattering.coefficient.random_incidence(s, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(s_rand.freq, desired)


def _create_coordinates(phi_rad, theta_rad):
    phi, theta = np.meshgrid(phi_rad, theta_rad)
    coords = Coordinates(
        phi, theta, np.ones(phi.shape), 'sph')
    return coords


def _create_frequencydata(shape, data_raw, frequencies):
    frequencies = np.atleast_1d(frequencies)
    shape_new = np.append(shape, frequencies.shape)
    if hasattr(data_raw, "__len__"):
        data_raw = np.repeat(
            data_raw[..., np.newaxis], len(frequencies), axis=-1)
    data = np.zeros(shape_new) + data_raw
    p_reference = FrequencyData(data, frequencies)
    return p_reference
