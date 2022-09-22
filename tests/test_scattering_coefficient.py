import pytest
import numpy as np

from imkar import scattering
from pyfar import FrequencyData, Coordinates


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1(frequencies):
    mics = _create_coordinates(
       np.arange(0, 370, 10), np.arange(0, 100, 10))
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
       np.arange(0, 370, 10), np.arange(0, 100, 10))
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
       np.arange(0, 370, 10), np.arange(0, 100, 10))
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
       np.arange(0, 370, 10), np.arange(0, 100, 10))
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
        np.arange(0, 370, 10), np.arange(0, 100, 10))
    incident_directions = _create_coordinates(
        np.arange(0, 360+30, 30), np.arange(0, 90+30, 30))
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
        np.arange(0, 370, 10), np.arange(0, 100, 10))
    incident_directions = _create_coordinates(
        np.arange(0, 360+30, 30), np.arange(0, 90+30, 30))
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
        np.arange(0, 370, 10), np.arange(0, 100, 10))
    incident_directions = _create_coordinates(
        np.arange(0, 360+30, 30), np.arange(0, 90+30, 30))
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


def _create_coordinates(phi_deg, theta_deg):
    phi, theta = np.meshgrid(phi_deg, theta_deg)
    coords = Coordinates(
        phi, theta, np.ones(phi.shape), 'sph', 'top_colat', 'deg')
    return coords


def _create_frequencydata(shape, value, frequencies):
    frequencies = np.atleast_1d(frequencies)
    shape_new = np.append(shape, frequencies.shape)
    data = np.zeros(shape_new) + value
    p_reference = FrequencyData(data, frequencies)
    return p_reference

