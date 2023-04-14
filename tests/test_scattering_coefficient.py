import pytest
import numpy as np
import pyfar as pf

from imkar import scattering
from imkar.testing import stub_utils


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1(frequencies):
    mics = pf.samplings.sph_gaussian(42)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 1, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_sample.freq[5, :] = 0
    p_reference.freq[5, :] = np.sum(p_sample.freq.flatten())/2
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_wrong_input(frequencies):
    mics = pf.samplings.sph_gaussian(42)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 1, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_sample.freq[5, :] = 0
    p_reference.freq[5, :] = np.sum(p_sample.freq.flatten())/2
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


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_05(frequencies):
    mics = pf.samplings.sph_gaussian(42)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_sample.freq[7, :] = 1
    p_sample.freq[28, :] = 1
    p_reference.freq[7, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0.5)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0(frequencies):
    mics = pf.samplings.sph_gaussian(42)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    p_sample = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        mics.cshape, 0, frequencies)
    p_reference.freq[5, :] = 1
    p_sample.freq[5, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0_with_incident(frequencies):
    mics = pf.samplings.sph_equal_area(42)
    mics.weights = pf.samplings.calculate_sph_voronoi_weights(mics)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.get_sph().T[1] <= np.pi/2]
    incident_directions = incident_directions[
        incident_directions.get_sph().T[0] <= np.pi/2]
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 2, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0)
    np.testing.assert_allclose(s_rand.freq, 0)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1_with_incidence(
        frequencies):
    mics = pf.samplings.sph_equal_angle(10)
    mics.weights = pf.samplings.calculate_sph_voronoi_weights(mics)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.get_sph().T[1] <= np.pi/2]
    incident_directions = incident_directions[
        incident_directions.get_sph().T[0] <= np.pi/2]
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_reference.freq[:, 2, :] = 1
    p_sample.freq[:, 3, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 1)
    np.testing.assert_allclose(s_rand.freq, 1)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
@pytest.mark.parametrize("mics",  [
    (pf.samplings.sph_equal_angle(10)),
    ])
def test_freefield_05_with_inci(frequencies, mics):
    mics.weights = pf.samplings.calculate_sph_voronoi_weights(mics)
    mics = mics[mics.get_sph().T[1] <= np.pi/2]  # delete lower part of sphere
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[
        incident_directions.get_sph().T[1] <= np.pi/2]
    incident_directions = incident_directions[
        incident_directions.get_sph().T[0] <= np.pi/2]
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequency_data_from_shape(
        data_shape, 0, frequencies)
    p_sample.freq[:, 7, :] = 1
    p_sample.freq[:, 5, :] = 1
    p_reference.freq[:, 5, :] = 1
    s = scattering.freefield(p_sample, p_reference, mics.weights)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s.freq, 0.5)
    np.testing.assert_allclose(s_rand.freq, 0.5)
    assert s.freq.shape[-1] == len(frequencies)
    assert s.cshape == incident_directions.cshape
    assert s_rand.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("s_value",  [
    (0), (0.2), (0.5), (0.8), (1)])
@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_constant_s(
        s_value, frequencies):
    coords = pf.samplings.sph_gaussian(10)
    incident_directions = coords[coords.get_sph().T[1] <= np.pi/2]
    s = stub_utils.frequency_data_from_shape(
        incident_directions.cshape, s_value, frequencies)
    s_rand = scattering.random(s, incident_directions)
    np.testing.assert_allclose(s_rand.freq, s_value)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_non_constant_s(frequencies):
    data = pf.samplings.sph_gaussian(10)
    incident_directions = data[data.get_sph().T[1] <= np.pi/2]
    incident_cshape = incident_directions.cshape
    s_value = np.arange(
        incident_cshape[0]).reshape(incident_cshape) / incident_cshape[0]
    theta = incident_directions.get_sph().T[1]
    actual_weight = np.cos(theta) * incident_directions.weights
    actual_weight /= np.sum(actual_weight)
    s = stub_utils.frequency_data_from_shape(
        [], s_value, frequencies)
    s_rand = scattering.random(s, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(s_rand.freq, desired)
