import pytest
import numpy as np

from imkar import scattering
from imkar.testing import stub_utils


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_1(frequencies, coords_half_sphere_10_deg):
    mics = coords_half_sphere_10_deg
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 1, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
    p_sample.freq[5, 0, :] = 0
    p_reference.freq[5, 0, :] = np.sum(p_sample.freq.flatten())/2
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 1)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_wrong_input(frequencies, coords_half_sphere_10_deg):
    mics = coords_half_sphere_10_deg
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 1, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
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
def test_freefield_05(frequencies, coords_half_sphere_10_deg):
    mics = coords_half_sphere_10_deg
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
    p_sample.freq[5, 7, :] = 1
    p_sample.freq[5, 5, :] = 1
    p_reference.freq[5, 5, :] = 1
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 0.5)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0(frequencies, coords_half_sphere_10_deg):
    mics = coords_half_sphere_10_deg
    p_sample = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        mics.cshape, 0, frequencies)
    p_reference.freq[5, 0, :] = 1
    p_sample.freq[5, 0, :] = 1
    s = scattering.coefficient.freefield(p_sample, p_reference, mics)
    np.testing.assert_allclose(s.freq, 0)
    assert s.freq.shape[-1] == len(frequencies)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_freefield_0_with_inci(
        frequencies, coords_half_sphere_10_deg, coords_half_sphere_30_deg):
    mics = coords_half_sphere_10_deg
    incident_directions = coords_half_sphere_30_deg
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
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
def test_freefield_1_with_inci(
        frequencies, coords_half_sphere_10_deg, coords_half_sphere_30_deg):
    mics = coords_half_sphere_10_deg
    incident_directions = coords_half_sphere_30_deg
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
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
def test_freefield_05_with_inci(
        frequencies, coords_half_sphere_10_deg, coords_half_sphere_30_deg):
    mics = coords_half_sphere_10_deg
    incident_directions = coords_half_sphere_30_deg
    data_shape = np.array((incident_directions.cshape, mics.cshape)).flatten()
    p_sample = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
    p_reference = stub_utils.frequencydata_from_shape(
        data_shape, 0, frequencies)
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
def test_random_incidence_constant_s(
        s_value, frequencies, coords_half_sphere_10_deg):
    incident_directions = coords_half_sphere_10_deg
    s = stub_utils.frequencydata_from_shape(
        incident_directions.cshape, s_value, frequencies)
    s_rand = scattering.coefficient.random_incidence(s, incident_directions)
    np.testing.assert_allclose(s_rand.freq, s_value)


@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_incidence_non_constant_s(
        frequencies, coords_half_sphere_10_deg):
    incident_directions = coords_half_sphere_10_deg
    s_value = np.arange(37*10).reshape(incident_directions.cshape) / 370
    sph = incident_directions.get_sph()
    actual_weight = np.sin(2*sph[..., 1])   # sin(2*theta)
    actual_weight /= np.sum(actual_weight)
    s = stub_utils.frequencydata_from_shape(
        [], s_value, frequencies)
    s_rand = scattering.coefficient.random_incidence(s, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(s_rand.freq, desired)
