import pytest
import numpy as np
import pyfar as pf

from imkar import utils


@pytest.mark.parametrize("c_value",  [
    (0), (0.2), (0.5), (0.8), (1)])
@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_constant_coefficient(
        c_value, frequencies):
    incident_directions = pf.Coordinates.from_spherical_colatitude(
        0, np.linspace(0, np.pi/2, 10), 1)
    incident_directions.weights = np.sin(incident_directions.colatitude)
    shape = np.append(incident_directions.cshape, len(frequencies))
    coefficient = pf.FrequencyData(np.zeros(shape)+c_value, frequencies)
    c_rand = utils.paris_formula(coefficient, incident_directions)
    np.testing.assert_allclose(c_rand.freq, c_value)
    assert c_rand.comment == 'random-incidence coefficient'


def test_random_non_constant_coefficient():
    incident_directions = pf.samplings.sph_gaussian(10)
    incident_directions = incident_directions[incident_directions.z >= 0]
    incident_cshape = incident_directions.cshape
    s_value = np.arange(
        incident_cshape[0]).reshape(incident_cshape) / incident_cshape[0]
    theta = incident_directions.colatitude
    actual_weight = np.cos(theta) * incident_directions.weights
    actual_weight /= np.sum(actual_weight)
    coefficient = pf.FrequencyData(s_value.reshape((50, 1)), [100])
    c_rand = utils.paris_formula(coefficient, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(c_rand.freq, desired)


def test_paris_formula_coefficients_type_error():
    incident_directions = pf.samplings.sph_gaussian(10)
    with pytest.raises(
            ValueError, match="coefficients has to be FrequencyData"):
        utils.paris_formula(np.zeros(10), incident_directions)


def test_paris_formula_incident_directions_type_error():
    coefficients = pf.FrequencyData(np.zeros((10, 1)), [100])
    with pytest.raises(
            ValueError,
            match="incident_directions have to be None or Coordinates"):
        utils.paris_formula(coefficients, None)


def test_paris_formula_shape_mismatch_error():
    incident_directions = pf.samplings.sph_gaussian(10)
    coefficients = pf.FrequencyData(np.zeros((5, 1)), [100])
    with pytest.raises(
            ValueError,
            match="the last dimension of coefficients needs be same as"):
        utils.paris_formula(coefficients, incident_directions)
