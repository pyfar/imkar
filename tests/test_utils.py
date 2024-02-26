import pytest
import numpy as np
import pyfar as pf

from imkar import utils


@pytest.mark.parametrize("c_value",  [
    (0), (0.2), (0.5), (0.8), (1)])
@pytest.mark.parametrize("frequencies",  [
    ([100, 200]), ([100])])
def test_random_constant_coefficient(
        c_value, frequencies, half_sphere):
    incident_directions = half_sphere
    shape = np.append(half_sphere.cshape, len(frequencies))
    coefficient = pf.FrequencyData(np.zeros(shape)+c_value, frequencies)
    c_rand = utils.paris_formula(coefficient, incident_directions)
    np.testing.assert_allclose(c_rand.freq, c_value)
    assert c_rand.comment == 'random-incidence coefficient'


def test_random_non_constant_coefficient():
    data = pf.samplings.sph_gaussian(10)
    incident_directions = data[data.get_sph().T[1] <= np.pi/2]
    incident_cshape = incident_directions.cshape
    s_value = np.arange(
        incident_cshape[0]).reshape(incident_cshape) / incident_cshape[0]
    theta = incident_directions.get_sph().T[1]
    actual_weight = np.cos(theta) * incident_directions.weights
    actual_weight /= np.sum(actual_weight)
    coefficient = pf.FrequencyData(s_value.reshape((50, 1)), [100])
    c_rand = utils.paris_formula(coefficient, incident_directions)
    desired = np.sum(s_value*actual_weight)
    np.testing.assert_allclose(c_rand.freq, desired)
