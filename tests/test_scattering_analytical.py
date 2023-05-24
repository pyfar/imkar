import pytest
import numpy as np

import imkar as ik


def test_rectangular_under_30_degree():
    # Test compares the directional scattering coefficient under 30° with
    # the one shown in Figure 2 in Embrechts et al.
    c_over_L = 343.901
    x = np.linspace(4, 8, num=5)
    result = ik.scattering.analytical.rectangular(
        x*c_over_L, np.array([30]), 0.50, 0.50, 0.49, 343.901)
    actual = np.reshape(np.array(
        [0.82, 0.11, 0.75, 0.67, 0.48]), (1, 5))
    np.testing.assert_allclose(result.freq, actual, atol=1e-2)


@pytest.mark.parametrize("frequencies",  [
    ([343.901 * 4, 343.901 * 8]), ([100]), np.array([343.901 * 4])])
@pytest.mark.parametrize("incident_angle",  [
    ([30, 30]), ([30]), np.array([30])])
def test_rectangular_different_inputs(frequencies, incident_angle):
    # Test compares the directional scattering coefficient under 30° with
    # the one shown in Figure 2 in Embrechts et al.
    ik.scattering.analytical.rectangular(
        frequencies, incident_angle, 0.50, 0.50, 0.49, 343.901)


def test_rectangular_wrong_input():
    with pytest.raises(ValueError, match='frequencies'):
        ik.scattering.analytical.rectangular(
            np.array([[500, 1000], [1500, 2000]]),
            np.array([30]), 0.5, 0.5, 0.49, 343.901)

    with pytest.raises(ValueError, match='incident_angles'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([[7, 25], [9, 35]]),
            0.5, 0.5, 0.49, 343.901)

    with pytest.raises(ValueError, match='incident_angles'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([30, 60, 90, 120]),
            0.5, 0.5, 0.49, 343.901)

    with pytest.raises(TypeError, match='rectangle_width'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([30]), 0, 0.5, 0.49)

    with pytest.raises(TypeError, match='gap_width'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([30]), 0.5, (1+5j), 0.49, 343.901)

    with pytest.raises(TypeError, match='height'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([30]), 0.5, 0.5, -3, 343.901)

    with pytest.raises(TypeError, match='speed_of_sound'):
        ik.scattering.analytical.rectangular(
            np.array([500]), np.array([30]), 0.5, 0.5, 0.49, -343.901)
