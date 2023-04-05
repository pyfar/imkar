import pytest
import numpy as np

from imkar import analytical

"""
References
----------
.. [#]J. Embrechts and A. Billon, "Theoretical Determination of the Random-
Incidence Scattering Coefficients of Infinite Rigid Surfaces with a
Periodic Rectangular Roughness Profile", Acta Acustica united with Acustica,
Bd. 97, Nr. 4, S. 607-617, Juli 2011, doi:10.3813/AAA.918441

"""


# Test compares the directional scattering coefficient under 30° with the one
# shown in Figure 2 in [#]_.
def test_rectangular_under_30_degree():
    c_over_L = 343.901
    x = np.linspace(4, 8, num=5)
    result = analytical.rectangular(x*c_over_L, np.array([30]), 0.50, 1,
                                    0.49, 343.901)
    actual = np.reshape(np.array([[0.82], [0.11], [0.75], [0.67],
                                  [0.48]]), (1, 5))
    np.testing.assert_allclose(result.freq, actual, atol=1e-2)


def test_rectangular_wrong_input():
    with pytest.raises(TypeError, match='frequency_vector'):
        analytical.rectangular(np.array([2+3j]), np.array([30]),
                               0.5, 1, 0.49, 343.901)

    with pytest.raises(ValueError, match='frequency_vector'):
        analytical.rectangular(np.array([[500, 1000], [1500, 2000]]),
                               np.array([30]), 0.5, 1, 0.49, 343.901)

    with pytest.raises(TypeError, match='phis'):
        analytical.rectangular(np.array([500]), np.array([7+3j]), 0.5, 1,
                               0.49, 343.901)

    with pytest.raises(ValueError, match='phis'):
        analytical.rectangular(np.array([500]), np.array([[7, 25], [9, 35]]),
                               0.5, 1, 0.49, 343.901)

    with pytest.raises(ValueError, match='phis'):
        analytical.rectangular(np.array([500]), np.array([30, 60, 90, 120]),
                               0.5, 1, 0.49, 343.901)

    with pytest.raises(TypeError, match='width'):
        analytical.rectangular(np.array([500]), np.array([30]), 0, 1,
                               0.49, 343.901)

    with pytest.raises(TypeError, match='length'):
        analytical.rectangular(np.array([500]), np.array([30]), 0.5, (1+5j),
                               0.49, 343.901)

    with pytest.raises(TypeError, match='height'):
        analytical.rectangular(np.array([500]), np.array([30]), 0.5, 1,
                               -3, 343.901)
