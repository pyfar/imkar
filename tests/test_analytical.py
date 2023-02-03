import numpy as np
import pyfar as pf

from imkar import analytical
from imkar.testing import stub_utils

"""
References
----------
.. [#]J. Embrechts and A. Billon, "Theoretical Determination of the Random-
Incidence Scattering Coefficients of Infinite Rigid Surfaces with a 
Periodic Rectangular Roughness Profile", Acta Acustica united with Acustica,
Bd. 97, Nr. 4, S. 607-617, Juli 2011, doi:10.3813/AAA.918441

"""

#Test compares the directional scattering coefficient under 30° with the one
#shown in Figure 2 in [#]_.
def test_rectangular_under_30_degree():
    c_over_L = 343.901
    x = np.arange(4,8.1,0.1)
    result = analytical.rectangular(x*c_over_L,np.array([30]),0.50,1,0.49)
    actual = np.reshape(np.array([[0.82],[0.46],[0.27],[0.42],[0.43],[0.52],\
        [0.74],[0.85],[0.54],[0.3],[0.11],[0.18],[0.92],[0.91],[0.73],[0.92],\
            [0.7],[0.36],[0.3],[0.45],[0.75],[0.98],[0.64],[0.34],[0.32],[0.7],\
                [0.64],[0.69],[0.79],[0.91],[0.67],[0.42],[0.18],[0.23],\
                    [0.82],[0.94],[0.74],[0.66],[0.7],[0.64],[0.48]]),(1,41))    

    np.testing.assert_allclose(result.freq, actual, atol=1e-2)

#def test_rectangular_wrong_input():
