import numpy as np

from imkar import analytical
from imkar.testing import stub_utils

def test_rectangular():
    result = analytical.rectangular(np.array([500]),np.array([30]),0.5,1,0.49)
    actual = 0.5927
    np.testing.assert_allclose(result, actual, rtol=5e-3)

#test_rectangular()