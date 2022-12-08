import numpy as np

from imkar import analytical
from imkar.testing import stub_utils

def test_rectangular():
    result = analytical.rectangular(np.array([100]),np.array([30]),0.5,1,0.49)
    actual = 7.554E-13
    np.testing.assert_allclose(result, actual, rtol=5e-3)

test_rectangular()