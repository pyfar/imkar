import pytest
import pyfar as pf
import numpy as np


@pytest.fixture
def half_sphere_gaussian():
    """return 42th order gaussian sampling for the half sphere and radius 1.

    Returns
    -------
    pf.Coordinates
        half sphere sampling grid
    """
    mics = pf.samplings.sph_gaussian(42)
    # delete lower part of sphere
    return mics[mics.get_sph().T[1] <= np.pi/2]
