import pytest
import numpy as np

from imkar.testing import stub_utils


@pytest.mark.parametrize(
    "shapes",  [
        (3, 2),
        (5, 2),
        (3, 2, 7),
    ])
@pytest.mark.parametrize(
    "data_in",  [
        0.1,
        0,
        np.array([0.1, 1]),
        np.arange(4*5).reshape(4, 5),
    ])
@pytest.mark.parametrize(
    "frequency",  [
        [100],
        [100, 200],
    ])
def test_frequency_data_from_shape(shapes, data_in, frequency):
    data = stub_utils.frequency_data_from_shape(shapes, data_in, frequency)
    # npt.assert_allclose(data.freq, data_in)
    if hasattr(data_in, '__len__'):
        for idx in range(len(data_in.shape)):
            assert data.cshape[idx] == data_in.shape[idx]
        for idx in range(len(shapes)):
            assert data.cshape[idx+len(data_in.shape)] == shapes[idx]

    else:
        for idx in range(len(shapes)):
            assert data.cshape[idx] == shapes[idx]
    assert data.n_bins == len(frequency)
