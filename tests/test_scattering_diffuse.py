import numpy as np
import pytest
import imkar.scattering.diffuse as isd
import pyfar as pf


def test_maximum_sample_absorption_coefficient_basic():
    freqs = np.array([100, 200, 400, 800])
    result = isd.maximum_sample_absorption_coefficient(freqs)
    assert isinstance(result, pf.FrequencyData)
    np.testing.assert_allclose(result.frequencies, freqs)
    np.testing.assert_allclose(result.freq, 0.5)
    assert "Maximum absorption coefficient" in result.comment


def test_maximum_sample_absorption_coefficient_list_input():
    freqs = [100, 200, 400]
    result = isd.maximum_sample_absorption_coefficient(freqs)
    np.testing.assert_allclose(result.frequencies, np.array(freqs))
    np.testing.assert_allclose(result.freq, 0.5)


def test_maximum_sample_absorption_coefficient_non_1d_input():
    freqs = np.array([[100, 200], [300, 400]])
    with pytest.raises(ValueError, match="frequencies must be a 1D array"):
        isd.maximum_sample_absorption_coefficient(freqs)


def test_maximum_sample_absorption_coefficient_invalid_type():
    freqs = "not_a_number"
    with pytest.raises(
            TypeError,
            match="frequencies must be convertible to a float array"):
        isd.maximum_sample_absorption_coefficient(freqs)


def test_maximum_baseplate_scattering_coefficient_default():
    result = isd.maximum_baseplate_scattering_coefficient()
    assert isinstance(result, pf.FrequencyData)
    expected_freqs = np.array([
        100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    ])
    expected_data = np.array([[
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10,
        0.10, 0.10, 0.15, 0.15, 0.15, 0.20, 0.20, 0.20, 0.25,
    ]])
    np.testing.assert_allclose(result.frequencies, expected_freqs)
    np.testing.assert_allclose(result.freq, expected_data)
    assert "baseplate" in result.comment


def test_maximum_baseplate_scattering_coefficient_with_scale():
    N = 2
    result = isd.maximum_baseplate_scattering_coefficient(N)
    expected_freqs = np.array([
        100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    ]) / N
    np.testing.assert_allclose(result.frequencies, expected_freqs)

def test_maximum_baseplate_scattering_coefficient_invalid_type():
    with pytest.raises(TypeError, match="N must be a positive integer."):
        isd.maximum_baseplate_scattering_coefficient(N=1.5)
    with pytest.raises(TypeError, match="N must be a positive integer."):
        isd.maximum_baseplate_scattering_coefficient(N=-1)

def test_maximum_baseplate_scattering_coefficient_negative_N():
    # Negative N is technically an integer, but let's check if it works
    N = 5
    result = isd.maximum_baseplate_scattering_coefficient(N)
    expected_freqs = np.array([
        100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    ]) / N
    np.testing.assert_allclose(result.frequencies, expected_freqs)
