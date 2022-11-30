import pytest
import numpy as np
from imkar.testing import stub_utils


@pytest.fixture
def coords_half_sphere_1_deg():
    delta = 1
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi/2, num=int(90/delta)+1))


@pytest.fixture
def coords_half_sphere_5_deg():
    delta = 5
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi/2, num=int(90/delta)+1))


@pytest.fixture
def coords_half_sphere_10_deg():
    delta = 10
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi/2, num=int(90/delta)+1))


@pytest.fixture
def coords_half_sphere_30_deg():
    delta = 30
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi/2, num=int(90/delta)+1))


@pytest.fixture
def coords_half_sphere_45_deg():
    delta = 45
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi/2, num=int(90/delta)+1))


@pytest.fixture
def coords_sphere_10_deg():
    delta = 10
    return stub_utils.create_coordinates_sph(
        np.linspace(0, 2*np.pi, num=int(360/delta)+1),
        np.linspace(0, np.pi, num=int(180/delta)+1))
