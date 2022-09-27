import pytest
import numpy as np
from imkar import analytical


def test_analytical():
    s = analytical.rectangular()
    assert s == 0
