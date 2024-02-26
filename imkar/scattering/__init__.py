"""
This module collects the functionality around sound scattering, such as,
(random incident) scattering coefficients, and analytical solutions.
"""

from .scattering import (
    freefield,
    random,
)

__all__ = [
    'freefield',
    'random',
    ]
