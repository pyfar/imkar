"""Top-level package for imkar."""

__author__ = """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.1.0'


from . import analytical


__all__ = [
    'sinusoid',
    'cartesian_to_theta_phi',
    'theta_phi_to_cartesian',
]
