import pytest


def test_import_imkar():
    try:
        import imkar           # noqa
    except ImportError:
        pytest.fail('import imkar failed')
