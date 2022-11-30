def test_import_imkar():
    try:
        import imkar           # noqa
    except ImportError:
        assert False
