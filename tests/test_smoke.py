import pi_ai


def test_package_exposes_version():
    assert isinstance(pi_ai.__version__, str)
    assert pi_ai.__version__ != ""
