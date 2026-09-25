"""pytest setup for the harness tests (CPU only)."""


def pytest_configure(config):
    config.addinivalue_line("markers", "overfit: the 200-step overfit test (about 10-60 s on CPU)")
