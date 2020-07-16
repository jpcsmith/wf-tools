"""Shared fixtures for classifier tests."""
import pytest


def pytest_addoption(parser):
    """Add options to pytest CLI."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """Update the list of available markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Modify the collected tests."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
