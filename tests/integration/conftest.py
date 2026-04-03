"""Integration test configuration."""


def pytest_collection_modifyitems(items):
    """Auto-apply the integration marker to all tests in this directory."""
    import pytest

    for item in items:
        item.add_marker(pytest.mark.integration)
