"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def sample_config() -> dict[str, int | float]:
    """Sample configuration for testing."""
    return {
        "camera_width": 640,
        "camera_height": 480,
        "detection_threshold": 0.7,
    }
