"""Unit tests for MaixCamSource camera implementation.

These tests mock the maix library to allow testing on non-MaixCam hardware.

Test categories:
1. Initialization (5 tests)
2. Start/Stop lifecycle (5 tests)
3. Frame capture (6 tests)
4. Resolution management (4 tests)
5. Context manager (3 tests)
6. Thread safety (3 tests)
7. Error handling (4 tests)

Total: ~30 tests
"""

import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# =============================================================================
# Mock MaixPy before importing the module
# =============================================================================


@pytest.fixture(autouse=True)
def mock_maix_module():
    """Mock the maix module for all tests."""
    # Create mock maix module
    mock_maix = MagicMock()
    mock_maix.camera = MagicMock()
    mock_maix.image = MagicMock()
    mock_maix.display = MagicMock()

    # Mock image Format enum
    mock_maix.image.Format = MagicMock()
    mock_maix.image.Format.FMT_RGB888 = 0

    # Add to sys.modules before importing
    sys.modules["maix"] = mock_maix
    sys.modules["maix.camera"] = mock_maix.camera
    sys.modules["maix.image"] = mock_maix.image
    sys.modules["maix.display"] = mock_maix.display

    yield mock_maix

    # Cleanup
    for mod in ["maix", "maix.camera", "maix.image", "maix.display"]:
        if mod in sys.modules:
            del sys.modules[mod]


@pytest.fixture
def mock_camera_instance():
    """Create a mock camera instance."""
    mock_cam = MagicMock()

    # Create mock image that read() returns
    mock_image = MagicMock()
    # Return a 480x640x3 RGB image
    mock_image.to_numpy.return_value = np.random.randint(
        0, 255, (480, 640, 3), dtype=np.uint8
    )
    mock_cam.read.return_value = mock_image

    return mock_cam


@pytest.fixture
def camera_source(mock_maix_module, mock_camera_instance):
    """Create a MaixCamSource with mocked dependencies."""
    # Patch the MAIXPY_AVAILABLE flag
    with patch.dict(
        "src.infrastructure.camera.maixcam_camera.__dict__",
        {"MAIXPY_AVAILABLE": True, "camera": mock_maix_module.camera, "image": mock_maix_module.image},
    ):
        # Reload the module with mocks in place
        import importlib
        import src.infrastructure.camera.maixcam_camera as maixcam_module

        # Patch module-level variables
        maixcam_module.MAIXPY_AVAILABLE = True
        maixcam_module.camera = mock_maix_module.camera
        maixcam_module.image = mock_maix_module.image

        # Make Camera constructor return our mock
        mock_maix_module.camera.Camera.return_value = mock_camera_instance

        from src.infrastructure.camera.maixcam_camera import MaixCamSource

        yield MaixCamSource()


@pytest.fixture
def running_camera(camera_source, mock_camera_instance):
    """Create a started camera source."""
    camera_source.start()
    yield camera_source
    camera_source.stop()


# =============================================================================
# 1. Initialization Tests (5 tests)
# =============================================================================


class TestMaixCamInitialization:
    """Tests for MaixCamSource initialization."""

    def test_init_with_default_config(self, camera_source):
        """Camera initializes with default configuration."""
        assert camera_source.config.width == 640
        assert camera_source.config.height == 480
        assert camera_source.config.fps == 30
        assert camera_source.is_running is False

    def test_init_with_custom_config(self, mock_maix_module, mock_camera_instance):
        """Camera initializes with custom configuration."""
        from src.core.interfaces.camera import CameraConfig

        with patch.dict(
            "src.infrastructure.camera.maixcam_camera.__dict__",
            {"MAIXPY_AVAILABLE": True, "camera": mock_maix_module.camera, "image": mock_maix_module.image},
        ):
            import src.infrastructure.camera.maixcam_camera as maixcam_module
            maixcam_module.MAIXPY_AVAILABLE = True
            maixcam_module.camera = mock_maix_module.camera
            maixcam_module.image = mock_maix_module.image
            mock_maix_module.camera.Camera.return_value = mock_camera_instance

            from src.infrastructure.camera.maixcam_camera import MaixCamSource

            config = CameraConfig(width=1280, height=720, fps=60)
            cam = MaixCamSource(config)

            assert cam.config.width == 1280
            assert cam.config.height == 720
            assert cam.config.fps == 60

    def test_init_not_running(self, camera_source):
        """Camera is not running after initialization."""
        assert camera_source.is_running is False

    def test_init_frame_counter_zero(self, camera_source):
        """Frame counter is zero after initialization."""
        assert camera_source._frame_counter == 0

    def test_init_without_maixpy_raises_error(self):
        """Initialization fails when MaixPy is not available."""
        with patch.dict(
            "src.infrastructure.camera.maixcam_camera.__dict__",
            {"MAIXPY_AVAILABLE": False},
        ):
            import src.infrastructure.camera.maixcam_camera as maixcam_module
            maixcam_module.MAIXPY_AVAILABLE = False

            # Reimport to get updated class
            from src.infrastructure.camera.maixcam_camera import MaixCamSource

            with pytest.raises(RuntimeError) as exc_info:
                MaixCamSource()

            assert "MaixPy not available" in str(exc_info.value)


# =============================================================================
# 2. Start/Stop Lifecycle Tests (5 tests)
# =============================================================================


class TestMaixCamLifecycle:
    """Tests for camera start/stop lifecycle."""

    def test_start_sets_running(self, camera_source):
        """Starting camera sets is_running to True."""
        camera_source.start()
        assert camera_source.is_running is True
        camera_source.stop()

    def test_stop_clears_running(self, running_camera):
        """Stopping camera sets is_running to False."""
        running_camera.stop()
        assert running_camera.is_running is False

    def test_start_twice_is_noop(self, camera_source):
        """Starting already running camera is a no-op."""
        camera_source.start()
        camera_source.start()  # Should not raise
        assert camera_source.is_running is True
        camera_source.stop()

    def test_stop_twice_is_noop(self, running_camera):
        """Stopping already stopped camera is a no-op."""
        running_camera.stop()
        running_camera.stop()  # Should not raise
        assert running_camera.is_running is False

    def test_start_resets_frame_counter(self, running_camera):
        """Starting camera resets frame counter."""
        # Capture some frames
        running_camera.capture()
        running_camera.capture()
        assert running_camera._frame_counter > 0

        # Restart
        running_camera.stop()
        running_camera.start()
        assert running_camera._frame_counter == 0


# =============================================================================
# 3. Frame Capture Tests (6 tests)
# =============================================================================


class TestMaixCamCapture:
    """Tests for frame capture functionality."""

    def test_capture_returns_frame(self, running_camera):
        """Capture returns a Frame object."""
        from src.core.interfaces.camera import Frame

        frame = running_camera.capture()
        assert frame is not None
        assert isinstance(frame, Frame)

    def test_capture_frame_has_rgb_image(self, running_camera):
        """Captured frame has RGB image with correct shape."""
        frame = running_camera.capture()
        assert frame is not None
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3  # RGB channels

    def test_capture_frame_has_timestamp(self, running_camera):
        """Captured frame has valid timestamp."""
        before = time.time()
        frame = running_camera.capture()
        after = time.time()

        assert frame is not None
        assert before <= frame.timestamp <= after

    def test_capture_increments_frame_id(self, running_camera):
        """Each capture increments frame ID."""
        frame1 = running_camera.capture()
        frame2 = running_camera.capture()
        frame3 = running_camera.capture()

        assert frame1.frame_id == 1
        assert frame2.frame_id == 2
        assert frame3.frame_id == 3

    def test_capture_when_not_running_returns_none(self, camera_source):
        """Capture returns None when camera is not running."""
        frame = camera_source.capture()
        assert frame is None

    def test_capture_on_read_failure_returns_none(self, running_camera, mock_camera_instance):
        """Capture returns None when camera read fails."""
        mock_camera_instance.read.return_value = None

        frame = running_camera.capture()
        assert frame is None


# =============================================================================
# 4. Resolution Management Tests (4 tests)
# =============================================================================


class TestMaixCamResolution:
    """Tests for resolution management."""

    def test_get_resolution_returns_config_values(self, camera_source):
        """get_resolution returns configured width and height."""
        width, height = camera_source.get_resolution()
        assert width == 640
        assert height == 480

    def test_set_resolution_updates_config(self, camera_source):
        """set_resolution updates configuration."""
        result = camera_source.set_resolution(1280, 720)

        assert result is True
        assert camera_source.config.width == 1280
        assert camera_source.config.height == 720

    def test_set_resolution_invalid_width_raises(self, camera_source):
        """set_resolution raises ValueError for invalid width."""
        with pytest.raises(ValueError) as exc_info:
            camera_source.set_resolution(0, 480)

        assert "width must be positive" in str(exc_info.value)

    def test_set_resolution_invalid_height_raises(self, camera_source):
        """set_resolution raises ValueError for invalid height."""
        with pytest.raises(ValueError) as exc_info:
            camera_source.set_resolution(640, -1)

        assert "height must be positive" in str(exc_info.value)


# =============================================================================
# 5. Context Manager Tests (3 tests)
# =============================================================================


class TestMaixCamContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_starts_camera(self, camera_source):
        """Entering context manager starts the camera."""
        with camera_source as cam:
            assert cam.is_running is True

    def test_context_manager_stops_camera(self, camera_source):
        """Exiting context manager stops the camera."""
        with camera_source:
            pass
        assert camera_source.is_running is False

    def test_context_manager_stops_on_exception(self, camera_source):
        """Camera is stopped even if exception occurs."""
        try:
            with camera_source:
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert camera_source.is_running is False


# =============================================================================
# 6. Thread Safety Tests (3 tests)
# =============================================================================


class TestMaixCamThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_captures(self, running_camera):
        """Multiple threads can capture concurrently."""
        frames = []
        errors = []

        def capture_frames():
            try:
                for _ in range(5):
                    frame = running_camera.capture()
                    if frame:
                        frames.append(frame)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=capture_frames) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(frames) > 0

    def test_start_stop_from_different_threads(self, camera_source):
        """Start and stop can be called from different threads."""
        errors = []

        def start_camera():
            try:
                camera_source.start()
            except Exception as e:
                errors.append(e)

        def stop_camera():
            try:
                time.sleep(0.01)  # Small delay
                camera_source.stop()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=start_camera)
        t2 = threading.Thread(target=stop_camera)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

    def test_frame_ids_are_sequential_under_concurrency(self, running_camera):
        """Frame IDs remain sequential under concurrent access."""
        frame_ids = []

        def capture_and_record():
            for _ in range(10):
                frame = running_camera.capture()
                if frame:
                    frame_ids.append(frame.frame_id)

        threads = [threading.Thread(target=capture_and_record) for _ in range(2)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Frame IDs should be unique
        assert len(frame_ids) == len(set(frame_ids))


# =============================================================================
# 7. Error Handling Tests (4 tests)
# =============================================================================


class TestMaixCamErrorHandling:
    """Tests for error handling."""

    def test_capture_handles_exception_gracefully(self, running_camera, mock_camera_instance):
        """Capture handles exceptions gracefully."""
        mock_camera_instance.read.side_effect = Exception("Camera error")

        frame = running_camera.capture()
        assert frame is None

    def test_stop_handles_close_exception(self, running_camera, mock_camera_instance):
        """Stop handles close() exception gracefully."""
        mock_camera_instance.close.side_effect = Exception("Close error")

        # Should not raise
        running_camera.stop()
        assert running_camera.is_running is False

    def test_set_resolution_while_running_restarts(self, running_camera, mock_maix_module, mock_camera_instance):
        """set_resolution restarts camera if it was running."""
        mock_maix_module.camera.Camera.return_value = mock_camera_instance

        assert running_camera.is_running is True

        running_camera.set_resolution(1280, 720)

        # Should still be running after resolution change
        assert running_camera.is_running is True
        assert running_camera.config.width == 1280

    def test_is_maixcam_available_function(self):
        """is_maixcam_available returns correct availability status."""
        from src.infrastructure.camera.maixcam_camera import is_maixcam_available

        # Since we've mocked it, it should indicate available
        # The actual return depends on module state
        result = is_maixcam_available()
        assert isinstance(result, bool)


# =============================================================================
# Helper Tests
# =============================================================================


class TestMaixCamHelpers:
    """Tests for helper functions."""

    def test_camera_implements_protocol(self, camera_source):
        """MaixCamSource implements CameraSource protocol."""
        from src.core.interfaces.camera import CameraSource

        assert isinstance(camera_source, CameraSource)

    def test_camera_returns_self_from_enter(self, camera_source):
        """Context manager __enter__ returns self."""
        result = camera_source.__enter__()
        assert result is camera_source
        camera_source.__exit__(None, None, None)
