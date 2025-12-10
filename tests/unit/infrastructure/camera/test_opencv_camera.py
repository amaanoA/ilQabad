"""Tests for OpenCVCameraSource implementation.

This module contains comprehensive tests for the OpenCV-based camera source
implementation that captures frames from webcams.
"""

import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.camera import CameraConfig, CameraSource, Frame

if TYPE_CHECKING:
    from src.infrastructure.camera.opencv_camera import OpenCVCameraSource


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> CameraConfig:
    """Create default camera configuration."""
    return CameraConfig(width=640, height=480, fps=30, device_id=0)


@pytest.fixture
def custom_config() -> CameraConfig:
    """Create custom camera configuration."""
    return CameraConfig(width=1280, height=720, fps=60, device_id=1)


@pytest.fixture
def mock_video_capture() -> MagicMock:
    """Create a mock cv2.VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.side_effect = lambda prop: {
        3: 640.0,  # cv2.CAP_PROP_FRAME_WIDTH
        4: 480.0,  # cv2.CAP_PROP_FRAME_HEIGHT
        5: 30.0,   # cv2.CAP_PROP_FPS
    }.get(prop, 0.0)
    mock_cap.set.return_value = True
    return mock_cap


@pytest.fixture
def camera_with_mock(
    default_config: CameraConfig, mock_video_capture: MagicMock
) -> "OpenCVCameraSource":
    """Create OpenCVCameraSource with mocked VideoCapture."""
    with patch("cv2.VideoCapture", return_value=mock_video_capture):
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource
        camera = OpenCVCameraSource(config=default_config)
        yield camera
        if camera.is_running:
            camera.stop()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestOpenCVCameraSourceInitialization:
    """Tests for OpenCVCameraSource initialization."""

    def test_initializes_with_default_config(self) -> None:
        """Test camera initializes with default configuration."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource()
        assert camera.config.device_id == 0
        assert camera.config.width == 640
        assert camera.config.height == 480

    def test_initializes_with_custom_config(
        self, custom_config: CameraConfig
    ) -> None:
        """Test camera initializes with custom configuration."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource(config=custom_config)
        assert camera.config.device_id == 1
        assert camera.config.width == 1280
        assert camera.config.height == 720

    def test_initializes_not_running(self) -> None:
        """Test camera starts in not-running state."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource()
        assert camera.is_running is False

    def test_implements_camera_source_protocol(
        self, camera_with_mock: "OpenCVCameraSource"
    ) -> None:
        """Test implementation satisfies CameraSource protocol."""
        assert isinstance(camera_with_mock, CameraSource)


# =============================================================================
# Start/Stop Tests
# =============================================================================


class TestOpenCVCameraSourceStartStop:
    """Tests for camera start and stop functionality."""

    def test_start_opens_camera(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test start() opens the camera."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            assert camera.is_running is True
            camera.stop()

    def test_start_sets_resolution(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test start() configures resolution."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            # Verify set was called for width and height
            assert mock_video_capture.set.called
            camera.stop()

    def test_start_failure_when_camera_unavailable(self) -> None:
        """Test start() handles unavailable camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            # Should not be running if camera failed to open
            assert camera.is_running is False

    def test_stop_releases_camera(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test stop() releases camera resources."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()
            camera.stop()

            mock_video_capture.release.assert_called_once()
            assert camera.is_running is False

    def test_double_start_is_safe(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test calling start() twice is safe."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()
            camera.start()  # Should not raise

            assert camera.is_running is True
            camera.stop()

    def test_double_stop_is_safe(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test calling stop() twice is safe."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()
            camera.stop()
            camera.stop()  # Should not raise

            assert camera.is_running is False


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestOpenCVCameraSourceContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_starts_camera(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test entering context starts camera."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()

            with camera:
                assert camera.is_running is True

    def test_context_manager_stops_camera(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test exiting context stops camera."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()

            with camera:
                pass

            assert camera.is_running is False

    def test_context_manager_stops_on_exception(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test camera stops even if exception occurs."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()

            with pytest.raises(ValueError):
                with camera:
                    raise ValueError("Test exception")

            assert camera.is_running is False


# =============================================================================
# Capture Tests
# =============================================================================


class TestOpenCVCameraSourceCapture:
    """Tests for frame capture functionality."""

    def test_capture_returns_frame(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test capture returns Frame object."""
        # Setup mock to return BGR image (OpenCV default)
        bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 255  # Blue channel
        mock_video_capture.read.return_value = (True, bgr_frame)

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame = camera.capture()

            assert frame is not None
            assert isinstance(frame, Frame)
            camera.stop()

    def test_capture_returns_none_when_not_running(self) -> None:
        """Test capture returns None when camera not started."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource()
        frame = camera.capture()

        assert frame is None

    def test_capture_converts_bgr_to_rgb(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test captured frame is converted from BGR to RGB."""
        # Create BGR image with distinct colors
        bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 100  # Blue in BGR
        bgr_frame[:, :, 1] = 150  # Green
        bgr_frame[:, :, 2] = 200  # Red in BGR
        mock_video_capture.read.return_value = (True, bgr_frame)

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame = camera.capture()

            # In RGB, red should be first channel
            assert frame is not None
            assert frame.image[0, 0, 0] == 200  # Red
            assert frame.image[0, 0, 1] == 150  # Green
            assert frame.image[0, 0, 2] == 100  # Blue
            camera.stop()

    def test_capture_returns_none_on_read_failure(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test capture returns None when read fails."""
        mock_video_capture.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame = camera.capture()

            assert frame is None
            camera.stop()

    def test_capture_includes_timestamp(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test captured frame includes timestamp."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            before = time.time()
            frame = camera.capture()
            after = time.time()

            assert frame is not None
            assert before <= frame.timestamp <= after
            camera.stop()

    def test_capture_increments_frame_id(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test frame_id increments with each capture."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame1 = camera.capture()
            frame2 = camera.capture()
            frame3 = camera.capture()

            assert frame1 is not None
            assert frame2 is not None
            assert frame3 is not None
            assert frame1.frame_id == 1
            assert frame2.frame_id == 2
            assert frame3.frame_id == 3
            camera.stop()

    def test_capture_frame_has_correct_shape(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test captured frame has correct shape."""
        mock_video_capture.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame = camera.capture()

            assert frame is not None
            assert frame.image.shape == (480, 640, 3)
            assert frame.image.dtype == np.uint8
            camera.stop()


# =============================================================================
# Resolution Tests
# =============================================================================


class TestOpenCVCameraSourceResolution:
    """Tests for resolution handling."""

    def test_get_resolution_returns_config(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test get_resolution returns configured values."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            config = CameraConfig(width=1280, height=720)
            camera = OpenCVCameraSource(config=config)

            width, height = camera.get_resolution()

            assert width == 1280
            assert height == 720

    def test_set_resolution_updates_config(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test set_resolution updates camera configuration."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            result = camera.set_resolution(1920, 1080)

            assert result is True
            assert camera.config.width == 1920
            assert camera.config.height == 1080
            camera.stop()

    def test_set_resolution_applies_to_capture(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test set_resolution calls VideoCapture.set()."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            camera.set_resolution(1920, 1080)

            # Verify set was called
            assert mock_video_capture.set.called
            camera.stop()

    def test_set_resolution_invalid_dimensions(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test set_resolution rejects invalid dimensions."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            with pytest.raises(ValueError):
                camera.set_resolution(0, 480)

            with pytest.raises(ValueError):
                camera.set_resolution(640, -1)

            camera.stop()

    def test_set_resolution_when_not_running(self) -> None:
        """Test set_resolution when camera not started."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource()

        # Should still update config even when not running
        result = camera.set_resolution(1920, 1080)

        assert result is True
        assert camera.config.width == 1920


# =============================================================================
# Properties Tests
# =============================================================================


class TestOpenCVCameraSourceProperties:
    """Tests for camera properties."""

    def test_is_running_false_initially(self) -> None:
        """Test is_running is False initially."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        camera = OpenCVCameraSource()
        assert camera.is_running is False

    def test_is_running_true_after_start(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test is_running is True after start."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            assert camera.is_running is True
            camera.stop()

    def test_device_id_from_config(self) -> None:
        """Test device_id matches configuration."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        config = CameraConfig(device_id=2)
        camera = OpenCVCameraSource(config=config)

        assert camera.config.device_id == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestOpenCVCameraSourceErrorHandling:
    """Tests for error handling."""

    def test_handles_device_not_found(self) -> None:
        """Test graceful handling when device not found."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            # Should handle gracefully
            assert camera.is_running is False
            frame = camera.capture()
            assert frame is None

    def test_handles_capture_exception(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test handling of exception during capture."""
        mock_video_capture.read.side_effect = RuntimeError("Capture failed")

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            # Should return None on exception, not raise
            frame = camera.capture()
            assert frame is None
            camera.stop()

    def test_handles_release_exception(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test handling of exception during release."""
        mock_video_capture.release.side_effect = RuntimeError("Release failed")

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            # Should not raise
            camera.stop()
            assert camera.is_running is False

    def test_capture_timeout_returns_none(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test capture returns None on timeout/slow read."""
        # Simulate slow read by returning immediately with failure
        mock_video_capture.read.return_value = (False, None)

        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame = camera.capture()

            assert frame is None
            camera.stop()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestOpenCVCameraSourceThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_captures(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test concurrent capture calls are safe."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            results = []
            errors = []

            def capture_frames():
                try:
                    for _ in range(10):
                        frame = camera.capture()
                        results.append(frame)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=capture_frames) for _ in range(3)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            camera.stop()

            assert len(errors) == 0
            assert len(results) == 30

    def test_start_stop_thread_safety(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test start/stop from multiple threads is safe."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            errors = []

            def toggle_camera():
                try:
                    for _ in range(5):
                        camera.start()
                        camera.stop()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=toggle_camera) for _ in range(2)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

    def test_frame_id_unique_across_threads(
        self, mock_video_capture: MagicMock
    ) -> None:
        """Test frame IDs are unique even with concurrent captures."""
        with patch("cv2.VideoCapture", return_value=mock_video_capture):
            from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

            camera = OpenCVCameraSource()
            camera.start()

            frame_ids = []
            lock = threading.Lock()

            def capture_and_store():
                for _ in range(10):
                    frame = camera.capture()
                    if frame:
                        with lock:
                            frame_ids.append(frame.frame_id)

            threads = [threading.Thread(target=capture_and_store) for _ in range(3)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            camera.stop()

            # All frame IDs should be unique
            assert len(frame_ids) == len(set(frame_ids))
