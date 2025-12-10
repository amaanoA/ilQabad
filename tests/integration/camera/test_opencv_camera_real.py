"""Integration tests for OpenCVCameraSource with real webcam.

These tests require a physical webcam and are skipped in CI environments.
Run manually with: pytest tests/integration/camera/test_opencv_camera_real.py -v

To run these tests, set the environment variable:
    RUN_WEBCAM_TESTS=1 pytest tests/integration/camera/ -v
"""

import os
import time

import numpy as np
import pytest

from src.core.interfaces.camera import CameraConfig


# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_WEBCAM_TESTS", "0") != "1",
    reason="Requires physical webcam. Set RUN_WEBCAM_TESTS=1 to run.",
)


@pytest.fixture
def opencv_camera():
    """Create OpenCVCameraSource for testing."""
    from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

    camera = OpenCVCameraSource()
    yield camera
    if camera.is_running:
        camera.stop()


class TestOpenCVCameraSourceRealWebcam:
    """Integration tests with real webcam."""

    def test_open_real_webcam(self, opencv_camera) -> None:
        """Test opening real webcam."""
        opencv_camera.start()
        assert opencv_camera.is_running is True
        opencv_camera.stop()

    def test_capture_real_frame(self, opencv_camera) -> None:
        """Test capturing frame from real webcam."""
        opencv_camera.start()

        frame = opencv_camera.capture()

        assert frame is not None
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3  # RGB
        assert frame.image.dtype == np.uint8

        opencv_camera.stop()

    def test_multiple_captures(self, opencv_camera) -> None:
        """Test capturing multiple frames."""
        opencv_camera.start()

        frames = []
        for _ in range(10):
            frame = opencv_camera.capture()
            if frame:
                frames.append(frame)
            time.sleep(0.033)  # ~30fps

        assert len(frames) >= 5  # At least half should succeed

        # Frame IDs should be sequential
        frame_ids = [f.frame_id for f in frames]
        assert frame_ids == sorted(frame_ids)

        opencv_camera.stop()

    def test_context_manager_with_real_webcam(self) -> None:
        """Test context manager with real webcam."""
        from src.infrastructure.camera.opencv_camera import OpenCVCameraSource

        with OpenCVCameraSource() as camera:
            assert camera.is_running is True
            frame = camera.capture()
            assert frame is not None

    def test_resolution_change(self, opencv_camera) -> None:
        """Test changing resolution on real webcam."""
        opencv_camera.start()

        # Try common resolutions
        resolutions = [(640, 480), (1280, 720), (320, 240)]

        for width, height in resolutions:
            result = opencv_camera.set_resolution(width, height)
            # Not all webcams support all resolutions
            if result:
                frame = opencv_camera.capture()
                if frame:
                    # Actual resolution may differ from requested
                    assert frame.image.shape[1] > 0
                    assert frame.image.shape[0] > 0

        opencv_camera.stop()

    def test_frame_rgb_format(self, opencv_camera) -> None:
        """Test that captured frames are in RGB format."""
        opencv_camera.start()

        # Capture a frame
        frame = opencv_camera.capture()
        assert frame is not None

        # The image should be RGB, not BGR
        # We can't easily verify this without a known color target
        # But we can check the shape is correct
        assert frame.image.shape[2] == 3

        opencv_camera.stop()

    def test_timestamp_accuracy(self, opencv_camera) -> None:
        """Test timestamp accuracy."""
        opencv_camera.start()

        before = time.time()
        frame = opencv_camera.capture()
        after = time.time()

        assert frame is not None
        assert before <= frame.timestamp <= after

        opencv_camera.stop()

    def test_continuous_capture_performance(self, opencv_camera) -> None:
        """Test continuous capture maintains reasonable framerate."""
        opencv_camera.start()

        start_time = time.time()
        frame_count = 0

        # Capture for 2 seconds
        while time.time() - start_time < 2.0:
            frame = opencv_camera.capture()
            if frame:
                frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        opencv_camera.stop()

        # Should achieve at least 10 FPS on most webcams
        assert fps >= 10, f"Only achieved {fps:.1f} FPS"

    def test_stop_and_restart(self, opencv_camera) -> None:
        """Test stopping and restarting camera."""
        opencv_camera.start()
        frame1 = opencv_camera.capture()
        assert frame1 is not None

        opencv_camera.stop()
        assert opencv_camera.is_running is False

        opencv_camera.start()
        assert opencv_camera.is_running is True

        frame2 = opencv_camera.capture()
        assert frame2 is not None

        opencv_camera.stop()
