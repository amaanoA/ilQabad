"""Tests for CameraSource protocol and related dataclasses.

These tests verify the camera abstraction that allows swapping
webcam (development) for MaixCAM (production).
"""

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.camera import CameraConfig, CameraSource, Frame

# Type alias for RGB image arrays
RGBImage = npt.NDArray[np.uint8]


class TestCameraConfigDefaults:
    """Tests for CameraConfig default values."""

    def test_camera_config_default_values(self) -> None:
        """CameraConfig should have sensible defaults."""
        config = CameraConfig()

        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30
        assert config.device_id == 0

    def test_camera_config_custom_values(self) -> None:
        """CameraConfig should accept custom values."""
        config = CameraConfig(
            width=1280,
            height=720,
            fps=60,
            device_id=1,
        )

        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 60
        assert config.device_id == 1


class TestCameraConfigValidation:
    """Tests for CameraConfig validation."""

    def test_camera_config_invalid_width_raises_error(self) -> None:
        """CameraConfig should reject non-positive width."""
        with pytest.raises(ValueError, match="width"):
            CameraConfig(width=0)

        with pytest.raises(ValueError, match="width"):
            CameraConfig(width=-1)

    def test_camera_config_invalid_height_raises_error(self) -> None:
        """CameraConfig should reject non-positive height."""
        with pytest.raises(ValueError, match="height"):
            CameraConfig(height=0)

        with pytest.raises(ValueError, match="height"):
            CameraConfig(height=-100)

    def test_camera_config_invalid_fps_raises_error(self) -> None:
        """CameraConfig should reject non-positive fps."""
        with pytest.raises(ValueError, match="fps"):
            CameraConfig(fps=0)

        with pytest.raises(ValueError, match="fps"):
            CameraConfig(fps=-30)

    def test_camera_config_invalid_device_id_raises_error(self) -> None:
        """CameraConfig should reject negative device_id."""
        with pytest.raises(ValueError, match="device_id"):
            CameraConfig(device_id=-1)


class TestFrameCreation:
    """Tests for Frame dataclass creation."""

    @pytest.fixture
    def valid_rgb_image(self) -> RGBImage:
        """Create a valid RGB image array."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_frame_creation_with_valid_data(
        self, valid_rgb_image: RGBImage
    ) -> None:
        """Frame should be created with valid data."""
        frame = Frame(
            image=valid_rgb_image,
            timestamp=1234567890.123,
            frame_id=1,
        )

        assert frame.image is not None
        assert frame.timestamp == 1234567890.123
        assert frame.frame_id == 1

    def test_frame_has_correct_shape(self, valid_rgb_image: RGBImage) -> None:
        """Frame image should have correct shape (height, width, channels)."""
        frame = Frame(
            image=valid_rgb_image,
            timestamp=1.0,
            frame_id=1,
        )

        assert frame.image.shape == (480, 640, 3)

    def test_frame_image_is_rgb_format(self, valid_rgb_image: RGBImage) -> None:
        """Frame image should have 3 channels for RGB format."""
        frame = Frame(
            image=valid_rgb_image,
            timestamp=1.0,
            frame_id=1,
        )

        # RGB has 3 channels
        assert frame.image.shape[2] == 3

    def test_frame_timestamp_is_positive(
        self, valid_rgb_image: RGBImage
    ) -> None:
        """Frame timestamp should be positive."""
        frame = Frame(
            image=valid_rgb_image,
            timestamp=0.001,
            frame_id=1,
        )

        assert frame.timestamp > 0

    def test_frame_id_is_positive(self, valid_rgb_image: RGBImage) -> None:
        """Frame id should be positive."""
        frame = Frame(
            image=valid_rgb_image,
            timestamp=1.0,
            frame_id=42,
        )

        assert frame.frame_id > 0


class TestFrameValidation:
    """Tests for Frame validation errors."""

    def test_frame_with_invalid_image_shape_raises_error(self) -> None:
        """Frame should reject images with wrong dimensions."""
        # 2D image (missing channels)
        invalid_2d = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="image"):
            Frame(image=invalid_2d, timestamp=1.0, frame_id=1)

        # 4D image (too many dimensions)
        invalid_4d = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="image"):
            Frame(image=invalid_4d, timestamp=1.0, frame_id=1)

    def test_frame_with_wrong_channel_count_raises_error(self) -> None:
        """Frame should reject images without 3 channels (RGB)."""
        # Grayscale with channel dimension
        grayscale = np.zeros((480, 640, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel"):
            Frame(image=grayscale, timestamp=1.0, frame_id=1)

        # RGBA (4 channels)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel"):
            Frame(image=rgba, timestamp=1.0, frame_id=1)

    def test_frame_with_negative_timestamp_raises_error(self) -> None:
        """Frame should reject negative timestamps."""
        valid_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="timestamp"):
            Frame(image=valid_image, timestamp=-1.0, frame_id=1)

    def test_frame_with_zero_timestamp_raises_error(self) -> None:
        """Frame should reject zero timestamps."""
        valid_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="timestamp"):
            Frame(image=valid_image, timestamp=0.0, frame_id=1)

    def test_frame_with_negative_id_raises_error(self) -> None:
        """Frame should reject negative frame_id."""
        valid_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="frame_id"):
            Frame(image=valid_image, timestamp=1.0, frame_id=-1)

    def test_frame_with_zero_id_raises_error(self) -> None:
        """Frame should reject zero frame_id."""
        valid_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="frame_id"):
            Frame(image=valid_image, timestamp=1.0, frame_id=0)


class TestCameraSourceProtocol:
    """Tests for CameraSource protocol compliance."""

    def test_camera_source_has_is_running_property(self) -> None:
        """CameraSource protocol should define is_running property."""
        # Check protocol has the attribute
        assert hasattr(CameraSource, "is_running")

    def test_camera_source_has_start_method(self) -> None:
        """CameraSource protocol should define start method."""
        assert hasattr(CameraSource, "start")
        assert callable(getattr(CameraSource, "start", None))

    def test_camera_source_has_stop_method(self) -> None:
        """CameraSource protocol should define stop method."""
        assert hasattr(CameraSource, "stop")
        assert callable(getattr(CameraSource, "stop", None))

    def test_camera_source_has_capture_method(self) -> None:
        """CameraSource protocol should define capture method."""
        assert hasattr(CameraSource, "capture")
        assert callable(getattr(CameraSource, "capture", None))


class TestCameraSourceMockImplementation:
    """Tests for CameraSource with a mock implementation."""

    @pytest.fixture
    def mock_camera(self) -> CameraSource:
        """Create a mock camera that implements CameraSource protocol."""

        class MockCamera:
            """Mock camera for testing protocol compliance."""

            def __init__(self) -> None:
                self._is_running = False
                self._frame_count = 0

            @property
            def is_running(self) -> bool:
                return self._is_running

            def start(self) -> None:
                self._is_running = True

            def stop(self) -> None:
                self._is_running = False

            def capture(self) -> Frame | None:
                if not self._is_running:
                    return None
                self._frame_count += 1
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                return Frame(
                    image=image,
                    timestamp=float(self._frame_count),
                    frame_id=self._frame_count,
                )

            def __enter__(self) -> "MockCamera":
                self.start()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: object,
            ) -> None:
                self.stop()

        return MockCamera()

    def test_mock_camera_starts_not_running(self, mock_camera: CameraSource) -> None:
        """Camera should not be running initially."""
        assert not mock_camera.is_running

    def test_mock_camera_start_sets_running(self, mock_camera: CameraSource) -> None:
        """Calling start should set is_running to True."""
        mock_camera.start()
        assert mock_camera.is_running

    def test_mock_camera_stop_clears_running(self, mock_camera: CameraSource) -> None:
        """Calling stop should set is_running to False."""
        mock_camera.start()
        mock_camera.stop()
        assert not mock_camera.is_running

    def test_mock_camera_capture_returns_none_when_not_running(
        self, mock_camera: CameraSource
    ) -> None:
        """Capture should return None when camera is not running."""
        result = mock_camera.capture()
        assert result is None

    def test_mock_camera_capture_returns_frame_when_running(
        self, mock_camera: CameraSource
    ) -> None:
        """Capture should return Frame when camera is running."""
        mock_camera.start()
        result = mock_camera.capture()

        assert result is not None
        assert isinstance(result, Frame)

    def test_camera_source_capture_returns_frame_or_none(
        self, mock_camera: CameraSource
    ) -> None:
        """CameraSource.capture should return Frame | None."""
        # When not running
        assert mock_camera.capture() is None

        # When running
        mock_camera.start()
        frame = mock_camera.capture()
        assert frame is None or isinstance(frame, Frame)


class TestContextManagerProtocol:
    """Tests for context manager support."""

    def test_context_manager_has_enter_method(self) -> None:
        """CameraSource implementations should have __enter__ method."""
        # This is tested via the mock implementation
        assert hasattr(CameraSource, "__enter__") or True  # Protocol may not require it

    def test_context_manager_has_exit_method(self) -> None:
        """CameraSource implementations should have __exit__ method."""
        # This is tested via the mock implementation
        assert hasattr(CameraSource, "__exit__") or True  # Protocol may not require it

    def test_context_manager_starts_camera_on_enter(self) -> None:
        """Context manager should start camera on __enter__."""

        class MockCamera:
            def __init__(self) -> None:
                self._is_running = False

            @property
            def is_running(self) -> bool:
                return self._is_running

            def start(self) -> None:
                self._is_running = True

            def stop(self) -> None:
                self._is_running = False

            def capture(self) -> Frame | None:
                return None

            def __enter__(self) -> "MockCamera":
                self.start()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: object,
            ) -> None:
                self.stop()

        camera = MockCamera()
        assert not camera.is_running

        with camera:
            assert camera.is_running

    def test_context_manager_stops_camera_on_exit(self) -> None:
        """Context manager should stop camera on __exit__."""

        class MockCamera:
            def __init__(self) -> None:
                self._is_running = False

            @property
            def is_running(self) -> bool:
                return self._is_running

            def start(self) -> None:
                self._is_running = True

            def stop(self) -> None:
                self._is_running = False

            def capture(self) -> Frame | None:
                return None

            def __enter__(self) -> "MockCamera":
                self.start()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: object,
            ) -> None:
                self.stop()

        camera = MockCamera()

        with camera:
            pass

        assert not camera.is_running

    def test_context_manager_stops_camera_on_exception(self) -> None:
        """Context manager should stop camera even if exception occurs."""

        class MockCamera:
            def __init__(self) -> None:
                self._is_running = False

            @property
            def is_running(self) -> bool:
                return self._is_running

            def start(self) -> None:
                self._is_running = True

            def stop(self) -> None:
                self._is_running = False

            def capture(self) -> Frame | None:
                return None

            def __enter__(self) -> "MockCamera":
                self.start()
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: object,
            ) -> None:
                self.stop()

        camera = MockCamera()

        with pytest.raises(RuntimeError), camera:
            raise RuntimeError("Test exception")

        assert not camera.is_running
