"""OpenCV-based camera source implementation.

This module provides a camera source implementation using OpenCV's VideoCapture
for capturing frames from webcams.
"""

import threading
import time
from typing import Self

import cv2
import numpy as np
import numpy.typing as npt

from src.core.interfaces.camera import CameraConfig, Frame


class OpenCVCameraSource:
    """OpenCV-based camera source for webcam capture.

    This implementation uses cv2.VideoCapture to capture frames from
    webcams or video devices. It converts BGR frames to RGB and provides
    thread-safe capture operations.

    Attributes:
        config: Camera configuration settings.

    Example:
        >>> camera = OpenCVCameraSource()
        >>> with camera:
        ...     frame = camera.capture()
        ...     if frame:
        ...         process(frame.image)
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        """Initialize the camera source.

        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None
        self._is_running = False
        self._frame_counter = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the camera is currently capturing frames."""
        return self._is_running

    def start(self) -> None:
        """Start the camera and begin capturing frames.

        Opens the video capture device and configures resolution.
        If the camera is already running, this is a no-op.
        """
        with self._lock:
            if self._is_running:
                return

            self._capture = cv2.VideoCapture(self.config.device_id)

            if not self._capture.isOpened():
                self._is_running = False
                return

            # Configure resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

            self._is_running = True
            self._frame_counter = 0

    def stop(self) -> None:
        """Stop the camera and release resources.

        Releases the video capture device. If the camera is not running,
        this is a no-op.
        """
        with self._lock:
            if not self._is_running:
                return

            self._is_running = False

            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass  # Ignore release errors
                self._capture = None

    def capture(self) -> Frame | None:
        """Capture a single frame from the camera.

        Captures a frame, converts it from BGR to RGB, and returns it
        wrapped in a Frame dataclass.

        Returns:
            Frame if capture was successful, None if camera is not running
            or capture failed.
        """
        with self._lock:
            if not self._is_running or self._capture is None:
                return None

            try:
                ret, bgr_frame = self._capture.read()

                if not ret or bgr_frame is None:
                    return None

                # Convert BGR (OpenCV default) to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                # Increment frame counter
                self._frame_counter += 1

                return Frame(
                    image=rgb_frame,
                    timestamp=time.time(),
                    frame_id=self._frame_counter,
                )

            except Exception:
                return None

    def get_resolution(self) -> tuple[int, int]:
        """Get the current resolution setting.

        Returns:
            Tuple of (width, height) from configuration.
        """
        return (self.config.width, self.config.height)

    def set_resolution(self, width: int, height: int) -> bool:
        """Set the camera resolution.

        Args:
            width: New frame width in pixels.
            height: New frame height in pixels.

        Returns:
            True if resolution was set successfully.

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        with self._lock:
            # Update config
            self.config = CameraConfig(
                width=width,
                height=height,
                fps=self.config.fps,
                device_id=self.config.device_id,
            )

            # Apply to capture if running
            if self._capture is not None and self._is_running:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return True

    def __enter__(self) -> Self:
        """Enter context manager, starting the camera."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, stopping the camera."""
        self.stop()
