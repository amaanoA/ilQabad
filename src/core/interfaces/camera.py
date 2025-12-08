"""Camera interface definitions for face recognition system.

This module defines the camera abstraction layer that allows swapping
webcam (development) for MaixCAM (production).
"""

from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass
class CameraConfig:
    """Configuration for camera capture settings.

    Attributes:
        width: Frame width in pixels. Must be positive.
        height: Frame height in pixels. Must be positive.
        fps: Frames per second. Must be positive.
        device_id: Camera device identifier. Must be non-negative.
    """

    width: int = 640
    height: int = 480
    fps: int = 30
    device_id: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.device_id < 0:
            raise ValueError(f"device_id must be non-negative, got {self.device_id}")


@dataclass
class Frame:
    """A single captured video frame.

    Attributes:
        image: RGB image as numpy array with shape (height, width, 3).
        timestamp: Capture timestamp in seconds since epoch. Must be positive.
        frame_id: Unique frame identifier. Must be positive.
    """

    image: npt.NDArray[np.uint8]
    timestamp: float
    frame_id: int

    def __post_init__(self) -> None:
        """Validate frame data after initialization."""
        if self.image.ndim != 3:
            raise ValueError(
                f"image must be 3-dimensional (height, width, channels), "
                f"got {self.image.ndim} dimensions"
            )
        if self.image.shape[2] != 3:
            raise ValueError(
                f"image must have 3 channels (RGB), got {self.image.shape[2]} channels"
            )
        if self.timestamp <= 0:
            raise ValueError(f"timestamp must be positive, got {self.timestamp}")
        if self.frame_id <= 0:
            raise ValueError(f"frame_id must be positive, got {self.frame_id}")


@runtime_checkable
class CameraSource(Protocol):
    """Protocol for camera sources providing video frames.

    Implementations should support context manager usage for automatic
    resource management (start on enter, stop on exit).

    Example:
        with camera_source as camera:
            frame = camera.capture()
            if frame is not None:
                process(frame.image)
    """

    @property
    def is_running(self) -> bool:
        """Whether the camera is currently capturing frames."""
        ...

    def start(self) -> None:
        """Start the camera and begin capturing frames."""
        ...

    def stop(self) -> None:
        """Stop the camera and release resources."""
        ...

    def capture(self) -> Frame | None:
        """Capture a single frame from the camera.

        Returns:
            Frame if capture was successful, None if camera is not running
            or capture failed.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager, starting the camera."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, stopping the camera."""
        ...
