"""Face detector interface definitions for face recognition system.

This module defines the face detection abstraction that will be
implemented by YuNet for detecting faces in images.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass
class BoundingBox:
    """Bounding box for a detected face.

    Attributes:
        x: X-coordinate of top-left corner. Can be negative for faces at image edge.
        y: Y-coordinate of top-left corner. Can be negative for faces at image edge.
        width: Width of bounding box. Must be positive.
        height: Height of bounding box. Must be positive.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate bounding box dimensions after initialization."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    @property
    def center(self) -> tuple[float, float]:
        """Calculate the center point of the bounding box.

        Returns:
            Tuple of (center_x, center_y) coordinates.
        """
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        return (center_x, center_y)

    @property
    def area(self) -> int:
        """Calculate the area of the bounding box.

        Returns:
            Area in pixels (width * height).
        """
        return self.width * self.height


@dataclass
class Landmarks:
    """Five-point facial landmarks.

    Attributes:
        left_eye: (x, y) coordinates of left eye center.
        right_eye: (x, y) coordinates of right eye center.
        nose: (x, y) coordinates of nose tip.
        mouth_left: (x, y) coordinates of left mouth corner.
        mouth_right: (x, y) coordinates of right mouth corner.
    """

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]


@dataclass
class DetectedFace:
    """A detected face with bounding box, landmarks, and confidence.

    Attributes:
        bounding_box: Bounding box around the detected face.
        landmarks: Optional 5-point facial landmarks.
        confidence: Detection confidence score between 0.0 and 1.0.
    """

    bounding_box: BoundingBox
    landmarks: Landmarks | None
    confidence: float

    def __post_init__(self) -> None:
        """Validate confidence score after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class DetectionResult:
    """Result of face detection containing all detected faces.

    Attributes:
        faces: List of detected faces in the image.
    """

    faces: list[DetectedFace]

    @property
    def has_faces(self) -> bool:
        """Check if any faces were detected.

        Returns:
            True if at least one face was detected.
        """
        return len(self.faces) > 0

    @property
    def face_count(self) -> int:
        """Get the number of detected faces.

        Returns:
            Number of faces detected.
        """
        return len(self.faces)

    @property
    def largest_face(self) -> DetectedFace | None:
        """Get the face with the largest bounding box area.

        Returns:
            DetectedFace with largest area, or None if no faces detected.
        """
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.bounding_box.area)

    @property
    def most_confident_face(self) -> DetectedFace | None:
        """Get the face with the highest confidence score.

        Returns:
            DetectedFace with highest confidence, or None if no faces detected.
        """
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.confidence)


@runtime_checkable
class FaceDetector(Protocol):
    """Protocol for face detection implementations.

    Implementations should detect faces in RGB images and return
    bounding boxes, landmarks, and confidence scores.
    """

    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
        """Detect all faces in an image.

        Args:
            image: RGB image as numpy array with shape (height, width, 3).

        Returns:
            DetectionResult containing all detected faces.
        """
        ...

    def detect_largest(self, image: npt.NDArray[np.uint8]) -> DetectedFace | None:
        """Detect and return only the largest face in an image.

        Args:
            image: RGB image as numpy array with shape (height, width, 3).

        Returns:
            The largest DetectedFace by bounding box area, or None if no faces.
        """
        ...
