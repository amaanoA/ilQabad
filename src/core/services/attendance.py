"""Attendance pipeline interface definitions.

This module defines the attendance processing abstraction that orchestrates
face detection, liveness checking, and recognition for the attendance system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass
class AttendanceResult:
    """Result of processing a face for attendance.

    Attributes:
        student_id: ID of the recognized student.
        confidence: Recognition confidence score between 0.0 and 1.0.
        is_live: Whether the face passed liveness check.
        timestamp: When the attendance was recorded.
        face_bbox: Bounding box as (x, y, width, height).
    """

    student_id: str
    confidence: float
    is_live: bool
    timestamp: datetime
    face_bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if len(self.face_bbox) != 4:
            raise ValueError(f"face_bbox must have 4 elements, got {len(self.face_bbox)}")


@runtime_checkable
class AttendancePipeline(Protocol):
    """Protocol for attendance processing pipeline.

    The pipeline orchestrates face detection, liveness checking, and
    recognition to process camera frames for attendance recording.

    Process flow:
    1. Detect faces in frame
    2. For each face, check liveness
    3. If live, extract embedding and match against enrolled students
    4. Return attendance results for matched students
    """

    def process_frame(
        self, frame: npt.NDArray[np.uint8]
    ) -> list[AttendanceResult]:
        """Process a camera frame and return attendance results.

        Args:
            frame: RGB image as numpy array with shape (height, width, 3).

        Returns:
            List of AttendanceResult for each recognized live face.
            Empty list if no faces, all spoofs, or no matches.
        """
        ...

    def enroll_student(
        self, student_id: str, photos: list[npt.NDArray[np.uint8]]
    ) -> bool:
        """Enroll a student with their face photos.

        Args:
            student_id: Unique identifier for the student.
            photos: List of RGB face photos for enrollment.

        Returns:
            True if enrollment was successful.

        Raises:
            StudentAlreadyEnrolledError: If student is already enrolled.
            InsufficientImagesError: If not enough valid photos provided.
        """
        ...

    def remove_student(self, student_id: str) -> bool:
        """Remove a student from the enrollment database.

        Args:
            student_id: ID of the student to remove.

        Returns:
            True if student was removed, False if not found.
        """
        ...

    def get_enrolled_count(self) -> int:
        """Get the number of enrolled students.

        Returns:
            Count of students currently enrolled.
        """
        ...
