"""Custom exception hierarchy for ilQabad system.

This module defines all custom exceptions used throughout the ilQabad
face recognition attendance system for consistent error handling.
"""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.interfaces.liveness import SpoofType


class IlQabadError(Exception):
    """Base exception for all ilQabad application errors.

    All custom exceptions in the ilQabad system inherit from this class,
    allowing for catch-all error handling when needed.
    """


# =============================================================================
# Camera Errors
# =============================================================================


class CameraError(IlQabadError):
    """Base exception for camera-related errors."""


class CameraNotFoundError(CameraError):
    """Raised when a camera device cannot be found.

    Attributes:
        device_id: The ID of the camera that was not found.
    """

    def __init__(self, device_id: int) -> None:
        """Initialize with device ID.

        Args:
            device_id: The camera device ID that was not found.
        """
        self.device_id = device_id
        super().__init__(f"Camera not found: device_id={device_id}")


class CameraPermissionError(CameraError):
    """Raised when permission to access camera is denied.

    Attributes:
        device_id: The ID of the camera with permission issues.
    """

    def __init__(self, device_id: int) -> None:
        """Initialize with device ID.

        Args:
            device_id: The camera device ID with permission issues.
        """
        self.device_id = device_id
        super().__init__(f"Camera permission denied: device_id={device_id}")


# =============================================================================
# Detection Errors
# =============================================================================


class DetectionError(IlQabadError):
    """Base exception for face detection errors."""


class NoFaceDetectedError(DetectionError):
    """Raised when no face is detected in an image."""

    def __init__(self) -> None:
        """Initialize with default message."""
        super().__init__("No face detected in image")


class MultipleFacesError(DetectionError):
    """Raised when multiple faces are detected but only one is expected.

    Attributes:
        face_count: The number of faces that were detected.
    """

    def __init__(self, face_count: int) -> None:
        """Initialize with face count.

        Args:
            face_count: The number of faces detected.
        """
        self.face_count = face_count
        super().__init__(f"Multiple faces detected: {face_count} faces found")


# =============================================================================
# Recognition Errors
# =============================================================================


class RecognitionError(IlQabadError):
    """Base exception for face recognition errors."""


class NoMatchFoundError(RecognitionError):
    """Raised when no matching student is found for a face.

    Attributes:
        confidence: The highest confidence score achieved (if any).
    """

    def __init__(self, confidence: float) -> None:
        """Initialize with confidence score.

        Args:
            confidence: The highest similarity score found.
        """
        self.confidence = confidence
        super().__init__(f"No matching student found: best confidence={confidence}")


class LowConfidenceError(RecognitionError):
    """Raised when match confidence is below threshold.

    Attributes:
        confidence: The achieved confidence score.
        threshold: The required minimum threshold.
    """

    def __init__(self, confidence: float, threshold: float) -> None:
        """Initialize with confidence and threshold.

        Args:
            confidence: The achieved confidence score.
            threshold: The required minimum threshold.
        """
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Match confidence too low: {confidence} < {threshold} threshold"
        )


# =============================================================================
# Liveness Errors
# =============================================================================


class LivenessError(IlQabadError):
    """Base exception for liveness detection errors."""


class SpoofDetectedError(LivenessError):
    """Raised when a spoof attack is detected.

    Attributes:
        spoof_type: The type of spoof attack detected.
    """

    def __init__(self, spoof_type: "SpoofType") -> None:
        """Initialize with spoof type.

        Args:
            spoof_type: The detected spoof attack type.
        """
        self.spoof_type = spoof_type
        super().__init__(f"Spoof attack detected: {spoof_type.name.lower()}")


# =============================================================================
# Enrollment Errors
# =============================================================================


class EnrollmentError(IlQabadError):
    """Base exception for student enrollment errors."""


class InsufficientImagesError(EnrollmentError):
    """Raised when not enough images are provided for enrollment.

    Attributes:
        provided: Number of images provided.
        required: Number of images required.
    """

    def __init__(self, provided: int, required: int) -> None:
        """Initialize with image counts.

        Args:
            provided: Number of images provided.
            required: Number of images required.
        """
        self.provided = provided
        self.required = required
        super().__init__(
            f"Insufficient images for enrollment: {provided} provided, {required} required"
        )


class StudentAlreadyEnrolledError(EnrollmentError):
    """Raised when attempting to enroll an already enrolled student.

    Attributes:
        student_id: The ID of the already enrolled student.
    """

    def __init__(self, student_id: str) -> None:
        """Initialize with student ID.

        Args:
            student_id: The ID of the student already enrolled.
        """
        self.student_id = student_id
        super().__init__(f"Student already enrolled: {student_id}")


# =============================================================================
# Attendance Errors
# =============================================================================


class AttendanceError(IlQabadError):
    """Base exception for attendance recording errors."""


class DuplicateScanError(AttendanceError):
    """Raised when a student scans attendance multiple times in a period.

    Attributes:
        student_id: The ID of the student who scanned.
        scan_time: The time of the previous scan.
    """

    def __init__(self, student_id: str, scan_time: datetime) -> None:
        """Initialize with student ID and scan time.

        Args:
            student_id: The ID of the student.
            scan_time: The time of the previous scan.
        """
        self.student_id = student_id
        self.scan_time = scan_time
        super().__init__(
            f"Duplicate scan: student {student_id} already scanned at {scan_time}"
        )


class StudentNotEnrolledError(AttendanceError):
    """Raised when an unenrolled student attempts to record attendance.

    Attributes:
        student_id: The ID of the unenrolled student.
    """

    def __init__(self, student_id: str) -> None:
        """Initialize with student ID.

        Args:
            student_id: The ID of the unenrolled student.
        """
        self.student_id = student_id
        super().__init__(f"Student not enrolled: {student_id}")


# =============================================================================
# Sync Errors
# =============================================================================


class SyncError(IlQabadError):
    """Base exception for data synchronization errors."""


class NetworkUnavailableError(SyncError):
    """Raised when network is unavailable for sync operations.

    Attributes:
        url: The URL that was unreachable.
    """

    def __init__(self, url: str) -> None:
        """Initialize with URL.

        Args:
            url: The URL that could not be reached.
        """
        self.url = url
        super().__init__(f"Network unavailable: cannot reach {url}")


# =============================================================================
# Storage Errors (kept for backwards compatibility)
# =============================================================================


class StorageError(IlQabadError):
    """Base exception for storage operation errors."""


class ConfigurationError(IlQabadError):
    """Base exception for configuration errors."""
