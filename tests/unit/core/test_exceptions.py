"""Tests for custom exception hierarchy.

These tests verify the exception classes used for consistent
error handling throughout the ilQabad system.
"""

from datetime import datetime

import pytest

from src.core.exceptions import (
    AttendanceError,
    CameraError,
    CameraNotFoundError,
    CameraPermissionError,
    DetectionError,
    DuplicateScanError,
    EnrollmentError,
    IlQabadError,
    InsufficientImagesError,
    LivenessError,
    LowConfidenceError,
    MultipleFacesError,
    NetworkUnavailableError,
    NoFaceDetectedError,
    NoMatchFoundError,
    RecognitionError,
    SpoofDetectedError,
    StudentAlreadyEnrolledError,
    StudentNotEnrolledError,
    SyncError,
)
from src.core.interfaces.liveness import SpoofType


class TestBaseException:
    """Tests for IlQabadError base exception."""

    def test_ilqabad_error_is_exception_subclass(self) -> None:
        """IlQabadError should inherit from Exception."""
        assert issubclass(IlQabadError, Exception)

    def test_ilqabad_error_can_be_instantiated_with_message(self) -> None:
        """IlQabadError should accept a message."""
        error = IlQabadError("Test error message")
        assert str(error) == "Test error message"


class TestInheritanceHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_camera_error_inherits_from_ilqabad_error(self) -> None:
        """CameraError should inherit from IlQabadError."""
        assert issubclass(CameraError, IlQabadError)

    def test_camera_not_found_error_inherits_from_camera_error(self) -> None:
        """CameraNotFoundError should inherit from CameraError."""
        assert issubclass(CameraNotFoundError, CameraError)

    def test_camera_permission_error_inherits_from_camera_error(self) -> None:
        """CameraPermissionError should inherit from CameraError."""
        assert issubclass(CameraPermissionError, CameraError)

    def test_detection_error_inherits_from_ilqabad_error(self) -> None:
        """DetectionError should inherit from IlQabadError."""
        assert issubclass(DetectionError, IlQabadError)

    def test_no_face_detected_error_inherits_from_detection_error(self) -> None:
        """NoFaceDetectedError should inherit from DetectionError."""
        assert issubclass(NoFaceDetectedError, DetectionError)

    def test_multiple_faces_error_inherits_from_detection_error(self) -> None:
        """MultipleFacesError should inherit from DetectionError."""
        assert issubclass(MultipleFacesError, DetectionError)

    def test_recognition_error_inherits_from_ilqabad_error(self) -> None:
        """RecognitionError should inherit from IlQabadError."""
        assert issubclass(RecognitionError, IlQabadError)

    def test_no_match_found_error_inherits_from_recognition_error(self) -> None:
        """NoMatchFoundError should inherit from RecognitionError."""
        assert issubclass(NoMatchFoundError, RecognitionError)

    def test_low_confidence_error_inherits_from_recognition_error(self) -> None:
        """LowConfidenceError should inherit from RecognitionError."""
        assert issubclass(LowConfidenceError, RecognitionError)

    def test_liveness_error_inherits_from_ilqabad_error(self) -> None:
        """LivenessError should inherit from IlQabadError."""
        assert issubclass(LivenessError, IlQabadError)

    def test_spoof_detected_error_inherits_from_liveness_error(self) -> None:
        """SpoofDetectedError should inherit from LivenessError."""
        assert issubclass(SpoofDetectedError, LivenessError)

    def test_enrollment_error_inherits_from_ilqabad_error(self) -> None:
        """EnrollmentError should inherit from IlQabadError."""
        assert issubclass(EnrollmentError, IlQabadError)

    def test_attendance_error_inherits_from_ilqabad_error(self) -> None:
        """AttendanceError should inherit from IlQabadError."""
        assert issubclass(AttendanceError, IlQabadError)

    def test_sync_error_inherits_from_ilqabad_error(self) -> None:
        """SyncError should inherit from IlQabadError."""
        assert issubclass(SyncError, IlQabadError)


class TestExceptionAttributes:
    """Tests for exception-specific attributes."""

    def test_camera_not_found_error_has_device_id(self) -> None:
        """CameraNotFoundError should have device_id attribute."""
        error = CameraNotFoundError(device_id=0)
        assert error.device_id == 0

    def test_camera_permission_error_has_device_id(self) -> None:
        """CameraPermissionError should have device_id attribute."""
        error = CameraPermissionError(device_id=1)
        assert error.device_id == 1

    def test_multiple_faces_error_has_face_count(self) -> None:
        """MultipleFacesError should have face_count attribute."""
        error = MultipleFacesError(face_count=3)
        assert error.face_count == 3

    def test_no_match_found_error_has_confidence(self) -> None:
        """NoMatchFoundError should have confidence attribute."""
        error = NoMatchFoundError(confidence=0.45)
        assert error.confidence == 0.45

    def test_low_confidence_error_has_confidence_and_threshold(self) -> None:
        """LowConfidenceError should have confidence and threshold attributes."""
        error = LowConfidenceError(confidence=0.55, threshold=0.6)
        assert error.confidence == 0.55
        assert error.threshold == 0.6

    def test_spoof_detected_error_has_spoof_type(self) -> None:
        """SpoofDetectedError should have spoof_type attribute."""
        error = SpoofDetectedError(spoof_type=SpoofType.PHOTO)
        assert error.spoof_type == SpoofType.PHOTO

    def test_insufficient_images_error_has_provided_and_required(self) -> None:
        """InsufficientImagesError should have provided and required attributes."""
        error = InsufficientImagesError(provided=2, required=5)
        assert error.provided == 2
        assert error.required == 5

    def test_duplicate_scan_error_has_student_id_and_scan_time(self) -> None:
        """DuplicateScanError should have student_id and scan_time attributes."""
        scan_time = datetime(2024, 1, 15, 8, 30, 0)
        error = DuplicateScanError(student_id="STU001", scan_time=scan_time)
        assert error.student_id == "STU001"
        assert error.scan_time == scan_time

    def test_student_already_enrolled_error_has_student_id(self) -> None:
        """StudentAlreadyEnrolledError should have student_id attribute."""
        error = StudentAlreadyEnrolledError(student_id="STU002")
        assert error.student_id == "STU002"

    def test_student_not_enrolled_error_has_student_id(self) -> None:
        """StudentNotEnrolledError should have student_id attribute."""
        error = StudentNotEnrolledError(student_id="STU003")
        assert error.student_id == "STU003"

    def test_network_unavailable_error_has_url(self) -> None:
        """NetworkUnavailableError should have url attribute."""
        error = NetworkUnavailableError(url="https://api.example.com")
        assert error.url == "https://api.example.com"


class TestExceptionMessages:
    """Tests for exception message formatting."""

    def test_camera_not_found_error_message_includes_device_id(self) -> None:
        """CameraNotFoundError message should include device_id."""
        error = CameraNotFoundError(device_id=2)
        assert "2" in str(error)

    def test_multiple_faces_error_message_includes_face_count(self) -> None:
        """MultipleFacesError message should include face_count."""
        error = MultipleFacesError(face_count=5)
        assert "5" in str(error)

    def test_low_confidence_error_message_includes_values(self) -> None:
        """LowConfidenceError message should include confidence and threshold."""
        error = LowConfidenceError(confidence=0.52, threshold=0.6)
        message = str(error)
        assert "0.52" in message or "52" in message
        assert "0.6" in message or "60" in message

    def test_spoof_detected_error_message_includes_spoof_type(self) -> None:
        """SpoofDetectedError message should include spoof type."""
        error = SpoofDetectedError(spoof_type=SpoofType.VIDEO)
        message = str(error).lower()
        assert "video" in message

    def test_insufficient_images_error_message_includes_counts(self) -> None:
        """InsufficientImagesError message should include provided and required."""
        error = InsufficientImagesError(provided=3, required=10)
        message = str(error)
        assert "3" in message
        assert "10" in message

    def test_duplicate_scan_error_message_includes_details(self) -> None:
        """DuplicateScanError message should include student_id."""
        scan_time = datetime(2024, 1, 15, 8, 30, 0)
        error = DuplicateScanError(student_id="STU001", scan_time=scan_time)
        message = str(error)
        assert "STU001" in message


class TestCatchingBehavior:
    """Tests for exception catching behavior."""

    def test_can_catch_specific_exception(self) -> None:
        """Should be able to catch specific exception type."""
        with pytest.raises(CameraNotFoundError):
            raise CameraNotFoundError(device_id=0)

    def test_can_catch_by_immediate_parent(self) -> None:
        """Should be able to catch exception by immediate parent class."""
        with pytest.raises(CameraError):
            raise CameraNotFoundError(device_id=0)

    def test_can_catch_by_base_ilqabad_error(self) -> None:
        """Should be able to catch any exception by IlQabadError."""
        with pytest.raises(IlQabadError):
            raise CameraNotFoundError(device_id=0)

        with pytest.raises(IlQabadError):
            raise MultipleFacesError(face_count=2)

        with pytest.raises(IlQabadError):
            raise SpoofDetectedError(spoof_type=SpoofType.MASK)

    def test_camera_errors_caught_by_camera_error(self) -> None:
        """All camera-related exceptions should be caught by CameraError."""
        caught_count = 0

        try:
            raise CameraNotFoundError(device_id=0)
        except CameraError:
            caught_count += 1

        try:
            raise CameraPermissionError(device_id=1)
        except CameraError:
            caught_count += 1

        assert caught_count == 2

    def test_detection_errors_caught_by_detection_error(self) -> None:
        """All detection-related exceptions should be caught by DetectionError."""
        caught_count = 0

        try:
            raise NoFaceDetectedError()
        except DetectionError:
            caught_count += 1

        try:
            raise MultipleFacesError(face_count=3)
        except DetectionError:
            caught_count += 1

        assert caught_count == 2
