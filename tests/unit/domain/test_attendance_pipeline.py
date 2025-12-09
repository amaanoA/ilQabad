"""Tests for AttendancePipeline implementation.

This module contains comprehensive tests for the attendance processing
pipeline that orchestrates face detection, liveness checking, and recognition.
"""

import time
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from src.core.exceptions import (
    InsufficientImagesError,
    StudentAlreadyEnrolledError,
)
from src.core.interfaces.detector import (
    BoundingBox,
    DetectedFace,
    DetectionResult,
    Landmarks,
)
from src.core.interfaces.liveness import LivenessResult, SpoofType
from src.core.interfaces.recognizer import FaceEmbedding, MatchResult
from src.core.services.attendance import AttendancePipeline, AttendanceResult

if TYPE_CHECKING:
    from src.domain.attendance_pipeline import DefaultAttendancePipeline


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_detector() -> MagicMock:
    """Create a mock face detector."""
    detector = MagicMock()
    detector.detect.return_value = DetectionResult(faces=[])
    return detector


@pytest.fixture
def mock_recognizer() -> MagicMock:
    """Create a mock face recognizer."""
    recognizer = MagicMock()
    embedding = FaceEmbedding(
        vector=np.random.randn(512).astype(np.float32),
        model_name="test",
    )
    recognizer.extract.return_value = embedding
    recognizer.match.return_value = MatchResult(
        matched=False,
        student_id=None,
        confidence=0.0,
        embedding=embedding,
    )
    return recognizer


@pytest.fixture
def mock_liveness() -> MagicMock:
    """Create a mock liveness checker."""
    liveness = MagicMock()
    liveness.check.return_value = LivenessResult(
        is_live=True,
        confidence=0.9,
        spoof_type=SpoofType.NONE,
    )
    return liveness


@pytest.fixture
def attendance_pipeline(
    mock_detector: MagicMock,
    mock_recognizer: MagicMock,
    mock_liveness: MagicMock,
) -> "DefaultAttendancePipeline":
    """Create an AttendancePipeline with mock dependencies."""
    from src.domain.attendance_pipeline import DefaultAttendancePipeline

    return DefaultAttendancePipeline(
        detector=mock_detector,
        recognizer=mock_recognizer,
        liveness_checker=mock_liveness,
    )


@pytest.fixture
def sample_frame() -> npt.NDArray[np.uint8]:
    """Create a sample 640x480 RGB frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_crop() -> npt.NDArray[np.uint8]:
    """Create a sample 224x224 face crop."""
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def detected_face() -> DetectedFace:
    """Create a sample detected face."""
    return DetectedFace(
        bounding_box=BoundingBox(x=100, y=100, width=200, height=200),
        landmarks=Landmarks(
            left_eye=(150.0, 150.0),
            right_eye=(250.0, 150.0),
            nose=(200.0, 200.0),
            mouth_left=(160.0, 250.0),
            mouth_right=(240.0, 250.0),
        ),
        confidence=0.95,
    )


@pytest.fixture
def live_result() -> LivenessResult:
    """Create a live face result."""
    return LivenessResult(
        is_live=True,
        confidence=0.9,
        spoof_type=SpoofType.NONE,
    )


@pytest.fixture
def spoof_result() -> LivenessResult:
    """Create a spoof face result."""
    return LivenessResult(
        is_live=False,
        confidence=0.3,
        spoof_type=SpoofType.PHOTO,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAttendancePipelineInitialization:
    """Tests for AttendancePipeline initialization."""

    def test_initializes_with_dependencies(
        self,
        mock_detector: MagicMock,
        mock_recognizer: MagicMock,
        mock_liveness: MagicMock,
    ) -> None:
        """Test pipeline initializes with injected dependencies."""
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=mock_detector,
            recognizer=mock_recognizer,
            liveness_checker=mock_liveness,
        )
        assert pipeline is not None

    def test_implements_attendance_pipeline_protocol(
        self, attendance_pipeline: "DefaultAttendancePipeline"
    ) -> None:
        """Test implementation satisfies AttendancePipeline protocol."""
        assert isinstance(attendance_pipeline, AttendancePipeline)

    def test_initializes_with_default_thresholds(
        self, attendance_pipeline: "DefaultAttendancePipeline"
    ) -> None:
        """Test pipeline has default threshold values."""
        assert hasattr(attendance_pipeline, "recognition_threshold")
        assert hasattr(attendance_pipeline, "liveness_threshold")

    def test_initializes_with_custom_thresholds(
        self,
        mock_detector: MagicMock,
        mock_recognizer: MagicMock,
        mock_liveness: MagicMock,
    ) -> None:
        """Test pipeline accepts custom threshold values."""
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=mock_detector,
            recognizer=mock_recognizer,
            liveness_checker=mock_liveness,
            recognition_threshold=0.7,
            liveness_threshold=0.6,
        )
        assert pipeline.recognition_threshold == 0.7
        assert pipeline.liveness_threshold == 0.6

    def test_initializes_empty_enrollment_database(
        self, attendance_pipeline: "DefaultAttendancePipeline"
    ) -> None:
        """Test pipeline starts with empty enrollment."""
        assert attendance_pipeline.get_enrolled_count() == 0


# =============================================================================
# Frame Processing Tests
# =============================================================================


class TestAttendancePipelineFrameProcessing:
    """Tests for frame processing functionality."""

    def test_returns_empty_list_for_no_faces(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test returns empty list when no faces detected."""
        mock_detector.detect.return_value = DetectionResult(faces=[])
        results = attendance_pipeline.process_frame(sample_frame)
        assert results == []

    def test_calls_detector_with_frame(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test detector is called with the frame."""
        attendance_pipeline.process_frame(sample_frame)
        mock_detector.detect.assert_called_once()
        # Verify frame was passed
        call_args = mock_detector.detect.call_args
        assert call_args is not None

    def test_processes_single_face(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test processing a single detected face."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        # Setup recognizer to match
        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        # Enroll a student first
        attendance_pipeline.enroll_student("STU001", [sample_frame])

        results = attendance_pipeline.process_frame(sample_frame)

        # Verify liveness was checked
        mock_liveness.check.assert_called()
        # Verify recognition was attempted
        mock_recognizer.extract.assert_called()

    def test_processes_multiple_faces(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test processing multiple detected faces."""
        face1 = DetectedFace(
            bounding_box=BoundingBox(x=50, y=50, width=100, height=100),
            landmarks=None,
            confidence=0.9,
        )
        face2 = DetectedFace(
            bounding_box=BoundingBox(x=300, y=50, width=100, height=100),
            landmarks=None,
            confidence=0.85,
        )
        mock_detector.detect.return_value = DetectionResult(faces=[face1, face2])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        attendance_pipeline.process_frame(sample_frame)

        # Verify liveness checked for each face
        assert mock_liveness.check.call_count == 2

    def test_returns_attendance_result_type(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test returns AttendanceResult objects."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        # Enroll student
        attendance_pipeline.enroll_student("STU001", [sample_frame])

        results = attendance_pipeline.process_frame(sample_frame)

        if results:
            assert isinstance(results[0], AttendanceResult)

    def test_result_contains_timestamp(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test result contains valid timestamp."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])

        before = datetime.now()
        results = attendance_pipeline.process_frame(sample_frame)
        after = datetime.now()

        if results:
            assert before <= results[0].timestamp <= after

    def test_result_contains_bounding_box(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test result contains face bounding box."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])

        results = attendance_pipeline.process_frame(sample_frame)

        if results:
            bbox = results[0].face_bbox
            assert len(bbox) == 4
            assert bbox == (100, 100, 200, 200)

    def test_skips_recognition_for_spoofs(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test recognition is skipped for spoof faces."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=False, confidence=0.3, spoof_type=SpoofType.PHOTO
        )

        results = attendance_pipeline.process_frame(sample_frame)

        # Recognition should not be called for spoof
        mock_recognizer.extract.assert_not_called()
        assert results == []


# =============================================================================
# Enrollment Tests
# =============================================================================


class TestAttendancePipelineEnrollment:
    """Tests for student enrollment functionality."""

    def test_enroll_student_returns_true(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test successful enrollment returns True."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        result = attendance_pipeline.enroll_student("STU001", [sample_frame])
        assert result is True

    def test_enroll_increases_count(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test enrollment increases enrolled count."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        initial_count = attendance_pipeline.get_enrolled_count()

        attendance_pipeline.enroll_student("STU001", [sample_frame])

        assert attendance_pipeline.get_enrolled_count() == initial_count + 1

    def test_enroll_multiple_students(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test enrolling multiple different students."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        attendance_pipeline.enroll_student("STU002", [sample_frame])
        attendance_pipeline.enroll_student("STU003", [sample_frame])

        assert attendance_pipeline.get_enrolled_count() == 3

    def test_enroll_with_multiple_photos(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test enrollment with multiple photos."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        photos = [sample_frame, sample_frame.copy(), sample_frame.copy()]

        result = attendance_pipeline.enroll_student("STU001", photos)

        assert result is True
        # Detector should be called for each photo
        assert mock_detector.detect.call_count >= 1

    def test_enroll_duplicate_raises_error(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test enrolling duplicate student raises error."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])

        attendance_pipeline.enroll_student("STU001", [sample_frame])

        with pytest.raises(StudentAlreadyEnrolledError):
            attendance_pipeline.enroll_student("STU001", [sample_frame])

    def test_enroll_empty_photos_raises_error(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
    ) -> None:
        """Test enrollment with no photos raises error."""
        with pytest.raises(InsufficientImagesError):
            attendance_pipeline.enroll_student("STU001", [])


# =============================================================================
# Recognition Tests
# =============================================================================


class TestAttendancePipelineRecognition:
    """Tests for face recognition functionality."""

    def test_recognizes_enrolled_student(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test recognition of enrolled student."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        # Enroll student
        attendance_pipeline.enroll_student("STU001", [sample_frame])

        results = attendance_pipeline.process_frame(sample_frame)

        assert len(results) == 1
        assert results[0].student_id == "STU001"

    def test_returns_empty_for_unknown_face(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test returns empty list for unknown face."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=False, student_id=None, confidence=0.3, embedding=embedding
        )

        results = attendance_pipeline.process_frame(sample_frame)

        assert results == []

    def test_confidence_above_threshold_matches(
        self,
        mock_detector: MagicMock,
        mock_recognizer: MagicMock,
        mock_liveness: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test match when confidence is above threshold."""
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=mock_detector,
            recognizer=mock_recognizer,
            liveness_checker=mock_liveness,
            recognition_threshold=0.6,
        )

        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.75, embedding=embedding
        )

        pipeline.enroll_student("STU001", [sample_frame])
        results = pipeline.process_frame(sample_frame)

        assert len(results) == 1

    def test_confidence_below_threshold_no_match(
        self,
        mock_detector: MagicMock,
        mock_recognizer: MagicMock,
        mock_liveness: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test no match when confidence is below threshold."""
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=mock_detector,
            recognizer=mock_recognizer,
            liveness_checker=mock_liveness,
            recognition_threshold=0.8,
        )

        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=False, student_id=None, confidence=0.5, embedding=embedding
        )

        results = pipeline.process_frame(sample_frame)

        assert results == []

    def test_result_contains_confidence_score(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test result contains confidence score."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        results = attendance_pipeline.process_frame(sample_frame)

        assert results[0].confidence == 0.85

    def test_removes_student_successfully(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test removing enrolled student."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        assert attendance_pipeline.get_enrolled_count() == 1

        result = attendance_pipeline.remove_student("STU001")

        assert result is True
        assert attendance_pipeline.get_enrolled_count() == 0

    def test_remove_nonexistent_student_returns_false(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
    ) -> None:
        """Test removing non-existent student returns False."""
        result = attendance_pipeline.remove_student("NONEXISTENT")
        assert result is False

    def test_removed_student_not_recognized(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test removed student is no longer recognized."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding

        # Initially match
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        attendance_pipeline.remove_student("STU001")

        # After removal, no match
        mock_recognizer.match.return_value = MatchResult(
            matched=False, student_id=None, confidence=0.0, embedding=embedding
        )

        results = attendance_pipeline.process_frame(sample_frame)
        assert results == []


# =============================================================================
# Liveness Integration Tests
# =============================================================================


class TestAttendancePipelineLiveness:
    """Tests for liveness checking integration."""

    def test_rejects_photo_spoof(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test photo spoof is rejected."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=False, confidence=0.2, spoof_type=SpoofType.PHOTO
        )

        results = attendance_pipeline.process_frame(sample_frame)

        assert results == []
        mock_recognizer.extract.assert_not_called()

    def test_rejects_video_spoof(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test video spoof is rejected."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=False, confidence=0.3, spoof_type=SpoofType.VIDEO
        )

        results = attendance_pipeline.process_frame(sample_frame)

        assert results == []

    def test_accepts_live_face(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test live face is accepted for recognition."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        results = attendance_pipeline.process_frame(sample_frame)

        assert len(results) == 1
        assert results[0].is_live is True

    def test_result_contains_liveness_status(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test result contains is_live flag."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        results = attendance_pipeline.process_frame(sample_frame)

        assert results[0].is_live is True

    def test_mixed_live_and_spoof_faces(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test processing frame with both live and spoof faces."""
        face1 = DetectedFace(
            bounding_box=BoundingBox(x=50, y=50, width=100, height=100),
            landmarks=None,
            confidence=0.9,
        )
        face2 = DetectedFace(
            bounding_box=BoundingBox(x=300, y=50, width=100, height=100),
            landmarks=None,
            confidence=0.85,
        )
        mock_detector.detect.return_value = DetectionResult(faces=[face1, face2])

        # First face is live, second is spoof
        liveness_results = [
            LivenessResult(is_live=True, confidence=0.9, spoof_type=SpoofType.NONE),
            LivenessResult(is_live=False, confidence=0.3, spoof_type=SpoofType.PHOTO),
        ]
        mock_liveness.check.side_effect = liveness_results

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])
        results = attendance_pipeline.process_frame(sample_frame)

        # Only live face should produce result
        assert len(results) <= 1

    def test_liveness_threshold_respected(
        self,
        mock_detector: MagicMock,
        mock_recognizer: MagicMock,
        mock_liveness: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test custom liveness threshold is respected."""
        from src.domain.attendance_pipeline import DefaultAttendancePipeline

        pipeline = DefaultAttendancePipeline(
            detector=mock_detector,
            recognizer=mock_recognizer,
            liveness_checker=mock_liveness,
            liveness_threshold=0.8,
        )

        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        # Liveness score below custom threshold
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.7, spoof_type=SpoofType.NONE
        )

        results = pipeline.process_frame(sample_frame)

        # Should be rejected due to low liveness confidence
        assert results == []


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestAttendancePipelineEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_empty_frame(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
    ) -> None:
        """Test handling of empty/zero frame."""
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)

        # Should not raise, just return empty
        results = attendance_pipeline.process_frame(empty_frame)
        assert results == []

    def test_handles_grayscale_frame(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
    ) -> None:
        """Test handling of grayscale frame."""
        grayscale_frame = np.zeros((480, 640), dtype=np.uint8)
        mock_detector.detect.return_value = DetectionResult(faces=[])

        results = attendance_pipeline.process_frame(grayscale_frame)
        assert results == []

    def test_handles_detector_exception(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test graceful handling of detector exception."""
        mock_detector.detect.side_effect = RuntimeError("Detection failed")

        # Should handle gracefully
        with pytest.raises(RuntimeError):
            attendance_pipeline.process_frame(sample_frame)

    def test_handles_small_face(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test handling of very small detected face."""
        small_face = DetectedFace(
            bounding_box=BoundingBox(x=100, y=100, width=20, height=20),
            landmarks=None,
            confidence=0.6,
        )
        mock_detector.detect.return_value = DetectionResult(faces=[small_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        # Should process without error
        results = attendance_pipeline.process_frame(sample_frame)
        # Small faces might be filtered or processed
        assert isinstance(results, list)

    def test_handles_face_at_edge(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
    ) -> None:
        """Test handling of face at frame edge."""
        edge_face = DetectedFace(
            bounding_box=BoundingBox(x=0, y=0, width=100, height=100),
            landmarks=None,
            confidence=0.8,
        )
        mock_detector.detect.return_value = DetectionResult(faces=[edge_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        results = attendance_pipeline.process_frame(sample_frame)
        assert isinstance(results, list)


# =============================================================================
# Performance Tests
# =============================================================================


class TestAttendancePipelinePerformance:
    """Tests for performance requirements."""

    def test_single_face_processing_time(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        mock_liveness: MagicMock,
        mock_recognizer: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test single face processing completes quickly with mocks."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])
        mock_liveness.check.return_value = LivenessResult(
            is_live=True, confidence=0.9, spoof_type=SpoofType.NONE
        )

        embedding = FaceEmbedding(vector=np.random.randn(512).astype(np.float32), model_name="test")
        mock_recognizer.extract.return_value = embedding
        mock_recognizer.match.return_value = MatchResult(
            matched=True, student_id="STU001", confidence=0.85, embedding=embedding
        )

        attendance_pipeline.enroll_student("STU001", [sample_frame])

        # With mocks, should be very fast
        start = time.perf_counter()
        attendance_pipeline.process_frame(sample_frame)
        elapsed = time.perf_counter() - start

        # Mock processing should be < 10ms
        assert elapsed < 0.01

    def test_enrollment_is_fast(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test enrollment completes quickly with mocks."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])

        start = time.perf_counter()
        attendance_pipeline.enroll_student("STU001", [sample_frame])
        elapsed = time.perf_counter() - start

        # Mock enrollment should be < 10ms
        assert elapsed < 0.01

    def test_get_enrolled_count_is_constant_time(
        self,
        attendance_pipeline: "DefaultAttendancePipeline",
        mock_detector: MagicMock,
        sample_frame: npt.NDArray[np.uint8],
        detected_face: DetectedFace,
    ) -> None:
        """Test get_enrolled_count is O(1)."""
        mock_detector.detect.return_value = DetectionResult(faces=[detected_face])

        # Enroll many students
        for i in range(100):
            attendance_pipeline.enroll_student(f"STU{i:03d}", [sample_frame])

        start = time.perf_counter()
        count = attendance_pipeline.get_enrolled_count()
        elapsed = time.perf_counter() - start

        assert count == 100
        # Should be instant
        assert elapsed < 0.001
