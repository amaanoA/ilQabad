"""Tests for FaceDetector protocol and related dataclasses.

These tests verify the face detection abstraction that will be
implemented by YuNet for detecting faces in images.
"""

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.detector import (
    BoundingBox,
    DetectedFace,
    DetectionResult,
    FaceDetector,
    Landmarks,
)

# Type alias for RGB image arrays
RGBImage = npt.NDArray[np.uint8]


class TestBoundingBoxCreation:
    """Tests for BoundingBox dataclass creation and validation."""

    def test_bounding_box_creation_with_valid_values(self) -> None:
        """BoundingBox should be created with valid coordinates and dimensions."""
        bbox = BoundingBox(x=100, y=150, width=200, height=250)

        assert bbox.x == 100
        assert bbox.y == 150
        assert bbox.width == 200
        assert bbox.height == 250

    def test_bounding_box_allows_negative_x(self) -> None:
        """BoundingBox should allow negative x for faces at left image edge."""
        bbox = BoundingBox(x=-50, y=100, width=200, height=200)

        assert bbox.x == -50

    def test_bounding_box_allows_negative_y(self) -> None:
        """BoundingBox should allow negative y for faces at top image edge."""
        bbox = BoundingBox(x=100, y=-30, width=200, height=200)

        assert bbox.y == -30

    def test_bounding_box_invalid_width_raises_error(self) -> None:
        """BoundingBox should reject non-positive width."""
        with pytest.raises(ValueError, match="width"):
            BoundingBox(x=0, y=0, width=0, height=100)

        with pytest.raises(ValueError, match="width"):
            BoundingBox(x=0, y=0, width=-10, height=100)

    def test_bounding_box_invalid_height_raises_error(self) -> None:
        """BoundingBox should reject non-positive height."""
        with pytest.raises(ValueError, match="height"):
            BoundingBox(x=0, y=0, width=100, height=0)

        with pytest.raises(ValueError, match="height"):
            BoundingBox(x=0, y=0, width=100, height=-20)


class TestBoundingBoxProperties:
    """Tests for BoundingBox computed properties."""

    def test_bounding_box_center_property(self) -> None:
        """BoundingBox center should be calculated correctly."""
        bbox = BoundingBox(x=100, y=100, width=200, height=100)

        center = bbox.center

        assert center == (200.0, 150.0)  # (x + width/2, y + height/2)

    def test_bounding_box_center_with_negative_coordinates(self) -> None:
        """BoundingBox center should handle negative coordinates."""
        bbox = BoundingBox(x=-50, y=-25, width=100, height=50)

        center = bbox.center

        assert center == (0.0, 0.0)  # (-50 + 50, -25 + 25)

    def test_bounding_box_area_property(self) -> None:
        """BoundingBox area should be width * height."""
        bbox = BoundingBox(x=0, y=0, width=100, height=50)

        assert bbox.area == 5000


class TestLandmarksCreation:
    """Tests for Landmarks dataclass creation."""

    def test_landmarks_creation_with_valid_coordinates(self) -> None:
        """Landmarks should be created with 5 facial points."""
        landmarks = Landmarks(
            left_eye=(100.0, 120.0),
            right_eye=(150.0, 120.0),
            nose=(125.0, 150.0),
            mouth_left=(105.0, 180.0),
            mouth_right=(145.0, 180.0),
        )

        assert landmarks.left_eye == (100.0, 120.0)
        assert landmarks.right_eye == (150.0, 120.0)
        assert landmarks.nose == (125.0, 150.0)
        assert landmarks.mouth_left == (105.0, 180.0)
        assert landmarks.mouth_right == (145.0, 180.0)

    def test_landmarks_all_points_accessible(self) -> None:
        """All 5 landmark points should be accessible."""
        landmarks = Landmarks(
            left_eye=(10.0, 20.0),
            right_eye=(30.0, 20.0),
            nose=(20.0, 35.0),
            mouth_left=(12.0, 50.0),
            mouth_right=(28.0, 50.0),
        )

        # Verify all points are tuples with 2 elements
        assert len(landmarks.left_eye) == 2
        assert len(landmarks.right_eye) == 2
        assert len(landmarks.nose) == 2
        assert len(landmarks.mouth_left) == 2
        assert len(landmarks.mouth_right) == 2

    def test_landmarks_accepts_float_coordinates(self) -> None:
        """Landmarks should accept float coordinates for sub-pixel precision."""
        landmarks = Landmarks(
            left_eye=(100.5, 120.75),
            right_eye=(150.25, 120.5),
            nose=(125.125, 150.875),
            mouth_left=(105.0, 180.5),
            mouth_right=(145.5, 180.25),
        )

        assert landmarks.left_eye == (100.5, 120.75)
        assert landmarks.nose == (125.125, 150.875)

    def test_landmarks_accepts_integer_coordinates(self) -> None:
        """Landmarks should accept integer coordinates."""
        landmarks = Landmarks(
            left_eye=(100, 120),
            right_eye=(150, 120),
            nose=(125, 150),
            mouth_left=(105, 180),
            mouth_right=(145, 180),
        )

        assert landmarks.left_eye == (100, 120)


class TestDetectedFaceCreation:
    """Tests for DetectedFace dataclass creation."""

    @pytest.fixture
    def sample_bbox(self) -> BoundingBox:
        """Create a sample bounding box for testing."""
        return BoundingBox(x=100, y=100, width=150, height=180)

    @pytest.fixture
    def sample_landmarks(self) -> Landmarks:
        """Create sample landmarks for testing."""
        return Landmarks(
            left_eye=(130.0, 140.0),
            right_eye=(180.0, 140.0),
            nose=(155.0, 170.0),
            mouth_left=(135.0, 210.0),
            mouth_right=(175.0, 210.0),
        )

    def test_detected_face_creation_with_all_fields(
        self, sample_bbox: BoundingBox, sample_landmarks: Landmarks
    ) -> None:
        """DetectedFace should be created with all fields."""
        face = DetectedFace(
            bounding_box=sample_bbox,
            landmarks=sample_landmarks,
            confidence=0.95,
        )

        assert face.bounding_box == sample_bbox
        assert face.landmarks == sample_landmarks
        assert face.confidence == 0.95

    def test_detected_face_creation_without_landmarks(
        self, sample_bbox: BoundingBox
    ) -> None:
        """DetectedFace should allow None landmarks."""
        face = DetectedFace(
            bounding_box=sample_bbox,
            landmarks=None,
            confidence=0.8,
        )

        assert face.bounding_box == sample_bbox
        assert face.landmarks is None
        assert face.confidence == 0.8


class TestDetectedFaceConfidenceValidation:
    """Tests for DetectedFace confidence validation."""

    @pytest.fixture
    def sample_bbox(self) -> BoundingBox:
        """Create a sample bounding box for testing."""
        return BoundingBox(x=0, y=0, width=100, height=100)

    def test_detected_face_confidence_at_minimum(
        self, sample_bbox: BoundingBox
    ) -> None:
        """DetectedFace should accept confidence of 0.0."""
        face = DetectedFace(
            bounding_box=sample_bbox,
            landmarks=None,
            confidence=0.0,
        )

        assert face.confidence == 0.0

    def test_detected_face_confidence_at_maximum(
        self, sample_bbox: BoundingBox
    ) -> None:
        """DetectedFace should accept confidence of 1.0."""
        face = DetectedFace(
            bounding_box=sample_bbox,
            landmarks=None,
            confidence=1.0,
        )

        assert face.confidence == 1.0

    def test_detected_face_confidence_below_zero_raises_error(
        self, sample_bbox: BoundingBox
    ) -> None:
        """DetectedFace should reject confidence below 0.0."""
        with pytest.raises(ValueError, match="confidence"):
            DetectedFace(
                bounding_box=sample_bbox,
                landmarks=None,
                confidence=-0.1,
            )

    def test_detected_face_confidence_above_one_raises_error(
        self, sample_bbox: BoundingBox
    ) -> None:
        """DetectedFace should reject confidence above 1.0."""
        with pytest.raises(ValueError, match="confidence"):
            DetectedFace(
                bounding_box=sample_bbox,
                landmarks=None,
                confidence=1.1,
            )


class TestDetectionResultCreation:
    """Tests for DetectionResult dataclass creation."""

    def test_detection_result_empty_no_faces(self) -> None:
        """DetectionResult should handle empty face list."""
        result = DetectionResult(faces=[])

        assert result.faces == []

    def test_detection_result_single_face(self) -> None:
        """DetectionResult should handle single face."""
        face = DetectedFace(
            bounding_box=BoundingBox(x=0, y=0, width=100, height=100),
            landmarks=None,
            confidence=0.9,
        )
        result = DetectionResult(faces=[face])

        assert len(result.faces) == 1
        assert result.faces[0] == face

    def test_detection_result_multiple_faces(self) -> None:
        """DetectionResult should handle multiple faces."""
        faces = [
            DetectedFace(
                bounding_box=BoundingBox(x=0, y=0, width=100, height=100),
                landmarks=None,
                confidence=0.9,
            ),
            DetectedFace(
                bounding_box=BoundingBox(x=200, y=100, width=80, height=90),
                landmarks=None,
                confidence=0.85,
            ),
            DetectedFace(
                bounding_box=BoundingBox(x=400, y=50, width=120, height=140),
                landmarks=None,
                confidence=0.75,
            ),
        ]
        result = DetectionResult(faces=faces)

        assert len(result.faces) == 3


class TestDetectionResultProperties:
    """Tests for DetectionResult computed properties."""

    @pytest.fixture
    def empty_result(self) -> DetectionResult:
        """Create empty detection result."""
        return DetectionResult(faces=[])

    @pytest.fixture
    def multi_face_result(self) -> DetectionResult:
        """Create detection result with multiple faces of varying sizes/confidence."""
        faces = [
            DetectedFace(
                bounding_box=BoundingBox(x=0, y=0, width=100, height=100),  # area=10000
                landmarks=None,
                confidence=0.7,
            ),
            DetectedFace(
                bounding_box=BoundingBox(x=200, y=100, width=150, height=200),  # area=30000
                landmarks=None,
                confidence=0.95,  # highest confidence
            ),
            DetectedFace(
                bounding_box=BoundingBox(x=400, y=50, width=200, height=250),  # area=50000, largest
                landmarks=None,
                confidence=0.8,
            ),
        ]
        return DetectionResult(faces=faces)

    def test_detection_result_has_faces_true(
        self, multi_face_result: DetectionResult
    ) -> None:
        """has_faces should return True when faces are detected."""
        assert multi_face_result.has_faces is True

    def test_detection_result_has_faces_false(
        self, empty_result: DetectionResult
    ) -> None:
        """has_faces should return False when no faces detected."""
        assert empty_result.has_faces is False

    def test_detection_result_face_count(
        self, multi_face_result: DetectionResult
    ) -> None:
        """face_count should return number of detected faces."""
        assert multi_face_result.face_count == 3

    def test_detection_result_face_count_empty(
        self, empty_result: DetectionResult
    ) -> None:
        """face_count should return 0 for empty result."""
        assert empty_result.face_count == 0

    def test_detection_result_largest_face(
        self, multi_face_result: DetectionResult
    ) -> None:
        """largest_face should return face with largest bounding box area."""
        largest = multi_face_result.largest_face

        assert largest is not None
        assert largest.bounding_box.area == 50000  # 200 * 250

    def test_detection_result_largest_face_empty_returns_none(
        self, empty_result: DetectionResult
    ) -> None:
        """largest_face should return None when no faces detected."""
        assert empty_result.largest_face is None

    def test_detection_result_most_confident_face(
        self, multi_face_result: DetectionResult
    ) -> None:
        """most_confident_face should return face with highest confidence."""
        most_confident = multi_face_result.most_confident_face

        assert most_confident is not None
        assert most_confident.confidence == 0.95

    def test_detection_result_most_confident_face_empty_returns_none(
        self, empty_result: DetectionResult
    ) -> None:
        """most_confident_face should return None when no faces detected."""
        assert empty_result.most_confident_face is None


class TestFaceDetectorProtocol:
    """Tests for FaceDetector protocol compliance."""

    def test_face_detector_has_detect_method(self) -> None:
        """FaceDetector protocol should define detect method."""
        assert hasattr(FaceDetector, "detect")
        assert callable(getattr(FaceDetector, "detect", None))

    def test_face_detector_has_detect_largest_method(self) -> None:
        """FaceDetector protocol should define detect_largest method."""
        assert hasattr(FaceDetector, "detect_largest")
        assert callable(getattr(FaceDetector, "detect_largest", None))


class TestFaceDetectorMockImplementation:
    """Tests for FaceDetector with a mock implementation."""

    @pytest.fixture
    def mock_detector(self) -> FaceDetector:
        """Create a mock detector that implements FaceDetector protocol."""

        class MockDetector:
            """Mock detector for testing protocol compliance."""

            def __init__(self) -> None:
                self._faces_to_return: list[DetectedFace] = []

            def set_faces(self, faces: list[DetectedFace]) -> None:
                """Configure faces to return on detection."""
                self._faces_to_return = faces

            def detect(self, image: RGBImage) -> DetectionResult:
                """Detect faces in the image."""
                return DetectionResult(faces=self._faces_to_return)

            def detect_largest(self, image: RGBImage) -> DetectedFace | None:
                """Detect and return only the largest face."""
                result = self.detect(image)
                return result.largest_face

        return MockDetector()

    @pytest.fixture
    def sample_image(self) -> RGBImage:
        """Create a sample RGB image for testing."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_face_detector_detect_returns_detection_result(
        self, mock_detector: FaceDetector, sample_image: RGBImage
    ) -> None:
        """FaceDetector.detect should return DetectionResult."""
        result = mock_detector.detect(sample_image)

        assert isinstance(result, DetectionResult)

    def test_face_detector_detect_largest_returns_face_or_none(
        self, mock_detector: FaceDetector, sample_image: RGBImage
    ) -> None:
        """FaceDetector.detect_largest should return DetectedFace or None."""
        result = mock_detector.detect_largest(sample_image)

        assert result is None or isinstance(result, DetectedFace)

    def test_face_detector_mock_detect_empty_image(
        self, mock_detector: FaceDetector, sample_image: RGBImage
    ) -> None:
        """Mock detector should return empty result by default."""
        result = mock_detector.detect(sample_image)

        assert result.face_count == 0
        assert result.has_faces is False

    def test_face_detector_mock_detect_with_faces(
        self, mock_detector: FaceDetector, sample_image: RGBImage
    ) -> None:
        """Mock detector should return configured faces."""
        # Configure mock to return faces
        faces = [
            DetectedFace(
                bounding_box=BoundingBox(x=100, y=100, width=150, height=180),
                landmarks=None,
                confidence=0.9,
            ),
        ]
        mock_detector.set_faces(faces)

        result = mock_detector.detect(sample_image)

        assert result.face_count == 1
        assert result.has_faces is True
        assert result.faces[0].confidence == 0.9
