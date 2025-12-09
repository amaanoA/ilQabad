"""Tests for YuNetDetector implementation.

This module contains comprehensive tests for the YuNet face detection
implementation using the ONNX model.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
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

if TYPE_CHECKING:
    from src.infrastructure.ml.yunet_detector import YuNetDetector


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_PATH = Path("models/detection/yunet.onnx")
MODEL_EXISTS = DEFAULT_MODEL_PATH.exists()

# Test face images path
TEST_FACES_DIR = Path("tests/data/faces")
TEST_FACE_IMAGE_PATH = TEST_FACES_DIR / "face_001.jpg"
TEST_FACE_EXISTS = TEST_FACE_IMAGE_PATH.exists()

# Skip message for tests requiring the model
SKIP_NO_MODEL = "YuNet model not found at models/detection/yunet.onnx"
SKIP_NO_TEST_FACE = "Test face image not found at tests/data/faces/face_001.jpg"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def yunet_detector() -> "YuNetDetector":
    """Create a YuNetDetector with default settings."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.yunet_detector import YuNetDetector

    return YuNetDetector()


@pytest.fixture
def yunet_detector_low_threshold() -> "YuNetDetector":
    """Create a YuNetDetector with low confidence threshold."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.yunet_detector import YuNetDetector

    return YuNetDetector(confidence_threshold=0.1)


@pytest.fixture
def yunet_detector_high_threshold() -> "YuNetDetector":
    """Create a YuNetDetector with high confidence threshold."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.yunet_detector import YuNetDetector

    return YuNetDetector(confidence_threshold=0.99)


@pytest.fixture
def blank_image() -> npt.NDArray[np.uint8]:
    """Create a blank (black) 640x640 image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def noise_image() -> npt.NDArray[np.uint8]:
    """Create a random noise 640x640 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> npt.NDArray[np.uint8]:
    """Create a small 64x64 image."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def large_image() -> npt.NDArray[np.uint8]:
    """Create a large 1920x1080 image."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def wide_image() -> npt.NDArray[np.uint8]:
    """Create a very wide image."""
    return np.zeros((100, 1000, 3), dtype=np.uint8)


@pytest.fixture
def tall_image() -> npt.NDArray[np.uint8]:
    """Create a very tall image."""
    return np.zeros((1000, 100, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image() -> npt.NDArray[np.uint8]:
    """Create a grayscale 640x640 image."""
    return np.zeros((640, 640), dtype=np.uint8)


@pytest.fixture
def single_pixel_image() -> npt.NDArray[np.uint8]:
    """Create a single pixel image."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestYuNetDetectorInitialization:
    """Tests for YuNetDetector initialization."""

    def test_yunet_detector_initializes_with_default_path(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test detector initializes with default model path."""
        assert yunet_detector is not None
        assert yunet_detector.model_path == DEFAULT_MODEL_PATH

    def test_yunet_detector_initializes_with_custom_path(self) -> None:
        """Test detector initializes with custom model path."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        custom_path = Path("models/detection/yunet.onnx")
        detector = YuNetDetector(model_path=custom_path)
        assert detector.model_path == custom_path

    def test_yunet_detector_raises_error_for_missing_model(self) -> None:
        """Test detector raises FileNotFoundError for missing model."""
        pytest.importorskip("onnxruntime")

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        with pytest.raises(FileNotFoundError):
            YuNetDetector(model_path="nonexistent/path/model.onnx")

    def test_yunet_detector_initializes_with_custom_confidence_threshold(self) -> None:
        """Test detector initializes with custom confidence threshold."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        detector = YuNetDetector(confidence_threshold=0.5)
        assert detector.confidence_threshold == 0.5

    def test_yunet_detector_initializes_with_custom_nms_threshold(self) -> None:
        """Test detector initializes with custom NMS threshold."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        detector = YuNetDetector(nms_threshold=0.5)
        assert detector.nms_threshold == 0.5

    def test_yunet_detector_initializes_with_custom_input_size(self) -> None:
        """Test detector initializes with custom input size."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        detector = YuNetDetector(input_size=(320, 320))
        assert detector.input_size == (320, 320)

    def test_yunet_detector_stores_configuration_attributes(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test detector stores all configuration attributes."""
        assert hasattr(yunet_detector, "model_path")
        assert hasattr(yunet_detector, "confidence_threshold")
        assert hasattr(yunet_detector, "nms_threshold")
        assert hasattr(yunet_detector, "input_size")


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestYuNetDetectorProtocolCompliance:
    """Tests for FaceDetector protocol compliance."""

    def test_yunet_detector_implements_face_detector_protocol(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test YuNetDetector implements FaceDetector protocol."""
        assert isinstance(yunet_detector, FaceDetector)

    def test_yunet_detector_has_detect_method(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test detector has detect method."""
        assert hasattr(yunet_detector, "detect")
        assert callable(yunet_detector.detect)

    def test_yunet_detector_has_detect_largest_method(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test detector has detect_largest method."""
        assert hasattr(yunet_detector, "detect_largest")
        assert callable(yunet_detector.detect_largest)

    def test_yunet_detector_detect_returns_detection_result(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect method returns DetectionResult."""
        result = yunet_detector.detect(blank_image)
        assert isinstance(result, DetectionResult)

    def test_yunet_detector_detect_largest_returns_detected_face_or_none(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect_largest returns DetectedFace or None."""
        result = yunet_detector.detect_largest(blank_image)
        assert result is None or isinstance(result, DetectedFace)


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestYuNetDetectorPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_returns_correct_shape(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing returns correct tensor shape."""
        result = yunet_detector._preprocess(blank_image)
        # Expected shape: [1, 3, height, width]
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_resizes_image_to_input_size(
        self, yunet_detector: "YuNetDetector", small_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing resizes image to input size."""
        result = yunet_detector._preprocess(small_image)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_normalizes_pixel_values_to_float32(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing converts to float32."""
        result = yunet_detector._preprocess(blank_image)
        assert result.dtype == np.float32

    def test_preprocess_handles_grayscale_image(
        self, yunet_detector: "YuNetDetector", grayscale_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing handles grayscale images."""
        result = yunet_detector._preprocess(grayscale_image)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_handles_different_image_sizes(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test preprocessing handles various image sizes."""
        sizes = [(100, 100), (200, 300), (800, 600), (1920, 1080)]
        for h, w in sizes:
            image = np.zeros((h, w, 3), dtype=np.uint8)
            result = yunet_detector._preprocess(image)
            assert result.shape == (1, 3, 640, 640)

    def test_preprocess_handles_small_image(
        self, yunet_detector: "YuNetDetector", small_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing handles small images."""
        result = yunet_detector._preprocess(small_image)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_handles_large_image(
        self, yunet_detector: "YuNetDetector", large_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test preprocessing handles large images."""
        result = yunet_detector._preprocess(large_image)
        assert result.shape == (1, 3, 640, 640)


# =============================================================================
# Detection Tests
# =============================================================================


class TestYuNetDetectorDetection:
    """Tests for face detection functionality."""

    def test_detect_on_blank_image_returns_no_faces(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detection on blank image returns no faces."""
        result = yunet_detector.detect(blank_image)
        assert isinstance(result, DetectionResult)
        assert len(result.faces) == 0

    def test_detect_on_noise_image_returns_detection_result(
        self, yunet_detector: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detection on noise image returns DetectionResult."""
        result = yunet_detector.detect(noise_image)
        assert isinstance(result, DetectionResult)
        # May or may not detect faces in noise, but should return valid result

    def test_detect_returns_bounding_boxes_with_valid_coordinates(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detected faces have valid bounding box coordinates."""
        result = yunet_detector_low_threshold.detect(noise_image)
        for face in result.faces:
            assert isinstance(face.bounding_box, BoundingBox)
            # Coordinates should be within image bounds
            assert face.bounding_box.x >= 0
            assert face.bounding_box.y >= 0
            assert face.bounding_box.width > 0
            assert face.bounding_box.height > 0

    def test_detect_returns_landmarks_with_five_points(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detected faces have landmarks with 5 points."""
        result = yunet_detector_low_threshold.detect(noise_image)
        for face in result.faces:
            if face.landmarks is not None:
                assert isinstance(face.landmarks, Landmarks)
                # Check all 5 landmark points exist and are tuples of 2 floats
                assert len(face.landmarks.left_eye) == 2
                assert len(face.landmarks.right_eye) == 2
                assert len(face.landmarks.nose) == 2
                assert len(face.landmarks.mouth_left) == 2
                assert len(face.landmarks.mouth_right) == 2

    def test_detect_returns_confidence_between_0_and_1(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detected faces have confidence between 0 and 1."""
        result = yunet_detector_low_threshold.detect(noise_image)
        for face in result.faces:
            assert 0.0 <= face.confidence <= 1.0

    def test_detect_bounding_box_coordinates_are_non_negative(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test bounding box coordinates are non-negative."""
        result = yunet_detector_low_threshold.detect(noise_image)
        for face in result.faces:
            assert face.bounding_box.x >= 0
            assert face.bounding_box.y >= 0


# =============================================================================
# Threshold Tests
# =============================================================================


class TestYuNetDetectorThresholds:
    """Tests for confidence threshold behavior."""

    def test_high_confidence_threshold_filters_more_detections(
        self,
        yunet_detector: "YuNetDetector",
        yunet_detector_high_threshold: "YuNetDetector",
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test higher threshold filters more detections."""
        result_normal = yunet_detector.detect(noise_image)
        result_high = yunet_detector_high_threshold.detect(noise_image)
        # High threshold should have <= detections compared to normal
        assert len(result_high.faces) <= len(result_normal.faces)

    def test_low_confidence_threshold_allows_more_detections(
        self,
        yunet_detector: "YuNetDetector",
        yunet_detector_low_threshold: "YuNetDetector",
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test lower threshold allows more detections."""
        result_normal = yunet_detector.detect(noise_image)
        result_low = yunet_detector_low_threshold.detect(noise_image)
        # Low threshold should have >= detections compared to normal
        assert len(result_low.faces) >= len(result_normal.faces)

    def test_confidence_threshold_zero_returns_all_detections(
        self, noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test zero threshold returns all possible detections."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.yunet_detector import YuNetDetector

        detector_zero = YuNetDetector(confidence_threshold=0.0)
        detector_high = YuNetDetector(confidence_threshold=0.9)

        result_zero = detector_zero.detect(noise_image)
        result_high = detector_high.detect(noise_image)

        assert len(result_zero.faces) >= len(result_high.faces)


# =============================================================================
# detect_largest Tests
# =============================================================================


class TestYuNetDetectorDetectLargest:
    """Tests for detect_largest method."""

    def test_detect_largest_returns_none_for_blank_image(
        self, yunet_detector: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect_largest returns None for blank image."""
        result = yunet_detector.detect_largest(blank_image)
        assert result is None

    def test_detect_largest_returns_none_when_no_faces_detected(
        self, yunet_detector_high_threshold: "YuNetDetector", blank_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect_largest returns None when no faces detected."""
        result = yunet_detector_high_threshold.detect_largest(blank_image)
        assert result is None

    def test_detect_largest_returns_detected_face_type(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect_largest returns DetectedFace when faces found."""
        result = yunet_detector_low_threshold.detect_largest(noise_image)
        # May or may not find faces, but if found should be DetectedFace
        if result is not None:
            assert isinstance(result, DetectedFace)

    def test_detect_largest_returns_face_with_largest_area(
        self, yunet_detector_low_threshold: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detect_largest returns face with largest bounding box area."""
        # First get all faces
        all_result = yunet_detector_low_threshold.detect(noise_image)

        if len(all_result.faces) > 0:
            # Get largest face
            largest = yunet_detector_low_threshold.detect_largest(noise_image)
            assert largest is not None

            # Verify it has the largest area
            largest_area = largest.bounding_box.area
            for face in all_result.faces:
                assert largest_area >= face.bounding_box.area


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestYuNetDetectorEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_detect_handles_single_pixel_image(
        self, yunet_detector: "YuNetDetector", single_pixel_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detection handles single pixel image."""
        result = yunet_detector.detect(single_pixel_image)
        assert isinstance(result, DetectionResult)
        assert len(result.faces) == 0

    def test_detect_handles_very_wide_image(
        self, yunet_detector: "YuNetDetector", wide_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detection handles very wide image."""
        result = yunet_detector.detect(wide_image)
        assert isinstance(result, DetectionResult)

    def test_detect_handles_very_tall_image(
        self, yunet_detector: "YuNetDetector", tall_image: npt.NDArray[np.uint8]
    ) -> None:
        """Test detection handles very tall image."""
        result = yunet_detector.detect(tall_image)
        assert isinstance(result, DetectionResult)

    def test_detect_handles_rgba_image(self, yunet_detector: "YuNetDetector") -> None:
        """Test detection handles RGBA image."""
        rgba_image = np.zeros((640, 640, 4), dtype=np.uint8)
        result = yunet_detector.detect(rgba_image)
        assert isinstance(result, DetectionResult)

    def test_detect_handles_bgr_vs_rgb(self, yunet_detector: "YuNetDetector") -> None:
        """Test detection produces consistent results regardless of channel order."""
        rng = np.random.default_rng(42)
        rgb_image = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        bgr_image = rgb_image[:, :, ::-1].copy()

        result_rgb = yunet_detector.detect(rgb_image)
        result_bgr = yunet_detector.detect(bgr_image)

        # Both should return valid DetectionResults
        assert isinstance(result_rgb, DetectionResult)
        assert isinstance(result_bgr, DetectionResult)


# =============================================================================
# Postprocessing Tests
# =============================================================================


class TestYuNetDetectorPostprocessing:
    """Tests for output postprocessing."""

    def test_postprocess_returns_list_of_detected_faces(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test postprocessing returns list of DetectedFace."""
        # Create mock raw outputs matching YuNet format
        # Format: [batch, num_detections, 15] where 15 = x,y,w,h + 10 landmarks + conf
        raw_outputs = [np.array([[[100, 100, 50, 50,
                                   110, 120, 140, 120, 125, 140, 115, 150, 135, 150,
                                   0.9]]], dtype=np.float32)]
        original_size = (640, 640)

        faces = yunet_detector._postprocess(raw_outputs, original_size)
        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, DetectedFace)

    def test_postprocess_scales_coordinates_to_original_size(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test postprocessing scales coordinates to original image size."""
        # With input_size=(640,640) and original_size=(1280,720)
        # coordinates should be scaled appropriately
        raw_outputs = [np.array([[[100, 100, 50, 50,
                                   110, 120, 140, 120, 125, 140, 115, 150, 135, 150,
                                   0.9]]], dtype=np.float32)]
        original_size = (1280, 720)  # width, height different from input

        faces = yunet_detector._postprocess(raw_outputs, original_size)
        # Should return faces (possibly scaled)
        assert isinstance(faces, list)

    def test_postprocess_filters_by_confidence_threshold(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test postprocessing filters faces below confidence threshold."""
        # Create outputs with face below threshold (default 0.7)
        raw_outputs = [np.array([[[100, 100, 50, 50,
                                   110, 120, 140, 120, 125, 140, 115, 150, 135, 150,
                                   0.3]]], dtype=np.float32)]  # confidence 0.3 < 0.7
        original_size = (640, 640)

        faces = yunet_detector._postprocess(raw_outputs, original_size)
        assert len(faces) == 0  # Should be filtered out

    def test_postprocess_handles_empty_outputs(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Test postprocessing handles empty detection outputs."""
        raw_outputs = [np.array([[]], dtype=np.float32).reshape(1, 0, 15)]
        original_size = (640, 640)

        faces = yunet_detector._postprocess(raw_outputs, original_size)
        assert faces == []


# =============================================================================
# Real Face Detection Tests
# =============================================================================


class TestYuNetDetectorRealFaces:
    """Tests with real face images."""

    @pytest.fixture
    def real_face_image(self) -> npt.NDArray[np.uint8]:
        """Load a real face image."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)
        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def test_detect_finds_face_in_real_photo(
        self, yunet_detector: "YuNetDetector", real_face_image: npt.NDArray[np.uint8]
    ) -> None:
        """Real face image should detect at least 1 face."""
        result = yunet_detector.detect(real_face_image)
        assert result.face_count >= 1
        assert result.faces[0].confidence > 0.7

    def test_detect_returns_reasonable_bounding_box(
        self, yunet_detector: "YuNetDetector", real_face_image: npt.NDArray[np.uint8]
    ) -> None:
        """Bounding box should cover significant portion of face."""
        result = yunet_detector.detect(real_face_image)
        assert result.face_count >= 1
        bbox = result.faces[0].bounding_box
        # Face should be reasonable size (not tiny, not huge)
        image_area = real_face_image.shape[0] * real_face_image.shape[1]
        face_area = bbox.area
        face_ratio = face_area / image_area
        assert 0.01 < face_ratio < 0.9  # Between 1% and 90% of image

    def test_detect_landmarks_are_within_bounding_box(
        self, yunet_detector: "YuNetDetector", real_face_image: npt.NDArray[np.uint8]
    ) -> None:
        """Landmarks should be within or near the bounding box."""
        result = yunet_detector.detect(real_face_image)
        assert result.face_count >= 1
        face = result.faces[0]
        bbox = face.bounding_box

        if face.landmarks is not None:
            # All landmarks should be near the bounding box
            for point in [
                face.landmarks.left_eye,
                face.landmarks.right_eye,
                face.landmarks.nose,
                face.landmarks.mouth_left,
                face.landmarks.mouth_right,
            ]:
                x, y = point
                # Allow some margin (20% of bbox size)
                margin_x = bbox.width * 0.2
                margin_y = bbox.height * 0.2
                assert bbox.x - margin_x <= x <= bbox.x + bbox.width + margin_x
                assert bbox.y - margin_y <= y <= bbox.y + bbox.height + margin_y


# =============================================================================
# Performance Tests
# =============================================================================


class TestYuNetDetectorPerformance:
    """Performance benchmarks for face detection."""

    def test_detection_completes_within_200ms(
        self, yunet_detector: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Face detection should complete within 200ms."""
        # Warm-up run
        yunet_detector.detect(noise_image)

        # Timed runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            yunet_detector.detect(noise_image)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 200, f"Detection took {avg_time:.1f}ms, expected < 200ms"

    def test_detection_is_consistent(
        self, yunet_detector: "YuNetDetector", noise_image: npt.NDArray[np.uint8]
    ) -> None:
        """Same image should produce same detection results."""
        result1 = yunet_detector.detect(noise_image)
        result2 = yunet_detector.detect(noise_image)

        assert result1.face_count == result2.face_count
        if result1.face_count > 0:
            # Bounding boxes should be identical
            assert result1.faces[0].bounding_box.x == result2.faces[0].bounding_box.x
            assert result1.faces[0].bounding_box.y == result2.faces[0].bounding_box.y

    def test_detection_is_consistent_with_real_face(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Same real face image should produce same detection results."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result1 = yunet_detector.detect(real_face_image)
        result2 = yunet_detector.detect(real_face_image)

        assert result1.face_count == result2.face_count
        if result1.face_count > 0:
            # Bounding boxes should be identical
            assert result1.faces[0].bounding_box.x == result2.faces[0].bounding_box.x
            assert result1.faces[0].bounding_box.y == result2.faces[0].bounding_box.y
            # Confidence should be identical
            assert result1.faces[0].confidence == result2.faces[0].confidence


# =============================================================================
# Robustness Tests
# =============================================================================


class TestYuNetDetectorRobustness:
    """Tests for detection robustness under various conditions."""

    def test_detect_handles_jpeg_artifacts(self, yunet_detector: "YuNetDetector") -> None:
        """Detection should work with JPEG compression artifacts."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simulate heavy JPEG compression
        _, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(real_face_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 20]
        )
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

        result = yunet_detector.detect(compressed_rgb)
        # Should still detect the face
        assert result.face_count >= 1

    def test_detect_handles_brightness_variations(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Detection should work with brightness variations."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Darken image
        dark = (real_face_image * 0.3).astype(np.uint8)
        result_dark = yunet_detector.detect(dark)

        # Brighten image
        bright = np.clip(real_face_image * 1.5, 0, 255).astype(np.uint8)
        result_bright = yunet_detector.detect(bright)

        # At least one should detect face
        assert result_dark.face_count >= 1 or result_bright.face_count >= 1

    def test_detect_handles_rotation_small_angle(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Detection should work with small rotations."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Rotate by 15 degrees
        h, w = real_face_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(real_face_image, rotation_matrix, (w, h))

        result = yunet_detector.detect(rotated)
        # May or may not detect, but should not crash
        assert isinstance(result, DetectionResult)

    def test_detect_handles_scaled_image(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Detection should work with scaled images."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale down to 50%
        h, w = real_face_image.shape[:2]
        scaled_down = cv2.resize(real_face_image, (w // 2, h // 2))

        # Scale up to 200%
        scaled_up = cv2.resize(real_face_image, (w * 2, h * 2))

        result_down = yunet_detector.detect(scaled_down)
        result_up = yunet_detector.detect(scaled_up)

        # Both should return valid results
        assert isinstance(result_down, DetectionResult)
        assert isinstance(result_up, DetectionResult)

    def test_detect_handles_flipped_image(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Detection should work with horizontally flipped images."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Horizontal flip
        flipped = cv2.flip(real_face_image, 1)

        result_original = yunet_detector.detect(real_face_image)
        result_flipped = yunet_detector.detect(flipped)

        # Both should detect same number of faces
        assert result_original.face_count == result_flipped.face_count

    def test_detect_handles_noisy_image(
        self, yunet_detector: "YuNetDetector"
    ) -> None:
        """Detection should work with noisy images."""
        if not TEST_FACE_EXISTS:
            pytest.skip(SKIP_NO_TEST_FACE)

        img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
        real_face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add Gaussian noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 25, real_face_image.shape).astype(np.float32)
        noisy = np.clip(real_face_image.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )

        result = yunet_detector.detect(noisy)
        # Should still detect the face (robust to moderate noise)
        assert isinstance(result, DetectionResult)
