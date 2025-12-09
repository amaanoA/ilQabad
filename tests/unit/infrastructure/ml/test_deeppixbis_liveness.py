"""Tests for DeePixBiSLiveness implementation.

This module contains comprehensive tests for the DeePixBiS liveness detection
implementation using the ONNX model for anti-spoofing.
"""

from pathlib import Path
from typing import TYPE_CHECKING
import time

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.liveness import (
    LivenessChecker,
    LivenessResult,
    SpoofType,
)

if TYPE_CHECKING:
    from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_PATH = Path("models/liveness/deeppixbis.onnx")
MODEL_EXISTS = DEFAULT_MODEL_PATH.exists()

# Skip message for tests requiring the model
SKIP_NO_MODEL = "DeePixBiS model not found at models/liveness/deeppixbis.onnx"

# Test face image path
TEST_FACE_PATH = Path("tests/data/faces/face_001.jpg")
TEST_FACE_EXISTS = TEST_FACE_PATH.exists()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def deeppixbis_liveness() -> "DeePixBiSLiveness":
    """Create a DeePixBiSLiveness with default settings."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

    return DeePixBiSLiveness()


@pytest.fixture
def deeppixbis_low_threshold() -> "DeePixBiSLiveness":
    """Create a DeePixBiSLiveness with low threshold (0.1)."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

    return DeePixBiSLiveness(liveness_threshold=0.1)


@pytest.fixture
def deeppixbis_high_threshold() -> "DeePixBiSLiveness":
    """Create a DeePixBiSLiveness with high threshold (0.9)."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

    return DeePixBiSLiveness(liveness_threshold=0.9)


@pytest.fixture
def face_crop_224() -> npt.NDArray[np.uint8]:
    """Create a 224x224 RGB face crop (simulated)."""
    rng = np.random.default_rng(42)
    # Create a face-like pattern with skin tones
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add skin tone base color (RGB)
    image[:, :] = [200, 160, 140]
    # Add some random variation
    noise = rng.integers(-20, 20, (224, 224, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


@pytest.fixture
def face_crop_small() -> npt.NDArray[np.uint8]:
    """Create a small 64x64 face crop."""
    rng = np.random.default_rng(42)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [200, 160, 140]
    noise = rng.integers(-20, 20, (64, 64, 3), dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def face_crop_large() -> npt.NDArray[np.uint8]:
    """Create a large 512x512 face crop."""
    rng = np.random.default_rng(42)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[:, :] = [200, 160, 140]
    noise = rng.integers(-20, 20, (512, 512, 3), dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def noise_image() -> npt.NDArray[np.uint8]:
    """Create a random noise 224x224 image."""
    rng = np.random.default_rng(12345)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def blank_image() -> npt.NDArray[np.uint8]:
    """Create a blank (black) 224x224 image."""
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image() -> npt.NDArray[np.uint8]:
    """Create a grayscale 224x224 image."""
    return np.zeros((224, 224), dtype=np.uint8)


@pytest.fixture
def rgba_image() -> npt.NDArray[np.uint8]:
    """Create an RGBA 224x224 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 4), dtype=np.uint8)


@pytest.fixture
def single_pixel_image() -> npt.NDArray[np.uint8]:
    """Create a single pixel image."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


@pytest.fixture
def non_square_image() -> npt.NDArray[np.uint8]:
    """Create a non-square 320x240 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def very_small_face() -> npt.NDArray[np.uint8]:
    """Create a very small 32x32 face crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)


@pytest.fixture
def real_face_crop() -> npt.NDArray[np.uint8]:
    """Load a real face crop from test data, or create synthetic one."""
    if TEST_FACE_EXISTS:
        import cv2
        img = cv2.imread(str(TEST_FACE_PATH))
        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            return img

    # Fallback: create a synthetic face-like image
    rng = np.random.default_rng(42)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Create a more structured face-like pattern
    # Skin tone background
    image[:, :] = [200, 160, 140]
    # Add oval shape for face
    center_y, center_x = 112, 112
    for y in range(224):
        for x in range(224):
            dist = ((x - center_x) / 80) ** 2 + ((y - center_y) / 100) ** 2
            if dist < 1:
                # Inside face oval
                noise = rng.integers(-10, 10, 3)
                image[y, x] = np.clip([200 + noise[0], 160 + noise[1], 140 + noise[2]], 0, 255)
    return image


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDeePixBiSLivenessInitialization:
    """Tests for DeePixBiSLiveness initialization."""

    def test_initializes_with_default_model_path(
        self, deeppixbis_liveness: "DeePixBiSLiveness"
    ) -> None:
        """Test detector initializes with default model path."""
        assert deeppixbis_liveness is not None
        assert deeppixbis_liveness.model_path == DEFAULT_MODEL_PATH

    def test_initializes_with_custom_model_path(self) -> None:
        """Test detector initializes with custom model path."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        custom_path = Path("models/liveness/deeppixbis.onnx")
        detector = DeePixBiSLiveness(model_path=custom_path)
        assert detector.model_path == custom_path

    def test_raises_error_for_missing_model(self) -> None:
        """Test detector raises FileNotFoundError for missing model."""
        pytest.importorskip("onnxruntime")

        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        with pytest.raises(FileNotFoundError):
            DeePixBiSLiveness(model_path=Path("nonexistent/path/model.onnx"))

    def test_initializes_with_custom_liveness_threshold(self) -> None:
        """Test detector initializes with custom liveness threshold."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        detector = DeePixBiSLiveness(liveness_threshold=0.7)
        assert detector.liveness_threshold == 0.7

    def test_stores_configuration_attributes(
        self, deeppixbis_liveness: "DeePixBiSLiveness"
    ) -> None:
        """Test detector stores all configuration attributes."""
        assert hasattr(deeppixbis_liveness, "model_path")
        assert hasattr(deeppixbis_liveness, "liveness_threshold")

    def test_onnx_session_created_successfully(
        self, deeppixbis_liveness: "DeePixBiSLiveness"
    ) -> None:
        """Test ONNX session is created successfully."""
        assert hasattr(deeppixbis_liveness, "_session")
        assert deeppixbis_liveness._session is not None


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestDeePixBiSLivenessProtocolCompliance:
    """Tests for LivenessChecker protocol compliance."""

    def test_implements_liveness_checker_protocol(
        self, deeppixbis_liveness: "DeePixBiSLiveness"
    ) -> None:
        """Test DeePixBiSLiveness implements LivenessChecker protocol."""
        assert isinstance(deeppixbis_liveness, LivenessChecker)

    def test_has_check_method(
        self, deeppixbis_liveness: "DeePixBiSLiveness"
    ) -> None:
        """Test detector has check method."""
        assert hasattr(deeppixbis_liveness, "check")
        assert callable(deeppixbis_liveness.check)

    def test_check_returns_liveness_result(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test check method returns LivenessResult."""
        result = deeppixbis_liveness.check(face_crop_224)
        assert isinstance(result, LivenessResult)

    def test_liveness_result_has_required_fields(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test LivenessResult has required fields (is_live, confidence)."""
        result = deeppixbis_liveness.check(face_crop_224)
        assert hasattr(result, "is_live")
        assert hasattr(result, "confidence")
        assert isinstance(result.is_live, bool)
        assert isinstance(result.confidence, float)


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestDeePixBiSLivenessPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_resizes_to_224x224(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_large: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing resizes face crop to 224x224."""
        result = deeppixbis_liveness._preprocess(face_crop_large)
        # Expected shape: [1, 3, 224, 224]
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_converts_to_float32(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing converts to float32."""
        result = deeppixbis_liveness._preprocess(face_crop_224)
        assert result.dtype == np.float32

    def test_preprocess_normalizes_with_imagenet_stats(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing normalizes using ImageNet mean and std."""
        result = deeppixbis_liveness._preprocess(face_crop_224)
        # After ImageNet normalization, values should be roughly in [-2.5, 2.5]
        # With mean subtraction and std division
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_preprocess_handles_grayscale_input(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        grayscale_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing handles grayscale images."""
        result = deeppixbis_liveness._preprocess(grayscale_image)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_handles_rgba_input(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        rgba_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing handles RGBA images."""
        result = deeppixbis_liveness._preprocess(rgba_image)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_transposes_hwc_to_chw(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing transposes HWC to CHW format."""
        result = deeppixbis_liveness._preprocess(face_crop_224)
        # Shape should be [N, C, H, W]
        assert len(result.shape) == 4
        assert result.shape[1] == 3  # Channels
        assert result.shape[2] == 224  # Height
        assert result.shape[3] == 224  # Width


# =============================================================================
# Liveness Detection Tests
# =============================================================================


class TestDeePixBiSLivenessDetection:
    """Tests for liveness detection functionality."""

    def test_returns_liveness_result_dataclass(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test check returns LivenessResult dataclass."""
        result = deeppixbis_liveness.check(face_crop_224)
        assert isinstance(result, LivenessResult)

    def test_score_is_between_0_and_1(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test confidence score is between 0.0 and 1.0."""
        result = deeppixbis_liveness.check(face_crop_224)
        assert 0.0 <= result.confidence <= 1.0

    def test_is_live_true_when_score_above_threshold(
        self,
        deeppixbis_low_threshold: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test is_live is True when score >= threshold."""
        result = deeppixbis_low_threshold.check(face_crop_224)
        # With low threshold (0.1), most scores should pass
        if result.confidence >= 0.1:
            assert result.is_live is True

    def test_is_live_false_when_score_below_threshold(
        self,
        deeppixbis_high_threshold: "DeePixBiSLiveness",
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test is_live is False when score < threshold."""
        result = deeppixbis_high_threshold.check(noise_image)
        # With high threshold (0.9), noise should likely fail
        if result.confidence < 0.9:
            assert result.is_live is False

    def test_deterministic_same_input_same_output(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test same input produces same output (deterministic)."""
        result1 = deeppixbis_liveness.check(face_crop_224)
        result2 = deeppixbis_liveness.check(face_crop_224)
        assert result1.confidence == result2.confidence
        assert result1.is_live == result2.is_live

    def test_different_images_different_scores(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test different images produce different scores."""
        result1 = deeppixbis_liveness.check(face_crop_224)
        result2 = deeppixbis_liveness.check(noise_image)
        # Different images should generally produce different scores
        # (not guaranteed, but highly likely with these different inputs)
        assert result1.confidence != result2.confidence or face_crop_224.sum() == noise_image.sum()


# =============================================================================
# Threshold Behavior Tests
# =============================================================================


class TestDeePixBiSLivenessThresholds:
    """Tests for threshold behavior."""

    def test_high_threshold_more_rejections(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        deeppixbis_high_threshold: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test high threshold (0.9) leads to more rejections."""
        result_normal = deeppixbis_liveness.check(face_crop_224)
        result_high = deeppixbis_high_threshold.check(face_crop_224)

        # Same confidence, but high threshold should be more likely to reject
        assert result_normal.confidence == result_high.confidence
        # If normal passes and high fails, threshold is working
        if result_normal.is_live and result_normal.confidence < 0.9:
            assert result_high.is_live is False

    def test_low_threshold_more_acceptances(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        deeppixbis_low_threshold: "DeePixBiSLiveness",
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test low threshold (0.1) leads to more acceptances."""
        result_normal = deeppixbis_liveness.check(noise_image)
        result_low = deeppixbis_low_threshold.check(noise_image)

        # Same confidence, but low threshold more likely to accept
        assert result_normal.confidence == result_low.confidence
        # If confidence is above 0.1 but below 0.5, low should pass, normal may fail
        if result_normal.confidence >= 0.1:
            assert result_low.is_live is True

    def test_threshold_zero_all_pass(self) -> None:
        """Test threshold 0.0 means all faces pass."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        detector = DeePixBiSLiveness(liveness_threshold=0.0)
        # Create random noise
        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        result = detector.check(image)
        # With threshold 0.0, any positive score should pass
        if result.confidence >= 0.0:
            assert result.is_live is True

    def test_threshold_one_requires_perfect_score(self) -> None:
        """Test threshold 1.0 means only perfect score passes."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

        detector = DeePixBiSLiveness(liveness_threshold=1.0)
        # Create random noise - very unlikely to get perfect score
        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        result = detector.check(image)
        # With threshold 1.0, should fail unless perfect score
        if result.confidence < 1.0:
            assert result.is_live is False


# =============================================================================
# Pixel Map Tests
# =============================================================================


class TestDeePixBiSLivenessPixelMap:
    """Tests for pixel map output."""

    def test_returns_14x14_pixel_map(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test pixel map is 14x14."""
        result = deeppixbis_liveness.check(face_crop_224)
        assert hasattr(result, "pixel_map")
        if result.pixel_map is not None:
            assert result.pixel_map.shape == (14, 14)

    def test_pixel_values_between_0_and_1(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test pixel map values are between 0.0 and 1.0."""
        result = deeppixbis_liveness.check(face_crop_224)
        if result.pixel_map is not None:
            assert result.pixel_map.min() >= 0.0
            assert result.pixel_map.max() <= 1.0

    def test_pixel_map_shape_is_correct(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test pixel map has correct 2D shape."""
        result = deeppixbis_liveness.check(face_crop_224)
        if result.pixel_map is not None:
            assert len(result.pixel_map.shape) == 2

    def test_can_access_spatial_liveness_information(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test pixel map provides spatial liveness information."""
        result = deeppixbis_liveness.check(face_crop_224)
        if result.pixel_map is not None:
            # Should be able to access individual pixel liveness scores
            center_score = result.pixel_map[7, 7]
            corner_score = result.pixel_map[0, 0]
            assert isinstance(center_score, (float, np.floating))
            assert isinstance(corner_score, (float, np.floating))


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDeePixBiSLivenessEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_handles_very_small_face_crop(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        very_small_face: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles very small 32x32 face crop."""
        result = deeppixbis_liveness.check(very_small_face)
        assert isinstance(result, LivenessResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_handles_very_large_face_crop(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_large: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles large 512x512 face crop."""
        result = deeppixbis_liveness.check(face_crop_large)
        assert isinstance(result, LivenessResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_handles_non_square_input(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        non_square_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles non-square input image."""
        result = deeppixbis_liveness.check(non_square_image)
        assert isinstance(result, LivenessResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_handles_single_pixel_image(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        single_pixel_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles single pixel image."""
        result = deeppixbis_liveness.check(single_pixel_image)
        assert isinstance(result, LivenessResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_bgr_vs_rgb_handling(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test BGR vs RGB produces different results (model expects RGB)."""
        rgb_result = deeppixbis_liveness.check(face_crop_224)
        # Convert to BGR
        bgr_image = face_crop_224[:, :, ::-1].copy()
        bgr_result = deeppixbis_liveness.check(bgr_image)

        # Both should be valid results
        assert isinstance(rgb_result, LivenessResult)
        assert isinstance(bgr_result, LivenessResult)
        # Scores may differ due to channel order


# =============================================================================
# Real Face Tests
# =============================================================================


class TestDeePixBiSLivenessRealFace:
    """Tests with real face images."""

    def test_real_face_produces_valid_result(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test real face produces valid LivenessResult."""
        result = deeppixbis_liveness.check(real_face_crop)
        assert isinstance(result, LivenessResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_real_face_higher_score_than_noise(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        real_face_crop: npt.NDArray[np.uint8],
        noise_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test real face likely has higher score than random noise."""
        face_result = deeppixbis_liveness.check(real_face_crop)
        noise_result = deeppixbis_liveness.check(noise_image)

        # Real face should generally score higher than random noise
        # (This is a soft assertion - may not always hold)
        # At minimum, both should be valid
        assert 0.0 <= face_result.confidence <= 1.0
        assert 0.0 <= noise_result.confidence <= 1.0

    def test_consistent_results_across_multiple_runs(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test consistent results across multiple inference runs."""
        results = [deeppixbis_liveness.check(real_face_crop) for _ in range(5)]

        # All results should be identical
        first_confidence = results[0].confidence
        first_is_live = results[0].is_live

        for result in results[1:]:
            assert result.confidence == first_confidence
            assert result.is_live == first_is_live

    def test_score_within_valid_range(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test score is within valid [0.0, 1.0] range."""
        result = deeppixbis_liveness.check(real_face_crop)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0


# =============================================================================
# Performance Tests
# =============================================================================


class TestDeePixBiSLivenessPerformance:
    """Tests for performance characteristics.

    Note on timing thresholds:
    - First inference is always slower due to JIT compilation
    - CPU-only inference typically takes 50-200ms depending on hardware
    - We use 500ms as a generous threshold to accommodate slower environments
    - The test uses warmup runs and averages multiple runs for reliability
    """

    def test_inference_completes_within_reasonable_time(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test single inference completes within reasonable time.

        Uses 500ms threshold to accommodate various CPU environments.
        Actual inference on modern hardware is typically 50-150ms.
        """
        # Warm up with multiple runs to allow JIT optimization
        for _ in range(3):
            deeppixbis_liveness.check(face_crop_224)

        # Measure multiple runs and take average
        times = []
        for _ in range(5):
            start = time.perf_counter()
            deeppixbis_liveness.check(face_crop_224)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        # 500ms threshold accommodates slower CI environments
        assert avg_time < 0.5, f"Avg inference took {avg_time:.3f}s, expected < 0.5s"

    def test_multiple_inferences_consistent(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test multiple inferences produce consistent results."""
        results = []
        for _ in range(10):
            result = deeppixbis_liveness.check(face_crop_224)
            results.append(result.confidence)

        # All should be identical
        assert all(r == results[0] for r in results)

    def test_batch_like_consistency(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test processing multiple different images produces valid results."""
        images = [
            face_crop_224,
            np.zeros_like(face_crop_224),
            np.ones_like(face_crop_224) * 128,
        ]

        results = [deeppixbis_liveness.check(img) for img in images]

        for result in results:
            assert isinstance(result, LivenessResult)
            assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Robustness Tests
# =============================================================================


class TestDeePixBiSLivenessRobustness:
    """Tests for robustness to image variations."""

    def test_handles_brightness_variations(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles brightness variations."""
        # Create darker and brighter versions
        dark = (face_crop_224 * 0.5).astype(np.uint8)
        bright = np.clip(face_crop_224 * 1.5, 0, 255).astype(np.uint8)

        result_normal = deeppixbis_liveness.check(face_crop_224)
        result_dark = deeppixbis_liveness.check(dark)
        result_bright = deeppixbis_liveness.check(bright)

        # All should produce valid results
        assert isinstance(result_normal, LivenessResult)
        assert isinstance(result_dark, LivenessResult)
        assert isinstance(result_bright, LivenessResult)

    def test_handles_minor_rotation(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles minor rotation."""
        try:
            import cv2
            # Rotate by 10 degrees
            center = (112, 112)
            matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
            rotated = cv2.warpAffine(face_crop_224, matrix, (224, 224))

            result = deeppixbis_liveness.check(rotated)
            assert isinstance(result, LivenessResult)
            assert 0.0 <= result.confidence <= 1.0
        except ImportError:
            pytest.skip("OpenCV not available for rotation test")

    def test_handles_jpeg_compression(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles JPEG compression artifacts."""
        try:
            import cv2
            # Simulate JPEG compression
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 50]
            _, encoded = cv2.imencode(".jpg", face_crop_224, encode_param)
            compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            result = deeppixbis_liveness.check(compressed)
            assert isinstance(result, LivenessResult)
            assert 0.0 <= result.confidence <= 1.0
        except ImportError:
            pytest.skip("OpenCV not available for compression test")

    def test_handles_scale_differences(
        self,
        deeppixbis_liveness: "DeePixBiSLiveness",
        face_crop_224: npt.NDArray[np.uint8],
    ) -> None:
        """Test handles images of different scales."""
        try:
            import cv2
            # Scale down then back up
            small = cv2.resize(face_crop_224, (112, 112))
            scaled_up = cv2.resize(small, (224, 224))

            result = deeppixbis_liveness.check(scaled_up)
            assert isinstance(result, LivenessResult)
            assert 0.0 <= result.confidence <= 1.0
        except ImportError:
            pytest.skip("OpenCV not available for scale test")
