"""Tests for MobileFaceNetRecognizer implementation.

This module contains comprehensive tests for the MobileFaceNet face recognition
implementation using the ONNX model.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.recognizer import (
    FaceEmbedding,
    FaceRecognizer,
    cosine_similarity,
)

if TYPE_CHECKING:
    from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_PATH = Path("models/recognition/mobilefacenet.onnx")
MODEL_EXISTS = DEFAULT_MODEL_PATH.exists()

# Test face images path
TEST_FACES_DIR = Path("tests/data/faces")
TEST_FACE_IMAGE_PATH = TEST_FACES_DIR / "face_001.jpg"
TEST_FACE_IMAGE_2_PATH = TEST_FACES_DIR / "face_002.jpg"
TEST_FACE_EXISTS = TEST_FACE_IMAGE_PATH.exists()
TEST_FACE_2_EXISTS = TEST_FACE_IMAGE_2_PATH.exists()

# Skip messages
SKIP_NO_MODEL = "MobileFaceNet model not found at models/recognition/mobilefacenet.onnx"
SKIP_NO_TEST_FACE = "Test face image not found at tests/data/faces/face_001.jpg"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mobilefacenet_recognizer() -> "MobileFaceNetRecognizer":
    """Create a MobileFaceNetRecognizer with default settings."""
    pytest.importorskip("onnxruntime")
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer

    return MobileFaceNetRecognizer()


@pytest.fixture
def face_crop_112() -> npt.NDArray[np.uint8]:
    """Create a 112x112 RGB face crop (random noise for testing)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def face_crop_small() -> npt.NDArray[np.uint8]:
    """Create a small 64x64 face crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def face_crop_large() -> npt.NDArray[np.uint8]:
    """Create a large 224x224 face crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def blank_face_crop() -> npt.NDArray[np.uint8]:
    """Create a blank (black) 112x112 face crop."""
    return np.zeros((112, 112, 3), dtype=np.uint8)


@pytest.fixture
def white_face_crop() -> npt.NDArray[np.uint8]:
    """Create a white 112x112 face crop."""
    return np.full((112, 112, 3), 255, dtype=np.uint8)


@pytest.fixture
def grayscale_face_crop() -> npt.NDArray[np.uint8]:
    """Create a grayscale 112x112 face crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (112, 112), dtype=np.uint8)


@pytest.fixture
def rgba_face_crop() -> npt.NDArray[np.uint8]:
    """Create an RGBA 112x112 face crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (112, 112, 4), dtype=np.uint8)


@pytest.fixture
def single_pixel_image() -> npt.NDArray[np.uint8]:
    """Create a single pixel image."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


@pytest.fixture
def real_face_crop() -> npt.NDArray[np.uint8]:
    """Load and crop a real face from test image."""
    if not TEST_FACE_EXISTS:
        pytest.skip(SKIP_NO_TEST_FACE)
    img = cv2.imread(str(TEST_FACE_IMAGE_PATH))
    if img is None:
        pytest.skip("Failed to load test face image")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 112x112 for face recognition
    return cv2.resize(rgb, (112, 112))  # type: ignore[return-value]


@pytest.fixture
def real_face_crop_2() -> npt.NDArray[np.uint8]:
    """Load and crop a second real face from test image."""
    if not TEST_FACE_2_EXISTS:
        pytest.skip("Second test face image not found")
    img = cv2.imread(str(TEST_FACE_IMAGE_2_PATH))
    if img is None:
        pytest.skip("Failed to load second test face image")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (112, 112))  # type: ignore[return-value]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMobileFaceNetRecognizerInitialization:
    """Tests for MobileFaceNetRecognizer initialization."""

    def test_initializes_with_default_path(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test recognizer initializes with default model path."""
        assert mobilefacenet_recognizer is not None
        assert mobilefacenet_recognizer.model_path == DEFAULT_MODEL_PATH

    def test_initializes_with_custom_path(self) -> None:
        """Test recognizer initializes with custom model path."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.mobilefacenet_recognizer import (
            MobileFaceNetRecognizer,
        )

        custom_path = Path("models/recognition/mobilefacenet.onnx")
        recognizer = MobileFaceNetRecognizer(model_path=custom_path)
        assert recognizer.model_path == custom_path

    def test_raises_error_for_missing_model(self) -> None:
        """Test recognizer raises FileNotFoundError for missing model."""
        pytest.importorskip("onnxruntime")

        from src.infrastructure.ml.mobilefacenet_recognizer import (
            MobileFaceNetRecognizer,
        )

        with pytest.raises(FileNotFoundError):
            MobileFaceNetRecognizer(model_path="nonexistent/path/model.onnx")

    def test_stores_configuration_attributes(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test recognizer stores all configuration attributes."""
        assert hasattr(mobilefacenet_recognizer, "model_path")
        assert hasattr(mobilefacenet_recognizer, "input_size")

    def test_onnx_session_created_successfully(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test ONNX runtime session is created."""
        assert hasattr(mobilefacenet_recognizer, "_session")
        assert mobilefacenet_recognizer._session is not None

    def test_initializes_with_custom_input_size(self) -> None:
        """Test recognizer initializes with custom input size."""
        pytest.importorskip("onnxruntime")
        if not MODEL_EXISTS:
            pytest.skip(SKIP_NO_MODEL)

        from src.infrastructure.ml.mobilefacenet_recognizer import (
            MobileFaceNetRecognizer,
        )

        recognizer = MobileFaceNetRecognizer(input_size=(112, 112))
        assert recognizer.input_size == (112, 112)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestMobileFaceNetRecognizerProtocolCompliance:
    """Tests for FaceRecognizer protocol compliance."""

    def test_implements_face_recognizer_protocol(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test MobileFaceNetRecognizer implements FaceRecognizer protocol."""
        assert isinstance(mobilefacenet_recognizer, FaceRecognizer)

    def test_has_extract_method(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test recognizer has extract method."""
        assert hasattr(mobilefacenet_recognizer, "extract")
        assert callable(mobilefacenet_recognizer.extract)

    def test_has_match_method(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer"
    ) -> None:
        """Test recognizer has match method."""
        assert hasattr(mobilefacenet_recognizer, "match")
        assert callable(mobilefacenet_recognizer.match)

    def test_extract_returns_face_embedding(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test extract method returns FaceEmbedding."""
        result = mobilefacenet_recognizer.extract(face_crop_112)
        assert isinstance(result, FaceEmbedding)


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestMobileFaceNetRecognizerPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocess_resizes_to_112x112(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_large: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing resizes to 112x112."""
        result = mobilefacenet_recognizer._preprocess(face_crop_large)
        # Expected shape: [1, 3, 112, 112]
        assert result.shape == (1, 3, 112, 112)

    def test_preprocess_converts_to_float32(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing converts to float32."""
        result = mobilefacenet_recognizer._preprocess(face_crop_112)
        assert result.dtype == np.float32

    def test_preprocess_normalizes_pixel_values(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing normalizes pixel values."""
        result = mobilefacenet_recognizer._preprocess(face_crop_112)
        # MobileFaceNet typically uses (pixel - 127.5) / 128.0 normalization
        # Values should be roughly in range [-1, 1]
        assert result.min() >= -2.0  # Allow some margin
        assert result.max() <= 2.0

    def test_preprocess_handles_grayscale_input(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        grayscale_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing handles grayscale images."""
        result = mobilefacenet_recognizer._preprocess(grayscale_face_crop)
        assert result.shape == (1, 3, 112, 112)

    def test_preprocess_handles_rgba_input(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        rgba_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing handles RGBA images."""
        result = mobilefacenet_recognizer._preprocess(rgba_face_crop)
        assert result.shape == (1, 3, 112, 112)

    def test_preprocess_transposes_hwc_to_chw(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test preprocessing transposes HWC to CHW format."""
        result = mobilefacenet_recognizer._preprocess(face_crop_112)
        # Shape should be [batch, channels, height, width]
        assert result.shape[1] == 3  # Channels
        assert result.shape[2] == 112  # Height
        assert result.shape[3] == 112  # Width


# =============================================================================
# Embedding Generation Tests
# =============================================================================


class TestMobileFaceNetRecognizerEmbeddingGeneration:
    """Tests for embedding generation."""

    def test_extract_returns_face_embedding_dataclass(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test extract returns FaceEmbedding dataclass."""
        result = mobilefacenet_recognizer.extract(face_crop_112)
        assert isinstance(result, FaceEmbedding)

    def test_embedding_has_512_dimensions(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding vector has 512 dimensions."""
        result = mobilefacenet_recognizer.extract(face_crop_112)
        assert result.dimension == 512

    def test_embedding_values_are_float(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding values are floating point."""
        result = mobilefacenet_recognizer.extract(face_crop_112)
        assert np.issubdtype(result.vector.dtype, np.floating)

    def test_embedding_is_l2_normalized(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding is L2 normalized (magnitude ≈ 1.0)."""
        result = mobilefacenet_recognizer.extract(face_crop_112)
        magnitude = np.linalg.norm(result.vector)
        assert abs(magnitude - 1.0) < 0.01  # Allow small tolerance

    def test_embedding_is_deterministic(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test same input produces same embedding."""
        result1 = mobilefacenet_recognizer.extract(face_crop_112)
        result2 = mobilefacenet_recognizer.extract(face_crop_112)
        np.testing.assert_array_almost_equal(result1.vector, result2.vector)

    def test_different_faces_produce_different_embeddings(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
        blank_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test different faces produce different embeddings."""
        result1 = mobilefacenet_recognizer.extract(face_crop_112)
        result2 = mobilefacenet_recognizer.extract(blank_face_crop)
        # Embeddings should not be identical
        assert not np.allclose(result1.vector, result2.vector)


# =============================================================================
# Embedding Comparison Tests
# =============================================================================


class TestMobileFaceNetRecognizerEmbeddingComparison:
    """Tests for embedding comparison functionality."""

    def test_same_embedding_has_similarity_one(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test same embedding compared to itself returns similarity ≈ 1.0."""
        embedding = mobilefacenet_recognizer.extract(face_crop_112)
        similarity = cosine_similarity(embedding, embedding)
        assert abs(similarity - 1.0) < 0.001

    def test_different_embeddings_have_lower_similarity(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
        blank_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test different embeddings have similarity < 1.0."""
        emb1 = mobilefacenet_recognizer.extract(face_crop_112)
        emb2 = mobilefacenet_recognizer.extract(blank_face_crop)
        similarity = cosine_similarity(emb1, emb2)
        assert similarity < 1.0

    def test_similarity_range_is_valid(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
        blank_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test similarity is in valid cosine range [-1, 1]."""
        emb1 = mobilefacenet_recognizer.extract(face_crop_112)
        emb2 = mobilefacenet_recognizer.extract(blank_face_crop)
        similarity = cosine_similarity(emb1, emb2)
        assert -1.0 <= similarity <= 1.0

    def test_comparison_is_symmetric(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
        blank_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test cosine_similarity(a, b) == cosine_similarity(b, a)."""
        emb1 = mobilefacenet_recognizer.extract(face_crop_112)
        emb2 = mobilefacenet_recognizer.extract(blank_face_crop)
        sim1 = cosine_similarity(emb1, emb2)
        sim2 = cosine_similarity(emb2, emb1)
        assert abs(sim1 - sim2) < 0.0001

    def test_zero_vector_returns_zero_similarity(self) -> None:
        """Test zero vector handling returns 0.0 similarity."""
        zero_emb = FaceEmbedding(
            vector=np.zeros(512, dtype=np.float32),
            model_name="test",
        )
        non_zero_emb = FaceEmbedding(
            vector=np.ones(512, dtype=np.float32),
            model_name="test",
        )
        similarity = cosine_similarity(zero_emb, non_zero_emb)
        assert similarity == 0.0

    def test_orthogonal_vectors_have_zero_similarity(self) -> None:
        """Test orthogonal vectors return similarity ≈ 0."""
        # Create orthogonal vectors
        vec1 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(512, dtype=np.float32)
        vec2[1] = 1.0

        emb1 = FaceEmbedding(vector=vec1, model_name="test")
        emb2 = FaceEmbedding(vector=vec2, model_name="test")
        similarity = cosine_similarity(emb1, emb2)
        assert abs(similarity) < 0.001


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestMobileFaceNetRecognizerEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_handles_very_small_face_crop(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
    ) -> None:
        """Test handling of very small 32x32 face crop."""
        small_crop = np.zeros((32, 32, 3), dtype=np.uint8)
        result = mobilefacenet_recognizer.extract(small_crop)
        assert isinstance(result, FaceEmbedding)
        assert result.dimension == 512

    def test_handles_very_large_face_crop(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
    ) -> None:
        """Test handling of very large 512x512 face crop."""
        large_crop = np.zeros((512, 512, 3), dtype=np.uint8)
        result = mobilefacenet_recognizer.extract(large_crop)
        assert isinstance(result, FaceEmbedding)
        assert result.dimension == 512

    def test_handles_non_square_input(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
    ) -> None:
        """Test handling of non-square input (160x120)."""
        non_square = np.zeros((120, 160, 3), dtype=np.uint8)
        result = mobilefacenet_recognizer.extract(non_square)
        assert isinstance(result, FaceEmbedding)
        assert result.dimension == 512

    def test_handles_single_pixel_image(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        single_pixel_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test handling of single pixel image."""
        result = mobilefacenet_recognizer.extract(single_pixel_image)
        assert isinstance(result, FaceEmbedding)
        assert result.dimension == 512

    def test_handles_bgr_vs_rgb(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
    ) -> None:
        """Test that BGR and RGB inputs produce valid embeddings."""
        rng = np.random.default_rng(42)
        rgb_image = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
        bgr_image = rgb_image[:, :, ::-1].copy()

        result_rgb = mobilefacenet_recognizer.extract(rgb_image)
        result_bgr = mobilefacenet_recognizer.extract(bgr_image)

        # Both should produce valid embeddings
        assert isinstance(result_rgb, FaceEmbedding)
        assert isinstance(result_bgr, FaceEmbedding)
        # They will differ since channel order matters


# =============================================================================
# Real Face Tests
# =============================================================================


class TestMobileFaceNetRecognizerRealFaces:
    """Tests with real face images."""

    def test_extract_from_real_face(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding extraction from real face image."""
        result = mobilefacenet_recognizer.extract(real_face_crop)
        assert isinstance(result, FaceEmbedding)
        assert result.dimension == 512

    def test_same_face_high_similarity(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test same face image produces high similarity."""
        emb1 = mobilefacenet_recognizer.extract(real_face_crop)
        emb2 = mobilefacenet_recognizer.extract(real_face_crop)
        similarity = cosine_similarity(emb1, emb2)
        assert similarity > 0.99  # Same image should be very similar

    def test_different_people_lower_similarity(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
        real_face_crop_2: npt.NDArray[np.uint8],
    ) -> None:
        """Test different people have lower similarity."""
        emb1 = mobilefacenet_recognizer.extract(real_face_crop)
        emb2 = mobilefacenet_recognizer.extract(real_face_crop_2)
        similarity = cosine_similarity(emb1, emb2)
        # Different people should have lower similarity
        # Not necessarily < 0.5, but less than same person
        assert similarity < 0.99

    def test_real_face_embedding_magnitude(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test real face embedding has magnitude ≈ 1.0."""
        result = mobilefacenet_recognizer.extract(real_face_crop)
        magnitude = np.linalg.norm(result.vector)
        assert abs(magnitude - 1.0) < 0.01

    def test_consistent_embeddings_across_runs(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test embeddings are consistent across multiple runs."""
        embeddings = [
            mobilefacenet_recognizer.extract(real_face_crop)
            for _ in range(5)
        ]
        for i in range(1, len(embeddings)):
            np.testing.assert_array_almost_equal(
                embeddings[0].vector, embeddings[i].vector
            )


# =============================================================================
# Performance Tests
# =============================================================================


class TestMobileFaceNetRecognizerPerformance:
    """Performance benchmarks for face recognition."""

    def test_embedding_generation_under_100ms(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding generation completes within 100ms."""
        # Warm-up run
        mobilefacenet_recognizer.extract(face_crop_112)

        # Timed runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            mobilefacenet_recognizer.extract(face_crop_112)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 100, f"Embedding took {avg_time:.1f}ms, expected < 100ms"

    def test_comparison_under_1ms(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
        blank_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test embedding comparison completes within 1ms."""
        emb1 = mobilefacenet_recognizer.extract(face_crop_112)
        emb2 = mobilefacenet_recognizer.extract(blank_face_crop)

        # Timed runs
        times = []
        for _ in range(100):
            start = time.perf_counter()
            cosine_similarity(emb1, emb2)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Comparison took {avg_time:.3f}ms, expected < 1ms"

    def test_batch_consistency(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        face_crop_112: npt.NDArray[np.uint8],
    ) -> None:
        """Test multiple extractions produce consistent results."""
        results = [
            mobilefacenet_recognizer.extract(face_crop_112)
            for _ in range(10)
        ]
        for i in range(1, len(results)):
            similarity = cosine_similarity(results[0], results[i])
            assert similarity > 0.999


# =============================================================================
# Robustness Tests
# =============================================================================


class TestMobileFaceNetRecognizerRobustness:
    """Tests for recognition robustness under various conditions."""

    def test_handles_brightness_variations(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test recognition handles brightness variations."""
        # Darken image
        dark = (real_face_crop * 0.5).astype(np.uint8)
        # Brighten image
        bright = np.clip(real_face_crop * 1.5, 0, 255).astype(np.uint8)

        emb_original = mobilefacenet_recognizer.extract(real_face_crop)
        emb_dark = mobilefacenet_recognizer.extract(dark)
        emb_bright = mobilefacenet_recognizer.extract(bright)

        # All should produce valid embeddings
        assert isinstance(emb_original, FaceEmbedding)
        assert isinstance(emb_dark, FaceEmbedding)
        assert isinstance(emb_bright, FaceEmbedding)

    def test_handles_minor_rotation(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test recognition handles minor rotations."""
        # Rotate by 5 degrees
        h, w = real_face_crop.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated = cv2.warpAffine(real_face_crop, rotation_matrix, (w, h))

        emb_original = mobilefacenet_recognizer.extract(real_face_crop)
        emb_rotated = mobilefacenet_recognizer.extract(rotated)

        # Both should produce valid embeddings
        assert isinstance(emb_original, FaceEmbedding)
        assert isinstance(emb_rotated, FaceEmbedding)

    def test_handles_jpeg_artifacts(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test recognition handles JPEG compression artifacts."""
        # Simulate heavy JPEG compression
        bgr_image = cv2.cvtColor(real_face_crop, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(
            ".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 30]
        )
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert compressed is not None
        compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

        emb_original = mobilefacenet_recognizer.extract(real_face_crop)
        emb_compressed = mobilefacenet_recognizer.extract(compressed_rgb)

        # Both should produce valid embeddings
        assert isinstance(emb_original, FaceEmbedding)
        assert isinstance(emb_compressed, FaceEmbedding)

    def test_handles_scale_differences(
        self, mobilefacenet_recognizer: "MobileFaceNetRecognizer",
        real_face_crop: npt.NDArray[np.uint8],
    ) -> None:
        """Test recognition handles slight scale differences."""
        # Scale down slightly
        h, w = real_face_crop.shape[:2]
        scaled_down = cv2.resize(real_face_crop, (int(w * 0.9), int(h * 0.9)))
        # Scale up slightly
        scaled_up = cv2.resize(real_face_crop, (int(w * 1.1), int(h * 1.1)))

        emb_original = mobilefacenet_recognizer.extract(real_face_crop)
        emb_down = mobilefacenet_recognizer.extract(scaled_down)
        emb_up = mobilefacenet_recognizer.extract(scaled_up)

        # All should produce valid embeddings
        assert isinstance(emb_original, FaceEmbedding)
        assert isinstance(emb_down, FaceEmbedding)
        assert isinstance(emb_up, FaceEmbedding)
