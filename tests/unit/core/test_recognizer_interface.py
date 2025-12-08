"""Tests for FaceRecognizer protocol and related dataclasses.

These tests verify the face recognition abstraction for extracting
embeddings and matching against enrolled students.
"""

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.recognizer import (
    FaceEmbedding,
    FaceRecognizer,
    MatchCandidate,
    MatchResult,
    cosine_similarity,
)

# Type alias for RGB image arrays
RGBImage = npt.NDArray[np.uint8]


class TestFaceEmbeddingCreation:
    """Tests for FaceEmbedding dataclass creation and validation."""

    def test_face_embedding_creation_with_128_dim_vector(self) -> None:
        """FaceEmbedding should accept 128-dimensional vectors."""
        vector = np.random.randn(128).astype(np.float32)
        embedding = FaceEmbedding(vector=vector, model_name="mobilefacenet")

        assert embedding.vector.shape == (128,)
        assert embedding.model_name == "mobilefacenet"

    def test_face_embedding_creation_with_512_dim_vector(self) -> None:
        """FaceEmbedding should accept 512-dimensional vectors."""
        vector = np.random.randn(512).astype(np.float32)
        embedding = FaceEmbedding(vector=vector, model_name="arcface")

        assert embedding.vector.shape == (512,)
        assert embedding.model_name == "arcface"

    def test_face_embedding_dimension_property(self) -> None:
        """FaceEmbedding.dimension should return vector length."""
        vector_128 = np.random.randn(128).astype(np.float32)
        embedding_128 = FaceEmbedding(vector=vector_128, model_name="model")

        vector_512 = np.random.randn(512).astype(np.float32)
        embedding_512 = FaceEmbedding(vector=vector_512, model_name="model")

        assert embedding_128.dimension == 128
        assert embedding_512.dimension == 512

    def test_face_embedding_model_name_stored(self) -> None:
        """FaceEmbedding should store model_name correctly."""
        vector = np.random.randn(128).astype(np.float32)
        embedding = FaceEmbedding(vector=vector, model_name="custom_model_v2")

        assert embedding.model_name == "custom_model_v2"

    def test_face_embedding_empty_vector_raises_error(self) -> None:
        """FaceEmbedding should reject empty vectors."""
        empty_vector = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="vector"):
            FaceEmbedding(vector=empty_vector, model_name="model")

    def test_face_embedding_2d_array_raises_error(self) -> None:
        """FaceEmbedding should reject 2D arrays."""
        vector_2d = np.random.randn(1, 128).astype(np.float32)

        with pytest.raises(ValueError, match="vector"):
            FaceEmbedding(vector=vector_2d, model_name="model")

    def test_face_embedding_accepts_float32(self) -> None:
        """FaceEmbedding should accept float32 dtype."""
        vector = np.random.randn(128).astype(np.float32)
        embedding = FaceEmbedding(vector=vector, model_name="model")

        assert embedding.vector.dtype == np.float32

    def test_face_embedding_accepts_float64(self) -> None:
        """FaceEmbedding should accept float64 dtype."""
        vector = np.random.randn(128).astype(np.float64)
        embedding = FaceEmbedding(vector=vector, model_name="model")

        assert embedding.vector.dtype == np.float64


class TestMatchResultCreation:
    """Tests for MatchResult dataclass creation."""

    @pytest.fixture
    def sample_embedding(self) -> FaceEmbedding:
        """Create a sample embedding for testing."""
        vector = np.random.randn(128).astype(np.float32)
        return FaceEmbedding(vector=vector, model_name="test_model")

    def test_match_result_creation_with_match(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should be created with matched=True and student_id."""
        result = MatchResult(
            matched=True,
            student_id="STU001",
            confidence=0.95,
            embedding=sample_embedding,
        )

        assert result.matched is True
        assert result.student_id == "STU001"
        assert result.confidence == 0.95
        assert result.embedding == sample_embedding

    def test_match_result_creation_without_match(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should allow matched=False with student_id=None."""
        result = MatchResult(
            matched=False,
            student_id=None,
            confidence=0.3,
            embedding=sample_embedding,
        )

        assert result.matched is False
        assert result.student_id is None
        assert result.confidence == 0.3


class TestMatchResultConfidenceValidation:
    """Tests for MatchResult confidence validation."""

    @pytest.fixture
    def sample_embedding(self) -> FaceEmbedding:
        """Create a sample embedding for testing."""
        vector = np.random.randn(128).astype(np.float32)
        return FaceEmbedding(vector=vector, model_name="test_model")

    def test_match_result_confidence_at_minimum(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should accept confidence of 0.0."""
        result = MatchResult(
            matched=False,
            student_id=None,
            confidence=0.0,
            embedding=sample_embedding,
        )

        assert result.confidence == 0.0

    def test_match_result_confidence_at_maximum(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should accept confidence of 1.0."""
        result = MatchResult(
            matched=True,
            student_id="STU001",
            confidence=1.0,
            embedding=sample_embedding,
        )

        assert result.confidence == 1.0

    def test_match_result_confidence_below_zero_raises_error(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should reject confidence below 0.0."""
        with pytest.raises(ValueError, match="confidence"):
            MatchResult(
                matched=False,
                student_id=None,
                confidence=-0.1,
                embedding=sample_embedding,
            )

    def test_match_result_confidence_above_one_raises_error(
        self, sample_embedding: FaceEmbedding
    ) -> None:
        """MatchResult should reject confidence above 1.0."""
        with pytest.raises(ValueError, match="confidence"):
            MatchResult(
                matched=True,
                student_id="STU001",
                confidence=1.5,
                embedding=sample_embedding,
            )


class TestMatchCandidateCreation:
    """Tests for MatchCandidate dataclass creation and validation."""

    def test_match_candidate_creation_with_valid_data(self) -> None:
        """MatchCandidate should be created with valid data."""
        candidate = MatchCandidate(student_id="STU001", confidence=0.85)

        assert candidate.student_id == "STU001"
        assert candidate.confidence == 0.85

    def test_match_candidate_confidence_at_minimum(self) -> None:
        """MatchCandidate should accept confidence of 0.0."""
        candidate = MatchCandidate(student_id="STU001", confidence=0.0)

        assert candidate.confidence == 0.0

    def test_match_candidate_confidence_at_maximum(self) -> None:
        """MatchCandidate should accept confidence of 1.0."""
        candidate = MatchCandidate(student_id="STU001", confidence=1.0)

        assert candidate.confidence == 1.0

    def test_match_candidate_confidence_below_zero_raises_error(self) -> None:
        """MatchCandidate should reject confidence below 0.0."""
        with pytest.raises(ValueError, match="confidence"):
            MatchCandidate(student_id="STU001", confidence=-0.1)

    def test_match_candidate_confidence_above_one_raises_error(self) -> None:
        """MatchCandidate should reject confidence above 1.0."""
        with pytest.raises(ValueError, match="confidence"):
            MatchCandidate(student_id="STU001", confidence=1.1)


class TestCosineSimilarity:
    """Tests for cosine_similarity utility function."""

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Identical vectors should have similarity of 1.0."""
        vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        embedding1 = FaceEmbedding(vector=vector.copy(), model_name="model")
        embedding2 = FaceEmbedding(vector=vector.copy(), model_name="model")

        similarity = cosine_similarity(embedding1, embedding2)

        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity of -1.0."""
        vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        embedding1 = FaceEmbedding(vector=vector, model_name="model")
        embedding2 = FaceEmbedding(vector=-vector, model_name="model")

        similarity = cosine_similarity(embedding1, embedding2)

        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity of 0.0."""
        vector1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vector2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        embedding1 = FaceEmbedding(vector=vector1, model_name="model")
        embedding2 = FaceEmbedding(vector=vector2, model_name="model")

        similarity = cosine_similarity(embedding1, embedding2)

        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_different_dimensions_raises_error(self) -> None:
        """Vectors with different dimensions should raise error."""
        vector1 = np.random.randn(128).astype(np.float32)
        vector2 = np.random.randn(512).astype(np.float32)
        embedding1 = FaceEmbedding(vector=vector1, model_name="model")
        embedding2 = FaceEmbedding(vector=vector2, model_name="model")

        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity(embedding1, embedding2)

    def test_cosine_similarity_normalized_result(self) -> None:
        """Cosine similarity should always be between -1.0 and 1.0."""
        for _ in range(10):
            vector1 = np.random.randn(128).astype(np.float32)
            vector2 = np.random.randn(128).astype(np.float32)
            embedding1 = FaceEmbedding(vector=vector1, model_name="model")
            embedding2 = FaceEmbedding(vector=vector2, model_name="model")

            similarity = cosine_similarity(embedding1, embedding2)

            assert -1.0 <= similarity <= 1.0

    def test_cosine_similarity_similar_vectors(self) -> None:
        """Similar vectors should have high similarity score."""
        vector1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # Small perturbation
        vector2 = vector1 + np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        embedding1 = FaceEmbedding(vector=vector1, model_name="model")
        embedding2 = FaceEmbedding(vector=vector2, model_name="model")

        similarity = cosine_similarity(embedding1, embedding2)

        assert similarity > 0.99  # Very similar

    def test_cosine_similarity_handles_zero_vector(self) -> None:
        """Zero vector should raise error or return 0.0."""
        vector1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        zero_vector = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embedding1 = FaceEmbedding(vector=vector1, model_name="model")
        embedding2 = FaceEmbedding(vector=zero_vector, model_name="model")

        # Should either raise an error or return 0.0
        try:
            similarity = cosine_similarity(embedding1, embedding2)
            assert similarity == 0.0 or np.isnan(similarity)
        except (ValueError, ZeroDivisionError):
            pass  # Also acceptable to raise an error


class TestFaceRecognizerProtocol:
    """Tests for FaceRecognizer protocol compliance."""

    def test_face_recognizer_has_extract_method(self) -> None:
        """FaceRecognizer protocol should define extract method."""
        assert hasattr(FaceRecognizer, "extract")
        assert callable(getattr(FaceRecognizer, "extract", None))

    def test_face_recognizer_has_match_method(self) -> None:
        """FaceRecognizer protocol should define match method."""
        assert hasattr(FaceRecognizer, "match")
        assert callable(getattr(FaceRecognizer, "match", None))

    def test_face_recognizer_has_match_top_k_method(self) -> None:
        """FaceRecognizer protocol should define match_top_k method."""
        assert hasattr(FaceRecognizer, "match_top_k")
        assert callable(getattr(FaceRecognizer, "match_top_k", None))


class TestFaceRecognizerMockImplementation:
    """Tests for FaceRecognizer with a mock implementation."""

    @pytest.fixture
    def mock_recognizer(self) -> FaceRecognizer:
        """Create a mock recognizer that implements FaceRecognizer protocol."""

        class MockRecognizer:
            """Mock recognizer for testing protocol compliance."""

            def __init__(self) -> None:
                self._enrolled: dict[str, FaceEmbedding] = {}
                self._threshold = 0.6

            def enroll(self, student_id: str, embedding: FaceEmbedding) -> None:
                """Enroll a student with their embedding."""
                self._enrolled[student_id] = embedding

            def extract(self, face_image: RGBImage) -> FaceEmbedding:
                """Extract embedding from face image."""
                # Return a random embedding for testing
                vector = np.random.randn(128).astype(np.float32)
                return FaceEmbedding(vector=vector, model_name="mock_model")

            def match(
                self, embedding: FaceEmbedding, threshold: float = 0.6
            ) -> MatchResult:
                """Match embedding against enrolled students."""
                best_match: str | None = None
                best_confidence = 0.0

                for student_id, enrolled_emb in self._enrolled.items():
                    sim = cosine_similarity(embedding, enrolled_emb)
                    # Convert from [-1, 1] to [0, 1]
                    confidence = (sim + 1) / 2
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = student_id

                matched = best_match is not None and best_confidence >= threshold
                return MatchResult(
                    matched=matched,
                    student_id=best_match if matched else None,
                    confidence=best_confidence,
                    embedding=embedding,
                )

            def match_top_k(
                self, embedding: FaceEmbedding, k: int, threshold: float = 0.6
            ) -> list[MatchCandidate]:
                """Return top-k matching candidates."""
                candidates: list[tuple[str, float]] = []

                for student_id, enrolled_emb in self._enrolled.items():
                    sim = cosine_similarity(embedding, enrolled_emb)
                    confidence = (sim + 1) / 2
                    if confidence >= threshold:
                        candidates.append((student_id, confidence))

                # Sort by confidence descending
                candidates.sort(key=lambda x: x[1], reverse=True)

                return [
                    MatchCandidate(student_id=sid, confidence=conf)
                    for sid, conf in candidates[:k]
                ]

        return MockRecognizer()

    @pytest.fixture
    def sample_image(self) -> RGBImage:
        """Create a sample RGB face image for testing."""
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def test_face_recognizer_extract_returns_embedding(
        self, mock_recognizer: FaceRecognizer, sample_image: RGBImage
    ) -> None:
        """FaceRecognizer.extract should return FaceEmbedding."""
        result = mock_recognizer.extract(sample_image)

        assert isinstance(result, FaceEmbedding)
        assert result.dimension > 0

    def test_face_recognizer_match_returns_match_result(
        self, mock_recognizer: FaceRecognizer, sample_image: RGBImage
    ) -> None:
        """FaceRecognizer.match should return MatchResult."""
        embedding = mock_recognizer.extract(sample_image)
        result = mock_recognizer.match(embedding)

        assert isinstance(result, MatchResult)

    def test_face_recognizer_match_top_k_returns_list(
        self, mock_recognizer: FaceRecognizer, sample_image: RGBImage
    ) -> None:
        """FaceRecognizer.match_top_k should return list of MatchCandidate."""
        embedding = mock_recognizer.extract(sample_image)
        result = mock_recognizer.match_top_k(embedding, k=5)

        assert isinstance(result, list)
        for candidate in result:
            assert isinstance(candidate, MatchCandidate)

    def test_face_recognizer_mock_no_match(
        self, mock_recognizer: FaceRecognizer, sample_image: RGBImage
    ) -> None:
        """Mock recognizer should return no match when database is empty."""
        embedding = mock_recognizer.extract(sample_image)
        result = mock_recognizer.match(embedding)

        assert result.matched is False
        assert result.student_id is None

    def test_face_recognizer_mock_with_match(
        self, mock_recognizer: FaceRecognizer, sample_image: RGBImage
    ) -> None:
        """Mock recognizer should find match when student is enrolled."""
        # Extract and enroll a student
        embedding = mock_recognizer.extract(sample_image)
        mock_recognizer.enroll("STU001", embedding)

        # Match should find the enrolled student
        result = mock_recognizer.match(embedding, threshold=0.5)

        assert result.matched is True
        assert result.student_id == "STU001"
        assert result.confidence >= 0.5
