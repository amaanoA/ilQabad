"""Face recognizer interface definitions for face recognition system.

This module defines the face recognition abstraction for extracting
embeddings and matching against enrolled students.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@dataclass
class FaceEmbedding:
    """Face embedding vector extracted from a face image.

    Attributes:
        vector: 1D numpy array of embedding values (128 or 512 dimensions).
        model_name: Name of the model used to generate the embedding.
    """

    vector: npt.NDArray[np.floating]
    model_name: str

    def __post_init__(self) -> None:
        """Validate embedding vector after initialization."""
        if self.vector.ndim != 1:
            raise ValueError(
                f"vector must be 1-dimensional, got {self.vector.ndim} dimensions"
            )
        if self.vector.size == 0:
            raise ValueError("vector must not be empty")

    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embedding vector.

        Returns:
            Length of the embedding vector (e.g., 128 or 512).
        """
        return len(self.vector)


@dataclass
class MatchResult:
    """Result of matching a face embedding against enrolled students.

    Attributes:
        matched: Whether a match was found above the threshold.
        student_id: ID of matched student, or None if no match.
        confidence: Similarity score between 0.0 and 1.0.
        embedding: The query embedding that was matched.
    """

    matched: bool
    student_id: str | None
    confidence: float
    embedding: FaceEmbedding

    def __post_init__(self) -> None:
        """Validate confidence score after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class MatchCandidate:
    """A candidate match from top-k matching.

    Attributes:
        student_id: ID of the candidate student.
        confidence: Similarity score between 0.0 and 1.0.
    """

    student_id: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate confidence score after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


def cosine_similarity(embedding1: FaceEmbedding, embedding2: FaceEmbedding) -> float:
    """Calculate cosine similarity between two face embeddings.

    Args:
        embedding1: First face embedding.
        embedding2: Second face embedding.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
        Returns 0.0 if either vector is zero.

    Raises:
        ValueError: If embeddings have different dimensions.
    """
    if embedding1.dimension != embedding2.dimension:
        raise ValueError(
            f"Embeddings must have same dimension, got {embedding1.dimension} "
            f"and {embedding2.dimension}"
        )

    vec1 = embedding1.vector
    vec2 = embedding2.vector

    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero vectors
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    # Ensure result is in valid range (handle floating point errors)
    return float(np.clip(similarity, -1.0, 1.0))


@runtime_checkable
class FaceRecognizer(Protocol):
    """Protocol for face recognition implementations.

    Implementations should extract face embeddings from images and
    match them against a database of enrolled students.
    """

    def extract(self, face_image: npt.NDArray[np.uint8]) -> FaceEmbedding:
        """Extract face embedding from a face image.

        Args:
            face_image: RGB face image as numpy array with shape (height, width, 3).

        Returns:
            FaceEmbedding containing the extracted feature vector.
        """
        ...

    def match(
        self, embedding: FaceEmbedding, threshold: float = 0.6
    ) -> MatchResult:
        """Match embedding against enrolled students.

        Args:
            embedding: Face embedding to match.
            threshold: Minimum similarity score to consider a match.

        Returns:
            MatchResult indicating whether a match was found.
        """
        ...

    def match_top_k(
        self, embedding: FaceEmbedding, k: int, threshold: float = 0.6
    ) -> list[MatchCandidate]:
        """Find top-k matching candidates above threshold.

        Args:
            embedding: Face embedding to match.
            k: Maximum number of candidates to return.
            threshold: Minimum similarity score for candidates.

        Returns:
            List of MatchCandidate sorted by confidence descending.
        """
        ...
