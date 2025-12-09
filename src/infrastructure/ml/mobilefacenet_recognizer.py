"""MobileFaceNet face recognizer implementation using ONNX Runtime.

This module provides a face recognition implementation using the MobileFaceNet model
for extracting face embeddings from aligned face images.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.core.interfaces.recognizer import (
    FaceEmbedding,
    MatchCandidate,
    MatchResult,
    cosine_similarity,
)


class MobileFaceNetRecognizer:
    """MobileFaceNet-based face recognizer using ONNX Runtime.

    This recognizer uses the MobileFaceNet model to extract 512-dimensional
    face embeddings from aligned face images.

    Attributes:
        model_path: Path to the ONNX model file.
        input_size: Input size for the model as (width, height).
    """

    def __init__(
        self,
        model_path: str | Path = "models/recognition/mobilefacenet.onnx",
        input_size: tuple[int, int] = (112, 112),
    ) -> None:
        """Initialize the MobileFaceNet face recognizer.

        Args:
            model_path: Path to the ONNX model file.
            input_size: Input size for the model as (width, height).

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model_path = Path(model_path)
        self.input_size = input_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize ONNX Runtime session
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The dimensionality of the embedding vector (512).
        """
        return 512

    def extract(self, face_image: npt.NDArray[np.uint8]) -> FaceEmbedding:
        """Extract face embedding from a face image.

        Args:
            face_image: RGB face image as numpy array with shape (height, width, 3).

        Returns:
            FaceEmbedding containing the extracted 512-dimensional feature vector.
        """
        # Preprocess the image
        input_tensor = self._preprocess(face_image)

        # Run inference
        outputs = self._session.run(None, {self._input_name: input_tensor})

        # Get the embedding and flatten
        embedding = outputs[0].flatten()

        # L2 normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return FaceEmbedding(
            vector=embedding.astype(np.float32),
            model_name="mobilefacenet",
        )

    def match(
        self, embedding: FaceEmbedding, threshold: float = 0.6
    ) -> MatchResult:
        """Match embedding against enrolled students.

        Note: This is a placeholder implementation. In a real system,
        this would match against a database of enrolled students.

        Args:
            embedding: Face embedding to match.
            threshold: Minimum similarity score to consider a match.

        Returns:
            MatchResult indicating whether a match was found.
        """
        # Placeholder: no enrolled students yet
        return MatchResult(
            matched=False,
            student_id=None,
            confidence=0.0,
            embedding=embedding,
        )

    def match_top_k(
        self, embedding: FaceEmbedding, k: int, threshold: float = 0.6
    ) -> list[MatchCandidate]:
        """Find top-k matching candidates above threshold.

        Note: This is a placeholder implementation. In a real system,
        this would match against a database of enrolled students.

        Args:
            embedding: Face embedding to match.
            k: Maximum number of candidates to return.
            threshold: Minimum similarity score for candidates.

        Returns:
            List of MatchCandidate sorted by confidence descending.
        """
        # Placeholder: no enrolled students yet
        return []

    def compare(
        self, embedding1: FaceEmbedding, embedding2: FaceEmbedding
    ) -> float:
        """Compare two face embeddings using cosine similarity.

        Args:
            embedding1: First face embedding.
            embedding2: Second face embedding.

        Returns:
            Cosine similarity score between -1.0 and 1.0.
        """
        return cosine_similarity(embedding1, embedding2)

    def _preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Preprocess image for model inference.

        Args:
            image: Input image as numpy array.

        Returns:
            Preprocessed image tensor with shape (1, 3, 112, 112).
        """
        # Handle grayscale images
        rgb_image: Any
        if image.ndim == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Handle RGBA images
        elif image.shape[2] == 4:
            rgb_image = image[:, :, :3]
        else:
            rgb_image = image

        # Resize to input size
        input_width, input_height = self.input_size
        resized = cv2.resize(rgb_image, (input_width, input_height))

        # Convert to float32 and normalize: (pixel - 127.5) / 128.0
        # This maps [0, 255] to approximately [-1, 1]
        normalized = (resized.astype(np.float32) - 127.5) / 128.0

        # Transpose from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched
