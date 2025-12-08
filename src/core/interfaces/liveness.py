"""Liveness checker interface definitions for face recognition system.

This module defines the anti-spoofing detection abstraction for
determining if a face is real or a spoof attempt.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


class SpoofType(Enum):
    """Type of spoof attack detected.

    Attributes:
        NONE: Not a spoof - live face detected.
        PHOTO: Printed photo attack detected.
        VIDEO: Video replay attack detected.
        MASK: 3D mask attack detected.
        UNKNOWN: Spoof detected but type could not be determined.
    """

    NONE = auto()
    PHOTO = auto()
    VIDEO = auto()
    MASK = auto()
    UNKNOWN = auto()


@dataclass
class LivenessResult:
    """Result of liveness detection check.

    Attributes:
        is_live: Whether the face is determined to be live.
        confidence: Confidence score between 0.0 and 1.0.
        spoof_type: Type of spoof if detected, NONE if live.
    """

    is_live: bool
    confidence: float
    spoof_type: SpoofType

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        # Validate confidence range
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate consistency between is_live and spoof_type
        if self.is_live and self.spoof_type != SpoofType.NONE:
            raise ValueError(
                f"inconsistent state: is_live=True but spoof_type={self.spoof_type.name}"
            )
        if not self.is_live and self.spoof_type == SpoofType.NONE:
            raise ValueError(
                "inconsistent state: is_live=False but spoof_type=NONE"
            )

    @property
    def is_spoof(self) -> bool:
        """Check if the result indicates a spoof.

        Returns:
            True if spoof detected, False if live face.
        """
        return not self.is_live


@runtime_checkable
class LivenessChecker(Protocol):
    """Protocol for liveness detection implementations.

    Implementations should detect spoof attacks including:
    - Photo attacks (printed photos)
    - Video replay attacks
    - 3D mask attacks
    """

    def check(self, face_image: npt.NDArray[np.uint8]) -> LivenessResult:
        """Check if a face image is live or a spoof.

        Args:
            face_image: RGB face image as numpy array with shape (height, width, 3).

        Returns:
            LivenessResult indicating if face is live and spoof type if detected.
        """
        ...

    def check_with_depth(
        self,
        face_image: npt.NDArray[np.uint8],
        depth_map: npt.NDArray[np.float32] | None = None,
    ) -> LivenessResult:
        """Check liveness with optional depth map for improved accuracy.

        Args:
            face_image: RGB face image as numpy array with shape (height, width, 3).
            depth_map: Optional depth map for enhanced spoof detection.

        Returns:
            LivenessResult indicating if face is live and spoof type if detected.
        """
        ...
