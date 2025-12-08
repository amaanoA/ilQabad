"""Tests for LivenessChecker protocol and related dataclasses.

These tests verify the anti-spoofing detection abstraction for
determining if a face is real or a spoof attempt.
"""

import numpy as np
import numpy.typing as npt
import pytest

from src.core.interfaces.liveness import (
    LivenessChecker,
    LivenessResult,
    SpoofType,
)

# Type alias for image arrays
RGBImage = npt.NDArray[np.uint8]
DepthMap = npt.NDArray[np.float32]


class TestSpoofTypeEnum:
    """Tests for SpoofType enum values."""

    def test_spoof_type_none_exists(self) -> None:
        """SpoofType should have NONE value for live faces."""
        assert SpoofType.NONE is not None
        assert SpoofType.NONE.name == "NONE"

    def test_spoof_type_photo_exists(self) -> None:
        """SpoofType should have PHOTO value for photo attacks."""
        assert SpoofType.PHOTO is not None
        assert SpoofType.PHOTO.name == "PHOTO"

    def test_spoof_type_video_exists(self) -> None:
        """SpoofType should have VIDEO value for video replay attacks."""
        assert SpoofType.VIDEO is not None
        assert SpoofType.VIDEO.name == "VIDEO"

    def test_spoof_type_mask_exists(self) -> None:
        """SpoofType should have MASK value for 3D mask attacks."""
        assert SpoofType.MASK is not None
        assert SpoofType.MASK.name == "MASK"

    def test_spoof_type_unknown_exists(self) -> None:
        """SpoofType should have UNKNOWN value for unidentified spoofs."""
        assert SpoofType.UNKNOWN is not None
        assert SpoofType.UNKNOWN.name == "UNKNOWN"

    def test_spoof_type_comparison(self) -> None:
        """SpoofType values should be comparable."""
        assert SpoofType.NONE == SpoofType.NONE
        assert SpoofType.PHOTO != SpoofType.VIDEO
        assert SpoofType.MASK != SpoofType.NONE


class TestLivenessResultCreation:
    """Tests for LivenessResult dataclass creation."""

    def test_liveness_result_creation_live_face(self) -> None:
        """LivenessResult should be created for live face."""
        result = LivenessResult(
            is_live=True,
            confidence=0.95,
            spoof_type=SpoofType.NONE,
        )

        assert result.is_live is True
        assert result.confidence == 0.95
        assert result.spoof_type == SpoofType.NONE

    def test_liveness_result_creation_photo_spoof(self) -> None:
        """LivenessResult should be created for photo attack."""
        result = LivenessResult(
            is_live=False,
            confidence=0.92,
            spoof_type=SpoofType.PHOTO,
        )

        assert result.is_live is False
        assert result.confidence == 0.92
        assert result.spoof_type == SpoofType.PHOTO

    def test_liveness_result_creation_video_spoof(self) -> None:
        """LivenessResult should be created for video replay attack."""
        result = LivenessResult(
            is_live=False,
            confidence=0.88,
            spoof_type=SpoofType.VIDEO,
        )

        assert result.is_live is False
        assert result.confidence == 0.88
        assert result.spoof_type == SpoofType.VIDEO

    def test_liveness_result_creation_mask_spoof(self) -> None:
        """LivenessResult should be created for mask attack."""
        result = LivenessResult(
            is_live=False,
            confidence=0.85,
            spoof_type=SpoofType.MASK,
        )

        assert result.is_live is False
        assert result.confidence == 0.85
        assert result.spoof_type == SpoofType.MASK

    def test_liveness_result_creation_unknown_spoof(self) -> None:
        """LivenessResult should be created for unknown spoof type."""
        result = LivenessResult(
            is_live=False,
            confidence=0.75,
            spoof_type=SpoofType.UNKNOWN,
        )

        assert result.is_live is False
        assert result.confidence == 0.75
        assert result.spoof_type == SpoofType.UNKNOWN

    def test_liveness_result_is_spoof_property(self) -> None:
        """is_spoof property should be inverse of is_live."""
        live_result = LivenessResult(
            is_live=True,
            confidence=0.9,
            spoof_type=SpoofType.NONE,
        )
        spoof_result = LivenessResult(
            is_live=False,
            confidence=0.9,
            spoof_type=SpoofType.PHOTO,
        )

        assert live_result.is_spoof is False
        assert spoof_result.is_spoof is True


class TestLivenessResultConfidenceValidation:
    """Tests for LivenessResult confidence validation."""

    def test_liveness_result_confidence_at_minimum(self) -> None:
        """LivenessResult should accept confidence of 0.0."""
        result = LivenessResult(
            is_live=False,
            confidence=0.0,
            spoof_type=SpoofType.UNKNOWN,
        )

        assert result.confidence == 0.0

    def test_liveness_result_confidence_at_maximum(self) -> None:
        """LivenessResult should accept confidence of 1.0."""
        result = LivenessResult(
            is_live=True,
            confidence=1.0,
            spoof_type=SpoofType.NONE,
        )

        assert result.confidence == 1.0

    def test_liveness_result_confidence_below_zero_raises_error(self) -> None:
        """LivenessResult should reject confidence below 0.0."""
        with pytest.raises(ValueError, match="confidence"):
            LivenessResult(
                is_live=True,
                confidence=-0.1,
                spoof_type=SpoofType.NONE,
            )

    def test_liveness_result_confidence_above_one_raises_error(self) -> None:
        """LivenessResult should reject confidence above 1.0."""
        with pytest.raises(ValueError, match="confidence"):
            LivenessResult(
                is_live=False,
                confidence=1.5,
                spoof_type=SpoofType.PHOTO,
            )


class TestLivenessResultConsistencyValidation:
    """Tests for LivenessResult consistency between is_live and spoof_type."""

    def test_liveness_result_live_with_photo_spoof_raises_error(self) -> None:
        """is_live=True with spoof_type=PHOTO is inconsistent."""
        with pytest.raises(ValueError, match="inconsistent|spoof_type|is_live"):
            LivenessResult(
                is_live=True,
                confidence=0.9,
                spoof_type=SpoofType.PHOTO,
            )

    def test_liveness_result_live_with_video_spoof_raises_error(self) -> None:
        """is_live=True with spoof_type=VIDEO is inconsistent."""
        with pytest.raises(ValueError, match="inconsistent|spoof_type|is_live"):
            LivenessResult(
                is_live=True,
                confidence=0.9,
                spoof_type=SpoofType.VIDEO,
            )

    def test_liveness_result_live_with_mask_spoof_raises_error(self) -> None:
        """is_live=True with spoof_type=MASK is inconsistent."""
        with pytest.raises(ValueError, match="inconsistent|spoof_type|is_live"):
            LivenessResult(
                is_live=True,
                confidence=0.9,
                spoof_type=SpoofType.MASK,
            )

    def test_liveness_result_not_live_with_none_spoof_raises_error(self) -> None:
        """is_live=False with spoof_type=NONE is inconsistent."""
        with pytest.raises(ValueError, match="inconsistent|spoof_type|is_live"):
            LivenessResult(
                is_live=False,
                confidence=0.9,
                spoof_type=SpoofType.NONE,
            )


class TestLivenessCheckerProtocol:
    """Tests for LivenessChecker protocol compliance."""

    def test_liveness_checker_has_check_method(self) -> None:
        """LivenessChecker protocol should define check method."""
        assert hasattr(LivenessChecker, "check")
        assert callable(getattr(LivenessChecker, "check", None))

    def test_liveness_checker_has_check_with_depth_method(self) -> None:
        """LivenessChecker protocol should define check_with_depth method."""
        assert hasattr(LivenessChecker, "check_with_depth")
        assert callable(getattr(LivenessChecker, "check_with_depth", None))


class TestLivenessCheckerMockImplementation:
    """Tests for LivenessChecker with a mock implementation."""

    @pytest.fixture
    def mock_checker(self) -> LivenessChecker:
        """Create a mock liveness checker that implements LivenessChecker protocol."""

        class MockLivenessChecker:
            """Mock liveness checker for testing protocol compliance."""

            def __init__(self) -> None:
                self._return_live = True
                self._spoof_type = SpoofType.NONE
                self._confidence = 0.95

            def set_response(
                self, is_live: bool, spoof_type: SpoofType, confidence: float
            ) -> None:
                """Configure the response for testing."""
                self._return_live = is_live
                self._spoof_type = spoof_type
                self._confidence = confidence

            def check(self, face_image: RGBImage) -> LivenessResult:
                """Check if face is live or spoof."""
                return LivenessResult(
                    is_live=self._return_live,
                    confidence=self._confidence,
                    spoof_type=self._spoof_type,
                )

            def check_with_depth(
                self, face_image: RGBImage, depth_map: DepthMap | None = None
            ) -> LivenessResult:
                """Check liveness with optional depth map for improved accuracy."""
                # Depth map could improve confidence
                confidence = self._confidence
                if depth_map is not None:
                    confidence = min(1.0, confidence + 0.03)

                return LivenessResult(
                    is_live=self._return_live,
                    confidence=confidence,
                    spoof_type=self._spoof_type,
                )

        return MockLivenessChecker()

    @pytest.fixture
    def sample_image(self) -> RGBImage:
        """Create a sample RGB face image for testing."""
        return np.zeros((224, 224, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_depth_map(self) -> DepthMap:
        """Create a sample depth map for testing."""
        return np.zeros((224, 224), dtype=np.float32)

    def test_liveness_checker_check_returns_liveness_result(
        self, mock_checker: LivenessChecker, sample_image: RGBImage
    ) -> None:
        """LivenessChecker.check should return LivenessResult."""
        result = mock_checker.check(sample_image)

        assert isinstance(result, LivenessResult)

    def test_liveness_checker_mock_live_face(
        self, mock_checker: LivenessChecker, sample_image: RGBImage
    ) -> None:
        """Mock checker should return live result by default."""
        result = mock_checker.check(sample_image)

        assert result.is_live is True
        assert result.spoof_type == SpoofType.NONE
        assert result.is_spoof is False

    def test_liveness_checker_mock_photo_attack(
        self, mock_checker: LivenessChecker, sample_image: RGBImage
    ) -> None:
        """Mock checker should return photo spoof when configured."""
        mock_checker.set_response(
            is_live=False, spoof_type=SpoofType.PHOTO, confidence=0.92
        )

        result = mock_checker.check(sample_image)

        assert result.is_live is False
        assert result.spoof_type == SpoofType.PHOTO
        assert result.is_spoof is True
        assert result.confidence == 0.92

    def test_liveness_checker_mock_with_depth_map(
        self,
        mock_checker: LivenessChecker,
        sample_image: RGBImage,
        sample_depth_map: DepthMap,
    ) -> None:
        """Mock checker should accept depth map for improved accuracy."""
        mock_checker.set_response(
            is_live=True, spoof_type=SpoofType.NONE, confidence=0.90
        )

        result_without_depth = mock_checker.check(sample_image)
        result_with_depth = mock_checker.check_with_depth(sample_image, sample_depth_map)

        assert isinstance(result_with_depth, LivenessResult)
        # Depth map should improve confidence
        assert result_with_depth.confidence >= result_without_depth.confidence
