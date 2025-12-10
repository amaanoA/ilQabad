"""Dependency injection for the REST API.

This module provides FastAPI dependencies for injecting the attendance
pipeline and related services into route handlers.
"""

from pathlib import Path
from typing import Generator

from src.domain.attendance_pipeline import DefaultAttendancePipeline


# Global pipeline instance (singleton)
_pipeline_instance: DefaultAttendancePipeline | None = None
_models_loaded: bool = False


def get_pipeline() -> DefaultAttendancePipeline | None:
    """Get the global pipeline instance.

    Returns:
        The pipeline instance or None if not initialized.
    """
    return _pipeline_instance


def is_models_loaded() -> bool:
    """Check if ML models are loaded.

    Returns:
        True if models are loaded and pipeline is ready.
    """
    return _models_loaded and _pipeline_instance is not None


def init_pipeline(
    models_dir: Path = Path("models"),
    liveness_threshold: float = 0.5,
    recognition_threshold: float = 0.6,
) -> DefaultAttendancePipeline | None:
    """Initialize the global pipeline instance.

    Args:
        models_dir: Directory containing the ONNX model files.
        liveness_threshold: Threshold for liveness detection.
        recognition_threshold: Threshold for face recognition.

    Returns:
        The initialized pipeline or None if initialization failed.
    """
    global _pipeline_instance, _models_loaded

    try:
        # Import ML implementations
        from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness
        from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer
        from src.infrastructure.ml.yunet_detector import YuNetDetector

        # Initialize models
        detector = YuNetDetector(
            model_path=models_dir / "detection" / "yunet.onnx"
        )
        recognizer = MobileFaceNetRecognizer(
            model_path=models_dir / "recognition" / "mobilefacenet.onnx"
        )
        liveness_checker = DeePixBiSLiveness(
            model_path=models_dir / "liveness" / "deeppixbis.onnx"
        )

        # Create pipeline
        _pipeline_instance = DefaultAttendancePipeline(
            detector=detector,
            recognizer=recognizer,
            liveness_checker=liveness_checker,
            recognition_threshold=recognition_threshold,
            liveness_threshold=liveness_threshold,
        )
        _models_loaded = True

        return _pipeline_instance

    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        _models_loaded = False
        return None
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        _models_loaded = False
        return None


def set_pipeline(pipeline: DefaultAttendancePipeline | None) -> None:
    """Set the global pipeline instance (for testing).

    Args:
        pipeline: The pipeline instance to use, or None to clear.
    """
    global _pipeline_instance, _models_loaded
    _pipeline_instance = pipeline
    _models_loaded = pipeline is not None


def reset_pipeline() -> None:
    """Reset the global pipeline instance."""
    global _pipeline_instance, _models_loaded
    _pipeline_instance = None
    _models_loaded = False
