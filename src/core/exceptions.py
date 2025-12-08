"""Core domain exceptions."""


class IlQabadError(Exception):
    """Base exception for ilQabad application."""


class DetectionError(IlQabadError):
    """Error during face detection."""


class RecognitionError(IlQabadError):
    """Error during face recognition."""


class LivenessError(IlQabadError):
    """Error during liveness detection."""


class CameraError(IlQabadError):
    """Error with camera operations."""


class StorageError(IlQabadError):
    """Error with storage operations."""


class ConfigurationError(IlQabadError):
    """Error with application configuration."""
