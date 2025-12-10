"""Pydantic schemas for the REST API.

This module defines request and response models for all API endpoints.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    models_loaded: bool = Field(..., description="Whether all ML models are loaded")
    enrolled_count: int = Field(..., description="Number of enrolled students")


# =============================================================================
# Enrollment
# =============================================================================


class EnrollRequest(BaseModel):
    """Request model for student enrollment."""

    student_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique student identifier"
    )
    photos: list[str] = Field(
        ..., min_length=1, description="List of base64-encoded face photos"
    )


class EnrollResponse(BaseModel):
    """Response model for successful enrollment."""

    success: bool = Field(..., description="Whether enrollment was successful")
    student_id: str = Field(..., description="ID of enrolled student")
    photos_processed: int = Field(..., description="Number of photos successfully processed")


# =============================================================================
# Recognition
# =============================================================================


class RecognizeRequest(BaseModel):
    """Request model for face recognition."""

    image: str = Field(..., description="Base64-encoded image containing faces")


class FaceResult(BaseModel):
    """Individual face recognition result."""

    student_id: str | None = Field(
        ..., description="Matched student ID or null if unknown"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence")
    is_live: bool = Field(..., description="Whether face passed liveness check")
    bbox: list[int] = Field(
        ..., min_length=4, max_length=4, description="Bounding box [x, y, width, height]"
    )


class RecognizeResponse(BaseModel):
    """Response model for face recognition."""

    faces: list[FaceResult] = Field(
        default_factory=list, description="List of detected and recognized faces"
    )


# =============================================================================
# Attendance Recording
# =============================================================================


class AttendanceRecordRequest(BaseModel):
    """Request model for recording attendance."""

    image: str = Field(..., description="Base64-encoded image containing faces")
    class_id: str = Field(
        ..., min_length=1, max_length=100, description="Class or session identifier"
    )


class AttendanceRecord(BaseModel):
    """Individual attendance record."""

    student_id: str = Field(..., description="Recognized student ID")
    timestamp: datetime = Field(..., description="Time of attendance recording")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence")


class AttendanceRecordResponse(BaseModel):
    """Response model for attendance recording."""

    recorded: list[AttendanceRecord] = Field(
        default_factory=list, description="List of recorded attendance entries"
    )
    class_id: str = Field(..., description="Class ID for which attendance was recorded")


# =============================================================================
# Student Management
# =============================================================================


class StudentListResponse(BaseModel):
    """Response model for listing enrolled students."""

    count: int = Field(..., ge=0, description="Total number of enrolled students")
    students: list[str] = Field(
        default_factory=list, description="List of enrolled student IDs"
    )


class StudentDeleteResponse(BaseModel):
    """Response model for student deletion."""

    success: bool = Field(..., description="Whether deletion was successful")
    student_id: str = Field(..., description="ID of deleted student")


# =============================================================================
# Error Responses
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(None, description="Additional error details")
