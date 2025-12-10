"""FastAPI route handlers for the REST API.

This module implements all API endpoints for the ilQabad attendance system.
"""

import base64
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status

from src.core.exceptions import (
    InsufficientImagesError,
    StudentAlreadyEnrolledError,
)
from src.presentation.api import dependencies
from src.presentation.api.schemas import (
    AttendanceRecord,
    AttendanceRecordRequest,
    AttendanceRecordResponse,
    EnrollRequest,
    EnrollResponse,
    ErrorResponse,
    FaceResult,
    HealthResponse,
    RecognizeRequest,
    RecognizeResponse,
    StudentDeleteResponse,
    StudentListResponse,
)


# Create router with prefix
router = APIRouter(prefix="/api/v1", tags=["attendance"])


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64-encoded image to numpy array.

    Args:
        base64_string: Base64-encoded image data.

    Returns:
        RGB image as numpy array.

    Raises:
        ValueError: If the base64 string is invalid or cannot be decoded.
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image data")

        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")


def require_pipeline():
    """Get pipeline or raise service unavailable error."""
    pipeline = dependencies.get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Pipeline not ready. Models may not be loaded.",
            },
        )
    return pipeline


# =============================================================================
# Health Check
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and ML models are ready.",
)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    pipeline = dependencies.get_pipeline()
    models_loaded = dependencies.is_models_loaded()

    enrolled_count = 0
    if pipeline is not None:
        enrolled_count = pipeline.get_enrolled_count()

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        enrolled_count=enrolled_count,
    )


# =============================================================================
# Enrollment
# =============================================================================


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image data"},
        409: {"model": ErrorResponse, "description": "Student already enrolled"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Enroll student",
    description="Enroll a new student with face photos for recognition.",
)
async def enroll_student(request: EnrollRequest) -> EnrollResponse:
    """Enroll a new student with face photos."""
    pipeline = require_pipeline()

    # Decode all photos
    photos = []
    for i, photo_b64 in enumerate(request.photos):
        try:
            img = decode_base64_image(photo_b64)
            photos.append(img)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_image",
                    "message": f"Invalid base64 image at index {i}: {e}",
                },
            )

    # Enroll student
    try:
        success = pipeline.enroll_student(request.student_id, photos)
        return EnrollResponse(
            success=success,
            student_id=request.student_id,
            photos_processed=len(photos),
        )
    except StudentAlreadyEnrolledError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "already_enrolled",
                "message": f"Student {request.student_id} is already enrolled.",
            },
        )
    except InsufficientImagesError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "insufficient_images",
                "message": f"Not enough valid face images: {e}",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "enrollment_failed",
                "message": f"Enrollment failed: {e}",
            },
        )


# =============================================================================
# Recognition
# =============================================================================


@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image data"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Recognize faces",
    description="Detect and recognize faces in an image.",
)
async def recognize_faces(request: RecognizeRequest) -> RecognizeResponse:
    """Detect and recognize faces in an image."""
    pipeline = require_pipeline()

    # Decode image
    try:
        image = decode_base64_image(request.image)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_image",
                "message": f"Invalid image data: {e}",
            },
        )

    # Process frame
    try:
        results = pipeline.process_frame(image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "processing_failed",
                "message": f"Face recognition failed: {e}",
            },
        )

    # Convert results to response
    faces = []
    for result in results:
        faces.append(
            FaceResult(
                student_id=result.student_id,
                confidence=result.confidence,
                is_live=result.is_live,
                bbox=list(result.face_bbox),
            )
        )

    return RecognizeResponse(faces=faces)


# =============================================================================
# Attendance Recording
# =============================================================================


@router.post(
    "/attendance/record",
    response_model=AttendanceRecordResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image data"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Record attendance",
    description="Process an image and record attendance for recognized students.",
)
async def record_attendance(
    request: AttendanceRecordRequest,
) -> AttendanceRecordResponse:
    """Record attendance from an image."""
    pipeline = require_pipeline()

    # Decode image
    try:
        image = decode_base64_image(request.image)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_image",
                "message": f"Invalid image data: {e}",
            },
        )

    # Process frame
    try:
        results = pipeline.process_frame(image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "processing_failed",
                "message": f"Attendance processing failed: {e}",
            },
        )

    # Convert to attendance records
    recorded = []
    for result in results:
        recorded.append(
            AttendanceRecord(
                student_id=result.student_id,
                timestamp=result.timestamp,
                confidence=result.confidence,
            )
        )

    return AttendanceRecordResponse(
        recorded=recorded,
        class_id=request.class_id,
    )


# =============================================================================
# Student Management
# =============================================================================


@router.get(
    "/students",
    response_model=StudentListResponse,
    summary="List students",
    description="Get a list of all enrolled students.",
)
async def list_students() -> StudentListResponse:
    """List all enrolled students."""
    pipeline = dependencies.get_pipeline()

    if pipeline is None:
        return StudentListResponse(count=0, students=[])

    students = list(pipeline._enrolled_students.keys())
    return StudentListResponse(
        count=len(students),
        students=students,
    )


@router.delete(
    "/students/{student_id}",
    response_model=StudentDeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Student not found"},
    },
    summary="Delete student",
    description="Remove an enrolled student from the system.",
)
async def delete_student(student_id: str) -> StudentDeleteResponse:
    """Delete an enrolled student."""
    pipeline = dependencies.get_pipeline()

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "message": f"Student {student_id} not found.",
            },
        )

    success = pipeline.remove_student(student_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "message": f"Student {student_id} not found.",
            },
        )

    return StudentDeleteResponse(
        success=True,
        student_id=student_id,
    )
