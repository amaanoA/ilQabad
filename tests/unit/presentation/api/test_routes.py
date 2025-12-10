"""Unit tests for the REST API routes.

Test categories:
1. Health endpoint (3 tests)
2. Enrollment (6 tests)
3. Recognition (5 tests)
4. Attendance recording (5 tests)
5. Student management (4 tests)
6. Error handling (4 tests)

Total: ~27 tests
"""

import base64
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.services.attendance import AttendanceResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pipeline():
    """Create a mock attendance pipeline."""
    pipeline = MagicMock()
    pipeline.get_enrolled_count.return_value = 5
    pipeline._enrolled_students = {"S001": [], "S002": [], "S003": [], "S004": [], "S005": []}
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a test client with mocked pipeline."""
    from src.presentation.api import dependencies
    from src.presentation.api.app import create_app

    # Set the mock pipeline
    dependencies.set_pipeline(mock_pipeline)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    # Reset after test
    dependencies.reset_pipeline()


@pytest.fixture
def client_no_pipeline():
    """Create a test client without pipeline (models not loaded)."""
    from src.presentation.api import dependencies
    from src.presentation.api.app import create_app

    # Clear the pipeline
    dependencies.reset_pipeline()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_base64_image() -> str:
    """Create a valid base64-encoded test image."""
    # Create a simple 100x100 RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 128]  # Gray

    # Encode as JPEG
    import cv2
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@pytest.fixture
def valid_face_image() -> str:
    """Create a base64 image that would contain a face."""
    # Create a 224x224 RGB image (typical face input size)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    import cv2
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# 1. Health Endpoint Tests (3 tests)
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /api/v1/health endpoint."""

    def test_health_check_healthy(self, client, mock_pipeline):
        """Health check returns healthy when pipeline is ready."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert data["enrolled_count"] == 5

    def test_health_check_unhealthy_no_models(self, client_no_pipeline):
        """Health check returns unhealthy when models not loaded."""
        response = client_no_pipeline.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["models_loaded"] is False
        assert data["enrolled_count"] == 0

    def test_health_check_returns_correct_count(self, client, mock_pipeline):
        """Health check returns correct enrolled student count."""
        mock_pipeline.get_enrolled_count.return_value = 42

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json()["enrolled_count"] == 42


# =============================================================================
# 2. Enrollment Tests (6 tests)
# =============================================================================


class TestEnrollmentEndpoint:
    """Tests for POST /api/v1/enroll endpoint."""

    def test_enroll_success(self, client, mock_pipeline, valid_face_image):
        """Successful enrollment with valid photos."""
        mock_pipeline.enroll_student.return_value = True

        response = client.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001",
                "photos": [valid_face_image, valid_face_image],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["student_id"] == "S001"
        assert data["photos_processed"] == 2

    def test_enroll_duplicate_student(self, client, mock_pipeline, valid_face_image):
        """Enrollment fails for already enrolled student."""
        from src.core.exceptions import StudentAlreadyEnrolledError

        mock_pipeline.enroll_student.side_effect = StudentAlreadyEnrolledError("S001")

        response = client.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001",
                "photos": [valid_face_image],
            },
        )

        assert response.status_code == 409
        data = response.json()
        assert "already enrolled" in data["message"].lower()

    def test_enroll_invalid_base64_image(self, client, mock_pipeline):
        """Enrollment fails with invalid base64 image."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001",
                "photos": ["not-valid-base64!!!"],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "invalid" in data["message"].lower() or "base64" in data["message"].lower()

    def test_enroll_missing_student_id(self, client, mock_pipeline, valid_face_image):
        """Enrollment fails when student_id is missing."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "photos": [valid_face_image],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_enroll_empty_photos(self, client, mock_pipeline):
        """Enrollment fails with empty photos list."""
        response = client.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001",
                "photos": [],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_enroll_pipeline_not_ready(self, client_no_pipeline, valid_face_image):
        """Enrollment fails when pipeline is not ready."""
        response = client_no_pipeline.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001",
                "photos": [valid_face_image],
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "not ready" in data["message"].lower() or "unavailable" in data["message"].lower()


# =============================================================================
# 3. Recognition Tests (5 tests)
# =============================================================================


class TestRecognitionEndpoint:
    """Tests for POST /api/v1/recognize endpoint."""

    def test_recognize_single_face(self, client, mock_pipeline, valid_face_image):
        """Recognition returns single face result."""
        mock_result = AttendanceResult(
            student_id="S001",
            confidence=0.87,
            is_live=True,
            timestamp=datetime.now(),
            face_bbox=(50, 30, 100, 100),
        )
        mock_pipeline.process_frame.return_value = [mock_result]

        response = client.post(
            "/api/v1/recognize",
            json={"image": valid_face_image},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["faces"]) == 1
        assert data["faces"][0]["student_id"] == "S001"
        assert data["faces"][0]["confidence"] == 0.87
        assert data["faces"][0]["is_live"] is True
        assert data["faces"][0]["bbox"] == [50, 30, 100, 100]

    def test_recognize_multiple_faces(self, client, mock_pipeline, valid_face_image):
        """Recognition handles multiple faces in image."""
        mock_results = [
            AttendanceResult(
                student_id="S001",
                confidence=0.9,
                is_live=True,
                timestamp=datetime.now(),
                face_bbox=(50, 30, 100, 100),
            ),
            AttendanceResult(
                student_id="S002",
                confidence=0.85,
                is_live=True,
                timestamp=datetime.now(),
                face_bbox=(200, 30, 100, 100),
            ),
        ]
        mock_pipeline.process_frame.return_value = mock_results

        response = client.post(
            "/api/v1/recognize",
            json={"image": valid_face_image},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["faces"]) == 2
        assert data["faces"][0]["student_id"] == "S001"
        assert data["faces"][1]["student_id"] == "S002"

    def test_recognize_no_faces(self, client, mock_pipeline, valid_face_image):
        """Recognition returns empty list when no faces detected."""
        mock_pipeline.process_frame.return_value = []

        response = client.post(
            "/api/v1/recognize",
            json={"image": valid_face_image},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["faces"] == []

    def test_recognize_invalid_image(self, client, mock_pipeline):
        """Recognition fails with invalid image data."""
        response = client.post(
            "/api/v1/recognize",
            json={"image": "definitely-not-an-image"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "invalid" in data["message"].lower()

    def test_recognize_pipeline_not_ready(self, client_no_pipeline, valid_face_image):
        """Recognition fails when pipeline is not ready."""
        response = client_no_pipeline.post(
            "/api/v1/recognize",
            json={"image": valid_face_image},
        )

        assert response.status_code == 503


# =============================================================================
# 4. Attendance Recording Tests (5 tests)
# =============================================================================


class TestAttendanceRecordingEndpoint:
    """Tests for POST /api/v1/attendance/record endpoint."""

    def test_attendance_record_success(self, client, mock_pipeline, valid_face_image):
        """Attendance recording succeeds with recognized faces."""
        mock_result = AttendanceResult(
            student_id="S001",
            confidence=0.92,
            is_live=True,
            timestamp=datetime.now(),
            face_bbox=(50, 30, 100, 100),
        )
        mock_pipeline.process_frame.return_value = [mock_result]

        response = client.post(
            "/api/v1/attendance/record",
            json={
                "image": valid_face_image,
                "class_id": "CLASS101",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["class_id"] == "CLASS101"
        assert len(data["recorded"]) == 1
        assert data["recorded"][0]["student_id"] == "S001"
        assert data["recorded"][0]["confidence"] == 0.92

    def test_attendance_record_no_matches(self, client, mock_pipeline, valid_face_image):
        """Attendance recording returns empty when no recognized faces."""
        mock_pipeline.process_frame.return_value = []

        response = client.post(
            "/api/v1/attendance/record",
            json={
                "image": valid_face_image,
                "class_id": "CLASS101",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recorded"] == []

    def test_attendance_record_spoof_rejected(self, client, mock_pipeline, valid_face_image):
        """Spoofed faces are not included in attendance."""
        # No results when spoof is detected (pipeline filters them out)
        mock_pipeline.process_frame.return_value = []

        response = client.post(
            "/api/v1/attendance/record",
            json={
                "image": valid_face_image,
                "class_id": "CLASS101",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recorded"] == []

    def test_attendance_record_missing_class_id(self, client, mock_pipeline, valid_face_image):
        """Attendance recording fails without class_id."""
        response = client.post(
            "/api/v1/attendance/record",
            json={"image": valid_face_image},
        )

        assert response.status_code == 422  # Validation error

    def test_attendance_record_pipeline_not_ready(self, client_no_pipeline, valid_face_image):
        """Attendance recording fails when pipeline is not ready."""
        response = client_no_pipeline.post(
            "/api/v1/attendance/record",
            json={
                "image": valid_face_image,
                "class_id": "CLASS101",
            },
        )

        assert response.status_code == 503


# =============================================================================
# 5. Student Management Tests (4 tests)
# =============================================================================


class TestStudentManagementEndpoints:
    """Tests for student listing and deletion endpoints."""

    def test_list_students_success(self, client, mock_pipeline):
        """List students returns all enrolled students."""
        response = client.get("/api/v1/students")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert len(data["students"]) == 5
        assert "S001" in data["students"]

    def test_list_students_empty(self, client, mock_pipeline):
        """List students returns empty when no students enrolled."""
        mock_pipeline.get_enrolled_count.return_value = 0
        mock_pipeline._enrolled_students = {}

        response = client.get("/api/v1/students")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["students"] == []

    def test_delete_student_success(self, client, mock_pipeline):
        """Delete student removes enrolled student."""
        mock_pipeline.remove_student.return_value = True

        response = client.delete("/api/v1/students/S001")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["student_id"] == "S001"

    def test_delete_student_not_found(self, client, mock_pipeline):
        """Delete student returns 404 for non-existent student."""
        mock_pipeline.remove_student.return_value = False

        response = client.delete("/api/v1/students/UNKNOWN")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["message"].lower()


# =============================================================================
# 6. Error Handling Tests (4 tests)
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_malformed_json(self, client, mock_pipeline):
        """Malformed JSON returns 422."""
        response = client.post(
            "/api/v1/enroll",
            content="{ this is not valid json }",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_invalid_base64_encoding(self, client, mock_pipeline):
        """Invalid base64 in image field returns 400."""
        response = client.post(
            "/api/v1/recognize",
            json={"image": "!!!invalid-base64!!!"},
        )

        assert response.status_code == 400

    def test_server_error_during_processing(self, client, mock_pipeline, valid_face_image):
        """Server errors during processing return 500."""
        mock_pipeline.process_frame.side_effect = RuntimeError("Unexpected error")

        response = client.post(
            "/api/v1/recognize",
            json={"image": valid_face_image},
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_unknown_endpoint_returns_404(self, client):
        """Unknown endpoints return 404."""
        response = client.get("/api/v1/unknown-endpoint")

        assert response.status_code == 404


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_very_large_image(self, client, mock_pipeline):
        """Large images are handled gracefully."""
        # Create a 2000x2000 image
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode(".jpg", img)
        large_image = base64.b64encode(buffer).decode("utf-8")

        mock_pipeline.process_frame.return_value = []

        response = client.post(
            "/api/v1/recognize",
            json={"image": large_image},
        )

        # Should handle gracefully (either succeed or return appropriate error)
        assert response.status_code in [200, 400, 413]

    def test_student_id_special_characters(self, client, mock_pipeline, valid_face_image):
        """Student IDs with special characters are handled."""
        mock_pipeline.enroll_student.return_value = True

        response = client.post(
            "/api/v1/enroll",
            json={
                "student_id": "S001-TEST_user",
                "photos": [valid_face_image],
            },
        )

        assert response.status_code == 201
        assert response.json()["student_id"] == "S001-TEST_user"

    def test_concurrent_enrollment(self, client, mock_pipeline, valid_face_image):
        """Multiple rapid enrollments don't cause issues."""
        mock_pipeline.enroll_student.return_value = True

        # Simulate rapid sequential requests
        for i in range(3):
            response = client.post(
                "/api/v1/enroll",
                json={
                    "student_id": f"S{i:03d}",
                    "photos": [valid_face_image],
                },
            )
            assert response.status_code == 201
