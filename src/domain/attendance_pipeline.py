"""Default implementation of the AttendancePipeline.

This module provides the main attendance processing pipeline that orchestrates
face detection, liveness checking, and recognition for the attendance system.
"""

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from src.core.exceptions import (
    InsufficientImagesError,
    StudentAlreadyEnrolledError,
)
from src.core.interfaces.detector import FaceDetector
from src.core.interfaces.liveness import LivenessChecker
from src.core.interfaces.recognizer import FaceEmbedding, FaceRecognizer
from src.core.services.attendance import AttendanceResult


class DefaultAttendancePipeline:
    """Default implementation of attendance processing pipeline.

    This pipeline orchestrates:
    1. Face detection using YuNet or similar detector
    2. Liveness checking using DeePixBiS or similar
    3. Face recognition using MobileFaceNet or similar

    Only live faces that match enrolled students produce attendance results.

    Attributes:
        detector: Face detector implementation.
        recognizer: Face recognizer implementation.
        liveness_checker: Liveness checker implementation.
        recognition_threshold: Minimum confidence for recognition match.
        liveness_threshold: Minimum confidence for liveness check.
    """

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        liveness_checker: LivenessChecker,
        recognition_threshold: float = 0.6,
        liveness_threshold: float = 0.5,
    ) -> None:
        """Initialize the attendance pipeline.

        Args:
            detector: Face detector for finding faces in frames.
            recognizer: Face recognizer for matching against enrolled students.
            liveness_checker: Liveness checker for anti-spoofing.
            recognition_threshold: Minimum confidence for a recognition match.
            liveness_threshold: Minimum liveness confidence to accept face.
        """
        self._detector = detector
        self._recognizer = recognizer
        self._liveness_checker = liveness_checker
        self.recognition_threshold = recognition_threshold
        self.liveness_threshold = liveness_threshold

        # Enrollment database: student_id -> list of embeddings
        self._enrolled_students: dict[str, list[FaceEmbedding]] = {}

    def process_frame(
        self, frame: npt.NDArray[np.uint8]
    ) -> list[AttendanceResult]:
        """Process a camera frame and return attendance results.

        Flow:
        1. Detect faces in frame
        2. For each face, check liveness
        3. If live, extract embedding and match against enrolled students
        4. Return results for matched students

        Args:
            frame: RGB image as numpy array with shape (height, width, 3).

        Returns:
            List of AttendanceResult for each recognized live face.
            Empty list if no faces, all spoofs, or no matches.
        """
        # Handle empty or invalid frames
        if frame.size == 0:
            return []

        # Detect faces
        detection_result = self._detector.detect(frame)

        if not detection_result.has_faces:
            return []

        results: list[AttendanceResult] = []

        for face in detection_result.faces:
            # Extract face crop from frame
            face_crop = self._crop_face(frame, face.bounding_box)

            # Check liveness
            liveness_result = self._liveness_checker.check(face_crop)

            # Skip if liveness confidence is below threshold
            # When threshold=0.0, this effectively bypasses liveness checking
            if liveness_result.confidence < self.liveness_threshold:
                continue

            # Extract embedding for recognition
            embedding = self._recognizer.extract(face_crop)

            # Match against enrolled students
            match_result = self._recognizer.match(
                embedding, threshold=self.recognition_threshold
            )

            # Skip if no match
            if not match_result.matched or match_result.student_id is None:
                continue

            # Verify student is still enrolled
            if match_result.student_id not in self._enrolled_students:
                continue

            # Create attendance result
            bbox = face.bounding_box
            result = AttendanceResult(
                student_id=match_result.student_id,
                confidence=match_result.confidence,
                is_live=True,
                timestamp=datetime.now(),
                face_bbox=(bbox.x, bbox.y, bbox.width, bbox.height),
            )
            results.append(result)

        return results

    def enroll_student(
        self, student_id: str, photos: list[npt.NDArray[np.uint8]]
    ) -> bool:
        """Enroll a student with their face photos.

        Args:
            student_id: Unique identifier for the student.
            photos: List of RGB face photos for enrollment.

        Returns:
            True if enrollment was successful.

        Raises:
            StudentAlreadyEnrolledError: If student is already enrolled.
            InsufficientImagesError: If no valid photos provided.
        """
        if not photos:
            raise InsufficientImagesError(provided=0, required=1)

        if student_id in self._enrolled_students:
            raise StudentAlreadyEnrolledError(student_id)

        embeddings: list[FaceEmbedding] = []

        for photo in photos:
            # Detect face in photo
            detection_result = self._detector.detect(photo)

            if detection_result.has_faces:
                # Use largest face
                face = detection_result.largest_face
                if face is not None:
                    face_crop = self._crop_face(photo, face.bounding_box)
                    embedding = self._recognizer.extract(face_crop)
                    embeddings.append(embedding)

        if not embeddings:
            raise InsufficientImagesError(provided=0, required=1)

        self._enrolled_students[student_id] = embeddings
        return True

    def remove_student(self, student_id: str) -> bool:
        """Remove a student from the enrollment database.

        Args:
            student_id: ID of the student to remove.

        Returns:
            True if student was removed, False if not found.
        """
        if student_id in self._enrolled_students:
            del self._enrolled_students[student_id]
            return True
        return False

    def get_enrolled_count(self) -> int:
        """Get the number of enrolled students.

        Returns:
            Count of students currently enrolled.
        """
        return len(self._enrolled_students)

    def _crop_face(
        self, frame: npt.NDArray[np.uint8], bbox: Any
    ) -> npt.NDArray[np.uint8]:
        """Crop face region from frame.

        Args:
            frame: Full image frame.
            bbox: Bounding box with x, y, width, height.

        Returns:
            Cropped face image.
        """
        h, w = frame.shape[:2] if frame.ndim >= 2 else (0, 0)

        # Get bounding box coordinates
        x = max(0, bbox.x)
        y = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)

        # Handle edge cases
        if x >= x2 or y >= y2:
            # Return small placeholder if bbox is invalid
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Crop the face region
        if frame.ndim == 2:
            # Grayscale - convert to RGB
            face_crop = frame[y:y2, x:x2]
            face_crop = np.stack([face_crop] * 3, axis=-1)
        else:
            face_crop = frame[y:y2, x:x2]

        return face_crop
