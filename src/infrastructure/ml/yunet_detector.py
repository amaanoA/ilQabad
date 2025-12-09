"""YuNet face detector implementation using OpenCV's FaceDetectorYN.

This module provides a face detection implementation using the YuNet model
via OpenCV's built-in FaceDetectorYN for detecting faces in images with
bounding boxes, landmarks, and confidence scores.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from src.core.interfaces.detector import (
    BoundingBox,
    DetectedFace,
    DetectionResult,
    Landmarks,
)


class YuNetDetector:
    """YuNet-based face detector using OpenCV's FaceDetectorYN.

    This detector uses the YuNet model via OpenCV to detect faces in images,
    returning bounding boxes, 5-point facial landmarks, and confidence scores.

    Attributes:
        model_path: Path to the ONNX model file.
        confidence_threshold: Minimum confidence score for detections.
        nms_threshold: IoU threshold for non-maximum suppression.
        input_size: Input size for the model as (width, height).
    """

    def __init__(
        self,
        model_path: str | Path = "models/detection/yunet.onnx",
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        """Initialize the YuNet face detector.

        Args:
            model_path: Path to the ONNX model file.
            confidence_threshold: Minimum confidence score for detections.
            nms_threshold: IoU threshold for non-maximum suppression.
            input_size: Input size for the model as (width, height).

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize OpenCV FaceDetectorYN
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            self.input_size,
            self.confidence_threshold,
            self.nms_threshold,
        )

    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
        """Detect all faces in an image.

        Args:
            image: RGB image as numpy array with shape (height, width, 3).

        Returns:
            DetectionResult containing all detected faces.
        """
        # Convert image to BGR format for OpenCV
        bgr_image = self._prepare_image(image)

        # Get image dimensions and set input size
        height, width = bgr_image.shape[:2]
        self._detector.setInputSize((width, height))

        # Run detection
        _, raw_detections = self._detector.detect(bgr_image)

        # Convert to list format for _postprocess
        if raw_detections is None:
            outputs: list[npt.NDArray[np.floating]] = [
                np.array([], dtype=np.float32).reshape(0, 15)
            ]
        else:
            outputs = [raw_detections.astype(np.float32)]

        # Postprocess the results (no scaling needed - coordinates are in original size)
        faces = self._postprocess(outputs, (width, height))

        return DetectionResult(faces=faces)

    def detect_largest(self, image: npt.NDArray[np.uint8]) -> DetectedFace | None:
        """Detect and return only the largest face in an image.

        Args:
            image: RGB image as numpy array with shape (height, width, 3).

        Returns:
            The largest DetectedFace by bounding box area, or None if no faces.
        """
        result = self.detect(image)
        return result.largest_face

    def _prepare_image(self, image: npt.NDArray[np.uint8]) -> Any:
        """Prepare image for OpenCV detection.

        Args:
            image: Input image as numpy array.

        Returns:
            BGR image suitable for OpenCV FaceDetectorYN.
        """
        # Handle grayscale images
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Handle RGBA images
        elif image.shape[2] == 4:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
        # Handle RGB images - convert to BGR for OpenCV
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Preprocess image for model inference.

        This method is kept for backward compatibility with tests.
        OpenCV's FaceDetectorYN handles preprocessing internally.

        Args:
            image: Input image as numpy array.

        Returns:
            Preprocessed image tensor with shape (1, 3, height, width).
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

        # Convert to float32
        normalized = resized.astype(np.float32)

        # Transpose from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _postprocess(
        self, outputs: list[npt.NDArray[np.floating]], original_size: tuple[int, int]
    ) -> list[DetectedFace]:
        """Postprocess model outputs to extract detected faces.

        Args:
            outputs: Raw model outputs from FaceDetectorYN.
            original_size: Original image size as (width, height).

        Returns:
            List of DetectedFace objects.
        """
        # Get the detection output
        # FaceDetectorYN output format: [num_detections, 15]
        # 15 values: x, y, w, h, right_eye(x,y), left_eye(x,y), nose(x,y),
        #            mouth_right(x,y), mouth_left(x,y), confidence
        detections = outputs[0]

        if detections.size == 0:
            return []

        # Remove batch dimension if present
        if detections.ndim == 3:
            detections = detections[0]

        if len(detections) == 0:
            return []

        # Calculate scale factors for coordinate conversion
        original_width, original_height = original_size
        input_width, input_height = self.input_size
        scale_x = original_width / input_width
        scale_y = original_height / input_height

        faces = []
        for det in detections:
            # Check confidence threshold
            confidence = float(det[14])
            if confidence < self.confidence_threshold:
                continue

            # Extract bounding box (already in original image coordinates from detect())
            # But for _postprocess called directly with scaled inputs, we need to scale
            x = float(det[0]) * scale_x
            y = float(det[1]) * scale_y
            w = float(det[2]) * scale_x
            h = float(det[3]) * scale_y

            # Ensure non-negative coordinates
            x = max(0.0, x)
            y = max(0.0, y)

            # Ensure positive dimensions
            w = max(1.0, w)
            h = max(1.0, h)

            bounding_box = BoundingBox(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
            )

            # Extract landmarks with correct OpenCV order and scale
            # OpenCV order: right_eye, left_eye, nose, mouth_right, mouth_left
            right_eye = (float(det[4]) * scale_x, float(det[5]) * scale_y)
            left_eye = (float(det[6]) * scale_x, float(det[7]) * scale_y)
            nose = (float(det[8]) * scale_x, float(det[9]) * scale_y)
            mouth_right = (float(det[10]) * scale_x, float(det[11]) * scale_y)
            mouth_left = (float(det[12]) * scale_x, float(det[13]) * scale_y)

            landmarks = Landmarks(
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_left=mouth_left,
                mouth_right=mouth_right,
            )

            faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    landmarks=landmarks,
                    confidence=confidence,
                )
            )

        return faces
