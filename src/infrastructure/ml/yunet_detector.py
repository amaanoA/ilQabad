"""YuNet face detector implementation using ONNX Runtime.

This module provides a face detection implementation using the YuNet model
for detecting faces in images with bounding boxes, landmarks, and confidence scores.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.core.interfaces.detector import (
    BoundingBox,
    DetectedFace,
    DetectionResult,
    Landmarks,
)


class YuNetDetector:
    """YuNet-based face detector using ONNX Runtime for inference.

    This detector uses the YuNet model to detect faces in images, returning
    bounding boxes, 5-point facial landmarks, and confidence scores.

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

        # Initialize ONNX Runtime session
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
        """Detect all faces in an image.

        Args:
            image: RGB image as numpy array with shape (height, width, 3).

        Returns:
            DetectionResult containing all detected faces.
        """
        # Get original image size (width, height)
        if image.ndim == 2:
            original_height, original_width = image.shape
        else:
            original_height, original_width = image.shape[:2]
        original_size = (original_width, original_height)

        # Preprocess the image
        input_tensor = self._preprocess(image)

        # Run inference
        outputs = self._session.run(None, {self._input_name: input_tensor})

        # Postprocess the results
        faces = self._postprocess(outputs, original_size)

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

    def _preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Preprocess image for model inference.

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

        # Convert to float32 and normalize
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
            outputs: Raw model outputs.
            original_size: Original image size as (width, height).

        Returns:
            List of DetectedFace objects.
        """
        # Get the detection output
        # YuNet output format: [batch, num_detections, 15]
        # 15 values: x, y, w, h, 5 landmark pairs (x,y), confidence
        detections = outputs[0]

        if detections.size == 0:
            return []

        # Remove batch dimension if present
        if detections.ndim == 3:
            detections = detections[0]

        if len(detections) == 0:
            return []

        # Calculate scale factors
        original_width, original_height = original_size
        input_width, input_height = self.input_size
        scale_x = original_width / input_width
        scale_y = original_height / input_height

        # Filter by confidence threshold first
        confidences = detections[:, 14]
        valid_mask = confidences >= self.confidence_threshold
        valid_detections = detections[valid_mask]

        if len(valid_detections) == 0:
            return []

        # Apply NMS
        boxes_for_nms = valid_detections[:, :4].copy()
        # Scale boxes for NMS
        boxes_for_nms[:, 0] *= scale_x
        boxes_for_nms[:, 1] *= scale_y
        boxes_for_nms[:, 2] *= scale_x
        boxes_for_nms[:, 3] *= scale_y

        confidences_for_nms = valid_detections[:, 14]

        # Convert to format expected by cv2.dnn.NMSBoxes: (x, y, w, h)
        boxes_list = boxes_for_nms.tolist()
        confidences_list = confidences_for_nms.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_list,
            confidences_list,
            self.confidence_threshold,
            self.nms_threshold,
        )

        # Handle different return types from NMSBoxes
        if len(indices) == 0:
            return []

        # Flatten indices if needed (OpenCV versions differ in return format)
        indices = indices.flatten() if isinstance(indices, np.ndarray) else list(indices)

        faces = []
        for idx in indices:
            det = valid_detections[idx]

            # Extract and scale bounding box
            x = det[0] * scale_x
            y = det[1] * scale_y
            w = det[2] * scale_x
            h = det[3] * scale_y

            # Ensure non-negative coordinates
            x = max(0, x)
            y = max(0, y)

            # Ensure positive dimensions
            w = max(1, w)
            h = max(1, h)

            bounding_box = BoundingBox(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
            )

            # Extract and scale landmarks
            # Landmarks are at indices 4-13 in pairs (x, y)
            left_eye = (det[4] * scale_x, det[5] * scale_y)
            right_eye = (det[6] * scale_x, det[7] * scale_y)
            nose = (det[8] * scale_x, det[9] * scale_y)
            mouth_left = (det[10] * scale_x, det[11] * scale_y)
            mouth_right = (det[12] * scale_x, det[13] * scale_y)

            landmarks = Landmarks(
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_left=mouth_left,
                mouth_right=mouth_right,
            )

            # Extract confidence
            confidence = float(det[14])

            faces.append(
                DetectedFace(
                    bounding_box=bounding_box,
                    landmarks=landmarks,
                    confidence=confidence,
                )
            )

        return faces
