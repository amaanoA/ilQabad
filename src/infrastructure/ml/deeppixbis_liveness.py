"""DeePixBiS liveness detection implementation.

This module provides anti-spoofing liveness detection using the DeePixBiS
ONNX model. DeePixBiS (Deep Pixel-wise Binary Supervision) is a lightweight
face anti-spoofing model that provides both binary liveness classification
and pixel-wise liveness maps.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.core.interfaces.liveness import LivenessResult, SpoofType


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeePixBiSLiveness:
    """DeePixBiS-based liveness detection using ONNX Runtime.

    This implementation uses the DeePixBiS model for face anti-spoofing.
    The model provides both a binary liveness score and a spatial pixel-wise
    liveness map for detailed analysis.

    Attributes:
        model_path: Path to the ONNX model file.
        liveness_threshold: Threshold for liveness classification (default 0.5).

    Example:
        >>> detector = DeePixBiSLiveness()
        >>> result = detector.check(face_image)
        >>> print(f"Live: {result.is_live}, Score: {result.confidence:.2f}")
    """

    def __init__(
        self,
        model_path: Path | str = Path("models/liveness/deeppixbis.onnx"),
        liveness_threshold: float = 0.5,
    ) -> None:
        """Initialize DeePixBiS liveness detector.

        Args:
            model_path: Path to the ONNX model file.
            liveness_threshold: Threshold for liveness decision (0.0-1.0).
                Scores >= threshold are classified as live.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model_path = Path(model_path)
        self.liveness_threshold = liveness_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Initialize ONNX Runtime session
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names from the model
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Create a mapping for output names to indices
        self._output_indices = {name: i for i, name in enumerate(self._output_names)}

    def check(self, face_image: npt.NDArray[np.uint8]) -> LivenessResult:
        """Check if a face image is live or a spoof.

        Args:
            face_image: RGB face crop as numpy array.
                Expected shape: (height, width, 3) or (height, width).

        Returns:
            LivenessResult with is_live decision, confidence score,
            spoof type, and optional pixel map.
        """
        # Preprocess the image
        input_tensor = self._preprocess(face_image)

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # Parse outputs - DeePixBiS has two outputs:
        # output_pixel: [1, 1, 14, 14] - spatial liveness map
        # output_binary: [1, 1] - binary liveness score
        # Use output names to get correct indices
        if "output_binary" in self._output_indices:
            binary_idx = self._output_indices["output_binary"]
            pixel_idx = self._output_indices["output_pixel"]
        else:
            # Fallback: assume pixel is first, binary is second based on shape
            if outputs[0].shape[-1] == 14:
                pixel_idx, binary_idx = 0, 1
            else:
                pixel_idx, binary_idx = 1, 0

        binary_output = outputs[binary_idx]
        pixel_output = outputs[pixel_idx]

        # Extract liveness score
        raw_score = float(binary_output.flatten()[0])

        # Check if output is logit or probability
        # If value is outside [0, 1], it's a logit and needs sigmoid
        if raw_score < 0.0 or raw_score > 1.0:
            confidence = 1.0 / (1.0 + np.exp(-raw_score))
        else:
            confidence = raw_score

        # Ensure confidence is in valid range
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Extract pixel map and squeeze to 2D
        pixel_map = pixel_output.squeeze()  # [14, 14]

        # Apply sigmoid if values are outside [0, 1]
        if pixel_map.min() < 0.0 or pixel_map.max() > 1.0:
            pixel_map = 1.0 / (1.0 + np.exp(-pixel_map))

        pixel_map = pixel_map.astype(np.float32)

        # Determine liveness based on threshold
        is_live = confidence >= self.liveness_threshold

        # Determine spoof type
        spoof_type = SpoofType.NONE if is_live else SpoofType.UNKNOWN

        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            spoof_type=spoof_type,
            pixel_map=pixel_map,
        )

    def check_with_depth(
        self,
        face_image: npt.NDArray[np.uint8],
        depth_map: npt.NDArray[np.float32] | None = None,
    ) -> LivenessResult:
        """Check liveness with optional depth map.

        Note: DeePixBiS does not use depth information. This method
        is provided for protocol compliance and simply calls check().

        Args:
            face_image: RGB face crop as numpy array.
            depth_map: Ignored by this implementation.

        Returns:
            LivenessResult from standard check() method.
        """
        return self.check(face_image)

    def _preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Preprocess face image for model input.

        Performs the following operations:
        1. Converts grayscale to RGB if needed
        2. Removes alpha channel if present
        3. Resizes to 224x224
        4. Converts to float32 and normalizes to [0, 1]
        5. Applies ImageNet normalization
        6. Transposes from HWC to CHW format
        7. Adds batch dimension

        Args:
            image: Input image as numpy array.

        Returns:
            Preprocessed tensor with shape [1, 3, 224, 224].
        """
        import cv2

        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Handle RGBA images
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize to model input size (224x224)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize to [0, 1]
        image_float = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD

        # Transpose from HWC to CHW
        image_float = np.transpose(image_float, (2, 0, 1))

        # Add batch dimension: [1, 3, 224, 224]
        input_tensor = np.expand_dims(image_float, axis=0)

        return input_tensor.astype(np.float32)
