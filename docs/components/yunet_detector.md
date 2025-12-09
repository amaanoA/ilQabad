# YuNetDetector

Face detection component using YuNet model via OpenCV's FaceDetectorYN.

## Overview

`YuNetDetector` detects faces in images, returning bounding boxes, 5-point facial landmarks, and confidence scores. It implements the `FaceDetector` protocol.

## Installation

The YuNet model must be downloaded separately:
```bash
curl -L -o models/detection/yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Usage

### Basic Detection
```python
from src.infrastructure.ml.yunet_detector import YuNetDetector

# Initialize detector
detector = YuNetDetector()

# Detect faces in image (RGB numpy array)
result = detector.detect(image)

for face in result.faces:
    print(f"Bounding box: {face.bounding_box}")
    print(f"Confidence: {face.confidence:.2%}")
    print(f"Landmarks: {face.landmarks}")
```

### Detect Largest Face Only
```python
# For single-face applications (e.g., attendance check-in)
face = detector.detect_largest(image)

if face:
    print(f"Found face at {face.bounding_box}")
else:
    print("No face detected")
```

### Custom Configuration
```python
detector = YuNetDetector(
    model_path="models/detection/yunet.onnx",
    confidence_threshold=0.7,  # Minimum confidence (0.0-1.0)
    nms_threshold=0.3,         # Non-maximum suppression IoU threshold
    input_size=(640, 640),     # Model input size
)
```

## API Reference

### YuNetDetector

#### Constructor
```python
YuNetDetector(
    model_path: str | Path = "models/detection/yunet.onnx",
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    input_size: tuple[int, int] = (640, 640),
)
```

**Parameters:**
- `model_path`: Path to ONNX model file
- `confidence_threshold`: Minimum confidence for detections (0.0-1.0)
- `nms_threshold`: IoU threshold for non-maximum suppression
- `input_size`: Internal processing size (width, height)

**Raises:**
- `FileNotFoundError`: If model file doesn't exist

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `detect(image)` | `DetectionResult` | Detect all faces in image |
| `detect_largest(image)` | `DetectedFace \| None` | Detect largest face only |

### DetectionResult
```python
@dataclass
class DetectionResult:
    faces: list[DetectedFace]

    @property
    def largest_face(self) -> DetectedFace | None: ...

    @property
    def face_count(self) -> int: ...
```

### DetectedFace
```python
@dataclass
class DetectedFace:
    bounding_box: BoundingBox
    landmarks: Landmarks
    confidence: float  # 0.0 to 1.0
```

### BoundingBox
```python
@dataclass
class BoundingBox:
    x: int       # Top-left X coordinate
    y: int       # Top-left Y coordinate
    width: int   # Box width
    height: int  # Box height

    @property
    def area(self) -> int: ...
```

### Landmarks

5-point facial landmarks:
```python
@dataclass
class Landmarks:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
```

## Input Requirements

| Property | Requirement |
|----------|-------------|
| Format | NumPy array (RGB) |
| Shape | (height, width, 3) or (height, width) for grayscale |
| Dtype | uint8 |
| Color | RGB (not BGR) |

Supported formats:
- RGB: `(H, W, 3)`
- Grayscale: `(H, W)` - auto-converted to RGB
- RGBA: `(H, W, 4)` - alpha channel dropped

## Performance

| Metric | Value |
|--------|-------|
| Model Size | 228 KB |
| Inference Time | ~5-20ms (CPU) |
| Memory Usage | ~50 MB |

## Limitations

1. **Minimum face size**: Very small faces (<20px) may not be detected
2. **Extreme angles**: Profile faces (>45) have lower accuracy
3. **Occlusion**: Heavily occluded faces may be missed
4. **Lighting**: Very dark images reduce accuracy

## Testing
```bash
# Run YuNetDetector tests
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py -v

# With coverage
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py --cov=src/infrastructure/ml/yunet_detector
```

## See Also

- [ADR-001: YuNet Selection](../architecture/decisions/001-yunet-face-detection.md)
- [MobileFaceNet Recognizer](./mobilefacenet_recognizer.md)
