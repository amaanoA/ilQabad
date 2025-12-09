# DeePixBiSLiveness

Face anti-spoofing component using DeePixBiS (Deep Pixel-wise Binary Supervision) model.

## Overview

`DeePixBiSLiveness` detects spoof attacks (photos, videos, masks) to ensure only live faces pass verification. It implements the `LivenessChecker` protocol and provides both binary classification and spatial liveness maps.

## Installation

The DeePixBiS model must be downloaded separately:
```bash
# Download from model repository
curl -L -o models/liveness/deeppixbis.onnx \
  https://github.com/your-repo/models/deeppixbis.onnx
```

## Usage

### Basic Liveness Check
```python
from src.infrastructure.ml.deeppixbis_liveness import DeePixBiSLiveness

# Initialize detector
liveness = DeePixBiSLiveness()

# Check if face is live (RGB face crop)
result = liveness.check(face_crop)

if result.is_live:
    print(f"Live face detected (confidence: {result.confidence:.2%})")
else:
    print(f"Spoof detected! Type: {result.spoof_type.name}")
```

### Access Spatial Liveness Map
```python
result = liveness.check(face_crop)

# Get 14x14 pixel-wise liveness map
if result.pixel_map is not None:
    # Each pixel indicates liveness probability (0.0-1.0)
    center_liveness = result.pixel_map[7, 7]
    print(f"Center region liveness: {center_liveness:.2%}")
```

### Custom Configuration
```python
liveness = DeePixBiSLiveness(
    model_path="models/liveness/deeppixbis.onnx",
    liveness_threshold=0.5,  # Threshold for live/spoof decision
)
```

### Adjusting Threshold
```python
# High security (fewer false accepts)
strict_liveness = DeePixBiSLiveness(liveness_threshold=0.7)

# Convenience mode (fewer false rejects)
relaxed_liveness = DeePixBiSLiveness(liveness_threshold=0.3)
```

## API Reference

### DeePixBiSLiveness

#### Constructor
```python
DeePixBiSLiveness(
    model_path: str | Path = "models/liveness/deeppixbis.onnx",
    liveness_threshold: float = 0.5,
)
```

**Parameters:**
- `model_path`: Path to ONNX model file
- `liveness_threshold`: Decision threshold (0.0-1.0). Scores >= threshold are live.

**Raises:**
- `FileNotFoundError`: If model file doesn't exist

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `check(face_image)` | `LivenessResult` | Check if face is live |
| `check_with_depth(face_image, depth_map)` | `LivenessResult` | Check with optional depth (depth ignored) |

### LivenessResult
```python
@dataclass
class LivenessResult:
    is_live: bool              # True if live face
    confidence: float          # Liveness score (0.0-1.0)
    spoof_type: SpoofType      # Type of spoof if detected
    pixel_map: NDArray | None  # 14x14 spatial liveness map
```

### SpoofType
```python
class SpoofType(Enum):
    NONE = auto()      # Not a spoof (live face)
    PHOTO = auto()     # Printed photo attack
    VIDEO = auto()     # Video replay attack
    MASK = auto()      # 3D mask attack
    UNKNOWN = auto()   # Spoof type undetermined
```

## Input Requirements

| Property | Requirement |
|----------|-------------|
| Format | NumPy array (RGB) |
| Shape | (height, width, 3) |
| Dtype | uint8 |
| Content | Cropped face image |
| Recommended Size | 224x224 (auto-resized if different) |

Supported formats:
- RGB: `(H, W, 3)` - standard input
- Grayscale: `(H, W)` - auto-converted to RGB
- RGBA: `(H, W, 4)` - alpha channel dropped

## Preprocessing Pipeline

1. **Channel handling**: Convert grayscale/RGBA to RGB
2. **Resize**: Scale to 224x224 (bilinear interpolation)
3. **Normalize**: Apply ImageNet statistics
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
4. **Transpose**: HWC to CHW format
5. **Batch**: Add batch dimension `[1, 3, 224, 224]`

## Model Architecture

DeePixBiS uses a ResNet-18 backbone with pixel-wise binary supervision:

| Component | Details |
|-----------|---------|
| Backbone | ResNet-18 |
| Input | 224x224x3 RGB |
| Output 1 | Binary score [1, 1] |
| Output 2 | Pixel map [1, 1, 14, 14] |

## Performance

| Environment | Inference Time |
|-------------|---------------|
| Production (4+ cores) | ~100-200ms |
| Development (2-4 cores) | ~300-500ms |
| CI/Codespace (2 cores) | ~800-1200ms |

| Metric | Value |
|--------|-------|
| Model Size | ~44 MB |
| Memory Usage | ~150 MB |

**Note:** DeePixBiS uses a ResNet backbone which is more compute-intensive than lightweight detection models like YuNet.

## Threshold Tuning

| Threshold | Use Case | FAR | FRR |
|-----------|----------|-----|-----|
| 0.3 | Convenience | Higher | Lower |
| 0.5 | Balanced (default) | Medium | Medium |
| 0.7 | Security | Lower | Higher |
| 0.9 | High Security | Lowest | Highest |

- **FAR** (False Accept Rate): Spoof attacks incorrectly accepted
- **FRR** (False Reject Rate): Live faces incorrectly rejected

## Limitations

1. **Compute intensive**: ResNet backbone requires more CPU than detection
2. **Face crop required**: Works best on properly cropped face images
3. **Lighting sensitivity**: Extreme lighting may affect accuracy
4. **Novel attacks**: May not detect unseen spoof types
5. **No depth support**: Depth map parameter ignored (single-image model)

## Testing
```bash
# Run DeePixBiSLiveness tests
poetry run pytest tests/unit/infrastructure/ml/test_deeppixbis_liveness.py -v

# Run with benchmark output
poetry run pytest tests/unit/infrastructure/ml/test_deeppixbis_liveness.py -v -s

# With coverage
poetry run pytest tests/unit/infrastructure/ml/test_deeppixbis_liveness.py \
  --cov=src/infrastructure/ml/deeppixbis_liveness
```

## Diagnostics

Run performance diagnostics to understand timing on your hardware:
```bash
poetry run python scripts/diagnose_deeppixbis_performance.py
```

## See Also

- [ADR-003: DeePixBiS Selection](../architecture/decisions/003-deeppixbis-liveness.md)
- [YuNet Detector](./yunet_detector.md)
- [MobileFaceNet Recognizer](./mobilefacenet_recognizer.md)
