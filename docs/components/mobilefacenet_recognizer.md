# MobileFaceNetRecognizer

Face recognition component using MobileFaceNet for generating face embeddings.

## Overview

`MobileFaceNetRecognizer` generates 512-dimensional embedding vectors from face images for identity comparison. It implements the `FaceRecognizer` protocol.

## Installation

Download the MobileFaceNet model from InsightFace:
```bash
# Download buffalo_sc model pack
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
unzip buffalo_sc.zip
mv buffalo_sc/w600k_mbf.onnx models/recognition/mobilefacenet.onnx
rm -rf buffalo_sc buffalo_sc.zip
```

## Usage

### Generate Embeddings
```python
from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer
from src.core.interfaces.recognizer import cosine_similarity

# Initialize recognizer
recognizer = MobileFaceNetRecognizer()

# Extract embedding from face crop (112x112 recommended)
embedding = recognizer.extract(face_crop)

print(f"Embedding dimension: {embedding.dimension}")  # 512
print(f"Embedding magnitude: {np.linalg.norm(embedding.vector)}")  # ~1.0
```

### Compare Faces
```python
# Extract embeddings from two face images
emb1 = recognizer.extract(face_crop_1)
emb2 = recognizer.extract(face_crop_2)

# Compare using cosine similarity
similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity:.3f}")  # -1.0 to 1.0

# Typical thresholds:
# > 0.5: Likely same person
# > 0.6: High confidence same person
# < 0.3: Likely different people
```

### Integration with YuNetDetector
```python
from src.infrastructure.ml.yunet_detector import YuNetDetector
from src.infrastructure.ml.mobilefacenet_recognizer import MobileFaceNetRecognizer
import cv2

detector = YuNetDetector()
recognizer = MobileFaceNetRecognizer()

# Detect face
image = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face = detector.detect_largest(image_rgb)

if face:
    # Crop face region
    bbox = face.bounding_box
    face_crop = image_rgb[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]

    # Generate embedding
    embedding = recognizer.extract(face_crop)
    print(f"Generated {embedding.dimension}-dim embedding")
```

## API Reference

### MobileFaceNetRecognizer

#### Constructor
```python
MobileFaceNetRecognizer(
    model_path: str | Path = "models/recognition/mobilefacenet.onnx",
    input_size: tuple[int, int] = (112, 112),
)
```

**Parameters:**
- `model_path`: Path to ONNX model file
- `input_size`: Expected input size (width, height)

**Raises:**
- `FileNotFoundError`: If model file doesn't exist

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `extract(face_image)` | `FaceEmbedding` | Generate embedding from face |
| `compare(emb1, emb2)` | `float` | Cosine similarity between embeddings |
| `match(emb, threshold)` | `MatchResult` | Match against enrolled students |
| `match_top_k(emb, k, threshold)` | `list[MatchCandidate]` | Find top-k matches |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dimension` | `int` | Embedding dimension (512) |

### FaceEmbedding
```python
@dataclass
class FaceEmbedding:
    vector: NDArray[np.floating]  # 512-dim normalized vector
    model_name: str               # "mobilefacenet"

    @property
    def dimension(self) -> int: ...  # len(vector)
```

### Utility Functions
```python
def cosine_similarity(emb1: FaceEmbedding, emb2: FaceEmbedding) -> float:
    """Calculate cosine similarity between embeddings.

    Returns:
        Similarity score between -1.0 and 1.0.
        Returns 0.0 if either vector is zero.
    """
```

## Input Requirements

| Property | Requirement |
|----------|-------------|
| Format | NumPy array (RGB) |
| Ideal Size | 112x112 pixels |
| Dtype | uint8 |
| Content | Cropped, roughly aligned face |

The recognizer automatically:
- Resizes to 112x112
- Converts grayscale to RGB
- Handles RGBA (drops alpha)
- Normalizes pixel values: `(pixel - 127.5) / 128.0`

## Performance

| Metric | Value |
|--------|-------|
| Model Size | 13 MB |
| Embedding Dim | 512 |
| Inference Time | ~10-20ms (CPU) |
| Comparison Time | <1ms |

## Similarity Thresholds

| Threshold | Interpretation |
|-----------|----------------|
| > 0.6 | High confidence match |
| 0.5 - 0.6 | Probable match |
| 0.3 - 0.5 | Uncertain |
| < 0.3 | Different people |

**Note:** Optimal thresholds vary by use case. Tune based on your FAR/FRR requirements.

## Testing
```bash
# Run MobileFaceNet tests
poetry run pytest tests/unit/infrastructure/ml/test_mobilefacenet_recognizer.py -v

# With coverage
poetry run pytest tests/unit/infrastructure/ml/test_mobilefacenet_recognizer.py --cov=src/infrastructure/ml/mobilefacenet_recognizer
```

## See Also

- [ADR-002: MobileFaceNet Selection](../architecture/decisions/002-mobilefacenet-recognition.md)
- [YuNet Detector](./yunet_detector.md)
