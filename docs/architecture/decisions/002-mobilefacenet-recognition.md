# [ADR-002] Use MobileFaceNet for Face Recognition Embeddings

## Status
Accepted

## Context
After face detection, the system needs to:
- Generate unique embeddings (feature vectors) for each face
- Compare embeddings to identify/verify individuals
- Support real-time recognition in attendance workflow

Requirements:
- 512-dimensional embeddings (industry standard)
- L2 normalized for cosine similarity comparison
- Fast inference for real-time use
- Good accuracy on Asian and diverse faces
- Reasonable model size for deployment

## Decision
Use MobileFaceNet (w600k_mbf) from InsightFace's buffalo_sc model pack.

**Model Details:**
- Source: InsightFace buffalo_sc
- Size: 13MB
- License: MIT
- Input: [1, 3, 112, 112] aligned face crop, RGB
- Output: [1, 512] L2-normalized embedding
- Training: WebFace600K dataset (600K identities)

**Implementation:**
- Wrap in `MobileFaceNetRecognizer` class implementing `FaceRecognizer` protocol
- Use ONNX Runtime for inference
- Normalize input: (pixel - 127.5) / 128.0
- L2 normalize output embeddings

## Consequences

### Positive
- Industry-standard 512-dim embeddings
- MobileNet backbone = fast inference (~10-20ms)
- Trained on large-scale dataset (600K identities)
- MIT license allows commercial use
- Good accuracy across ethnicities

### Negative
- Requires aligned face crops (112x112)
- 13MB larger than detection model
- Less accurate than larger ArcFace models

### Neutral
- ONNX format = portable across frameworks

## Alternatives Considered

| Model | Embedding Dim | Size | Accuracy | Verdict |
|-------|---------------|------|----------|---------|
| ArcFace-R100 | 512 | ~250MB | Highest | Too large |
| ArcFace-R50 | 512 | ~166MB | Very High | Still large |
| **MobileFaceNet** | 512 | **13MB** | Good | **Selected** |
| FaceNet | 128/512 | ~90MB | Good | Larger, older |
| SFace | 512 | ~40MB | Good | Less tested |

## References
- [InsightFace](https://github.com/deepinsight/insightface)
- [MobileFaceNets Paper](https://arxiv.org/abs/1804.07573)
- [buffalo_sc Model Pack](https://github.com/deepinsight/insightface/tree/master/model_zoo)
