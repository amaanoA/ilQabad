# [ADR-003] Use DeePixBiS for Face Anti-Spoofing

## Status
Accepted

## Context
The ilQabad attendance system requires face anti-spoofing (liveness detection) to prevent:
- Photo attacks (printed photos held up to camera)
- Video replay attacks (face video played on screen)
- 3D mask attacks (silicone/paper masks)

Requirements:
- Real-time performance (<1.5s per check on limited hardware)
- High accuracy on common attack types (photos, screens)
- Single RGB image input (no depth sensor required)
- Pixel-wise liveness map for visualization/debugging
- Permissive license for commercial use

## Decision
Use DeePixBiS (Deep Pixel-wise Binary Supervision) for face anti-spoofing.

**Model Details:**
- Architecture: ResNet-18 backbone with pixel-wise supervision
- Size: ~44MB ONNX
- License: MIT
- Input: 224x224 RGB face crop
- Outputs:
  - Binary liveness score [1, 1]
  - Spatial liveness map [1, 1, 14, 14]

**Implementation:**
- Wrap in `DeePixBiSLiveness` class implementing `LivenessChecker` protocol
- Use ONNX Runtime for inference (CPUExecutionProvider)
- Apply ImageNet normalization in preprocessing
- Environment-aware performance thresholds

## Consequences

### Positive
- **Pixel-wise supervision**: Provides spatial liveness map for debugging
- **Good generalization**: Trained on multiple attack types
- **Single-image**: No depth sensor or multi-frame analysis required
- **MIT license**: No commercial restrictions
- **Interpretable output**: Can visualize which regions appear spoofed

### Negative
- **Compute intensive**: ResNet-18 backbone requires ~100-1000ms (CPU-dependent)
- **Model size**: 44MB larger than lightweight alternatives
- **No attack type classification**: Binary output only (live vs spoof)

### Neutral
- Requires face cropping before liveness check (integrates with YuNet)
- May need threshold tuning per deployment environment

## Alternatives Considered

| Model | Accuracy | Speed | Size | Pixel Map | Verdict |
|-------|----------|-------|------|-----------|---------|
| MiniFASNet | Good | Fast | ~1MB | No | Too simple for varied attacks |
| Silent-Face-Anti-Spoofing | High | Medium | ~20MB | No | Complex multi-scale fusion |
| FAS-SGTD | High | Slow | ~100MB | Yes | Too slow, research-only license |
| CDCN | High | Medium | ~50MB | Yes | Complex training, no pretrained |
| **DeePixBiS** | Good | Medium | ~44MB | **Yes** | **Selected** |

### Why not MiniFASNet?
- Only 1MB but lower accuracy on diverse attacks
- No spatial liveness information
- Overfits to specific datasets

### Why not Silent-Face-Anti-Spoofing?
- Better accuracy but complex multi-scale architecture
- Harder to debug without pixel-wise output
- Less interpretable decisions

### Why DeePixBiS?
1. **Interpretability**: Pixel-wise map shows which regions are suspicious
2. **Balanced trade-off**: Good accuracy with reasonable compute
3. **Proven architecture**: ResNet-18 is well-understood and optimized
4. **Training flexibility**: Can fine-tune on local attack samples if needed

## Performance Characteristics

| CPU Cores | Expected Latency |
|-----------|-----------------|
| 2 (CI/Codespace) | 800-1200ms |
| 4 (Development) | 300-500ms |
| 8+ (Production) | 100-200ms |

Tests use environment-aware thresholds to accommodate different hardware.

## Integration Notes

```python
# Typical usage in face verification pipeline
detector = YuNetDetector()
recognizer = MobileFaceNetRecognizer()
liveness = DeePixBiSLiveness(liveness_threshold=0.5)

# 1. Detect face
result = detector.detect(frame)
if not result.faces:
    raise NoFaceDetectedError()

# 2. Crop face
face_crop = crop_face(frame, result.faces[0].bounding_box)

# 3. Check liveness BEFORE recognition
liveness_result = liveness.check(face_crop)
if not liveness_result.is_live:
    raise SpoofDetectedError(liveness_result.spoof_type)

# 4. Extract embedding and match
embedding = recognizer.extract(face_crop)
match = recognizer.match(embedding)
```

## References
- [DeePixBiS Paper](https://arxiv.org/abs/1908.03850) - "Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection"
- [Face Anti-Spoofing Survey](https://arxiv.org/abs/2106.14948)
- [OULU-NPU Dataset](https://sites.google.com/site/aboraborabor/oulu-npu-database)
