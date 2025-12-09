# API Reference

API documentation for ilQabad components.

## Protocols

| Protocol | Description | Location |
|----------|-------------|----------|
| `FaceDetector` | Face detection interface | `src/core/interfaces/detector.py` |
| `FaceRecognizer` | Face recognition interface | `src/core/interfaces/recognizer.py` |
| `LivenessDetector` | Liveness detection interface | `src/core/interfaces/liveness.py` |
| `Camera` | Camera capture interface | `src/core/interfaces/camera.py` |

## Data Classes

### Detection

- `BoundingBox` - Face bounding box coordinates
- `Landmarks` - 5-point facial landmarks
- `DetectedFace` - Complete face detection result
- `DetectionResult` - Collection of detected faces

### Recognition

- `FaceEmbedding` - 512-dimensional face embedding
- `MatchResult` - Result of embedding comparison
- `MatchCandidate` - Candidate match with confidence

## See Also

- [Component Documentation](../components/README.md)
- [Architecture Decisions](../architecture/decisions/)
