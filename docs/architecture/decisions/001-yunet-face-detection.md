# [ADR-001] Use YuNet via cv2.FaceDetectorYN for Face Detection

## Status
Accepted

## Context
The ilQabad attendance system requires real-time face detection for:
- Identifying faces in camera frames
- Extracting facial landmarks for alignment
- Providing bounding boxes for face recognition pipeline

Requirements:
- Real-time performance (<200ms per frame)
- High accuracy on frontal and slightly angled faces
- 5-point facial landmarks (eyes, nose, mouth corners)
- Lightweight for potential edge deployment
- Permissive license for commercial use

## Decision
Use YuNet face detector via OpenCV's `cv2.FaceDetectorYN` API.

**Model Details:**
- Source: OpenCV Zoo
- Size: 228KB
- License: Apache 2.0
- Input: RGB image (any size)
- Output: Bounding boxes + 5 landmarks + confidence

**Implementation:**
- Wrap in `YuNetDetector` class implementing `FaceDetector` protocol
- Use OpenCV's built-in decoder (not raw ONNX) for simplicity
- Handle grayscale, RGBA, and various image sizes

## Consequences

### Positive
- Extremely lightweight (228KB vs 100MB+ for alternatives)
- Fast inference (~1.6ms on modern CPU)
- OpenCV handles complex anchor decoding internally
- Battle-tested by OpenCV community
- 81.1% mAP on WIDER FACE hard set

### Negative
- Lower accuracy than RetinaFace-ResNet50 on very small faces
- Requires OpenCV 4.5.4+ for FaceDetectorYN API
- Less control over internal parameters than raw ONNX

### Neutral
- 5 landmarks sufficient for basic alignment (not 68/106 point)

## Alternatives Considered

| Model | Accuracy | Speed | Size | Verdict |
|-------|----------|-------|------|---------|
| RetinaFace-ResNet50 | Highest | Slow | ~100MB | Too slow for real-time |
| MTCNN | Good | Medium | ~2MB | Cascade architecture complex |
| MediaPipe Face | Fast | Fast | ~5MB | Misses faces at different scales |
| SCRFD | High | Fast | ~10MB | Adds InsightFace dependency |
| **YuNet** | Good | **Fastest** | **228KB** | **Selected** |

## References
- [YuNet Paper](https://link.springer.com/article/10.1007/s11633-023-1423-y)
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- [LearnOpenCV Comparison](https://learnopencv.com/what-is-face-detection-the-ultimate-guide/)
