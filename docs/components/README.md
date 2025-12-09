# Components

This directory contains documentation for the ML components of the ilQabad attendance system.

## Face Detection

| Component | Model | Status | Docs |
|-----------|-------|--------|------|
| [YuNetDetector](yunet_detector.md) | YuNet | Complete | [Guide](yunet_detector.md) |

## Face Recognition

| Component | Model | Status | Docs |
|-----------|-------|--------|------|
| [MobileFaceNetRecognizer](mobilefacenet_recognizer.md) | MobileFaceNet | Complete | [Guide](mobilefacenet_recognizer.md) |

## Liveness Detection

| Component | Model | Status | Docs |
|-----------|-------|--------|------|
| DeePixBiSLiveness | DeePixBiS | Planned | TBD |

## Pipeline Overview

```
Camera Frame (RGB)
       |
       v
+----------------+
| YuNetDetector  |  --> BoundingBox, Landmarks, Confidence
+----------------+
       |
       v
   Face Crop
       |
       v
+----------------------+
| MobileFaceNetRecog.  |  --> 512-dim Embedding
+----------------------+
       |
       v
  Embedding DB
       |
       v
   Match Result
```

## Model Files

Models are stored in the `models/` directory:

```
models/
├── detection/
│   └── yunet.onnx           # YuNet face detector (228KB)
├── recognition/
│   └── mobilefacenet.onnx   # MobileFaceNet (13MB)
└── liveness/
    └── (planned)
```

## See Also

- [Architecture Overview](../architecture/overview.md)
- [Architecture Decisions](../architecture/decisions/)
