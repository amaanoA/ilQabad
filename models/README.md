# ONNX Models for ilQabad

This directory contains the ONNX models required for the ilQabad face recognition attendance system.

## Directory Structure

```
models/
├── detection/
│   └── yunet.onnx           # Face detection model
├── recognition/
│   └── mobilefacenet.onnx   # Face embedding/recognition model
├── liveness/
│   ├── minifas_2_7.onnx     # Anti-spoofing (2.7x scale)
│   └── minifas_4_0.onnx     # Anti-spoofing (4.0x scale)
└── README.md
```

## Automatic Download

Run the download script to automatically fetch all models:

```bash
poetry run python scripts/download_models.py
```

To verify existing models without downloading:

```bash
poetry run python scripts/download_models.py --verify
```

## Manual Download Instructions

If automatic download fails (due to network restrictions), download manually:

### 1. YuNet Face Detection

- **File**: `models/detection/yunet.onnx`
- **Size**: ~233 KB
- **License**: Apache-2.0
- **Source**: [OpenCV Zoo](https://github.com/opencv/opencv_zoo)

**Download Options:**
1. GitHub: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   - Click "Download raw file" button
2. Clone the repo:
   ```bash
   git clone --depth 1 https://github.com/opencv/opencv_zoo.git /tmp/opencv_zoo
   cp /tmp/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx models/detection/yunet.onnx
   ```

**Model Details:**
- Input: `[1, 3, H, W]` - RGB image (dynamic height/width)
- Output: Face bounding boxes with 5-point landmarks
- Performance: 0.834/0.824/0.708 on WIDER Face (easy/medium/hard)

### 2. MobileFaceNet / ArcFace Recognition

- **File**: `models/recognition/mobilefacenet.onnx`
- **Size**: ~12-240 MB (varies by variant)
- **License**: MIT
- **Source**: [InsightFace](https://github.com/deepinsight/insightface) / [ONNX Model Zoo](https://github.com/onnx/models)

**Download Options:**
1. ONNX Model Zoo ArcFace:
   - https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface
2. InsightFace models (need conversion):
   - https://github.com/deepinsight/insightface/tree/master/model_zoo
3. Pre-converted options:
   - yakhyo's repo: https://github.com/yakhyo/face-reidentification
   - Hugging Face: https://huggingface.co/models?search=arcface+onnx

**Recommended Models:**
| Model | Size | Embedding Dim | Notes |
|-------|------|---------------|-------|
| MobileFaceNet | ~12 MB | 512 | Fast, good for edge devices |
| ArcFace R50 | ~166 MB | 512 | Better accuracy |
| ArcFace R100 | ~249 MB | 512 | Best accuracy |

**Model Details:**
- Input: `[1, 3, 112, 112]` - Aligned face image (RGB, normalized)
- Output: `[1, 512]` - Face embedding vector
- Use cosine similarity for face matching

### 3. MiniFASNet Anti-Spoofing

- **Files**: `models/liveness/minifas_2_7.onnx`, `models/liveness/minifas_4_0.onnx`
- **Size**: ~400 KB each
- **License**: Apache-2.0
- **Source**: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

**Download Options:**
1. ONNX Conversions:
   - Kazuhito00's ONNX version: https://github.com/Kazuhito00/Silent-Face-Anti-Spoofing-ONNX
   - hairymax's conversion: https://github.com/hairymax/Face-AntiSpoofing
2. Convert from PyTorch yourself:
   ```python
   import torch
   from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

   # Load PyTorch model
   model = MiniFASNetV2(conv6_kernel=(5, 5))
   model.load_state_dict(torch.load("2.7_80x80_MiniFASNetV2.pth"))
   model.eval()

   # Export to ONNX
   dummy_input = torch.randn(1, 3, 80, 80)
   torch.onnx.export(model, dummy_input, "minifas_2_7.onnx", opset_version=11)
   ```

**Model Variants:**
| Model | Scale | Input Size | Description |
|-------|-------|------------|-------------|
| MiniFASNetV2 | 2.7x | 80x80 | Faster, slightly less accurate |
| MiniFASNetV1SE | 4.0x | 80x80 | More accurate, includes SE blocks |

**Model Details:**
- Input: `[1, 3, 80, 80]` - Cropped face region (RGB)
- Output: `[1, 3]` - [fake_prob, real_prob, spoof_type]
- Multi-scale fusion recommended for best results

## Model Verification

After downloading, verify all models load correctly:

```bash
poetry run python scripts/download_models.py --verify
```

Expected output:
```
[OK] yunet: 233.2 KB
[OK] mobilefacenet: 12.1 MB
[OK] minifas_2_7: 435.6 KB
[OK] minifas_4_0: 412.3 KB
```

## Usage Example

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("models/detection/yunet.onnx")

# Get input/output info
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# Run inference
image = np.random.randn(1, 3, 320, 320).astype(np.float32)
results = session.run(output_names, {input_name: image})
```

## License Information

| Model | License | Commercial Use |
|-------|---------|----------------|
| YuNet | Apache-2.0 | Yes |
| ArcFace/MobileFaceNet | MIT | Yes |
| MiniFASNet | Apache-2.0 | Yes |

All models are compatible with commercial use in the ilQabad attendance system.

## Troubleshooting

### Model fails to load
- Ensure ONNX Runtime is installed: `poetry install`
- Check file isn't corrupted: re-download if size seems wrong
- Verify ONNX opset version compatibility

### Download fails
- Check network connectivity
- Try alternative download sources listed above
- Use a VPN if GitHub is blocked

### Performance issues
- Use ONNX Runtime with GPU: `pip install onnxruntime-gpu`
- Quantize models for faster inference (int8 versions available)
- Reduce input resolution if needed
