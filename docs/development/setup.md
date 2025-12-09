# Development Setup

Guide for setting up the ilQabad development environment.

## Prerequisites

- Python 3.12 or higher
- Poetry (dependency management)
- Git
- OpenCV 4.5.4+ (for FaceDetectorYN support)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/amaanoA/ilQabad.git
cd ilQabad
```

### 2. Install Dependencies
```bash
# Install all dependencies including dev tools
poetry install

# Activate virtual environment
poetry shell
```

### 3. Download Models

Download the required ML models:

```bash
# Create model directories
mkdir -p models/detection models/recognition models/liveness

# Download YuNet (face detection)
curl -L -o models/detection/yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Download MobileFaceNet (face recognition)
# Option 1: From InsightFace
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
unzip buffalo_sc.zip
mv buffalo_sc/w600k_mbf.onnx models/recognition/mobilefacenet.onnx
rm -rf buffalo_sc buffalo_sc.zip
```

### 4. Verify Installation
```bash
# Run tests
poetry run pytest

# Check linting
poetry run ruff check .
poetry run mypy src/
```

## IDE Setup

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

### PyCharm

1. Open project directory
2. Configure Poetry interpreter: Settings > Project > Python Interpreter
3. Enable Ruff: Settings > Tools > Ruff

## Project Structure

```
ilQabad/
├── src/
│   ├── core/              # Domain layer
│   │   ├── entities/      # Domain entities
│   │   ├── interfaces/    # Protocols/interfaces
│   │   └── services/      # Domain services
│   ├── infrastructure/    # Infrastructure layer
│   │   ├── ml/            # ML implementations
│   │   ├── camera/        # Camera implementations
│   │   └── storage/       # Database implementations
│   ├── application/       # Application layer
│   │   └── use_cases/     # Use cases
│   └── presentation/      # Presentation layer
│       ├── cli/           # CLI interface
│       └── gui/           # GUI interface
├── tests/                 # Test suite
├── models/                # ML models (not in git)
├── docs/                  # Documentation
└── pyproject.toml         # Project configuration
```

## Common Commands

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py -v

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Lint code
poetry run ruff check .
poetry run ruff format .

# Type check
poetry run mypy src/

# Pre-commit hooks
poetry run pre-commit run --all-files
```

## Troubleshooting

### OpenCV FaceDetectorYN not found
Ensure OpenCV 4.5.4+ is installed:
```bash
python -c "import cv2; print(cv2.__version__)"
```

### Model file not found
Verify models are downloaded:
```bash
ls -la models/detection/
ls -la models/recognition/
```

## Next Steps

- Read the [Testing Guide](testing.md)
- Review [Architecture Decisions](../architecture/decisions/)
- Explore [Component Documentation](../components/README.md)
