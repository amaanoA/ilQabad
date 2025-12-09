# Testing Guide

Comprehensive guide for testing ilQabad components.

## Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── core/                  # Domain layer tests
│   └── infrastructure/
│       └── ml/                # ML component tests
│           ├── test_yunet_detector.py
│           └── test_mobilefacenet_recognizer.py
├── integration/               # Integration tests
└── conftest.py                # Shared fixtures
```

## Running Tests

### All Tests
```bash
poetry run pytest
```

### Specific Test File
```bash
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py -v
```

### Specific Test Class
```bash
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py::TestYuNetDetectorDetection -v
```

### Specific Test
```bash
poetry run pytest tests/unit/infrastructure/ml/test_yunet_detector.py::TestYuNetDetectorDetection::test_detect_on_blank_image_returns_no_faces -v
```

### With Coverage
```bash
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Parallel Execution
```bash
poetry run pytest -n auto
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
def test_detector_initializes_with_default_path(
    self, yunet_detector: "YuNetDetector"
) -> None:
    """Test detector initializes with default model path."""
    assert yunet_detector.model_path == DEFAULT_MODEL_PATH
```

### Integration Tests

Test component interactions:

```python
def test_detection_to_recognition_pipeline(
    self, detector, recognizer, test_image
) -> None:
    """Test full detection to recognition pipeline."""
    face = detector.detect_largest(test_image)
    assert face is not None

    embedding = recognizer.extract(face_crop)
    assert embedding.dimension == 512
```

### Performance Tests

Verify performance requirements:

```python
def test_detection_completes_within_200ms(
    self, detector, test_image
) -> None:
    """Test detection completes within 200ms."""
    times = []
    for _ in range(10):
        start = time.perf_counter()
        detector.detect(test_image)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    assert avg_time < 200
```

## Writing Tests

### Test Naming Convention
```python
def test_<method>_<scenario>_<expected_result>(self):
    """<Clear description of what is being tested>."""
```

Examples:
- `test_detect_on_blank_image_returns_no_faces`
- `test_extract_returns_512_dimensional_embedding`
- `test_preprocess_handles_grayscale_input`

### Fixture Pattern
```python
@pytest.fixture
def yunet_detector() -> "YuNetDetector":
    """Create a YuNetDetector with default settings."""
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)

    from src.infrastructure.ml.yunet_detector import YuNetDetector
    return YuNetDetector()
```

### Graceful Skipping
```python
MODEL_PATH = Path("models/detection/yunet.onnx")
MODEL_EXISTS = MODEL_PATH.exists()
SKIP_NO_MODEL = "Model not available"

def test_requires_model(self, detector):
    if not MODEL_EXISTS:
        pytest.skip(SKIP_NO_MODEL)
    # Test code...
```

## Test Data

### Synthetic Images
```python
@pytest.fixture
def blank_image() -> npt.NDArray[np.uint8]:
    """Create a blank 640x640 RGB image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def noise_image() -> npt.NDArray[np.uint8]:
    """Create random noise image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
```

### Real Test Images
```python
TEST_FACE_PATH = Path("tests/data/faces/face_001.jpg")

@pytest.fixture
def real_face_image():
    if not TEST_FACE_PATH.exists():
        pytest.skip("Test face image not found")
    img = cv2.imread(str(TEST_FACE_PATH))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

## Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| Core Domain | 95% |
| ML Components | 90% |
| Use Cases | 90% |
| Infrastructure | 85% |

## Continuous Integration

Tests run automatically on:
- Pull requests
- Merges to main branch

CI Configuration (`.github/workflows/test.yml`):
```yaml
- name: Run tests
  run: poetry run pytest --cov=src --cov-fail-under=85
```

## Debugging Tests

### Verbose Output
```bash
poetry run pytest -v --tb=long
```

### Print Statements
```bash
poetry run pytest -s
```

### Debug Single Test
```bash
poetry run pytest tests/path/to/test.py::test_name -v --tb=long -s
```

### PDB Breakpoint
```python
def test_something():
    import pdb; pdb.set_trace()
    # Test code...
```

## Best Practices

1. **One assertion per test** (when practical)
2. **Descriptive test names** that explain intent
3. **Use fixtures** for common setup
4. **Test edge cases** explicitly
5. **Keep tests fast** (<100ms per unit test)
6. **Isolate tests** - no dependencies between tests
7. **Use type hints** in test code
8. **Document fixtures** with docstrings

## See Also

- [Development Setup](setup.md)
- [Contributing Guidelines](contributing.md)
