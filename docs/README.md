# ilQabad Documentation

Face recognition attendance system built with Clean Architecture and TDD.

## Quick Links

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/overview.md) | System design |
| [Development Setup](development/setup.md) | Getting started |
| [Testing Guide](development/testing.md) | How to test |

## Components

| Component | Status | Docs |
|-----------|--------|------|
| YuNetDetector | Complete | [Guide](components/yunet_detector.md) |
| MobileFaceNetRecognizer | Complete | [Guide](components/mobilefacenet_recognizer.md) |
| DeePixBiSLiveness | Planned | TBD |

## Architecture Decisions

| ADR | Title | Status |
|-----|-------|--------|
| [001](architecture/decisions/001-yunet-face-detection.md) | YuNet for Face Detection | Accepted |
| [002](architecture/decisions/002-mobilefacenet-recognition.md) | MobileFaceNet for Recognition | Accepted |

## Directory Structure

```
docs/
├── README.md                          # This file
├── architecture/
│   ├── overview.md                    # System architecture
│   ├── clean-architecture.md          # Clean Architecture explanation
│   └── decisions/                     # ADRs (Architecture Decision Records)
│       ├── 001-yunet-face-detection.md
│       ├── 002-mobilefacenet-recognition.md
│       └── template.md
├── components/
│   ├── README.md                      # Components overview
│   ├── yunet_detector.md              # YuNet documentation
│   └── mobilefacenet_recognizer.md    # MobileFaceNet documentation
├── development/
│   ├── setup.md                       # Development environment setup
│   ├── testing.md                     # Testing guide
│   └── contributing.md                # Contribution guidelines
└── api/
    └── README.md                      # API reference (future)
```

## Getting Started

### Prerequisites
- Python 3.12+
- Poetry for dependency management
- OpenCV 4.5.4+

### Quick Start
```bash
# Clone repository
git clone https://github.com/amaanoA/ilQabad.git
cd ilQabad

# Install dependencies
poetry install

# Download models
./scripts/download_models.sh

# Run tests
poetry run pytest
```

## Project Status

| Metric | Value |
|--------|-------|
| Test Coverage | 94% |
| Tests Passing | 98 |
| Phase | 2 (ML Components) |

## License

MIT License - see [LICENSE](../LICENSE) for details.
