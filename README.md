# ilQabad (عين قبض) - Eye Catch

Face Recognition Attendance System for Somali Schools.

## Overview

ilQabad automates student attendance using face recognition technology, designed for deployment on low-cost MaixCAM Pro devices ($55) in Somali schools.

## Features

- Face detection with hijab support
- Face recognition (≥95% accuracy)
- Anti-spoofing (photo/video attack detection)
- Offline-first with cloud sync
- Integration with KaamilSMS

## Requirements

- Python 3.12+
- Poetry

## Setup
```bash
# Install dependencies
poetry install

# Run tests
make test

# Run linting
make lint
```

## Architecture

Clean Architecture with 4 layers:
- **Core**: Domain entities, interfaces, business logic
- **Infrastructure**: Camera, ML models, database, API clients
- **Application**: Use cases orchestrating domain logic
- **Presentation**: CLI and GUI interfaces

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/README.md) directory:

| Document | Description |
|----------|-------------|
| [Components](docs/components/README.md) | ML component guides |
| [YuNetDetector](docs/components/yunet_detector.md) | Face detection |
| [MobileFaceNetRecognizer](docs/components/mobilefacenet_recognizer.md) | Face recognition |
| [Architecture Decisions](docs/architecture/decisions/) | ADRs |

## License

MIT
