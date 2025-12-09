# Contributing Guidelines

Guide for contributing to the ilQabad project.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Follow [Development Setup](setup.md)
4. Create a feature branch

## Development Workflow

### 1. Create Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow these principles:
- **Test-Driven Development**: Write tests first
- **Clean Architecture**: Respect layer boundaries
- **Single Responsibility**: One purpose per class/function
- **Type Safety**: Use type hints everywhere

### 3. Run Checks
```bash
# Format code
poetry run ruff format .

# Lint
poetry run ruff check .

# Type check
poetry run mypy src/

# Run tests
poetry run pytest
```

### 4. Commit Changes
```bash
# Use conventional commits
git commit -m "feat(ml): add new face detector"
git commit -m "fix(detector): handle empty images"
git commit -m "docs(readme): update installation"
git commit -m "test(recognizer): add edge case tests"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

## Code Style

### Python Style
- Follow PEP 8
- Use Ruff for formatting
- Maximum line length: 100 characters
- Use Google-style docstrings

### Type Hints
```python
def detect(self, image: npt.NDArray[np.uint8]) -> DetectionResult:
    """Detect faces in image.

    Args:
        image: RGB image as numpy array.

    Returns:
        DetectionResult containing detected faces.
    """
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `YuNetDetector` |
| Functions | snake_case | `detect_faces` |
| Constants | UPPER_SNAKE | `DEFAULT_THRESHOLD` |
| Private | _prefix | `_preprocess` |

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

## Pull Request Process

### PR Title
Use conventional commit format:
```
feat(ml): implement YuNet face detector
```

### PR Description Template
```markdown
## Summary
Brief description of changes.

## Changes
- Added X
- Modified Y
- Fixed Z

## Test Plan
- [ ] Unit tests added
- [ ] Integration tests pass
- [ ] Manual testing completed

## Related Issues
Closes #123
```

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage maintained
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Backwards compatible (or breaking changes documented)

## Architecture Guidelines

### Layer Boundaries

```
Presentation → Application → Core ← Infrastructure
```

- **Core** has no external dependencies
- **Application** depends only on Core
- **Infrastructure** implements Core interfaces
- **Presentation** orchestrates Application use cases

### Adding New ML Component

1. Define protocol in `src/core/interfaces/`
2. Write tests in `tests/unit/infrastructure/ml/`
3. Implement in `src/infrastructure/ml/`
4. Create ADR in `docs/architecture/decisions/`
5. Document in `docs/components/`

## Testing Requirements

- Unit tests for all new code
- Integration tests for component interactions
- Performance tests for ML components
- Minimum 85% coverage for PRs

## Documentation

Update docs for:
- New features
- API changes
- Configuration changes
- Breaking changes

## Getting Help

- Open an issue for bugs
- Discussions for questions
- Check existing issues before creating new ones

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.
