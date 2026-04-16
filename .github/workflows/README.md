# GitHub Workflows

This directory contains GitHub Actions workflows for CI/CD.

## Workflows

### test.yml

Runs automated tests on every push and pull request.

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests targeting these branches

**Test Matrix:**
- Python versions: 3.10, 3.11
- Operating systems: Ubuntu, Windows, macOS

**Steps:**
1. Checkout code
2. Set up Python
3. Install dependencies
4. Lint with flake8 (syntax check)
5. Run pytest with coverage
6. Upload coverage to Codecov

### Adding New Tests

1. Create test file in `tests/` directory
2. Name it `test_*.py`
3. Run tests locally: `pytest`

### Running Tests Locally

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model_utils.py

# Run specific test
pytest tests/test_model_utils.py::TestBuildModel
```

### Skipping CI for Documentation Changes

Add `[ci skip]` or `[skip ci]` to your commit message to skip CI for documentation-only changes.
