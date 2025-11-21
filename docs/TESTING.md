# Blasphemer Testing Guide

Comprehensive testing suite for Blasphemer to ensure reliability and correctness.

## Overview

Blasphemer includes a multi-tier testing framework:

- **Unit Tests**: Test individual components (utils, config)
- **Integration Tests**: Test system interactions (checkpoints, environment)
- **Shell Script Tests**: Validate all bash scripts

## Quick Start

### Run All Tests

```bash
./run_tests.sh
```

### Run Specific Test Suites

```bash
# Unit tests only
./run_tests.sh --unit

# Integration tests only
./run_tests.sh --integration

# Shell script tests only
./run_tests.sh --scripts

# With coverage report
./run_tests.sh --coverage
```

## Installation

### Install Test Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install -e ".[test]"

# Or for development (includes linters)
pip install -e ".[dev]"
```

## Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_config.py         # Configuration tests
│   └── test_utils.py          # Utility function tests
├── integration/               # Integration tests
│   └── test_checkpoint_system.py  # Checkpoint system tests
└── scripts/                   # Shell script tests
    └── test_shell_scripts.sh  # Bash script validation
```

## Unit Tests

### Configuration Tests (`test_config.py`)

Tests for the Pydantic configuration system:

- Default settings validation
- Checkpoint directory naming (uses `blasphemer`)
- Environment variable prefix (`BLASPHEMER_`)
- Refusal markers configuration
- Data type settings
- Batch size and trial validation

**Run:**
```bash
pytest tests/unit/test_config.py -v
```

### Utility Tests (`test_utils.py`)

Tests for utility functions:

- `format_duration()` - Time formatting
- `batchify()` - List batching
- `empty_cache()` - GPU cache management
- `get_readme_intro()` - README generation with proper attribution

**Run:**
```bash
pytest tests/unit/test_utils.py -v
```

## Integration Tests

### Checkpoint System Tests (`test_checkpoint_system.py`)

Tests for the checkpoint/resume functionality:

- Checkpoint directory creation
- SQLite database operations
- Checkpoint naming conventions (`blasphemer_` prefix)
- Resume detection
- Environment variable handling
- Model naming conventions

**Run:**
```bash
pytest tests/integration/ -v
```

## Shell Script Tests

### Script Validation (`test_shell_scripts.sh`)

Validates all bash scripts:

**Checks:**
- ✓ File exists
- ✓ Executable permissions
- ✓ Valid bash syntax
- ✓ Proper shebang (`#!/usr/bin/env bash`)
- ✓ Has help/usage information
- ✓ Error handling (`set -e`, exit codes)
- ✓ Blasphemer branding (no outdated "Heretic" references)
- ✓ Help flag works (`--help`)

**Scripts Tested:**
- `blasphemer.sh`
- `convert-to-gguf.sh`
- `install-macos.sh`

**Run:**
```bash
./tests/scripts/test_shell_scripts.sh
```

## Coverage Reports

### Generate Coverage Report

```bash
./run_tests.sh --coverage
```

This creates:
- Terminal output with line coverage
- HTML report at `htmlcov/index.html`

### View HTML Coverage Report

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

## Writing New Tests

### Unit Test Example

```python
# tests/unit/test_new_feature.py
import pytest
from heretic.new_module import new_function

class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_operation(self):
        """Test basic operation of new feature."""
        result = new_function("input")
        assert result == "expected"
    
    def test_error_handling(self):
        """Test that errors are handled correctly."""
        with pytest.raises(ValueError):
            new_function(None)
```

### Integration Test Example

```python
# tests/integration/test_new_integration.py
import tempfile
from pathlib import Path

class TestNewIntegration:
    """Test integration between components."""
    
    def setup_method(self):
        """Create test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_component_interaction(self):
        """Test that components work together."""
        # Test code here
        assert True
```

### Shell Script Test Example

```bash
# Add to tests/scripts/test_shell_scripts.sh

test_new_script_feature() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if grep -q "expected_feature" "$script_path"; then
        test_pass "$script_name has expected feature"
        return 0
    else
        test_fail "$script_name has expected feature" "Feature not found"
        return 1
    fi
}
```

## Continuous Integration

### Pre-commit Testing

Before committing code, run:

```bash
./run_tests.sh
```

All tests should pass before pushing changes.

### Automated Testing

The test suite is designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -e ".[test]"
      - name: Run tests
        run: ./run_tests.sh
```

## Test Markers

Pytest markers for selective testing:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only GPU-dependent tests (if GPU available)
pytest -m requires_gpu
```

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure Blasphemer is installed in development mode
pip install -e .
```

### Shell Script Tests Fail

```bash
# Ensure scripts are executable
chmod +x blasphemer.sh convert-to-gguf.sh install-macos.sh
```

### Coverage Report Not Generated

```bash
# Install coverage dependencies
pip install pytest-cov
```

### Tests Pass Locally But Fail in CI

- Check Python version compatibility (3.10+)
- Verify all dependencies are in `pyproject.toml`
- Ensure test environment matches CI environment

## Best Practices

### Do:
- ✅ Write tests for new features
- ✅ Update tests when changing functionality
- ✅ Run tests before committing
- ✅ Aim for high code coverage
- ✅ Test edge cases and error handling
- ✅ Use descriptive test names
- ✅ Keep tests independent

### Don't:
- ❌ Skip failing tests
- ❌ Commit without running tests
- ❌ Test implementation details
- ❌ Create interdependent tests
- ❌ Leave commented-out test code

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage of `src/heretic/`
- **Integration Tests**: Cover critical workflows
- **Shell Scripts**: 100% script validation

## Reporting Issues

If you find a bug or test failure:

1. Run tests to confirm: `./run_tests.sh`
2. Check if it's a known issue
3. Create detailed bug report with:
   - Test output
   - System information (`uname -a`, Python version)
   - Steps to reproduce

## Contributing Tests

When contributing:

1. Add tests for new features
2. Update existing tests if changing behavior
3. Ensure all tests pass: `./run_tests.sh`
4. Check coverage: `./run_tests.sh --coverage`
5. Document new test patterns

## Resources

- **pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Blasphemer Source**: `src/heretic/`

---

For questions or issues with testing, please open an issue at:
https://github.com/sunkencity999/blasphemer/issues
