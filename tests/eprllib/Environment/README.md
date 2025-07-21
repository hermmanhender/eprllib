# Environment Tests

This directory contains tests for the Environment module of the eprllib package.

## Test Files

- `test_Environment.py`: Tests for the `Environment` class
- `test_EnvironmentConfig.py`: Tests for the `EnvironmentConfig` class
- `test_EnvironmentRunner.py`: Tests for the `EnvironmentRunner` class

## Running the Tests

To run the tests, use pytest:

```bash
# Run all tests in the Environment directory
pytest tests/eprllib/Environment

# Run a specific test file
pytest tests/eprllib/Environment/test_Environment.py

# Run a specific test
pytest tests/eprllib/Environment/test_Environment.py::TestEnvironment::test_environment_initialization
```

## Test Coverage

The tests cover the following functionality:

### Environment Class
- Initialization
- Reset method
- Step method
- Close method
- Handling of simulation completion and failures
- Episode truncation
- Observation history

### EnvironmentConfig Class
- Initialization
- Configuration methods (generals, agents, connector, episodes)
- Configuration validation
- Dictionary-like access

### EnvironmentRunner Class
- Initialization
- Start and stop methods
- Handling of simulation failures
- EnergyPlus argument generation
- Progress handling
- Queue management