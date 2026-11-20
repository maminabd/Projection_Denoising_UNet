#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e 
echo "Running tests with coverage..."

# Clean up any previous coverage data
coverage erase

# Run all tests in the 'tests/' folder 
# along with coverage measurement, by using the specified .coveragerc configuration file
coverage run --rcfile=.coveragerc -m pytest tests/

# Generate a coverage report in the terminal with missing lines
coverage report -m

# Generate an HTML coverage report in the 'htmlcov/' directory
coverage html

echo "All tests completed."
echo "Open 'htmlcov/index.html' to view the coverage report."
echo "Open 'htmlcov/function_index.html' to view the coverage report for each function."
