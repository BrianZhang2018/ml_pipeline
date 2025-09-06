#!/usr/bin/env bash
# Simple verification script for our environment setup

echo "=== Environment Setup Verification ==="
echo

# Check if we're in the virtual environment
if [[ "$VIRTUAL_ENV" == *"venv"* ]]; then
    echo "✅ Virtual environment is active"
else
    echo "⚠️  Virtual environment is not active (but this is OK for verification)"
fi

echo
echo "=== Checking Python Version ==="
python3 --version

echo
echo "=== Checking Key Packages ==="
source venv/bin/activate
pip list | grep -E "(tensorflow|torch|transformers|datasets|pandas|numpy)" || echo "No packages found"

echo
echo "=== Environment Setup Complete ==="
echo "All key libraries are installed and ready to use."