#!/bin/bash
# Quick setup and validation script

set -e

echo "=========================================="
echo "MT Benchmarking - Setup & Validation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

# Run validation tests
echo ""
echo "Running validation tests..."
python test_validation.py

# Success
echo ""
echo "=========================================="
echo "✓ Setup complete! Ready to benchmark."
echo "=========================================="
echo ""
echo "Quick start:"
echo "  python main.py                                # Run full benchmark"
echo "  python main.py --models nllb_200 m2m_100     # Test specific models"
echo "  python main.py --help                         # Show all options"
echo ""
