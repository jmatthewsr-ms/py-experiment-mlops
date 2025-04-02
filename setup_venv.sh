#!/bin/bash
echo "Creating Python virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

echo "Setup complete! You can now run:"
echo "jupyter notebook"
