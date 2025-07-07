#!/bin/bash
set -e  # exit on first error

# Optional: echo current working directory
echo "Running from: $(pwd)"

# Install Python dependencies
pip install -r requirements.txt

# Run your automation
python main.py
