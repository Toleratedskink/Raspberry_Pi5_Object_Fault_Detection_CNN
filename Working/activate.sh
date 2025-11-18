#!/bin/bash
# Quick activation script for the virtual environment
# Usage: source activate.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"
echo "Virtual environment activated!"
echo "You can now run: python main.py --mode train"

