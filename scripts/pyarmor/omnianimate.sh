#!/bin/bash

# Define a global variable for the output directory
OUTPUT_DIR="/home/ubuntu/DiskW/OmniAnimate"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Generate files with PyArmor
pyarmor gen -O "$OUTPUT_DIR" --exclude "*.pyc" -r ./omni_animate
pyarmor gen -O "$OUTPUT_DIR" api.py
pyarmor gen -O "$OUTPUT_DIR" webui.py

# Copy assets and third-party directories
cp -r ./assets "$OUTPUT_DIR/"
cp -r ./third_party "$OUTPUT_DIR/"