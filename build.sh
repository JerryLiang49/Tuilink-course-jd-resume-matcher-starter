#!/bin/bash

# Build script for AI Auto Reply project
# Creates a dist folder with all necessary files and folders

set -e  # Exit on any error

echo "🚀 Starting build process..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"

# Remove existing dist folder if it exists
if [ -d "$DIST_DIR" ]; then
    echo "Removing existing dist folder..."
    rm -rf "$DIST_DIR"
fi

# Create dist folder
mkdir -p "$DIST_DIR"
echo "Created $DIST_DIR"

# Copy individual files
echo "Copying files..."
files=(".env" "quick_handler.py" "worker_handler.py")
for file in "${files[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        cp "$SCRIPT_DIR/$file" "$DIST_DIR/"
        echo "  ✓ Copied $file"
    else
        echo "  ⚠ Warning: $file not found"
    fi
done

# Copy and rename requirements-compact.txt to requirements.txt
if [ -f "$SCRIPT_DIR/requirements-compact.txt" ]; then
    cp "$SCRIPT_DIR/requirements-compact.txt" "$DIST_DIR/requirements.txt"
    echo "  ✓ Copied requirements-compact.txt as requirements.txt"
else
    echo "  ⚠ Warning: requirements-compact.txt not found"
fi

# Copy folders
echo "Copying folders..."
folders=("matcher" "models" "nodes" "utils")
for folder in "${folders[@]}"; do
    if [ -d "$SCRIPT_DIR/$folder" ]; then
        # Create destination folder
        mkdir -p "$DIST_DIR/$folder"
        
        # Copy contents, excluding __pycache__ and .DS_Store
        find "$SCRIPT_DIR/$folder" -type f \( -name "*.py" -o -name "*.json" -o -name "*.xlsx" -o -name "*.txt" -o -name "*.md" \) -exec cp {} "$DIST_DIR/$folder/" \;
        
        echo "  ✓ Copied $folder/"
    else
        echo "  ⚠ Warning: $folder/ not found"
    fi
done

# Create __init__.py files in Python package folders
echo "Creating __init__.py files..."
python_packages=("models" "nodes" "utils")
for package in "${python_packages[@]}"; do
    if [ -d "$DIST_DIR/$package" ]; then
        touch "$DIST_DIR/$package/__init__.py"
        echo "  ✓ Created $package/__init__.py"
    fi
done

# Calculate total size
total_size=$(du -sh "$DIST_DIR" | cut -f1)
echo ""
echo "✅ Build completed! Distribution package created in $DIST_DIR"
echo "📦 Total size: $total_size"

# List contents
echo ""
echo "📋 Contents of $DIST_DIR:"
for item in "$DIST_DIR"/*; do
    if [ -f "$item" ]; then
        size=$(du -h "$item" | cut -f1)
        echo "  📄 $(basename "$item") ($size)"
    elif [ -d "$item" ]; then
        count=$(find "$item" -type f | wc -l)
        echo "  📁 $(basename "$item")/ ($count items)"
    fi
done

echo ""
echo "🎉 Build successful! Your distribution package is ready in: $DIST_DIR" 