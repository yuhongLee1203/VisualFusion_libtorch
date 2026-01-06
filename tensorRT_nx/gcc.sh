#!/bin/bash
# Clean the build directory to ensure a fresh build
echo "🧹 Cleaning build directory..."
rm -rf ./build
mkdir -p ./build

# Navigate to the build directory
cd ./build

# Run CMake and Make
echo "🛠️ Running CMake..."
cmake ..
echo "🏗️ Building project with Make..."
make -j$(nproc)


