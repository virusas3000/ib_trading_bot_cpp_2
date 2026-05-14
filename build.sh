#!/usr/bin/env bash
# build.sh — compile C++ trading_engine module and copy to project root
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/cpp/build"

echo "==> Installing/upgrading pybind11..."
pip3 install --quiet --upgrade pybind11

echo "==> Configuring CMake..."
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR/cpp" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR"

echo "==> Building..."
cmake --build "$BUILD_DIR" --config Release -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"

echo "==> Installing to project root..."
cmake --install "$BUILD_DIR"

# pybind11 names it trading_engine.cpython-*.so — create a stable symlink
SO_FILE=$(find "$SCRIPT_DIR" -maxdepth 1 -name "trading_engine*.so" | head -1)
if [[ -n "$SO_FILE" ]]; then
    echo "==> Built: $SO_FILE"
    echo "==> Build complete. Import with: import trading_engine"
else
    echo "ERROR: .so file not found after build" >&2
    exit 1
fi
