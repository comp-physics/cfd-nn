#!/usr/bin/env bash
set -euo pipefail

CASE="${1:-}"
if [[ -z "$CASE" ]]; then
  echo "usage: $0 <case>"
  echo ""
  echo "cases:"
  ls -1 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cases" | sed -n 's/\.cfg$//p'
  exit 2
fi

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

CFG="$EXAMPLE_DIR/cases/${CASE}.cfg"
OUT="$EXAMPLE_DIR/output/${CASE}/"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  exit 2
fi

if [[ ! -x "$BUILD_DIR/periodic_hills" ]]; then
  echo "ERROR: periodic_hills binary not found at: $BUILD_DIR/periodic_hills"
  echo "Build first:"
  echo "  cd $PROJECT_ROOT && mkdir -p build && cd build && cmake .. && make -j4"
  exit 1
fi

mkdir -p "$OUT"

echo "=== Periodic hills example ==="
echo "case:   $CASE"
echo "config: $CFG"
echo "out:    $OUT"
echo ""

cd "$BUILD_DIR"
./periodic_hills --config "$CFG" --output "$OUT" "${@:2}"


