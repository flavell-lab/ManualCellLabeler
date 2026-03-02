#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:?Usage: $0 <data_dir> [output_dir]}"
OUTPUT_DIR="${2:-$DATA_DIR/relabelled}"
# Auto-pick a free port in 3001-3030, or use PORT env var
if [ -n "${PORT:-}" ]; then
    if ss -tln | grep -q ":$PORT "; then
        echo "Error: port $PORT is already in use" >&2
        exit 1
    fi
else
    PORT=""
    for p in $(seq 3001 3030); do
        if ! ss -tln | grep -q ":$p "; then
            PORT=$p
            break
        fi
    done
    if [ -z "$PORT" ]; then
        echo "Error: no free port in range 3001-3030" >&2
        exit 1
    fi
fi

PODMAN_ROOT="/tmp/$(whoami)-podman-storage/root"
PODMAN_RUNROOT="/tmp/$(whoami)-podman-storage/runroot"
IMAGE="manual-cell-labeler"

# Build if image doesn't exist
if ! podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" image exists "$IMAGE" 2>/dev/null; then
    echo "Building image (first run or after reboot)..."
    podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" build -t "$IMAGE" "$(dirname "$0")"
fi

echo "Starting ManualCellLabeler on port $PORT"
echo "  Data dir:   $DATA_DIR (read-only)"
echo "  Output dir: $OUTPUT_DIR (read-write)"
echo "  URL:        http://localhost:$PORT"

mkdir -p "$OUTPUT_DIR"

podman --root "$PODMAN_ROOT" --runroot "$PODMAN_RUNROOT" run --rm -p "$PORT:8501" \
    -v "$DATA_DIR:$DATA_DIR:ro" \
    -v "$OUTPUT_DIR:$OUTPUT_DIR" \
    "$IMAGE"
