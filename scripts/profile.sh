#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
REPORT_DIR="${PROJECT_ROOT}/reports"

REPORT_NAME="fps_nsys"
REPORT_BASE="${REPORT_DIR}/${REPORT_NAME}"

NSYS_BIN=$(command -v nsys || true)

if [[ -z "$NSYS_BIN" ]]; then
    echo "nsys not found"
    exit 1
fi

bash "${SCRIPT_DIR}/build.sh"

mkdir -p "${REPORT_DIR}"

# Run profiling
cd "${BUILD_DIR}"

$NSYS_BIN profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --force-overwrite=true \
    -o "${REPORT_BASE}" \
    ./fps | tee "${REPORT_BASE}.log"

# Stats
$NSYS_BIN stats "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_stats.txt"

echo ""
echo "Report: ${REPORT_BASE}.nsys-rep"
echo "Stats:  ${REPORT_BASE}_stats.txt"
echo "Log:    ${REPORT_BASE}.log"