#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
REPORT_DIR="${PROJECT_ROOT}/reports"
REPORT_NAME="fps_nsys"
REPORT_BASE="${REPORT_DIR}/${REPORT_NAME}"
BUNDLE_PATH="${REPORT_DIR}/fps_profile_bundle.tar.gz"

NSYS_BIN="${NSYS_BIN:-}"
if [[ -z "${NSYS_BIN}" ]]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS_BIN="$(command -v nsys)"
    elif compgen -G "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys" >/dev/null; then
        NSYS_BIN="$(ls -1 /opt/nvidia/nsight-systems/*/target-linux-x64/nsys | tail -n 1)"
    fi
fi

if [[ -z "${NSYS_BIN}" || ! -x "${NSYS_BIN}" ]]; then
    cat >&2 <<'EOF'
nsys was not found on this server/container.

Try these on the server:
  command -v nsys
  find /opt /usr/local/cuda -name nsys -type f 2>/dev/null
  module avail nsight 2>/dev/null

If your cluster uses modules, load Nsight Systems, for example:
  module load nsight-systems

If nsys exists but is not on PATH, run this script with:
  NSYS_BIN=/full/path/to/nsys bash scripts/profile.sh
EOF
    exit 1
fi

bash "${SCRIPT_DIR}/build.sh"
mkdir -p "${REPORT_DIR}"
rm -f "${REPORT_BASE}.nsys-rep" "${REPORT_BASE}.sqlite" "${REPORT_BASE}_stats.txt" "${BUNDLE_PATH}"

if [[ ! -f "${BUILD_DIR}/sphere.pcd" ]]; then
    if [[ -f "${PROJECT_ROOT}/src/sphere.pcd" ]]; then
        cp "${PROJECT_ROOT}/src/sphere.pcd" "${BUILD_DIR}/sphere.pcd"
    elif [[ -x "${BUILD_DIR}/sphere" ]]; then
        (cd "${BUILD_DIR}" && ./sphere)
        cp "${PROJECT_ROOT}/sphere.pcd" "${BUILD_DIR}/sphere.pcd"
    else
        echo "Missing sphere.pcd and sphere generator after build." >&2
        exit 1
    fi
fi

(
    cd "${BUILD_DIR}"
    "${NSYS_BIN}" profile \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --stats=true \
        --force-overwrite=true \
        -o "${REPORT_BASE}" \
        ./fps
) 2>&1 | tee "${REPORT_BASE}_stats.txt"

"${NSYS_BIN}" stats "${REPORT_BASE}.nsys-rep" >> "${REPORT_BASE}_stats.txt"

tar -czf "${BUNDLE_PATH}" \
    -C "${REPORT_DIR}" "${REPORT_NAME}.nsys-rep" "${REPORT_NAME}_stats.txt" \
    -C "${BUILD_DIR}" "sphere_fps.pcd"

echo
echo "Nsight Systems report: ${REPORT_BASE}.nsys-rep"
echo "Text metrics: ${REPORT_BASE}_stats.txt"
echo "Program output PCD: ${BUILD_DIR}/sphere_fps.pcd"
echo "Copy bundle: ${BUNDLE_PATH}"
