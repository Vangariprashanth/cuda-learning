#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
REPORT_DIR="${PROJECT_ROOT}/reports"
REPORT_NAME="fps_nsys"
REPORT_BASE="${REPORT_DIR}/${REPORT_NAME}"
BUNDLE_PATH="${REPORT_DIR}/fps_profile_bundle.tar.gz"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys was not found. Install Nsight Systems CLI or load the CUDA/Nsight module on the server." >&2
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
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=cpu \
        --stats=true \
        --force-overwrite=true \
        -o "${REPORT_BASE}" \
        ./fps
) 2>&1 | tee "${REPORT_BASE}_stats.txt"

nsys stats "${REPORT_BASE}.nsys-rep" >> "${REPORT_BASE}_stats.txt"

tar -czf "${BUNDLE_PATH}" \
    -C "${REPORT_DIR}" "${REPORT_NAME}.nsys-rep" "${REPORT_NAME}_stats.txt" \
    -C "${BUILD_DIR}" "sphere_fps.pcd"

echo
echo "Nsight Systems report: ${REPORT_BASE}.nsys-rep"
echo "Text metrics: ${REPORT_BASE}_stats.txt"
echo "Program output PCD: ${BUILD_DIR}/sphere_fps.pcd"
echo "Copy bundle: ${BUNDLE_PATH}"
