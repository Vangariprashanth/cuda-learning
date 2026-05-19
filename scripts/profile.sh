#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
REPORT_DIR="${PROJECT_ROOT}/reports"

REPORT_NAME="fps_nsys"
REPORT_BASE="${REPORT_DIR}/${REPORT_NAME}"
BUNDLE_PATH="${REPORT_DIR}/fps_profile_bundle.tar.gz"

NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"

if [[ -z "${NSYS_BIN}" || ! -x "${NSYS_BIN}" ]]; then
    echo "nsys not found. Install Nsight Systems CLI or run with NSYS_BIN=/full/path/to/nsys bash scripts/profile.sh" >&2
    exit 1
fi

bash "${SCRIPT_DIR}/build.sh"
mkdir -p "${REPORT_DIR}"
rm -f \
    "${REPORT_BASE}.nsys-rep" \
    "${REPORT_BASE}.qdstrm" \
    "${REPORT_BASE}.sqlite" \
    "${REPORT_BASE}.log" \
    "${REPORT_BASE}_stats.txt" \
    "${BUNDLE_PATH}"

cd "${BUILD_DIR}"

"${NSYS_BIN}" profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --force-overwrite=true \
    -o "${REPORT_BASE}" \
    ./fps 2>&1 | tee "${REPORT_BASE}.log"

if [[ -f "${REPORT_BASE}.nsys-rep" ]]; then
    "${NSYS_BIN}" stats "${REPORT_BASE}.nsys-rep" > "${REPORT_BASE}_stats.txt" || true
else
    cat > "${REPORT_BASE}_stats.txt" <<EOF
Nsight Systems did not generate ${REPORT_NAME}.nsys-rep.

This usually means the server/container has the nsys target collector but is missing the importer binary or one of its dependencies.
The raw trace was generated as ${REPORT_NAME}.qdstrm if that file exists.

To fix report conversion on the server, install the full Nsight Systems CLI/importer package or use a CUDA/Nsight container image that includes it.
EOF
fi

bundle_items=()
[[ -f "${REPORT_BASE}.nsys-rep" ]] && bundle_items+=("${REPORT_NAME}.nsys-rep")
[[ -f "${REPORT_BASE}.qdstrm" ]] && bundle_items+=("${REPORT_NAME}.qdstrm")
[[ -f "${REPORT_BASE}.log" ]] && bundle_items+=("${REPORT_NAME}.log")
[[ -f "${REPORT_BASE}_stats.txt" ]] && bundle_items+=("${REPORT_NAME}_stats.txt")

if [[ ${#bundle_items[@]} -gt 0 ]]; then
    tar -czf "${BUNDLE_PATH}" -C "${REPORT_DIR}" "${bundle_items[@]}" -C "${BUILD_DIR}" "sphere_fps.pcd"
fi

echo ""
echo "Report directory: ${REPORT_DIR}"
[[ -f "${REPORT_BASE}.nsys-rep" ]] && echo "Nsight report: ${REPORT_BASE}.nsys-rep"
[[ -f "${REPORT_BASE}.qdstrm" ]] && echo "Raw trace: ${REPORT_BASE}.qdstrm"
echo "Stats/log: ${REPORT_BASE}_stats.txt"
echo "Program output PCD: ${BUILD_DIR}/sphere_fps.pcd"
echo "Copy bundle: ${BUNDLE_PATH}"
