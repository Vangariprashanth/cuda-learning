#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

if [[ ! -x "${BUILD_DIR}/fps" ]]; then
    bash "${SCRIPT_DIR}/build.sh"
fi

if [[ ! -f "${BUILD_DIR}/sphere.pcd" ]]; then
    if [[ -f "${PROJECT_ROOT}/src/sphere.pcd" ]]; then
        cp "${PROJECT_ROOT}/src/sphere.pcd" "${BUILD_DIR}/sphere.pcd"
    elif [[ -x "${BUILD_DIR}/sphere" ]]; then
        (cd "${BUILD_DIR}" && ./sphere)
        cp "${PROJECT_ROOT}/sphere.pcd" "${BUILD_DIR}/sphere.pcd"
    else
        echo "Missing sphere.pcd and sphere generator. Run ${SCRIPT_DIR}/build.sh first." >&2
        exit 1
    fi
fi

cd "${BUILD_DIR}"
./fps
