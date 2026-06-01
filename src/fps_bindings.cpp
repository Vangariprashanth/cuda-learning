/*
fps_bindings.cpp — pybind11 + PyTorch Extension Bindings
=========================================================

This file bridges your CUDA kernel and Python.

HOW PYTORCH EXTENSIONS WORK:
    torch.utils.cpp_extension.CUDAExtension compiles this file
    alongside fps_cuda.cu using nvcc. The result is a shared library
    (.so) that Python imports like any other module.

    Python side:  import fps_cuda_ext; fps_cuda_ext.fps(points_tensor, K)
    C++ side:     this file receives torch::Tensor, validates it,
                  extracts the raw GPU pointer, calls fps_cuda().

TENSOR VALIDATION (critical habit):
    Always check: device (must be CUDA), dtype (must be float32),
    contiguous (must be True — non-contiguous breaks pointer arithmetic).
    Never skip this. Silent wrong results are worse than an error.

WHY pybind11 OVER ctypes:
    pybind11 understands PyTorch's tensor type natively via torch's
    extension API. It handles reference counting, error propagation,
    and GIL management automatically.
*/

#include <torch/extension.h>
#include <vector>
#include <stdexcept>

// Declaration of the CUDA function defined in fps_cuda.cu
extern "C" void fps_cuda(
    const float* points_d,
    int*         sampled_idx_d,
    int N, int K,
    float* timing_ms_out
);


// ─── MAIN BINDING FUNCTION ────────────────────────────────────
// Called from Python as: fps_cuda_ext.fps(points, K)
//
// points:  torch.Tensor of shape (N, 3), dtype=float32, device=cuda
// K:       number of points to sample
// returns: dict with 'indices' (K,) int32 and 'sampled' (K, 3) float32
std::vector<torch::Tensor> fps_forward(
    torch::Tensor points,
    int K
) {
    // ── Validation ─────────────────────────────────────────────
    TORCH_CHECK(points.is_cuda(),
        "fps: points must be a CUDA tensor. Got CPU tensor. "
        "Call points = points.cuda() first.");

    TORCH_CHECK(points.dtype() == torch::kFloat32,
        "fps: points must be float32. Got ", points.dtype(),
        ". Call points = points.float() first.");

    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
        "fps: expected shape (N, 3), got ", points.sizes());

    TORCH_CHECK(points.is_contiguous(),
        "fps: points must be contiguous. Call points = points.contiguous() first.");

    int N = points.size(0);
    TORCH_CHECK(K <= N,
        "fps: K=", K, " > N=", N, ". Cannot sample more points than exist.");
    TORCH_CHECK(K > 0, "fps: K must be positive.");

    // ── Allocate output tensors on the same device ─────────────
    auto options_int   = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

    torch::Tensor indices = torch::zeros({K}, options_int);

    // ── Call CUDA kernel ───────────────────────────────────────
    fps_cuda(
        points.data_ptr<float>(),
        indices.data_ptr<int>(),
        N, K,
        nullptr  // no per-step timing in fast path
    );

    // Synchronize to ensure kernel completion before returning
    // (Important: Python caller shouldn't need to manage this)
    cudaDeviceSynchronize();

    // ── Gather sampled points using indices ────────────────────
    // indices.to(torch::kLong): index_select requires LongTensor
    torch::Tensor sampled = points.index_select(0, indices.to(torch::kLong));

    return {indices, sampled};
}


// ─── TIMED VERSION — for profiling ───────────────────────────
// Same as fps_forward but returns per-iteration timing.
// Use this in your profiling scripts, not in production.
std::vector<torch::Tensor> fps_forward_timed(
    torch::Tensor points,
    int K
) {
    TORCH_CHECK(points.is_cuda(), "fps: points must be CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "fps: need float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "fps: need (N,3)");
    TORCH_CHECK(points.is_contiguous(), "fps: must be contiguous");

    int N = points.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
    torch::Tensor indices = torch::zeros({K}, options_int);

    // Allocate timing array on CPU (small, K floats)
    std::vector<float> timing_ms(K);

    fps_cuda(
        points.data_ptr<float>(),
        indices.data_ptr<int>(),
        N, K,
        timing_ms.data()
    );
    cudaDeviceSynchronize();

    torch::Tensor sampled = points.index_select(0, indices.to(torch::kLong));

    // Return timing as a CPU tensor
    torch::Tensor timing_tensor = torch::from_blob(
        timing_ms.data(), {K}, torch::kFloat32
    ).clone();  // clone because from_blob doesn't own the memory

    return {indices, sampled, timing_tensor};
}


// ─── MODULE REGISTRATION ──────────────────────────────────────
// PYBIND11_MODULE: creates the Python module named "fps_cuda_ext"
// The m.def calls register Python-callable functions.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA FPS — Farthest Point Sampling GPU Extension";

    m.def("fps",
          &fps_forward,
          "Farthest Point Sampling (CUDA)\n"
          "Args:\n"
          "  points (Tensor): (N, 3) float32 CUDA tensor\n"
          "  K (int): number of points to sample\n"
          "Returns:\n"
          "  indices (Tensor): (K,) int32 — indices into original N points\n"
          "  sampled (Tensor): (K, 3) float32 — sampled point coordinates",
          py::arg("points"), py::arg("K"));

    m.def("fps_timed",
          &fps_forward_timed,
          "FPS with per-iteration timing (for profiling only)\n"
          "Returns: (indices, sampled, timing_ms) where timing_ms is (K,) float32 CPU tensor",
          py::arg("points"), py::arg("K"));
}
