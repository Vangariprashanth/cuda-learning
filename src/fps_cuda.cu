/*
fps_cuda.cu — Farthest Point Sampling, CUDA Implementation
===========================================================

ALGORITHM REVIEW:
    FPS selects K points from N such that selected points are
    maximally spread. Each new point is the one farthest from
    the already-selected set.

    Step i:
      1. For every unselected point p_j, compute dist(p_j, p_selected_i).
      2. Update min_distances[j] = min(min_distances[j], dist).
      3. Next selected = argmax(min_distances).

    Steps 1+2 are PARALLEL over N. Step 3 is a REDUCTION.
    The K-iteration loop itself is SEQUENTIAL (each step needs step i-1).

WHY THIS MATTERS FOR PROFILING:
    The parallel parts (distance update) are memory-bandwidth bound.
    Each point: read 3 floats (12B) + read 1 float (4B) + write 1 float (4B) = 20 bytes.
    Compute: 3 subs + 3 muls + 2 adds + 1 min = ~9 FLOPs.
    Arithmetic intensity = 9/20 = 0.45 FLOP/byte → firmly MEMORY BOUND.

    This is the key insight for your interview:
    "FPS's distance kernel has arithmetic intensity ~0.45 FLOP/byte,
     placing it deep in the memory-bound regime on the roofline model.
     No amount of compute optimization will help — only better memory
     access patterns or reduced memory traffic can improve it."
*/

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// ─── ERROR CHECKING ───────────────────────────────────────────
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while(0)


// ─── KERNEL 1: UPDATE DISTANCES ───────────────────────────────
// Called once per FPS iteration.
// Updates min_distances[i] = min(min_distances[i], dist(points[i], new_point))
// for all N points in parallel.
//
// MEMORY ACCESS PATTERN:
//   points:        read  (N, 3) → stride-3 access to X, Y, Z
//   min_distances: read + write (N,) → coalesced
//   new_xyz:       broadcast — same value for all threads → L1 cache hit after first access
//
// COALESCING ANALYSIS:
//   Thread i reads points[i*3 + 0,1,2].
//   Consecutive threads: thread 0 → addr 0,1,2; thread 1 → addr 3,4,5 ...
//   This IS coalesced for the full (N,3) row-major layout.
//   Sector loads: 4 sectors per warp (128B covering 32 threads × 3 floats + padding)
__global__ void fps_update_distances_kernel(
    const float* __restrict__ points,       // (N, 3) input cloud
    float*       __restrict__ min_dists,    // (N,)   running minimum distances
    float new_x, float new_y, float new_z,  // newly selected point coords
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Compute squared Euclidean distance (no sqrt — monotone, saves FLOPs)
    float dx = points[i * 3 + 0] - new_x;
    float dy = points[i * 3 + 1] - new_y;
    float dz = points[i * 3 + 2] - new_z;
    float dist = dx*dx + dy*dy + dz*dz;

    // Update running minimum
    if (dist < min_dists[i]) {
        min_dists[i] = dist;
    }
}


// ─── KERNEL 2: ARGMAX REDUCTION ───────────────────────────────
// Finds the index of the maximum value in min_distances.
// Uses shared memory tree reduction, then atomic on global result.
//
// WHY SEPARATE FROM UPDATE:
//   We need ALL distance updates to complete before finding the max.
//   CUDA guarantees: all threads in a grid finish before the next kernel.
//   So: kernel 1 (update) → synchronize → kernel 2 (argmax).
//
// REDUCTION PATTERN (taught in Lesson 5):
//   Each block reduces its 256 elements to 1 using shared memory.
//   Then blocks compete via atomicMax on a global result.
//
// ATOMICS:
//   atomicMax works on int/unsigned int, not float.
//   Trick: cast float bits to unsigned int — works because IEEE 754
//   positive floats have the same ordering as their bit representations.
__global__ void fps_argmax_kernel(
    const float* __restrict__ min_dists,  // (N,)
    int*         __restrict__ result_idx, // (1,) output
    float*       __restrict__ result_val, // (1,) output
    int N
) {
    extern __shared__ float smem[];       // shared memory: [0..blockDim-1] = values
    int* smem_idx = (int*)(smem + blockDim.x);  // [blockDim..2*blockDim-1] = indices

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Load: each thread loads one value (or -inf if out of range)
    float val = (i < N) ? min_dists[i] : -FLT_MAX;
    int   idx = (i < N) ? i : -1;

    smem[tid]     = val;
    smem_idx[tid] = idx;
    __syncthreads();

    // Tree reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smem[tid + stride] > smem[tid]) {
                smem[tid]     = smem[tid + stride];
                smem_idx[tid] = smem_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Each block's winner tries to become the global winner via atomic
    if (tid == 0 && idx >= 0) {
        // atomicMax on float using int reinterpretation:
        // Positive floats compare the same way as their uint32 bit patterns.
        unsigned int old_bits = atomicMax(
            (unsigned int*)result_val,
            __float_as_uint(smem[0])
        );
        // If we won, also update the index
        if (__uint_as_float(old_bits) < smem[0]) {
            atomicExch(result_idx, smem_idx[0]);
        }
    }
}


// ─── HOST FUNCTION: FULL FPS ──────────────────────────────────
// Orchestrates K iterations of (update + argmax) from CPU.
// points_d:        GPU pointer to (N, 3) float32
// sampled_idx_d:   GPU pointer to (K,) int32 output (pre-allocated)
// N, K:            sizes
// timing_ms:       if non-null, records per-iteration timing
extern "C" void fps_cuda(
    const float* points_d,
    int*         sampled_idx_d,
    int N, int K,
    float* timing_ms_out    // optional: array of K floats for per-step timing
) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // Allocate working memory on GPU
    float *min_dists_d, *result_val_d;
    int   *result_idx_d;
    CUDA_CHECK(cudaMalloc(&min_dists_d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result_val_d, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result_idx_d, sizeof(int)));

    // Initialize min_distances to +infinity
    // We use a kernel rather than cudaMemset because cudaMemset works in bytes.
    // FLT_MAX bit pattern: 0x7F7FFFFF. We set manually via a fill kernel.
    // Shortcut: use thrust or just cudaMemset with a custom value.
    // Here we use a simple approach: fill via kernel.
    auto fill_inf = [&]() {
        // kernel: set every float to FLT_MAX
        // We inline this as a lambda-launched kernel
    };
    // Simpler: use cudaMemset to 0x7F (each byte 0x7F → float ~3.4e38 ≈ FLT_MAX)
    CUDA_CHECK(cudaMemset(min_dists_d, 0x7F, N * sizeof(float)));

    // Host buffer for reading selected point coords back
    float selected_xyz[3];

    // Random starting point (index 0 — deterministic for reproducibility)
    int first_idx = 0;
    CUDA_CHECK(cudaMemcpy(sampled_idx_d, &first_idx, sizeof(int),
                          cudaMemcpyHostToDevice));

    // Read first point's coordinates to host for first kernel call
    CUDA_CHECK(cudaMemcpy(selected_xyz, points_d, 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ─── K-iteration FPS loop ──────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int step = 0; step < K; step++) {
        if (timing_ms_out) CUDA_CHECK(cudaEventRecord(t0));

        // Step A: Update all distances from latest selected point
        fps_update_distances_kernel<<<grid, BLOCK>>>(
            points_d, min_dists_d,
            selected_xyz[0], selected_xyz[1], selected_xyz[2],
            N
        );

        // Step B: Find point with maximum min-distance (next to select)
        // Reset result before argmax
        float neg_inf = -FLT_MAX;
        int   neg_one = -1;
        CUDA_CHECK(cudaMemcpy(result_val_d, &neg_inf, sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(result_idx_d, &neg_one, sizeof(int),
                              cudaMemcpyHostToDevice));

        // Shared memory: 2 * BLOCK * 4 bytes (values + indices)
        fps_argmax_kernel<<<grid, BLOCK, 2 * BLOCK * sizeof(float)>>>(
            min_dists_d, result_idx_d, result_val_d, N
        );

        if (timing_ms_out) {
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&timing_ms_out[step], t0, t1));
        }

        if (step + 1 < K) {
            // Read selected index back to host
            int next_idx;
            CUDA_CHECK(cudaMemcpy(&next_idx, result_idx_d, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            // Store in output array
            CUDA_CHECK(cudaMemcpy(sampled_idx_d + step + 1, &next_idx,
                                  sizeof(int), cudaMemcpyHostToDevice));
            // Read its coordinates for next iteration
            CUDA_CHECK(cudaMemcpy(selected_xyz,
                                  points_d + next_idx * 3,
                                  3 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(min_dists_d));
    CUDA_CHECK(cudaFree(result_val_d));
    CUDA_CHECK(cudaFree(result_idx_d));
}
