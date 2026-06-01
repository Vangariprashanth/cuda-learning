# RESUME BULLET TEMPLATES
# ========================
# Run profiling/profile_all.py and replace ALL_CAPS with your numbers.
# The script also prints these with placeholders labeled for you.
#
# WHICH BULLETS TO PICK:
#   For NVIDIA (systems/GPU role):   A1, B1, B2, C1, C2, D1, D2
#   For ML Engineer (general):       A1, A2, B1, D2, E1, E2
#   For Research roles:              A2, B1, C1, E1, E2
#   Maximum 6 bullets on one job entry.


# ═══════════════════════════════════════════
# A — ARCHITECTURE & SYSTEM DESIGN
# ═══════════════════════════════════════════

A1_PIPELINE:
  "Engineered an end-to-end GPU point cloud inference pipeline coupling
   a custom CUDA FPS kernel (exposed to Python via pybind11 and PyTorch's
   CUDAExtension API) with a PointNet-style encoder and MLP regression head,
   processing N=100K-point depth clouds to K=1,024 spatially uniform samples
   per inference call."

A2_CUDA_EXTENSION:
  "Implemented a C++/CUDA FPS extension using pybind11, writing a CUDA kernel
   for the O(N)-parallel distance-update step and a shared-memory tree-reduction
   kernel for argmax — exposing the full pipeline to Python with validated tensor
   dtype, device, and contiguity checks at the binding layer."

A3_TRAINING:
  "Trained a PointNet encoder + MLP regression head for 3D centroid prediction
   on sphere point clouds, achieving MSE loss of FINAL_LOSS after EPOCHS epochs
   with cosine-annealed learning rate scheduling — validating end-to-end gradient
   flow through the CUDA-extended sampling and PyTorch inference pipeline."


# ═══════════════════════════════════════════
# B — PROFILING & BOTTLENECK IDENTIFICATION
# (Most valuable for GPU/systems roles)
# ═══════════════════════════════════════════

B1_STAGE_BREAKDOWN:
  "Profiled a 3-stage GPU pipeline (FPS → PointNet encoder → MLP) using CUDA
   events and Nsight Systems, producing a stage-level latency breakdown
   (FPS: FPS_MS ms, encoder: ENC_MS ms, MLP: MLP_MS ms) that identified FPS
   as the dominant contributor at FPS_PCT% of total runtime — establishing that
   encoder-side optimizations would have near-zero impact on end-to-end latency."

B2_PER_STEP_FPS:
  "Quantified per-iteration FPS latency across all K=1,024 sampling steps,
   confirming near-constant STEP_MS ms per iteration consistent with an O(N)
   parallel distance-update kernel executing K sequential times — and providing
   a data-driven basis for evaluating approximate sampling strategies as a
   replacement under tighter latency budgets."

B3_NCU_COALESCING:
  "Used Nsight Compute to measure sectors-per-request for the FPS distance-update
   kernel (SECTORS sectors/request vs. ideal 4), quantifying the memory transaction
   overhead introduced by the stride-3 access pattern across the (N, 3) coordinate
   tensor — identifying a concrete coalescing optimization opportunity."

B4_OCCUPANCY:
  "Profiled SM occupancy of OCC_PCT% and warp efficiency of WARP_EFF% for the FPS
   distance-update kernel using Nsight Compute, attributing sub-peak occupancy to
   register pressure (REG registers/thread from --ptxas-options=-v output) — and
   evaluating the occupancy-latency tradeoff against block size configurations
   ranging from 128 to 512 threads."


# ═══════════════════════════════════════════
# C — ROOFLINE ANALYSIS
# (Differentiating for senior GPU roles)
# ═══════════════════════════════════════════

C1_ROOFLINE_POSITION:
  "Applied roofline analysis to characterize the compute-memory tradeoff of each
   pipeline stage on a GPU_NAME GPU (peak bandwidth: BW_GBPS GB/s, peak FP32:
   PEAK_TF TFLOPS, ridge point: RIDGE FLOP/byte): FPS distance kernel at
   FPS_AI FLOP/byte and PointNet encoder at ENC_AI FLOP/byte — both deeply
   memory-bound — establishing that reduced memory traffic, not faster compute,
   is the productive optimization direction."

C2_ROOFLINE_ACHIEVED:
  "Measured achieved throughput of ENC_GFLOPS GFLOPS/s for the encoder at
   batch size BS, representing ENC_PCT% of the memory-bandwidth-limited
   attainable peak — demonstrating high memory utilization efficiency while
   confirming the kernel remains BW_GBPS GB/s-constrained rather than
   compute-constrained."


# ═══════════════════════════════════════════
# D — THROUGHPUT, LATENCY, MEMORY
# ═══════════════════════════════════════════

D1_THROUGHPUT:
  "Benchmarked encoder+MLP inference across batch sizes 1–64, achieving peak
   throughput of BEST_FPS frames/sec at batch size BEST_BS before GPU memory
   saturation at PEAK_MB MB — and establishing per-frame latency of BS1_MS ms
   at batch size 1, characterizing the full latency-throughput tradeoff envelope
   for deployment planning."

D2_MEMORY:
  "Profiled peak GPU memory allocation across batch sizes 1–64, identifying
   PEAK_MB MB at batch size BEST_BS as the VRAM ceiling, and characterizing
   the memory growth curve — enabling data-driven batch size selection for
   constrained-hardware deployment (e.g., edge GPUs with 6–8 GB VRAM)."

D3_H2D_TRANSFER:
  "Measured host-to-device transfer overhead at STD_MS ms (standard) vs
   PIN_MS ms (pinned memory) for a SIZE_MB MB point cloud tensor over PCIe,
   demonstrating a SPEEDUP× latency reduction with page-locked allocation —
   informing DataLoader design for low-latency streaming inference pipelines."

D4_SCALING:
  "Swept encoder inference latency across point cloud sizes K=128 to K=4,096,
   confirming O(K) scaling at MSPT ms/point — consistent with PointNet-style
   per-point shared-MLP processing — and establishing the architecture's
   suitability for variable-density sensor inputs without runtime modification."


# ═══════════════════════════════════════════
# E — MODEL & ACCURACY
# ═══════════════════════════════════════════

E1_MODEL:
  "Implemented a PointNet-style feature encoder — shared MLP (3→64→128→256)
   with global max-pooling — producing 256-dimensional global feature vectors
   from 1,024 FPS-sampled points, combined with an MLP regression head for
   3D centroid prediction; confirmed O(K) inference scaling empirically across
   point cloud sizes 128–4,096."

E2_ACCURACY:
  "Achieved final MSE loss of FINAL_LOSS on sphere centroid regression after
   EPOCHS training epochs, with the model converging to centroid predictions
   within RMSE mm of ground truth — validating end-to-end pipeline correctness
   from CUDA FPS sampling through PyTorch inference."


# ═══════════════════════════════════════════
# HOW TO FILL IN THE VALUES
# ═══════════════════════════════════════════
#
# Run: python profiling/profile_all.py --pcd data/sphere.pcd
#
# Then fill in:
#
#   FPS_MS       ← Section 4, FPS row, Time column
#   ENC_MS       ← Section 4, Encoder row
#   MLP_MS       ← Section 4, MLP Head row
#   TOTAL_MS     ← Section 4, PIPELINE TOTAL row
#   FPS_PCT      ← Section 4, STAGE BREAKDOWN, FPS row %
#   ENC_PCT      ← Section 4, STAGE BREAKDOWN, Encoder row %
#   STEP_MS      ← Section 2B, Mean over all steps
#   BEST_FPS     ← Section 3, highest FPS value in table
#   BEST_BS      ← Section 3, batch size at BEST_FPS
#   BS1_MS       ← Section 3, Latency at BS=1
#   PEAK_MB      ← Section 3, Peak Mem MB at BEST_BS
#   STD_MS       ← Section 6, Standard H2D
#   PIN_MS       ← Section 6, Pinned memory H2D
#   SPEEDUP      ← Section 6, Speedup row
#   SIZE_MB      ← Section 6, Tensor size
#   STD_BW       ← Section 6, BW for standard
#   PIN_BW       ← Section 6, BW for pinned
#   FPS_AI       ← Section 5, FPS kernel AI column
#   ENC_AI       ← Section 5, Encoder kernel AI column
#   RIDGE        ← Section 5, Ridge point line
#   BW_GBPS      ← Section 1, peak_bw_gbps
#   PEAK_TF      ← Section 1, peak_tflops_fp32
#   GPU_NAME     ← Section 1, name
#   ENC_GFLOPS   ← Section 5, Encoder Achieved GF/s
#   ENC_PCT      ← Section 5, Encoder % of roof
#   SECTORS      ← ncu output, sectors_per_request for fps kernel
#   OCC_PCT      ← ncu output, warps_active %
#   WARP_EFF     ← ncu output, thread_inst_executed_pred_on %
#   REG          ← nvcc build output --ptxas-options=-v, registers/thread
#   FINAL_LOSS   ← outputs/training_history.json, last epoch loss
#   EPOCHS       ← however many you ran
#   RMSE         ← sqrt(FINAL_LOSS), in coordinate units
#   MSPT         ← ms per point from Section 3 scaling sweep
