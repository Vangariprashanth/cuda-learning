#!/bin/bash
# nsight_profile.sh — Nsight Systems + Nsight Compute profiling
# Run AFTER building the extension: pip install -e .

set -e
mkdir -p outputs

PCD_ARG=${1:-"--synthetic"}   # pass --pcd data/sphere.pcd as $1, or use synthetic

echo "═══════════════════════════════════════════════"
echo "  NSIGHT SYSTEMS — System-level timeline"
echo "═══════════════════════════════════════════════"

nsys profile \
    --trace=cuda,osrt,nvtx \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --delay=1 \
    --duration=20 \
    --force-overwrite=true \
    -o outputs/nsys_trace \
    python profiling/profile_all.py $PCD_ARG --skip_torch_profiler 2>&1

echo ""
echo "Nsight Systems trace: outputs/nsys_trace.nsys-rep"
echo "Open with: nsys-ui outputs/nsys_trace.nsys-rep"
echo ""

echo "═══════════════════════════════════════════════"
echo "  NSIGHT COMPUTE — Kernel-level metrics"
echo "═══════════════════════════════════════════════"
echo ""

# Key metrics explained:
# dram__bytes_read/write         : total DRAM traffic — for bandwidth utilization
# sm__throughput.pct             : SM compute utilization
# l1tex__average_t_sectors_per_request : coalescing quality (4=perfect, 32=worst)
# sm__warps_active.pct           : occupancy
# smsp__thread_inst_executed_pred_on.pct : warp efficiency (divergence indicator)
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared : shared memory bank conflicts

METRICS="dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_active"

echo "Profiling fps_update_distances_kernel..."
ncu \
    --metrics $METRICS \
    --kernel-name fps_update_distances_kernel \
    --launch-count 5 \
    --force-overwrite \
    -o outputs/ncu_fps \
    python profiling/profile_all.py $PCD_ARG --skip_torch_profiler 2>/dev/null

echo ""
echo "═══════════════════════════════════════════════"
echo "  HOW TO INTERPRET NCU OUTPUT"
echo "═══════════════════════════════════════════════"
echo ""
echo "dram__bytes_read.sum:"
echo "  What you expect: ~N*5*4 bytes per FPS step (read xyz + read+write dist)"
echo "  If much higher: cache is not helping, all accesses go to DRAM"
echo ""
echo "sm__throughput (%):"
echo "  For FPS: expect LOW (20-40%). FPS is memory-bound, not compute-bound."
echo "  Low throughput here is EXPECTED — it means compute units wait on DRAM."
echo ""
echo "sectors_per_request (load):"
echo "  Ideal: 4 (128 bytes per warp = 4 × 32-byte sectors)"
echo "  If >8: coalescing issue — your (N,3) access pattern may be strided"
echo ""
echo "warps_active (%):"
echo "  Occupancy. For FPS: should be moderate (50-80%)."
echo "  Low occupancy + memory-bound = GPU stalls waiting on DRAM."
echo ""
echo "thread_inst_executed_pred_on (%):"
echo "  Warp efficiency. Should be >90% for FPS (no branching in inner loop)."
echo "  If <80%: look for if-statements inside the hot path."
echo ""
echo "NCU report: outputs/ncu_fps.ncu-rep"
echo "Open with: ncu-ui outputs/ncu_fps.ncu-rep"
