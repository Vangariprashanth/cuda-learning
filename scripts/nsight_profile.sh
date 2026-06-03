#!/bin/bash
# nsight_profile.sh — Nsight Systems + Nsight Compute profiling (Vast.ai safe)

set -e
mkdir -p outputs

PCD_ARG=${1:-"--synthetic"}   # pass --pcd data/sphere.pcd or default synthetic

echo "═══════════════════════════════════════════════"
echo "  NSIGHT SYSTEMS — System-level timeline"
echo "═══════════════════════════════════════════════"

# NOTE: GPU metrics removed (not supported on Vast.ai / restricted GPUs)
nsys profile \
    --trace=cuda,osrt,nvtx \
    --cuda-memory-usage=true \
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

# Kernel-level metrics (safe to run if NCU is installed properly)
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
echo "  OUTPUTS"
echo "═══════════════════════════════════════════════"
echo ""
echo "Nsight Systems: outputs/nsys_trace.nsys-rep"
echo "Nsight Compute: outputs/ncu_fps.ncu-rep"
echo ""
echo "Open with:"
echo "  nsys-ui outputs/nsys_trace.nsys-rep"
echo "  ncu-ui outputs/ncu_fps.ncu-rep"