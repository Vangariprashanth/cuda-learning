# FPS + MLP Pipeline — CUDA Profiling Project

A complete CUDA/PyTorch profiling project for your resume.

**What this produces:**
- Real performance numbers from your GPU
- Roofline analysis positioning your kernels
- Stage-level breakdown showing where time goes
- Chrome trace for visual inspection
- Resume bullet templates with placeholder labels

---

## Project Structure

```
fps_project/
├── src/
│   ├── fps_cuda.cu          CUDA kernels (distance update + argmax)
│   └── fps_bindings.cpp     pybind11 bindings exposing kernels to Python
├── python/
│   ├── model.py             PointNet encoder + MLP head
│   ├── train.py             Training loop
│   └── dataset.py           PCD loader + FPS dataset
├── profiling/
│   └── profile_all.py       MAIN SCRIPT — all 8 measurement sections
├── scripts/
│   └── nsight_profile.sh    Nsight Systems + Compute profiling
├── setup.py                 Build script for CUDA extension
├── RESUME_BULLETS.md        All bullet templates (fill in your numbers)
└── README.md                This file
```

---

## Step 1 — Prerequisites

```bash
# Verify your environment
nvidia-smi                          # see your GPU + CUDA version
nvcc --version                      # CUDA compiler
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
# These two CUDA versions must match.

pip install torch open3d numpy      # if not already installed
```

**Important:** the `sm_80` arch flag in `setup.py` is for A100. Change it to match your GPU:
- RTX 2080/2080Ti → `sm_75`
- RTX 3090/3080   → `sm_86`
- RTX 4090        → `sm_89`
- A100            → `sm_80`

---

## Step 2 — Build the CUDA Extension

```bash
cd fps_project

# Edit setup.py: change '-arch=sm_80' to match your GPU
# Then build:
pip install -e .

# Verify:
python -c "import fps_cuda_ext; print('CUDA FPS extension OK')"
```

You should see the extension compile (takes ~60 seconds first time).
The `--ptxas-options=-v` flag will print register usage per kernel during build — note these numbers for your resume.

---

## Step 3 — Add Your PCD File

```bash
cp /path/to/your/sphere.pcd fps_project/data/sphere.pcd
```

If you don't have the file yet, use `--synthetic` in all commands below.

---

## Step 4 — Train (optional but recommended for real loss numbers)

```bash
python python/train.py --pcd data/sphere.pcd --epochs 30

# Or with synthetic data:
python python/train.py --synthetic --epochs 30
```

Loss should drop from ~0.1 to <0.01. Saves checkpoint to `checkpoints/best.pth`.

---

## Step 5 — Run the Full Profiler (generates your resume numbers)

```bash
python profiling/profile_all.py --pcd data/sphere.pcd
# Or:
python profiling/profile_all.py --synthetic
```

This prints 8 sections. Each section tells you which numbers to copy into `RESUME_BULLETS.md`.

**Section 1** — GPU specs (peak BW and TFLOPS for roofline)
**Section 2** — FPS latency vs K, per-step timing
**Section 3** — Encoder+MLP latency and throughput vs batch size
**Section 4** — Stage breakdown: FPS% vs Encoder% vs MLP%
**Section 5** — Roofline: arithmetic intensity, achieved GFLOPS, memory/compute bound
**Section 6** — H2D transfer: standard vs pinned memory
**Section 7** — torch.profiler Chrome trace
**Section 8** — Resume bullet templates (printed with your actual values)

Results saved to `outputs/profile_results.json`.

---

## Step 6 — Run Nsight Profiling (for the deep kernel metrics)

```bash
chmod +x scripts/nsight_profile.sh
./scripts/nsight_profile.sh --pcd data/sphere.pcd
# Or:
./scripts/nsight_profile.sh --synthetic
```

This runs Nsight Systems (timeline) and Nsight Compute (kernel metrics).

**Key ncu metrics to note for your resume:**
- `sectors_per_request` → coalescing quality (note for B3 bullet)
- `warps_active %` → occupancy (note for B4 bullet)
- `sm__throughput %` → confirms memory-bound (note for C1 bullet)

---

## Step 7 — Fill In Your Resume Bullets

Open `RESUME_BULLETS.md`. The bottom section shows a complete mapping:
which number goes where, and which section of the profiler output it comes from.

Pick 6 bullets total. For NVIDIA: use A2, B1, B2, C1, D1, D3.
For general ML: use A1, B1, D1, D2, E1, E2.

---

## Common Errors

| Error | Fix |
|---|---|
| `nvcc not found` | `export PATH=/usr/local/cuda/bin:$PATH` |
| CUDA version mismatch | Rebuild PyTorch or match nvcc to PyTorch CUDA version |
| `fps_cuda_ext not found` | Run `pip install -e .` from project root |
| `CUDA error: no kernel image` | Wrong `-arch=sm_XX` in setup.py — change to match your GPU |
| `OOM` during profiling | Reduce `--batch_size` argument |
| `Empty PCD` | Check file path; use `--synthetic` to verify extension works |
