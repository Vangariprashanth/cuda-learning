"""
dataset.py — Data loading for FPS + MLP pipeline
==================================================

Supports:
  1. Loading your real sphere.pcd file
  2. Generating synthetic sphere data (fallback)

TASK DEFINITION:
    Input:  K=1024 FPS-sampled points from a sphere (K, 3)
    Target: Predict the centroid of the sampled cloud (3,)
    Why:    For a noisy sphere, the centroid should be near (0,0,0).
            This is a real regression problem that exercises the full pipeline.
            It's simple enough to converge fast but meaningful enough to
            make the training numbers credible.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


def load_pcd_numpy(filepath: str) -> np.ndarray:
    """Load a .pcd file, return (N, 3) float32 numpy array."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("pip install open3d")

    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise ValueError(f"Empty PCD: {filepath}")
    print(f"[Dataset] Loaded {pts.shape[0]:,} points from {Path(filepath).name}")
    return pts


def generate_sphere_numpy(n_points: int = 100_000,
                           radius: float = 1.0,
                           noise: float = 0.01) -> np.ndarray:
    """Uniform sphere surface, optional Gaussian noise."""
    raw  = np.random.randn(n_points, 3).astype(np.float32)
    nrm  = np.linalg.norm(raw, axis=1, keepdims=True)
    pts  = raw / nrm * radius
    pts += np.random.randn(*pts.shape).astype(np.float32) * noise
    print(f"[Dataset] Generated synthetic sphere: {n_points:,} pts, r={radius}")
    return pts


class SphereDataset(Dataset):
    """
    Dataset that wraps a point cloud and applies FPS per sample.

    Each __getitem__ call:
      1. Takes the full N-point cloud.
      2. Adds per-sample jitter (data augmentation).
      3. Applies FPS to get K points.
      4. Returns (sampled_points, centroid_target).

    WHY FPS IN DATALOADER vs PRE-COMPUTED:
      In production you might pre-compute FPS to save time.
      Here we compute it on-the-fly so you can profile the
      DataLoader + FPS + model chain end-to-end.
      This also makes the dataset size = n_samples regardless of
      how many point clouds you have (we use one cloud, augmented).

    IMPORTANT: FPS here uses the Python-wrapped CUDA extension.
    This means the DataLoader must use num_workers=0 (CUDA context
    cannot be shared across forked processes without extra setup).
    """

    def __init__(self,
                 points: np.ndarray,         # (N, 3) full cloud
                 n_samples: int = 1024,      # K — FPS output size
                 dataset_size: int = 512,    # how many items to return
                 jitter_std: float = 0.005,  # per-sample point noise
                 use_cuda_fps: bool = True): # use our CUDA extension

        self.points_np    = points
        self.n_samples    = n_samples
        self.dataset_size = dataset_size
        self.jitter_std   = jitter_std
        self.use_cuda_fps = use_cuda_fps
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Pre-load full cloud to GPU once (avoid repeated H2D transfers)
        self.points_gpu = torch.from_numpy(points).to(self.device)

        if use_cuda_fps:
            try:
                import fps_cuda_ext
                self.fps_fn = fps_cuda_ext.fps
                print(f"[Dataset] Using CUDA FPS extension")
            except ImportError:
                print("[Dataset] WARN: fps_cuda_ext not built. Using PyTorch fallback.")
                self.use_cuda_fps = False

    def _fps_pytorch_fallback(self, points: torch.Tensor, K: int) -> torch.Tensor:
        """Pure PyTorch FPS — used if CUDA extension not available."""
        N = points.shape[0]
        selected = torch.zeros(K, dtype=torch.long, device=points.device)
        dists    = torch.full((N,), float('inf'), device=points.device)
        farthest = torch.randint(0, N, (1,)).item()
        for i in range(K):
            selected[i] = farthest
            centroid     = points[farthest].unsqueeze(0)
            dist         = ((points - centroid) ** 2).sum(-1)
            dists        = torch.minimum(dists, dist)
            farthest     = dists.argmax().item()
        return points[selected]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Add per-sample jitter for augmentation
        jitter = torch.randn_like(self.points_gpu) * self.jitter_std
        pts    = self.points_gpu + jitter

        # Apply FPS
        if self.use_cuda_fps:
            import fps_cuda_ext
            _, sampled = fps_cuda_ext.fps(pts.contiguous(), self.n_samples)
        else:
            sampled = self._fps_pytorch_fallback(pts, self.n_samples)

        # Target: centroid of sampled cloud (ground truth ≈ 0,0,0 for sphere)
        centroid = sampled.mean(dim=0)  # (3,)

        return sampled, centroid   # (K, 3), (3,)


def make_dataloader(points: np.ndarray,
                    n_samples: int = 1024,
                    batch_size: int = 8,
                    dataset_size: int = 256,
                    use_cuda_fps: bool = True) -> DataLoader:
    """
    Create a DataLoader for training.

    IMPORTANT: num_workers=0 required when using CUDA in __getitem__.
    CUDA operations cannot run in forked DataLoader worker processes
    without explicitly creating a new CUDA context per worker.
    For production, pre-compute FPS and save to disk to enable
    num_workers > 0.
    """
    ds = SphereDataset(points, n_samples, dataset_size,
                       use_cuda_fps=use_cuda_fps)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # must be 0 for CUDA in worker
        pin_memory=False,
        drop_last=True,
    )
