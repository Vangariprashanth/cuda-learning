"""
model.py — PointNet Encoder + MLP Regression Head
===================================================

INPUT:  (B, K, 3) — batch of FPS-sampled point clouds
OUTPUT: (B, 3)    — predicted centroid XYZ per cloud

ARCHITECTURE:
    PointNetEncoder:  (B,K,3) → (B,256) global feature vector
    MLPHead:          (B,256) → (B,3)   centroid prediction

WHY THIS ARCHITECTURE FOR THE EXERCISE:
    - Simple enough to train in minutes on a laptop GPU
    - Has enough compute to make profiling meaningful
    - Directly uses the FPS output — the pipeline is: PCD → FPS → Encoder → MLP
    - Centroid regression is a real task with a clear ground truth

PROFILING TARGETS (fill in after running profiling/profile_mlp.py):
    Encoder forward latency:    X ms at batch size Y
    MLP forward latency:        X ms at batch size Y
    Peak GPU memory:            X MB at batch size Y
    Encoder FLOPs:              X GFLOPs (computed by torchinfo)
    Achieved TFLOPS:            X TFLOPS (FLOPs / latency)
    Arithmetic intensity:       X FLOP/byte (for roofline plot)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


# ─── SHARED MLP BLOCK ─────────────────────────────────────────
class PointMLP(nn.Module):
    """
    Per-point MLP: Linear + BatchNorm + ReLU
    Applied independently to each of the K points.
    'Shared' = same weights for all K points = order-invariant.

    NOTE ON BATCH NORM WITH POINT CLOUDS:
        Input shape: (B, K, C).
        BatchNorm1d expects (B, C) or (B, C, L).
        We reshape to (B*K, C) for BN, then restore to (B, K, C).
        This normalizes across both batch AND point dimensions.
    """
    def __init__(self, in_dim: int, out_dim: int, bn: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=not bn)
        self.bn     = nn.BatchNorm1d(out_dim) if bn else nn.Identity()
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, in_dim)
        B, K, _ = x.shape
        x = self.linear(x)                    # (B, K, out_dim)
        x = x.reshape(B * K, -1)
        x = self.bn(x)
        x = x.reshape(B, K, -1)
        return self.act(x)


# ─── POINTNET ENCODER ─────────────────────────────────────────
class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder.
    Applies a shared MLP to each point, then max-pools globally.

    Architecture:
        (B, K, 3) → [MLP 3→64] → [MLP 64→128] → [MLP 128→256]
                  → [MaxPool over K] → (B, 256)

    FLOP COUNT (for roofline):
        Layer 3→64:   2 * 3 * 64 = 384 FLOPs per point (mul+add per weight)
        Layer 64→128: 2 * 64 * 128 = 16,384 FLOPs per point
        Layer 128→256:2 * 128 * 256 = 65,536 FLOPs per point
        Total per point: ~82,304 FLOPs
        For K=1024: ~84M FLOPs per sample
        For B=8:    ~672M FLOPs per batch = ~0.67 GFLOPs
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            PointMLP(3,   64),
            PointMLP(64,  128),
            PointMLP(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, K, 3)
        Returns:
            global_feat:    (B, out_dim)  — one vector per cloud
            per_point_feat: (B, K, out_dim) — before pooling (for segmentation)
        """
        per_point = self.mlp(x)                         # (B, K, 256)
        global_f  = per_point.max(dim=1).values         # (B, 256)
        return global_f, per_point


# ─── MLP REGRESSION HEAD ──────────────────────────────────────
class MLPHead(nn.Module):
    """
    Simple MLP: 256 → 128 → 64 → 3

    FLOP COUNT:
        Layer 256→128: 2 * 256 * 128 = 65,536 FLOPs per sample
        Layer 128→64:  2 * 128 * 64  = 16,384 FLOPs per sample
        Layer 64→3:    2 * 64  * 3   = 384   FLOPs per sample
        Total: ~82K FLOPs per sample — much less than encoder
        This is why MLP is never the bottleneck: encoder dominates.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── FULL PIPELINE MODEL ──────────────────────────────────────
class FPSMLPNet(nn.Module):
    """
    Complete model: encoder + head.
    Does NOT include FPS — FPS is done in the DataLoader.
    This separation is intentional: it lets you profile FPS
    and model inference independently.
    """
    def __init__(self, encoder_dim: int = 256, out_dim: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(out_dim=encoder_dim)
        self.head    = MLPHead(in_dim=encoder_dim, out_dim=out_dim,
                               dropout=dropout)

        self._init_weights()
        self._print_stats()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _print_stats(self):
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = enc_params + head_params
        print(f"[Model] Encoder params: {enc_params:,}")
        print(f"[Model] Head params:    {head_params:,}")
        print(f"[Model] Total params:   {total:,}")
        print(f"[Model] Model size:     {total * 4 / 1024:.1f} KB (float32)")

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        points: (B, K, 3) — FPS-sampled coordinates
        Returns dict for easy inspection during profiling.
        """
        global_feat, per_point = self.encoder(points)
        pred = self.head(global_feat)
        return {
            'pred':        pred,         # (B, 3)
            'global_feat': global_feat,  # (B, 256)
            'per_point':   per_point,    # (B, K, 256)
        }
