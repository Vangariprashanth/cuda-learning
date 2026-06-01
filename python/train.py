"""
train.py — Training Loop
=========================

Trains FPSMLPNet to predict the centroid of FPS-sampled sphere clouds.

RUN:
    python python/train.py --pcd data/sphere.pcd
    python python/train.py --synthetic

WHAT GOOD TRAINING LOOKS LIKE:
    Loss should drop from ~0.1 to ~0.001 within 20 epochs.
    The centroid of a sphere centered at origin should converge to (0,0,0).
    If loss plateaus > 0.01, FPS is not sampling uniformly (check your PCD).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import argparse
import time
import json
from pathlib import Path

from python.model import FPSMLPNet
from python.dataset import load_pcd_numpy, generate_sphere_numpy, make_dataloader


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ───────────────────────────────────────────────────
    if args.pcd:
        points = load_pcd_numpy(args.pcd)
    else:
        points = generate_sphere_numpy(100_000)

    loader = make_dataloader(
        points,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
    )

    # ── Model ──────────────────────────────────────────────────
    model = FPSMLPNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────
    history = []
    best_loss = float('inf')

    print(f"\n[Train] Starting: {args.epochs} epochs, bs={args.batch_size}, K={args.n_samples}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()

        for batch_pts, batch_targets in loader:
            # batch_pts:     (B, K, 3) — already on GPU from dataset
            # batch_targets: (B, 3)    — centroid ground truth

            batch_pts     = batch_pts.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            out  = model(batch_pts)
            loss = criterion(out['pred'], batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(loader)
        epoch_time  = (time.perf_counter() - epoch_start) * 1000  # ms

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'loss':       best_loss,
            }, 'checkpoints/best.pth')

        if epoch % 5 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {epoch_loss:.6f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {epoch_time:.0f} ms")

        history.append({'epoch': epoch, 'loss': epoch_loss, 'time_ms': epoch_time})

    # Save training history
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[Train] Best loss: {best_loss:.6f}")
    print(f"[Train] Checkpoint saved to checkpoints/best.pth")
    print(f"[Train] History saved to outputs/training_history.json")
    print(f"\nNext step: python profiling/profile_mlp.py")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pcd',          type=str,  default=None)
    p.add_argument('--synthetic',    action='store_true')
    p.add_argument('--epochs',       type=int,  default=30)
    p.add_argument('--batch_size',   type=int,  default=8)
    p.add_argument('--n_samples',    type=int,  default=1024)
    p.add_argument('--dataset_size', type=int,  default=256)
    train(p.parse_args())
