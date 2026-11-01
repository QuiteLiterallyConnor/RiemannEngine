#!/usr/bin/env python
"""Measure the CPU Gaussian-to-adaptive-voxel reference path."""

import argparse
import statistics
import time

import torch

import riemannengine as re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=16,
                        help="voxels per axis in the cubic reference grid")
    parser.add_argument("--gaussians", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.resolution, args.gaussians, args.iterations) < 1:
        parser.error("resolution, gaussians, and iterations must be positive")
    if torch.cuda.is_available():
        parser.error("this benchmark measures the CPU-only reference path")

    n = args.resolution
    coordinates = torch.cartesian_prod(
        torch.arange(n), torch.arange(n), torch.arange(n)
    ).to(torch.int64)
    levels = torch.zeros(len(coordinates), dtype=torch.int32)
    config = re.adaptive.AdaptiveVoxelConfig(
        [1.0, 1.0, 1.0], origin=[-n / 2.0] * 3
    )
    reference = re.adaptive.AdaptiveSparseTensor.from_voxels(
        coordinates, levels, torch.zeros(len(coordinates), 1), config
    )
    generator = torch.Generator().manual_seed(7)
    positions = torch.rand(args.gaussians, 3, generator=generator) * (n * 0.6)
    positions -= n * 0.3
    rotations = torch.zeros(args.gaussians, 4)
    rotations[:, 0] = 1.0
    scales = torch.rand(args.gaussians, 3, generator=generator) * 0.35 + 0.2
    opacities = torch.rand(args.gaussians, 1, generator=generator) * 0.5 + 0.25
    colors = torch.rand(args.gaussians, 3, generator=generator)
    features = torch.rand(args.gaussians, 8, generator=generator)
    gaussians = re.gaussian.GaussianTensor(
        positions, rotations, scales, opacities, colors, features=features
    )

    re.gaussian_voxel.rasterize(gaussians, reference)
    elapsed = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = re.gaussian_voxel.rasterize(gaussians, reference)
        elapsed.append(time.perf_counter() - started)
    print(f"torch={torch.__version__} device=cpu")
    print(
        f"gaussians={args.gaussians} voxels={len(reference)} "
        f"channels={features.shape[1]} iterations={args.iterations}"
    )
    print(
        f"median_seconds={statistics.median(elapsed):.6f} "
        f"mean_seconds={statistics.mean(elapsed):.6f} "
        f"density_mass={output.density.sum().item():.6f}"
    )


if __name__ == "__main__":
    main()
