#!/usr/bin/env python
"""Measure CPU/CUDA Gaussian voxelization, optionally with refinement/backward."""

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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--adaptive-refinement", action="store_true")
    parser.add_argument("--max-voxels", type=int, default=100_000)
    args = parser.parse_args()
    if min(args.resolution, args.gaussians, args.iterations) < 1:
        parser.error("resolution, gaussians, and iterations must be positive")
    if args.device == "cuda" and (
            not torch.cuda.is_available() or not re.gaussian_voxel.cuda_available()):
        parser.error("CUDA benchmarking requires CUDA PyTorch, a GPU, and CUDA bindings")
    device = torch.device(args.device)

    n = args.resolution
    coordinates = torch.cartesian_prod(
        torch.arange(n), torch.arange(n), torch.arange(n)
    ).to(torch.int64)
    levels = torch.zeros(len(coordinates), dtype=torch.int32)
    config = re.adaptive.AdaptiveVoxelConfig(
        [1.0, 1.0, 1.0], max_depth=2 if args.adaptive_refinement else 0,
        origin=[-n / 2.0] * 3
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
    fields = [value.to(device).requires_grad_(args.backward) for value in
              (positions, rotations, scales, opacities, features)]
    gaussians = re.gaussian.GaussianTensor(
        *fields[:4], colors.to(device), features=fields[4]
    )
    refinement = re.gaussian_voxel.AdaptiveRefinementOptions()
    refinement.max_voxels = args.max_voxels

    def run():
        for field in fields:
            field.grad = None
        if args.adaptive_refinement:
            output = re.gaussian_voxel.rasterize_adaptive(
                gaussians, reference, refinement=refinement
            ).rasterized
        else:
            output = re.gaussian_voxel.rasterize(gaussians, reference)
        if args.backward:
            (output.density.sum() + output.voxels.features.square().sum()).backward()
        return output

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    run()
    synchronize()
    elapsed = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = run()
        synchronize()
        elapsed.append(time.perf_counter() - started)
    print(f"torch={torch.__version__} device={device} backward={args.backward}")
    print(
        f"gaussians={args.gaussians} voxels={len(output.voxels)} "
        f"channels={features.shape[1]} iterations={args.iterations}"
    )
    print(
        f"median_seconds={statistics.median(elapsed):.6f} "
        f"mean_seconds={statistics.mean(elapsed):.6f} "
        f"density_mass={output.density.sum().item():.6f}"
    )


if __name__ == "__main__":
    main()
