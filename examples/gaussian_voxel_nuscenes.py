#!/usr/bin/env python
"""Train or run image-conditioned Gaussian voxel occupancy on nuScenes mini.

The native voxelizer and LiDAR target builder currently run on CPU. This
example intentionally rejects CUDA rather than silently copying trainable
Gaussian values between devices.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import riemannengine as re


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
GRID_SIZE = (40, 40, 8)
GRID_ORIGIN = (-40.0, -40.0, -4.0)
VOXEL_SIZE = (2.0, 2.0, 2.0)
FEATURE_CHANNELS = 16


def transform_matrix(translation, quaternion):
    """Return a rigid transform; nuScenes quaternions are ordered w, x, y, z."""
    w, x, y, z = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


class NuScenesMini:
    def __init__(self, root, image_size=(256, 144)):
        self.root = Path(root).expanduser().resolve()
        version = self.root / "v1.0-mini"
        if not version.is_dir():
            raise FileNotFoundError(f"nuScenes metadata not found: {version}")
        self.samples = self._table(version, "sample")
        self.sample_data = self._table(version, "sample_data")
        self.calibrations = self._table(version, "calibrated_sensor")
        self.poses = self._table(version, "ego_pose")
        sensors = self._table(version, "sensor")
        self.sensor_channels = {row["token"]: row["channel"] for row in sensors}
        self.calibrations_by_token = {
            row["token"]: row for row in self.calibrations
        }
        self.poses_by_token = {row["token"]: row for row in self.poses}
        self.rows_by_sample = {}
        for row in self.sample_data:
            if row["is_key_frame"]:
                self.rows_by_sample.setdefault(row["sample_token"], {})[
                    self._channel(row)
                ] = row
        self.image_size = image_size

    @staticmethod
    def _table(version, name):
        with (version / f"{name}.json").open(encoding="utf-8") as file:
            return json.load(file)

    def _channel(self, row):
        calibration = self.calibrations_by_token[row["calibrated_sensor_token"]]
        return self.sensor_channels[calibration["sensor_token"]]

    def _sensor_to_global(self, row):
        calibration = self.calibrations_by_token[row["calibrated_sensor_token"]]
        pose = self.poses_by_token[row["ego_pose_token"]]
        ego_to_global = transform_matrix(pose["translation"], pose["rotation"])
        sensor_to_ego = transform_matrix(
            calibration["translation"], calibration["rotation"]
        )
        return ego_to_global @ sensor_to_ego

    def load(self, sample_index, include_lidar=True):
        sample = self.samples[sample_index % len(self.samples)]
        rows = self.rows_by_sample.get(sample["token"], {})
        missing = [channel for channel in (*CAMERAS, "LIDAR_TOP")
                   if channel not in rows]
        if missing:
            raise RuntimeError(
                f"sample {sample['token']} is missing keyframes: {missing}"
            )

        reference_lidar = rows["LIDAR_TOP"]
        reference_pose = self.poses_by_token[reference_lidar["ego_pose_token"]]
        reference_ego_to_global = transform_matrix(
            reference_pose["translation"], reference_pose["rotation"]
        )
        reference_sensor_to_ego = transform_matrix(
            self.calibrations_by_token[
                reference_lidar["calibrated_sensor_token"]
            ]["translation"],
            self.calibrations_by_token[
                reference_lidar["calibrated_sensor_token"]
            ]["rotation"],
        )

        images, camera_from_reference, intrinsics = [], [], []
        width, height = self.image_size
        for channel in CAMERAS:
            row = rows[channel]
            calibration = self.calibrations_by_token[
                row["calibrated_sensor_token"]
            ]
            path = self.root / row["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"nuScenes image not found: {path}")
            with Image.open(path) as image:
                original_width, original_height = image.size
                image = image.convert("RGB").resize((width, height))
                image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
            images.append(torch.from_numpy(image_array).permute(2, 0, 1))

            camera_to_global = self._sensor_to_global(row)
            camera_from_reference.append(
                np.linalg.inv(camera_to_global) @ reference_ego_to_global
            )
            intrinsic = np.asarray(calibration["camera_intrinsic"],
                                   dtype=np.float32).reshape(3, 3).copy()
            intrinsic[0, :] *= width / original_width
            intrinsic[1, :] *= height / original_height
            intrinsics.append(intrinsic)

        result = {
            "sample_token": sample["token"],
            "images": torch.stack(images),
            "camera_from_reference": torch.from_numpy(
                np.asarray(camera_from_reference, dtype=np.float32)
            ),
            "intrinsics": torch.from_numpy(
                np.asarray(intrinsics, dtype=np.float32)
            ),
        }
        if include_lidar:
            lidar_path = self.root / reference_lidar["filename"]
            if not lidar_path.is_file():
                raise FileNotFoundError(
                    f"nuScenes LiDAR scan not found: {lidar_path}"
                )
            raw_points = np.fromfile(lidar_path, dtype=np.float32)
            if raw_points.size % 5:
                raise ValueError(
                    f"unexpected nuScenes LiDAR record size: {lidar_path}"
                )
            sensor_points = raw_points.reshape(-1, 5)[:, :3]
            finite = np.isfinite(sensor_points).all(axis=1)
            sensor_points = sensor_points[finite]
            points_h = np.concatenate(
                [
                    sensor_points,
                    np.ones((len(sensor_points), 1), dtype=np.float32),
                ],
                axis=1,
            )
            points_ego = (points_h @ reference_sensor_to_ego.T)[:, :3]
            within_range = np.linalg.norm(points_ego, axis=1) <= 80.0
            points_ego = points_ego[within_range].copy()
            result["points"] = torch.from_numpy(points_ego)
            result["sensor_origin"] = torch.from_numpy(
                reference_sensor_to_ego[:3, 3].astype(np.float32)
            )
        return result


def make_reference_grid():
    coordinates = torch.tensor(
        [
            [x, y, z]
            for x in range(GRID_SIZE[0])
            for y in range(GRID_SIZE[1])
            for z in range(GRID_SIZE[2])
        ],
        dtype=torch.int64,
    )
    levels = torch.zeros(len(coordinates), dtype=torch.int32)
    config = re.adaptive.AdaptiveVoxelConfig(
        list(VOXEL_SIZE), origin=list(GRID_ORIGIN)
    )
    return re.adaptive.AdaptiveSparseTensor.from_voxels(
        coordinates, levels, torch.zeros(len(coordinates), 1), config
    )


def make_anchors():
    xs = torch.arange(-38.0, 40.0, 4.0)
    ys = torch.arange(-38.0, 40.0, 4.0)
    zs = torch.tensor([-3.0, 1.0, 5.0, 9.0])
    return torch.cartesian_prod(xs, ys, zs)


class ImageToGaussianVoxel(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
        )
        self.gaussian_head = nn.Sequential(
            nn.Linear(35, 96),
            nn.GELU(),
            nn.Linear(96, 27),
        )
        self.occupancy_head = nn.Sequential(
            nn.Linear(FEATURE_CHANNELS + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch, reference):
        images = batch["images"]
        view_features = self.encoder(images)
        _, _, image_height, image_width = images.shape
        anchors = self.anchors
        anchors_h = torch.cat(
            [anchors, torch.ones_like(anchors[:, :1])], dim=1
        )
        camera_points = torch.einsum(
            "vij,kj->vki", batch["camera_from_reference"], anchors_h
        )[..., :3]
        projected = torch.einsum(
            "vij,vkj->vki", batch["intrinsics"], camera_points
        )
        depth = projected[..., 2]
        uv = projected[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
        grid = torch.stack(
            [
                2.0 * (uv[..., 0] + 0.5) / image_width - 1.0,
                2.0 * (uv[..., 1] + 0.5) / image_height - 1.0,
            ],
            dim=-1,
        ).unsqueeze(2)
        sampled = F.grid_sample(
            view_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1).permute(0, 2, 1)
        valid = (
            (depth > 1e-3)
            & (grid[..., 0, 0] >= -1.0)
            & (grid[..., 0, 0] <= 1.0)
            & (grid[..., 0, 1] >= -1.0)
            & (grid[..., 0, 1] <= 1.0)
        )
        weights = valid.to(sampled.dtype).unsqueeze(-1)
        fused = (sampled * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)

        anchor_input = anchors / anchors.new_tensor([40.0, 40.0, 8.0])
        raw = self.gaussian_head(torch.cat([fused, anchor_input], dim=1))
        positions = anchors + 0.5 * torch.tanh(raw[:, :3])
        rotations = F.normalize(
            raw[:, 3:7] + raw.new_tensor([1.0, 0.0, 0.0, 0.0]),
            dim=1,
        )
        scales = 0.15 + 1.35 * torch.sigmoid(raw[:, 7:10])
        opacities = torch.sigmoid(raw[:, 10:11])
        features = raw[:, 11:]
        gaussians = re.gaussian.GaussianTensor(
            positions,
            rotations,
            scales,
            opacities,
            torch.zeros_like(positions),
            features=features,
        )
        rasterized = re.gaussian_voxel.rasterize(gaussians, reference)
        decoder_input = torch.cat(
            [rasterized.voxels.features, rasterized.density], dim=1
        )
        occupancy = torch.sigmoid(self.occupancy_head(decoder_input)).T
        return rasterized, occupancy


def build_targets(batch, reference):
    points = batch["points"]
    origins = batch["sensor_origin"].expand_as(points)
    ray_batch_ids = torch.zeros(len(points), dtype=torch.int64)
    options = re.occupancy.LidarRayOptions()
    options.max_ray_distance = 80.0
    return re.occupancy.from_lidar(
        points, origins, ray_batch_ids, 1, reference, options
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\nuScenes"))
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("gaussian_voxel_nuscenes.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("gaussian_voxel_prediction.npz"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError(
            "This example requires a CUDA-enabled native voxelizer; the "
            "current Gaussian voxel and LiDAR target APIs are CPU-only."
        )
    dataset = NuScenesMini(args.root)
    reference = make_reference_grid()
    model = ImageToGaussianVoxel(make_anchors())
    if args.mode == "infer" and not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"inference checkpoint not found: {args.checkpoint}"
        )
    if args.checkpoint.is_file():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                         weights_only=True))

    if args.mode == "train":
        if args.steps < 1:
            parser.error("--steps must be positive")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        for step in range(args.steps):
            batch = dataset.load(args.sample_index + step, include_lidar=True)
            targets = build_targets(batch, reference)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            rasterized, predicted = model(batch, reference)
            loss = re.occupancy.masked_binary_cross_entropy(predicted, targets)
            loss.backward()
            optimizer.step()
            print(
                f"step={step + 1}/{args.steps} loss={loss.item():.6f} "
                f"sample={batch['sample_token']} "
                f"observed={int(targets.observed.sum())} "
                f"seconds={time.perf_counter() - started:.2f}"
            )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    else:
        batch = dataset.load(args.sample_index, include_lidar=False)
        model.eval()
        with torch.no_grad():
            rasterized, predicted = model(batch, reference)

    centers = reference.centers().detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_token=batch["sample_token"],
        centers=centers,
        occupancy=predicted.detach().cpu().numpy().reshape(-1),
        density=rasterized.density.detach().cpu().numpy().reshape(-1),
        voxel_coordinates=reference.coordinates().detach().cpu().numpy(),
    )
    print(f"wrote {args.output} for sample {batch['sample_token']}")


if __name__ == "__main__":
    main()
