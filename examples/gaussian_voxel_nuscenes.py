#!/usr/bin/env python
"""Train, evaluate, overfit-check, or infer camera-only occupancy on nuScenes."""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import random
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

import riemannengine as re
from riemannengine.camera_occupancy import (
    CameraOccupancyModel,
    GridSpec,
    OccupancyMetrics,
    RefinementSpec,
    column_ceiling_free_prior,
    forward_batch,
    limit_samples,
    load_checkpoint,
    masked_occupancy_loss,
    save_checkpoint,
    scene_split,
    seed_everything,
)


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
GRID_SIZE = (40, 40, 8)
# z=-5 keeps the road surface (ego z ~ 0) inside the [-1, 1) layer and the
# cameras (z ~ 1.5) in the layer above, rather than on a voxel boundary.
GRID_ORIGIN = (-40.0, -40.0, -5.0)
VOXEL_SIZE = (2.0, 2.0, 2.0)
# Ego-frame box (rear-axle origin) of the nuScenes Renault Zoe, with margin.
EGO_BODY_BOX = (np.array([-1.5, -1.2, -0.5]), np.array([4.0, 1.2, 2.5]))
FEATURE_CHANNELS = 16


def image_from_array(values):
    return Image.fromarray(np.asarray(values, dtype=np.uint8))


def heatmap_colors(values):
    values = np.clip(values, 0.0, 1.0)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    palette = np.array(
        [
            [35, 15, 75],
            [45, 85, 180],
            [30, 190, 210],
            [210, 230, 75],
            [245, 100, 35],
        ],
        dtype=np.float32,
    )
    return np.stack(
        [
            np.interp(values, stops, palette[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    ).astype(np.uint8)


def make_bev_maps(reference, predicted_occupancy, density, targets=None,
                  spec=GridSpec()):
    coordinates = reference.coordinates().detach().cpu().numpy()
    predicted = predicted_occupancy.detach().cpu().numpy().reshape(-1)
    density = density.detach().cpu().numpy().reshape(-1)
    nx, ny, nz = spec.shape
    predicted_by_height = np.zeros((nz, ny, nx), dtype=np.float32)
    density_by_height = np.zeros_like(predicted_by_height)
    x, y, z = coordinates.T
    predicted_by_height[z, y, x] = predicted
    density_by_height[z, y, x] = density
    predicted_bev = predicted_by_height.max(axis=0)
    density_bev = density_by_height.sum(axis=0)

    occupied_bev = np.zeros((ny, nx), dtype=bool)
    free_bev = np.zeros((ny, nx), dtype=bool)
    if targets is not None:
        occupied = targets.occupied[0].detach().cpu().numpy()
        free = targets.free[0].detach().cpu().numpy()
        np.logical_or.at(occupied_bev, (y[occupied], x[occupied]), True)
        np.logical_or.at(free_bev, (y[free], x[free]), True)
        free_bev &= ~occupied_bev

    return (predicted_by_height, density_by_height, predicted_bev,
            density_bev, occupied_bev, free_bev)


def render_occupancy_images(reference, predicted_occupancy, density, output_dir,
                            sample_token, targets=None, spec=GridSpec()):
    output_dir.mkdir(parents=True, exist_ok=True)
    (_, density_by_height, predicted_bev, density_bev,
     occupied_bev, free_bev) = make_bev_maps(
         reference, predicted_occupancy, density, targets, spec
     )
    positive_density = np.log1p(density_bev)
    nonzero_density = positive_density[positive_density > 0.0]
    density_scale = (
        float(np.percentile(nonzero_density, 99.0))
        if nonzero_density.size else 0.0
    )
    if density_scale > 0.0:
        density_bev = np.clip(positive_density / density_scale, 0.0, 1.0)
    else:
        density_bev = positive_density

    nx, ny, nz = spec.shape
    cell_size = 12
    title_height = 30
    panel_width, panel_height = nx * cell_size, ny * cell_size
    panel_count = 4 if targets is not None else 2
    footer_height = 58
    canvas = Image.new(
        "RGB",
        (panel_count * panel_width, panel_height + title_height + footer_height),
        (28, 30, 36),
    )
    draw = ImageDraw.Draw(canvas)
    titles = [
        "Predicted occupancy (max over height)",
        "Gaussian density (log scale, sum over height)",
    ]
    panels = [
        heatmap_colors(np.flipud(predicted_bev)),
        heatmap_colors(np.flipud(density_bev)),
    ]
    if targets is not None:
        target_map = np.full((ny, nx, 3), (55, 58, 64), dtype=np.uint8)
        target_map[free_bev] = (45, 115, 175)
        target_map[occupied_bev] = (235, 75, 55)
        target_map = np.flipud(target_map)
        titles.extend(["LiDAR target (unknown / free / occupied)",
                       "Prediction with LiDAR occupied cells"])
        panels.append(target_map)

        overlay = heatmap_colors(np.flipud(predicted_bev))
        overlay[occupied_bev[::-1]] = (
            0.45 * overlay[occupied_bev[::-1]] +
            0.55 * np.array([255, 45, 35])
        ).astype(np.uint8)
        panels.append(overlay)

    for index, (title, panel) in enumerate(zip(titles, panels)):
        left = index * panel_width
        canvas.paste(image_from_array(panel).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        ), (left, title_height))
        draw.text((left + 6, 8), title, fill=(245, 245, 245))
    draw.multiline_text(
        (6, panel_height + title_height + 8),
        textwrap.fill(
            "Occupancy: purple=0, orange=1. Density: log scale, nonzero p99. "
            "LiDAR: gray=unknown, blue=free, red=occupied. Up=+y, right=+x.",
            width=max(20, panel_count * panel_width // 6),
        ),
        fill="white",
    )
    bev_path = output_dir / f"{sample_token}_occupancy_bev.png"
    canvas.save(bev_path)

    columns = 4
    rows = (nz + columns - 1) // columns
    slice_canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + title_height)),
        (28, 30, 36),
    )
    draw = ImageDraw.Draw(slice_canvas)
    all_log_density = np.log1p(density_by_height)
    nonzero_slices = all_log_density[all_log_density > 0.0]
    slice_density_scale = (
        float(np.percentile(nonzero_slices, 99.0))
        if nonzero_slices.size else 0.0
    )
    for level in range(nz):
        column, row = level % columns, level // columns
        left, top = column * panel_width, row * (panel_height + title_height)
        log_density = np.log1p(density_by_height[level])
        if slice_density_scale > 0.0:
            log_density = np.clip(log_density / slice_density_scale, 0.0, 1.0)
        slice_panel = heatmap_colors(np.flipud(log_density))
        slice_canvas.paste(image_from_array(slice_panel).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        ), (left, top + title_height))
        height_center = spec.origin[2] + (level + 0.5) * spec.voxel_size[2]
        draw.text(
            (left + 6, top + 8),
            f"z={height_center:.1f} m Gaussian density",
            fill=(245, 245, 245),
        )
    slices_path = output_dir / f"{sample_token}_occupancy_slices.png"
    slice_canvas.save(slices_path)
    return bev_path, slices_path


def render_camera_gaussians(batch, gaussians, output_dir, sample_token):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = batch["images"].detach().cpu().numpy()
    positions = gaussians.positions.detach().cpu().numpy()
    opacities = gaussians.opacities.detach().cpu().numpy().reshape(-1)
    camera_from_reference = batch["camera_from_reference"].cpu().numpy()
    intrinsics = batch["intrinsics"].cpu().numpy()
    view_count, _, height, width = images.shape
    scale = 2
    title_height = 30
    panel_width, panel_height = width * scale, height * scale
    columns = 3
    rows = (view_count + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * panel_width, rows * (panel_height + title_height)),
        (28, 30, 36),
    )
    draw = ImageDraw.Draw(canvas)

    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1), dtype=positions.dtype)],
        axis=1,
    )
    for view in range(view_count):
        camera_points = homogeneous @ camera_from_reference[view].T
        projected = camera_points[:, :3] @ intrinsics[view].T
        depth = projected[:, 2]
        safe_depth = np.maximum(depth, 1e-6)
        uv = projected[:, :2] / safe_depth[:, None]
        valid = (
            (depth > 1e-3)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )

        column, row = view % columns, view // columns
        left, top = column * panel_width, row * (panel_height + title_height)
        rgb = np.clip(images[view].transpose(1, 2, 0) * 255.0, 0, 255)
        panel = image_from_array(rgb.astype(np.uint8)).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(panel, (left, top + title_height))
        draw.text(
            (left + 6, top + 8),
            f"{CAMERAS[view]} - projected Gaussian centers",
            fill=(245, 245, 245),
        )
        for u, v, opacity in zip(uv[valid, 0], uv[valid, 1], opacities[valid]):
            center_x = left + int(round(u * scale))
            center_y = top + title_height + int(round(v * scale))
            radius = 2 + int(round(2.0 * float(opacity)))
            draw.ellipse(
                (center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius),
                fill=(255, 80, 35),
                outline=(255, 245, 220),
            )

    camera_path = output_dir / f"{sample_token}_camera_gaussians.png"
    canvas.save(camera_path)
    return camera_path


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
        self.sample_data_by_token = {row["token"]: row for row in self.sample_data}
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

    def load(self, sample_index, include_lidar=True, lidar_sweeps=0):
        if sample_index < 0 or sample_index >= len(self.samples):
            raise IndexError("sample index is outside the dataset")
        if lidar_sweeps < 0:
            raise ValueError("lidar_sweeps must be non-negative")
        sample = self.samples[sample_index]
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
            "scene_token": sample["scene_token"],
            "images": torch.stack(images),
            "camera_from_reference": torch.from_numpy(
                np.asarray(camera_from_reference, dtype=np.float32)
            ),
            "intrinsics": torch.from_numpy(
                np.asarray(intrinsics, dtype=np.float32)
            ),
        }
        if include_lidar:
            points_ego = self._read_lidar(reference_lidar) @ \
                reference_sensor_to_ego[:3, :3].T + reference_sensor_to_ego[:3, 3]
            result["points"] = torch.from_numpy(points_ego.astype(np.float32))
            result["sensor_origin"] = torch.from_numpy(
                reference_sensor_to_ego[:3, 3].astype(np.float32)
            )
            # Earlier sweeps are expressed in the reference ego frame and are
            # used for free-space rays only, so moving objects do not smear.
            reference_from_global = np.linalg.inv(reference_ego_to_global)
            sweep_points, sweep_origins = [], []
            neighbours = []
            for direction in ("prev", "next"):
                row = reference_lidar
                for _ in range(lidar_sweeps):
                    if not row[direction]:
                        break
                    row = self.sample_data_by_token[row[direction]]
                    neighbours.append(row)
            for row in neighbours:
                reference_from_sensor = (
                    reference_from_global @ self._sensor_to_global(row)
                )
                points = (self._read_lidar(row) @ reference_from_sensor[:3, :3].T
                          + reference_from_sensor[:3, 3])
                sweep_points.append(points)
                sweep_origins.append(np.broadcast_to(
                    reference_from_sensor[:3, 3], points.shape
                ))
            if sweep_points:
                result["sweep_points"] = torch.from_numpy(
                    np.concatenate(sweep_points).astype(np.float32)
                )
                result["sweep_origins"] = torch.from_numpy(
                    np.concatenate(sweep_origins).astype(np.float32)
                )
        return result

    def _read_lidar(self, row):
        lidar_path = self.root / row["filename"]
        if not lidar_path.is_file():
            raise FileNotFoundError(f"nuScenes LiDAR scan not found: {lidar_path}")
        raw_points = np.fromfile(lidar_path, dtype=np.float32)
        if raw_points.size % 5:
            raise ValueError(f"unexpected nuScenes LiDAR record size: {lidar_path}")
        sensor_points = raw_points.reshape(-1, 5)[:, :3].astype(np.float64)
        if not np.isfinite(sensor_points).all():
            raise ValueError(f"nonfinite LiDAR returns: {lidar_path}")
        # About a quarter of LIDAR_TOP returns hit the ego vehicle itself; as
        # targets they would mark the cells around the cameras as occupied.
        calibration = self.calibrations_by_token[row["calibrated_sensor_token"]]
        sensor_to_ego = transform_matrix(
            calibration["translation"], calibration["rotation"]
        )
        ego_points = sensor_points @ sensor_to_ego[:3, :3].T + sensor_to_ego[:3, 3]
        low, high = EGO_BODY_BOX
        on_body = np.all((ego_points >= low) & (ego_points <= high), axis=1)
        return sensor_points[~on_body]


def build_targets(batch, reference):
    points = batch["points"]
    origins = batch["sensor_origin"].expand_as(points)
    ray_batch_ids = torch.zeros(len(points), dtype=torch.int64)
    options = re.occupancy.LidarRayOptions()
    options.max_ray_distance = 80.0
    targets = re.occupancy.from_lidar(
        points, origins, ray_batch_ids, 1, reference, options
    )
    if "sweep_points" not in batch:
        return targets
    sweep_points = batch["sweep_points"]
    sweeps = re.occupancy.from_lidar(
        sweep_points, batch["sweep_origins"],
        torch.zeros(len(sweep_points), dtype=torch.int64), 1, reference,
        options,
    )
    free = (targets.free | sweeps.free) & ~targets.occupied
    return MergedTargets(targets.occupied, free, free | targets.occupied)


@dataclass(frozen=True)
class MergedTargets:
    occupied: torch.Tensor
    free: torch.Tensor
    observed: torch.Tensor


class PreparedSamples:
    def __init__(self, dataset, reference, capacity=16, lidar_sweeps=0,
                 ceiling_margin=0):
        self.dataset = dataset
        self.reference = reference
        self.capacity = capacity
        self.lidar_sweeps = lidar_sweeps
        self.ceiling_margin = ceiling_margin
        self.cache = OrderedDict()

    def get(self, index):
        if index not in self.cache:
            batch = self.dataset.load(
                index, include_lidar=True, lidar_sweeps=self.lidar_sweeps
            )
            targets = build_targets(batch, self.reference)
            if not targets.observed.any():
                raise ValueError(f"sample {index} has no observations inside the grid")
            batch["prior_free"] = column_ceiling_free_prior(
                targets, self.reference, self.ceiling_margin
            )
            self.cache[index] = (batch, targets)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


@torch.no_grad()
def evaluate(model, prepared, indices, reference, threshold):
    if not indices:
        raise ValueError("evaluation set must not be empty")
    model.eval()
    metrics = OccupancyMetrics(threshold)
    balanced_loss = 0.0
    for index in indices:
        batch, targets = prepared.get(index)
        logits, _, _ = forward_batch(model, batch, reference)
        metrics.update(logits, targets, batch["prior_free"])
        balanced_loss += float(masked_occupancy_loss(logits, targets))
    result = metrics.compute()
    result["balanced_loss"] = balanced_loss / len(indices)
    result["samples"] = len(indices)
    return result


def cube_vertices(centers, voxel_size):
    offsets = np.array(
        [[x, y, z] for z in (-0.5, 0.5)
         for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
    )
    return centers[:, None, :] + offsets[None] * np.asarray(voxel_size)


def render_voxel_geometry(batch, reference, prediction, output_dir,
                          sample_token, spec, threshold, targets=None):
    """Render solid voxel faces, rather than projected Gaussian-center dots."""
    centers = reference.centers().cpu().numpy()
    selected = prediction.detach().cpu().numpy().reshape(-1) >= threshold
    sets = [("Predicted occupied voxels (including unobserved)", selected)]
    if targets is not None:
        sets.append(("Prediction within observed mask (diagnostic)",
                     selected & targets.observed[0].cpu().numpy()))
        sets.append(("LiDAR occupied voxels", targets.occupied[0].cpu().numpy()))
    width, height = 800, 700
    canvas = Image.new("RGB", (width * len(sets), height), (24, 27, 35))
    draw = ImageDraw.Draw(canvas)
    bounds = np.array([spec.origin,
                       np.asarray(spec.origin) +
                       np.asarray(spec.shape) * np.asarray(spec.voxel_size)])

    def iso(points):
        return np.stack(
            [points[..., 0] - points[..., 1],
             0.45 * (points[..., 0] + points[..., 1]) - 1.2 * points[..., 2]],
            axis=-1,
        )

    projected_bounds = iso(cube_vertices(
        bounds.mean(0, keepdims=True), bounds[1] - bounds[0]
    )[0])
    lower, upper = projected_bounds.min(0), projected_bounds.max(0)
    scale = min((width - 70) / (upper[0] - lower[0]),
                (height - 110) / (upper[1] - lower[1]))
    faces = ([1, 3, 7, 5], [2, 6, 7, 3], [4, 5, 7, 6])
    for panel_index, (title, mask) in enumerate(sets):
        active = centers[mask]
        vertices = cube_vertices(active, spec.voxel_size)
        screen = (iso(vertices) - lower) * scale + np.array([35, 65])
        colors = heatmap_colors(
            (active[:, 2] - spec.origin[2]) /
            (spec.shape[2] * spec.voxel_size[2])
        )
        order = np.argsort(active[:, 0] + active[:, 1] + active[:, 2])
        for index in order:
            for face, shade in zip(faces, (0.65, 0.8, 1.0)):
                polygon = [
                    (float(screen[index, corner, 0] + panel_index * width),
                     float(screen[index, corner, 1])) for corner in face
                ]
                color = tuple(int(c * shade) for c in colors[index])
                draw.polygon(polygon, fill=color, outline=(20, 25, 30))
        draw.text((panel_index * width + 15, 15), title, fill="white")
        draw.text(
            (panel_index * width + 15, 35),
            f"{int(mask.sum())} cells | threshold={threshold:.2f} | color=height",
            fill="white",
        )
        if not len(active):
            draw.text((panel_index * width + 50, 150),
                      "No occupied voxels above threshold", fill="white")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / f"{sample_token}_voxel_geometry.png"
    canvas.save(geometry_path)

    images = batch["images"].cpu().numpy()
    transforms = batch["camera_from_reference"].cpu().numpy()
    intrinsics = batch["intrinsics"].cpu().numpy()
    view_count, _, image_height, image_width = images.shape
    columns = 3
    panel_width, panel_height = image_width * 2, image_height * 2
    camera_canvas = Image.new(
        "RGB", (columns * panel_width,
                math.ceil(view_count / columns) * (panel_height + 30)),
        (24, 27, 35),
    )
    vertices = cube_vertices(centers[selected], spec.voxel_size)
    colors = heatmap_colors(
        (centers[selected, 2] - spec.origin[2]) /
        (spec.shape[2] * spec.voxel_size[2])
    )
    all_faces = ([0, 1, 3, 2], [4, 5, 7, 6], [0, 4, 5, 1],
                 [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3])
    for view in range(view_count):
        rgb = (np.clip(images[view].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        panel = image_from_array(rgb).resize((panel_width, panel_height)).convert("RGBA")
        overlay = Image.new("RGBA", panel.size)
        overlay_draw = ImageDraw.Draw(overlay)
        camera = (vertices @ transforms[view, :3, :3].T +
                  transforms[view, :3, 3])
        projection = camera @ intrinsics[view].T
        depth = camera[..., 2]
        uv = projection[..., :2] / np.maximum(depth[..., None], 0.01) * 2
        polygons = []
        for index in range(len(vertices)):
            if depth[index].min() <= 0.1:
                continue
            for face in all_faces:
                polygon = uv[index, face]
                if (polygon[:, 0].max() < 0 or polygon[:, 0].min() >= panel_width or
                        polygon[:, 1].max() < 0 or polygon[:, 1].min() >= panel_height):
                    continue
                polygons.append((float(depth[index, face].mean()), index, face))
        for _, index, face in sorted(polygons, key=lambda item: -item[0]):
            overlay_draw.polygon(
                [tuple(point) for point in uv[index, face]],
                fill=tuple(int(c) for c in colors[index]) + (150,),
                outline=(255, 255, 255, 190),
            )
        panel = Image.alpha_composite(panel, overlay).convert("RGB")
        left, top = (view % columns) * panel_width, (view // columns) * (panel_height + 30)
        camera_canvas.paste(panel, (left, top + 30))
        ImageDraw.Draw(camera_canvas).text(
            (left + 6, top + 8), f"{CAMERAS[view]} - predicted voxel faces",
            fill="white",
        )
    camera_path = output_dir / f"{sample_token}_camera_voxels.png"
    camera_canvas.save(camera_path)
    return geometry_path, camera_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\nuScenes"))
    parser.add_argument("--mode", choices=("train", "eval", "infer", "overfit"),
                        default="train")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--steps", type=int, default=1000,
                        help="additional optimizer steps, also when resuming")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="scenes accumulated before each optimizer step")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--adaptive-refinement", action="store_true",
                        help="opt in to budgeted mixed-resolution Gaussian integration")
    parser.add_argument("--refine-depth", type=int, default=2)
    parser.add_argument("--refine-max-voxels", type=int, default=100_000)
    parser.add_argument("--refine-splits", type=int, default=1024)
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--refine-threshold", type=float, default=0.25)
    parser.add_argument("--refine-min-density", type=float, default=1e-4)
    parser.add_argument("--refine-criterion", choices=("uncertainty", "density"),
                        default="uncertainty")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lidar-sweeps", type=int, default=20,
                        help="neighbouring LiDAR sweeps per direction added as free-space rays")
    parser.add_argument("--ceiling-prior-weight", type=float, default=0.1,
                        help="weak free loss above each column's highest return")
    parser.add_argument("--ceiling-margin", type=int, default=0,
                        help="levels kept unconstrained above the highest return")
    parser.add_argument("--position-encoding", default="height_range",
                        choices=("height_range", "absolute"),
                        help="new runs only; absolute x/y lets the decoder memorize layout")
    parser.add_argument("--sample-cache", type=int, default=16,
                        help="prepared frames (images + targets) kept in memory")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--grid-shape", type=int, nargs=3, default=GRID_SIZE)
    parser.add_argument("--grid-origin", type=float, nargs=3, default=GRID_ORIGIN)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=VOXEL_SIZE)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 144),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/camera_occupancy/latest.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/camera_occupancy/prediction.npz"))
    parser.add_argument("--report", type=Path,
                        help="JSON metrics and acceptance report")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="directory for PNGs (default: output path without .npz)",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--minimum-overfit-iou", type=float, default=0.8)
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.eval_every, args.threads,
           args.sample_cache, *args.image_size) < 1:
        parser.error("steps, batch size, eval interval, threads, sample cache and image size must be positive")
    if (args.max_train_samples < 0 or args.max_eval_samples < 0 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("sample limits must be nonnegative and learning rate positive")
    if not 0 < args.threshold < 1 or not 0 < args.minimum_overfit_iou <= 1:
        parser.error("threshold must be in (0, 1) and minimum overfit IoU in (0, 1]")
    if (args.lidar_sweeps < 0 or args.ceiling_margin < 0 or
            not math.isfinite(args.ceiling_prior_weight) or
            args.ceiling_prior_weight < 0):
        parser.error("LiDAR sweeps, ceiling margin and prior weight must be nonnegative")
    if args.resume and args.mode not in ("train", "overfit"):
        parser.error("--resume is only valid for training")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requires a CUDA-enabled PyTorch and available GPU")
        if not re.gaussian_voxel.cuda_available():
            parser.error("--device cuda requires bindings built with CPU_ONLY=OFF")
    seed_everything(args.seed)
    loaded = None
    if args.mode in ("infer", "eval") or args.resume:
        model, loaded = load_checkpoint(args.checkpoint)
        spec = model.spec
        image_size = tuple(loaded["image_size"])
    else:
        if args.checkpoint.exists():
            raise FileExistsError(
                f"{args.checkpoint} exists; use --resume or choose a new checkpoint"
            )
        spec = GridSpec(tuple(args.grid_shape), tuple(args.grid_origin),
                        tuple(args.voxel_size))
        model = CameraOccupancyModel(spec, encoding=args.position_encoding)
        image_size = tuple(args.image_size)

    if args.adaptive_refinement:
        refinement = RefinementSpec(
            max_depth=args.refine_depth, max_voxels=args.refine_max_voxels,
            max_splits_per_iteration=args.refine_splits,
            max_iterations=args.refine_iterations,
            score_threshold=args.refine_threshold, min_density=args.refine_min_density,
            criterion=args.refine_criterion,
        )
        if args.resume and model.refinement != refinement:
            parser.error("resume requires the saved refinement configuration")
        model.refinement = refinement
    model.to(args.device)
    dataset = NuScenesMini(args.root, image_size=image_size)
    tokens = [sample["token"] for sample in dataset.samples]
    if loaded is not None:
        if tokens != loaded["sample_tokens"]:
            raise ValueError("dataset sample tokens/order do not match the checkpoint")
        split = loaded["split"]
        if args.mode == "train" and split.get("overfit_indices"):
            raise ValueError("an overfit checkpoint cannot resume as general training")
    else:
        split = scene_split(dataset.samples, args.validation_fraction, args.seed)
    reference = spec.reference(
        max_depth=model.refinement.max_depth if model.refinement else 0
    )
    supervision = {
        "lidar_sweeps": args.lidar_sweeps,
        "ceiling_prior_weight": args.ceiling_prior_weight,
        "ceiling_margin": args.ceiling_margin,
    }
    if args.resume:
        saved = {key: split[key] for key in supervision if key in split}
        if saved and saved != supervision:
            raise ValueError(f"resume requires the saved supervision {saved}")
    if args.mode in ("train", "overfit"):
        split.update(supervision)
    prepared = PreparedSamples(
        dataset, reference, capacity=args.sample_cache,
        lidar_sweeps=args.lidar_sweeps,
        ceiling_margin=args.ceiling_margin,
    )
    report = {
        "mode": args.mode, "split": split, "supervision": supervision, "grid": {
            "shape": spec.shape, "origin": spec.origin,
            "voxel_size": spec.voxel_size,
        },
        "device": args.device, "checkpoint": str(args.checkpoint),
    }
    training = args.mode in ("train", "overfit")
    if training:
        train_indices = split["train"]
        validation_indices = split["validation"]
        if args.mode == "overfit":
            index = args.sample_index if args.sample_index is not None else train_indices[0]
            if index not in train_indices:
                raise ValueError("overfit sample must belong to the training scenes")
            if loaded and split.get("overfit_indices") != [index]:
                raise ValueError("resume must use the same overfit sample")
            split["overfit_indices"] = [index]
            train_indices = validation_indices = [index]
            report["warning"] = "Single-scene overfit, NOT held-out generalization"
        elif args.sample_index is not None:
            parser.error("--sample-index selects a sample only for overfit/eval/infer")
        if loaded is not None and "training_indices" in split:
            train_indices = split["training_indices"]
            validation_indices = split["evaluation_indices"]
            if (args.batch_size != split["batch_size"] or
                    args.threshold != split["threshold"]):
                raise ValueError("resume requires the saved batch size and threshold")
            if args.max_train_samples or args.max_eval_samples:
                raise ValueError("resumed sample subsets come from the checkpoint")
        else:
            train_indices = limit_samples(
                train_indices, dataset.samples, args.max_train_samples
            )
            validation_indices = limit_samples(
                validation_indices, dataset.samples, args.max_eval_samples
            )
            split.update(
                training_indices=train_indices,
                evaluation_indices=validation_indices,
                batch_size=args.batch_size, threshold=args.threshold,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        start = 0
        best_iou = -1.0
        history = []
        if loaded is not None:
            optimizer.load_state_dict(loaded["optimizer"])
            start = loaded["step"]
            best_iou = loaded["best_iou"]
            history = loaded["history"]
            torch.set_rng_state(loaded["torch_rng"])
            if args.device == "cuda" and loaded.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(loaded["cuda_rng"])
            random.setstate(loaded["python_rng"])
        baseline = evaluate(
            model, prepared, validation_indices, reference, args.threshold
        )
        if not history:
            history.append({"step": start, **baseline})
        elif args.mode == "overfit":
            report["resume_baseline"] = baseline
            baseline = {key: value for key, value in history[0].items()
                        if key != "step"}
        report["baseline"] = baseline
        print(f"baseline: {json.dumps(baseline)}", flush=True)
        for step in range(start + 1, start + args.steps + 1):
            model.train()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.batch_size):
                batch, targets = prepared.get(random.choice(train_indices))
                logits, _, _ = forward_batch(model, batch, reference)
                loss = masked_occupancy_loss(
                    logits, targets, batch["prior_free"],
                    split.get("ceiling_prior_weight", 0.0),
                )
                (loss / args.batch_size).backward()
                step_loss += float(loss.detach()) / args.batch_size
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step == start + 1 or step % 10 == 0:
                print(
                    f"step={step} loss={step_loss:.5f} "
                    f"seconds={time.perf_counter() - started:.2f}", flush=True
                )
            if step % args.eval_every != 0 and step != start + args.steps:
                continue
            result = evaluate(
                model, prepared, validation_indices, reference, args.threshold
            )
            history.append({"step": step, **result})
            improved = result["occupied_iou"] > best_iou
            best_iou = max(best_iou, result["occupied_iou"])
            save_checkpoint(
                args.checkpoint, model, optimizer, step, split, tokens,
                image_size, loaded["seed"] if loaded else args.seed,
                best_iou, history,
            )
            if improved:
                best_path = args.checkpoint.with_name(
                    args.checkpoint.stem + "_best" + args.checkpoint.suffix
                )
                save_checkpoint(
                    best_path, model, optimizer, step, split, tokens, image_size,
                    loaded["seed"] if loaded else args.seed, best_iou, history,
                )
            print(f"evaluation step={step}: {json.dumps(result)}", flush=True)
        report["metrics"] = result
        report["training_metrics"] = evaluate(
            model, prepared, train_indices, reference, args.threshold
        )
        report["history"] = history
        report["training_samples"] = len(train_indices)
        if args.mode == "overfit":
            report["acceptance"] = {
                "minimum_occupied_iou": args.minimum_overfit_iou,
                "maximum_bce_ratio": 0.6,
                "passed": (
                    result["occupied_iou"] >= args.minimum_overfit_iou and
                    result["bce"] <= baseline["bce"] * 0.6
                ),
            }
        selected_index = validation_indices[0]
        batch, targets = prepared.get(selected_index)
    else:
        selected_index = args.sample_index
        if selected_index is None:
            selected_index = split["validation"][0]
        if args.mode == "eval":
            if selected_index not in split["validation"]:
                raise ValueError("evaluation sample must belong to held-out scenes")
            indices = [selected_index] if args.sample_index is not None else split["validation"]
            indices = limit_samples(
                indices, dataset.samples, args.max_eval_samples
            )
            report["metrics"] = evaluate(
                model, prepared, indices, reference, args.threshold
            )
            batch, targets = prepared.get(selected_index)
        else:
            batch = dataset.load(selected_index, include_lidar=False)
            targets = None

    model.eval()
    with torch.no_grad():
        logits, rasterized, gaussians = forward_batch(model, batch, reference)
        predicted = logits.sigmoid()
        if training and args.mode == "overfit":
            ablated = dict(batch)
            ablated["images"] = torch.zeros_like(batch["images"])
            ablated_logits, _, _ = forward_batch(model, ablated, reference)
            report["image_ablation_mean_probability_change"] = float(
                (ablated_logits.sigmoid() - predicted).abs().mean()
            )

    centers = reference.centers().detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_token=batch["sample_token"],
        centers=centers,
        occupancy=predicted.detach().cpu().numpy().reshape(-1),
        density=rasterized.density.detach().cpu().numpy().reshape(-1),
        voxel_coordinates=reference.coordinates().detach().cpu().numpy(),
        voxel_size=np.asarray(spec.voxel_size),
        grid_origin=np.asarray(spec.origin),
        threshold=args.threshold,
    )
    if model.refinement is not None:
        refined = rasterized.adaptive
        fine = refined.rasterized
        adaptive_path = args.output.with_name(args.output.stem + "_adaptive.npz")
        np.savez_compressed(
            adaptive_path,
            centers=fine.voxels.centers().numpy(),
            voxel_coordinates=fine.voxels.coordinates().numpy(),
            levels=fine.voxels.levels().numpy(),
            voxel_sizes=fine.voxels.voxel_sizes().numpy(),
            density=fine.density.detach().cpu().numpy(),
            gaussian_occupancy=fine.occupancy.detach().cpu().numpy(),
            features=fine.voxels.features.detach().cpu().numpy(),
        )
        report["refinement"] = {
            "leaf_counts": refined.leaf_counts, "stop_reason": refined.stop_reason,
            "output": str(adaptive_path),
            "note": "Adaptive Gaussian integration; decoded occupancy stays on the base grid",
        }
    visualization_dir = args.visualization_dir or args.output.with_suffix("")
    camera_path = render_camera_gaussians(
        batch, gaussians, visualization_dir, batch["sample_token"]
    )
    bev_path, slices_path = render_occupancy_images(
        reference, predicted, rasterized.density, visualization_dir,
        batch["sample_token"],
        targets=targets, spec=spec,
    )
    geometry_path, camera_voxels_path = render_voxel_geometry(
        batch, reference, predicted, visualization_dir,
        batch["sample_token"], spec, args.threshold, targets,
    )
    report["sample_token"] = batch["sample_token"]
    report["images"] = [str(path) for path in
                        (camera_path, bev_path, slices_path,
                         geometry_path, camera_voxels_path)]
    report_path = args.report or args.output.with_suffix(".json")
    write_json(report_path, report)
    print(f"wrote {args.output}, {report_path} and PNGs in {visualization_dir}",
          flush=True)
    if args.mode == "overfit" and not report["acceptance"]["passed"]:
        raise RuntimeError(
            "overfit acceptance FAILED; inspect the report rather than treating "
            "this run as a learned reconstruction"
        )


if __name__ == "__main__":
    main()
