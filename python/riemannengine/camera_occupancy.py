"""CPU/CUDA camera-to-Gaussian-to-voxel occupancy learning.

LiDAR labels belong to the loss/evaluator, never the model's forward inputs.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from . import adaptive, gaussian, gaussian_voxel


@dataclass(frozen=True)
class GridSpec:
    shape: tuple = (40, 40, 8)
    origin: tuple = (-40.0, -40.0, -4.0)
    voxel_size: tuple = (2.0, 2.0, 2.0)

    def __post_init__(self):
        if (len(self.shape) != 3 or len(self.origin) != 3 or
                len(self.voxel_size) != 3):
            raise ValueError("grid shape, origin, and voxel size must be 3D")
        if any(not isinstance(n, int) or n < 1 for n in self.shape):
            raise ValueError("grid shape must contain positive integers")
        if any(not math.isfinite(x) for x in self.origin):
            raise ValueError("grid origin must be finite")
        if any(not math.isfinite(x) or x <= 0 for x in self.voxel_size):
            raise ValueError("voxel sizes must be positive and finite")

    def reference(self, max_depth=0):
        coordinates = torch.tensor(
            [[x, y, z] for x in range(self.shape[0])
             for y in range(self.shape[1]) for z in range(self.shape[2])],
            dtype=torch.int64,
        )
        return adaptive.AdaptiveSparseTensor.from_voxels(
            coordinates, torch.zeros(len(coordinates), dtype=torch.int32),
            torch.zeros(len(coordinates), 1),
            adaptive.AdaptiveVoxelConfig(
                list(self.voxel_size), max_depth=max_depth, origin=list(self.origin)
            ),
        )


@dataclass(frozen=True)
class RefinementSpec:
    max_depth: int = 2
    max_voxels: int = 100_000
    max_splits_per_iteration: int = 1024
    max_iterations: int = 2
    score_threshold: float = 0.25
    min_density: float = 1e-4
    criterion: str = "uncertainty"

    def __post_init__(self):
        if self.criterion not in ("uncertainty", "density"):
            raise ValueError("refinement criterion must be uncertainty or density")
        if (not isinstance(self.max_depth, int) or not 0 <= self.max_depth <= 30 or
                not isinstance(self.max_iterations, int) or self.max_iterations < 0):
            raise ValueError("refinement depth must be in [0, 30] and iterations nonnegative")
        if any(not isinstance(value, int) or value < 1 for value in
               (self.max_voxels, self.max_splits_per_iteration)):
            raise ValueError("refinement voxel and split budgets must be positive integers")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.score_threshold, self.min_density)):
            raise ValueError("refinement thresholds must be nonnegative and finite")
        if self.criterion == "uncertainty" and self.score_threshold > 1:
            raise ValueError("uncertainty threshold must be in [0, 1]")

    def options(self):
        options = gaussian_voxel.AdaptiveRefinementOptions()
        options.criterion = (
            gaussian_voxel.RefinementCriterion.Density
            if self.criterion == "density"
            else gaussian_voxel.RefinementCriterion.OccupancyUncertainty
        )
        for name in ("max_depth", "max_voxels", "max_splits_per_iteration",
                     "max_iterations", "score_threshold", "min_density"):
            setattr(options, name, getattr(self, name))
        return options


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def scene_split(samples, validation_fraction=0.2, seed=7):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    scenes = sorted({sample["scene_token"] for sample in samples})
    if len(scenes) < 2:
        raise ValueError("scene-disjoint validation requires at least two scenes")
    random.Random(seed).shuffle(scenes)
    count = min(len(scenes) - 1, max(1, round(len(scenes) * validation_fraction)))
    validation_scenes = set(scenes[:count])
    return {
        "train": [i for i, sample in enumerate(samples)
                  if sample["scene_token"] not in validation_scenes],
        "validation": [i for i, sample in enumerate(samples)
                       if sample["scene_token"] in validation_scenes],
        "train_scenes": sorted(set(scenes) - validation_scenes),
        "validation_scenes": sorted(validation_scenes),
    }


def limit_samples(indices, samples, maximum):
    """Keep limited debug runs representative of all selected scenes."""
    if maximum < 0:
        raise ValueError("sample limit must be nonnegative")
    if maximum == 0 or maximum >= len(indices):
        return list(indices)
    groups = {}
    for index in indices:
        groups.setdefault(samples[index]["scene_token"], []).append(index)
    selected = []
    offset = 0
    while len(selected) < maximum:
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) == maximum:
                    return selected
        offset += 1
    return selected


POSITION_ENCODINGS = ("height_range", "absolute")


def position_encoding(positions, spec, kind="height_range"):
    """Fourier features of voxel centers.

    ``height_range`` encodes only height and horizontal distance from the ego
    origin, so the network cannot memorize a per-cell dataset layout (such as
    "the left curb is usually occupied") and must infer layout from images.
    ``absolute`` is the legacy x/y/z encoding kept for older checkpoints.
    """
    origin = positions.new_tensor(spec.origin)
    extent = positions.new_tensor(spec.shape) * positions.new_tensor(spec.voxel_size)
    if kind == "absolute":
        normalized = 2.0 * (positions - origin) / extent - 1.0
    elif kind == "height_range":
        height = 2.0 * (positions[:, 2:] - origin[2]) / extent[2] - 1.0
        max_range = 0.5 * extent[:2].norm()
        distance = positions[:, :2].norm(dim=1, keepdim=True)
        normalized = torch.cat([height, 2.0 * distance / max_range - 1.0], dim=1)
    else:
        raise ValueError(f"position encoding must be one of {POSITION_ENCODINGS}")
    frequencies = positions.new_tensor([1.0, 2.0, 4.0, 8.0]) * math.pi
    angles = normalized.unsqueeze(-1) * frequencies
    return torch.cat(
        [normalized, angles.sin().flatten(1), angles.cos().flatten(1)], dim=1
    )


def project_image_features(features, positions, camera_from_reference,
                           intrinsics, image_size):
    height, width = image_size
    homogeneous = torch.cat(
        [positions, torch.ones_like(positions[:, :1])], dim=1
    )
    camera_points = torch.einsum(
        "vij,nj->vni", camera_from_reference, homogeneous
    )[..., :3]
    projected = torch.einsum("vij,vnj->vni", intrinsics, camera_points)
    depth = camera_points[..., 2]
    uv = projected[..., :2] / depth.clamp_min(1e-4).unsqueeze(-1)
    valid = (
        (depth > 1e-3) & (uv[..., 0] >= 0) & (uv[..., 0] < width)
        & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    )
    grid = torch.stack(
        [2 * (uv[..., 0] + 0.5) / width - 1,
         2 * (uv[..., 1] + 0.5) / height - 1], dim=-1
    )
    # Invalid projections must not overflow grid_sample or contribute features.
    grid = torch.where(valid.unsqueeze(-1), grid, torch.zeros_like(grid))
    sampled = F.grid_sample(
        features, grid.unsqueeze(2), align_corners=False,
        mode="bilinear", padding_mode="zeros",
    ).squeeze(-1).permute(0, 2, 1)
    weights = valid.to(sampled.dtype).unsqueeze(-1)
    return ((sampled * weights).sum(0) / weights.sum(0).clamp_min(1),
            weights.sum(0) / features.shape[0])


class CameraOccupancyModel(nn.Module):
    """Image-only, shared-weight Gaussian prediction and spatial 3D decoding."""

    def __init__(self, spec=GridSpec(), channels=16, refinement=None,
                 encoding="height_range"):
        super().__init__()
        if gaussian_voxel is None:
            raise RuntimeError("RiemannEngine was built without Gaussian voxel support")
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("Gaussian feature channels must be a positive integer")
        self.spec = spec
        self.channels = channels
        self.refinement = refinement
        self.encoding_kind = encoding
        reference = spec.reference()
        self.register_buffer("anchors", reference.centers().float())
        self.register_buffer("coordinates", reference.coordinates().long())
        self.register_buffer(
            "encoding", position_encoding(self.anchors, spec, encoding)
        )
        encoding_channels = self.encoding.shape[1]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
        )
        self.gaussian_head = nn.Sequential(
            nn.Linear(32 + encoding_channels + 1, 96), nn.GELU(),
            nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, 11 + channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(channels + 1 + encoding_channels, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 1, 1),
        )
        self.options = gaussian_voxel.GaussianVoxelOptions()
        self.options.support_radius = 2.5
        self.options.opacity_density_scale = 8.0

    def forward(self, images, camera_from_reference, intrinsics, reference):
        if images.device != self.anchors.device:
            raise ValueError("images and model must use the same device")
        if (images.device.type == "cuda" and
                not gaussian_voxel.cuda_available()):
            raise ValueError("CUDA voxelization requires a CPU_ONLY=OFF extension build")
        if (camera_from_reference.device != images.device or
                intrinsics.device != images.device):
            raise ValueError("images and calibration must use the same device")
        if (images.ndim != 4 or images.shape[1] != 3 or
                images.shape[0] < 1):
            raise ValueError("images must have shape [views, 3, H, W]")
        if camera_from_reference.shape != (len(images), 4, 4):
            raise ValueError("camera_from_reference must have shape [views, 4, 4]")
        if intrinsics.shape != (len(images), 3, 3):
            raise ValueError("intrinsics must have shape [views, 3, 3]")
        if (images.dtype != torch.float32 or
                camera_from_reference.dtype != torch.float32 or
                intrinsics.dtype != torch.float32):
            raise ValueError("images and calibration must use float32")
        if not all(torch.isfinite(value).all() for value in
                   (images, camera_from_reference, intrinsics)):
            raise ValueError("images and calibration must be finite")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("images must contain RGB values in [0, 1]")
        if (not reference.coordinates().equal(self.coordinates.cpu()) or
                reference.levels().ne(0).any()):
            raise ValueError("reference topology does not match the model grid")
        if (tuple(reference.config.origin) != self.spec.origin or
                tuple(reference.config.base_voxel_size) != self.spec.voxel_size):
            raise ValueError("reference geometry does not match the model grid")
        features = self.encoder(images * 2.0 - 1.0)
        fused, visibility = project_image_features(
            features, self.anchors, camera_from_reference, intrinsics,
            images.shape[-2:],
        )
        raw = self.gaussian_head(torch.cat(
            [fused, self.encoding, visibility], dim=1
        ))
        spacing = raw.new_tensor(self.spec.voxel_size)
        positions = self.anchors + spacing * torch.tanh(raw[:, :3])
        rotations = F.normalize(
            raw[:, 3:7] + raw.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=1
        )
        scales = spacing * (0.25 + 0.75 * torch.sigmoid(raw[:, 7:10]))
        gaussians = gaussian.GaussianTensor(
            positions, rotations, scales, torch.sigmoid(raw[:, 10:11]),
            torch.zeros_like(positions), features=raw[:, 11:],
        )
        if self.refinement is None:
            rasterized = gaussian_voxel.rasterize(gaussians, reference, self.options)
        else:
            refined = gaussian_voxel.rasterize_adaptive(
                gaussians, reference, self.options, self.refinement.options()
            )
            fine = refined.rasterized
            # The spatial decoder keeps its checkpoint-compatible uniform grid.
            # Aggregate integrated mass, not an unweighted mean of mixed leaves.
            mass = fine.voxels.with_features(fine.density).resample_to(
                reference, adaptive.Reduction.Sum
            ).features
            weighted = fine.voxels.with_features(
                fine.voxels.features * fine.density
            ).resample_to(reference, adaptive.Reduction.Sum)
            features = weighted.features / mass.clamp_min(
                torch.finfo(mass.dtype).tiny
            )
            rasterized = SimpleNamespace(
                voxels=weighted.with_features(features), density=mass,
                occupancy=-torch.expm1(-mass), adaptive=refined,
            )
        rows = torch.cat(
            [rasterized.voxels.features, torch.log1p(rasterized.density),
             self.encoding], dim=1
        )
        nx, ny, nz = self.spec.shape
        volume = rows.new_zeros((nx, ny, nz, rows.shape[1]))
        x, y, z = self.coordinates.T
        volume[x, y, z] = rows
        logits_volume = self.decoder(
            volume.permute(3, 2, 1, 0).unsqueeze(0)
        )[0, 0]
        logits = logits_volume[z, y, x].unsqueeze(0)
        return logits, rasterized, gaussians


def forward_batch(model, batch, reference):
    device = model.anchors.device
    return model(
        batch["images"].to(device), batch["camera_from_reference"].to(device),
        batch["intrinsics"].to(device), reference,
    )


def column_ceiling_free_prior(targets, reference, margin_levels=0):
    """Unknown cells above the highest LiDAR return in their column.

    Columns without a return use the sample's highest return level. The mask
    is meant for a weak free-space prior, never as a hard label: it suppresses
    unsupported roofs and floating slabs where LiDAR provides no evidence.
    """
    if margin_levels < 0:
        raise ValueError("margin_levels must be non-negative")
    coordinates = reference.coordinates().long().to(targets.occupied.device)
    if coordinates.shape[0] != targets.occupied.shape[1]:
        raise ValueError("reference topology does not match target width")
    x, y, z = coordinates.unbind(1)
    columns = (x - x.min()) * (int(y.max() - y.min()) + 1) + (y - y.min())
    column_count = int(columns.max()) + 1
    batch = targets.occupied.shape[0]
    occupied_level = torch.where(
        targets.occupied, z.expand(batch, -1), torch.full_like(z, -1)
    )
    ceiling = torch.full((batch, column_count), -1, dtype=z.dtype,
                         device=z.device)
    ceiling = ceiling.scatter_reduce(
        1, columns.expand(batch, -1), occupied_level, reduce="amax"
    )
    sample_ceiling = ceiling.amax(dim=1, keepdim=True)
    ceiling = torch.where(ceiling >= 0, ceiling, sample_ceiling)
    above = z.expand(batch, -1) > ceiling.gather(
        1, columns.expand(batch, -1)
    ) + margin_levels
    return above & ~targets.observed & (sample_ceiling >= 0)


def masked_occupancy_loss(logits, targets, prior_free=None, prior_weight=0.0):
    if logits.shape != targets.occupied.shape:
        raise ValueError("logits and targets must have matching [B, V] shapes")
    if (targets.observed.dtype != torch.bool or
            targets.free.dtype != torch.bool or
            targets.occupied.dtype != torch.bool):
        raise ValueError("target masks must be boolean")
    if not torch.equal(targets.occupied | targets.free, targets.observed):
        raise ValueError("observed must equal occupied OR free")
    if (targets.occupied & targets.free).any():
        raise ValueError("occupied and free masks must not overlap")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not targets.observed.any():
        raise ValueError("cannot train on a sample with no observed voxels")
    occupied = targets.occupied.to(logits.device)
    free = targets.free.to(logits.device)
    observed = targets.observed.to(logits.device)
    labels = occupied.to(logits.dtype)
    per_voxel = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    classes = [per_voxel[mask].mean()
               for mask in (occupied, free) if mask.any()]
    balanced_bce = torch.stack(classes).mean()
    probabilities = logits.sigmoid()[observed]
    observed_labels = labels[observed]
    intersection = (probabilities * observed_labels).sum()
    dice = (2 * intersection + 1) / (
        probabilities.sum() + observed_labels.sum() + 1
    )
    loss = balanced_bce + 0.5 * (1 - dice)
    if prior_free is not None and prior_weight:
        if prior_weight < 0:
            raise ValueError("prior_weight must be non-negative")
        prior_free = prior_free.to(logits.device)
        if prior_free.shape != logits.shape or prior_free.dtype != torch.bool:
            raise ValueError("prior_free must be a boolean [B, V] mask")
        if (prior_free & observed).any():
            raise ValueError("prior_free must only cover unobserved cells")
        if prior_free.any():
            loss = loss + prior_weight * F.binary_cross_entropy_with_logits(
                logits[prior_free], torch.zeros_like(logits[prior_free])
            )
    return loss


class OccupancyMetrics:
    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("occupancy threshold must be between zero and one")
        self.threshold = threshold
        self.tp = self.fp = self.tn = self.fn = 0
        self.bce_sum = 0.0
        self.observed = 0
        self.unknown = self.unknown_predicted = 0
        self.prior = self.prior_predicted = 0

    def update(self, logits, targets, prior_free=None):
        if logits.shape != targets.occupied.shape:
            raise ValueError("logits and targets must have matching [B, V] shapes")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must be finite")
        mask = targets.observed.to(logits.device)
        labels = targets.occupied.to(logits.device)[mask]
        predictions = logits.detach().sigmoid()[mask] >= self.threshold
        self.tp += int((predictions & labels).sum())
        self.fp += int((predictions & ~labels).sum())
        self.tn += int((~predictions & ~labels).sum())
        self.fn += int((~predictions & labels).sum())
        self.observed += int(mask.sum())
        self.bce_sum += float(F.binary_cross_entropy_with_logits(
            logits.detach()[mask], labels.float(), reduction="sum"
        ))
        predicted = logits.detach().sigmoid() >= self.threshold
        self.unknown += int((~mask).sum())
        self.unknown_predicted += int((predicted & ~mask).sum())
        if prior_free is not None:
            prior_free = prior_free.to(logits.device)
            self.prior += int(prior_free.sum())
            self.prior_predicted += int((predicted & prior_free).sum())

    def compute(self):
        if not self.observed:
            raise ValueError("no observed cells available for evaluation")
        occupied_union = self.tp + self.fp + self.fn
        free_union = self.tn + self.fp + self.fn
        occupied_iou = self.tp / occupied_union if occupied_union else 1.0
        free_iou = self.tn / free_union if free_union else 1.0
        return {
            "occupied_iou": occupied_iou, "free_iou": free_iou,
            "mean_iou": (occupied_iou + free_iou) / 2,
            "precision": self.tp / max(1, self.tp + self.fp),
            "recall": self.tp / max(1, self.tp + self.fn),
            "bce": self.bce_sum / self.observed,
            "observed": self.observed,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "threshold": self.threshold,
            "always_occupied_iou": (self.tp + self.fn) / self.observed,
            "always_free_iou": 0.0 if self.tp + self.fn else 1.0,
            "unknown_cells": self.unknown,
            "unknown_predicted_occupied": self.unknown_predicted,
            "above_ceiling_cells": self.prior,
            "above_ceiling_predicted_occupied": self.prior_predicted,
        }


CHECKPOINT_VERSION = 1


def save_checkpoint(path, model, optimizer, step, split, sample_tokens,
                    image_size, seed, best_iou, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "grid": asdict(model.spec), "channels": model.channels,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "split": split, "sample_tokens": sample_tokens,
        "image_size": list(image_size), "seed": seed,
        "best_iou": best_iou, "history": history,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "refinement": asdict(model.refinement) if model.refinement else None,
        "position_encoding": model.encoding_kind,
        "cuda_rng": torch.cuda.get_rng_state_all()
        if model.anchors.device.type == "cuda" else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or
            checkpoint.get("format_version") != CHECKPOINT_VERSION):
        raise ValueError(
            "unsupported checkpoint; old smoke-example weights are not compatible"
        )
    required = {
        "grid", "channels", "model", "optimizer", "step", "split",
        "sample_tokens", "image_size", "seed", "best_iou", "history",
        "torch_rng", "python_rng",
    }
    if missing := required - checkpoint.keys():
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    grid = {name: tuple(value) for name, value in checkpoint["grid"].items()}
    refinement = checkpoint.get("refinement")
    model = CameraOccupancyModel(
        GridSpec(**grid), checkpoint["channels"],
        RefinementSpec(**refinement) if refinement else None,
        # Checkpoints written before this field existed used absolute x/y/z.
        checkpoint.get("position_encoding", "absolute"),
    )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
