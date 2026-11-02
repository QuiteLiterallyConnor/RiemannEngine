import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("PIL.Image")
from PIL import Image

import riemannengine as re


pytest.importorskip("riemannengine.gaussian_voxel")
pytest.importorskip("riemannengine.occupancy")

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "gaussian_voxel_nuscenes.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gaussian_voxel_nuscenes", _SCRIPT
)
visualization = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(visualization)


def make_reference_grid():
    coordinates = torch.tensor(
        [
            [x, y, z]
            for x in range(visualization.GRID_SIZE[0])
            for y in range(visualization.GRID_SIZE[1])
            for z in range(visualization.GRID_SIZE[2])
        ],
        dtype=torch.int64,
    )
    levels = torch.zeros(len(coordinates), dtype=torch.int32)
    config = re.adaptive.AdaptiveVoxelConfig(
        list(visualization.VOXEL_SIZE),
        origin=list(visualization.GRID_ORIGIN),
    )
    return re.adaptive.AdaptiveSparseTensor.from_voxels(
        coordinates, levels, torch.zeros(len(coordinates), 1), config
    )


def test_occupancy_visualizer_writes_nonblank_bev_and_height_slices(tmp_path):
    reference = make_reference_grid()
    voxel_count = len(reference)
    prediction = torch.linspace(0.0, 1.0, voxel_count).reshape(1, -1)
    density = torch.zeros(voxel_count, 1)
    density[::17] = 1.0
    points = torch.tensor([[2.0, -1.0, -1.0]])
    origins = torch.tensor([[-2.0, -1.0, -1.0]])
    targets = re.occupancy.from_lidar(
        points, origins, torch.zeros(1, dtype=torch.int64), 1, reference
    )

    bev_path, slices_path = visualization.render_occupancy_images(
        reference, prediction, density, tmp_path, "sample", targets
    )

    for path in (bev_path, slices_path):
        assert path.is_file()
        with Image.open(path) as image:
            assert image.width > 0 and image.height > 0
            assert len(image.getcolors(maxcolors=1_000_000)) > 8


def test_camera_visualizer_projects_gaussian_centers(tmp_path):
    gaussian = re.gaussian.GaussianTensor(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.full((1, 3), 0.5),
        torch.full((1, 1), 0.8),
        torch.zeros(1, 3),
    )
    batch = {
        "images": torch.zeros(1, 3, 12, 16),
        "camera_from_reference": torch.eye(4).reshape(1, 4, 4),
        "intrinsics": torch.tensor(
            [[[4.0, 0.0, 8.0], [0.0, 4.0, 6.0], [0.0, 0.0, 1.0]]]
        ),
    }

    path = visualization.render_camera_gaussians(
        batch, gaussian, tmp_path, "sample"
    )

    assert path.is_file()
    with Image.open(path) as image:
        assert image.size == (3 * 16 * 2, 12 * 2 + 30)
        assert image.getpixel((16, 42)) == (255, 80, 35)


def test_geometry_renderer_draws_solid_voxels_not_only_text(tmp_path):
    spec = visualization.GridSpec((2, 2, 2), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    reference = spec.reference()
    prediction = torch.zeros(1, len(reference))
    prediction[0, 0] = 1.0
    batch = {
        "images": torch.zeros(1, 3, 32, 32),
        "camera_from_reference": torch.eye(4).reshape(1, 4, 4),
        "intrinsics": torch.tensor(
            [[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0], [0.0, 0.0, 1.0]]]
        ),
    }
    geometry, cameras = visualization.render_voxel_geometry(
        batch, reference, prediction, tmp_path, "sample", spec, 0.5
    )
    with Image.open(geometry) as image:
        body = np.asarray(image)[65:]
        assert np.count_nonzero(body[..., 0] != 24) > 100
    with Image.open(cameras) as image:
        body = np.asarray(image)[30:94, :64]
        assert np.count_nonzero(body[..., :3]) > 50


def test_loader_inference_never_opens_lidar_returns(tmp_path, monkeypatch):
    metadata = tmp_path / "v1.0-mini"
    metadata.mkdir()
    Image.new("RGB", (32, 24), (25, 50, 75)).save(tmp_path / "camera.png")
    samples = [{"token": "sample", "scene_token": "scene"}]
    sensors, calibrations, rows = [], [], []
    for channel in (*visualization.CAMERAS, "LIDAR_TOP"):
        sensors.append({"token": channel, "channel": channel})
        calibrations.append({
            "token": channel, "sensor_token": channel,
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [[10.0, 0.0, 16.0],
                                 [0.0, 10.0, 12.0],
                                 [0.0, 0.0, 1.0]],
        })
        rows.append({
            "token": channel,
            "sample_token": "sample", "is_key_frame": True,
            "calibrated_sensor_token": channel, "ego_pose_token": "pose",
            "filename": "not-present.bin" if channel == "LIDAR_TOP" else "camera.png",
        })
    tables = {
        "sample": samples, "sample_data": rows,
        "sensor": sensors, "calibrated_sensor": calibrations,
        "ego_pose": [{"token": "pose", "translation": [1.0, 2.0, 3.0],
                      "rotation": [1.0, 0.0, 0.0, 0.0]}],
    }
    for name, values in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")
    dataset = visualization.NuScenesMini(tmp_path, image_size=(16, 12))

    def forbidden(*args, **kwargs):
        pytest.fail("inference attempted to read LiDAR")

    monkeypatch.setattr(np, "fromfile", forbidden)
    batch = dataset.load(0, include_lidar=False)
    assert "points" not in batch and "sensor_origin" not in batch
    assert batch["images"].shape == (6, 3, 12, 16)
    torch.testing.assert_close(batch["camera_from_reference"][0], torch.eye(4))
    assert batch["intrinsics"][0, 0, 0] == 5.0
    with pytest.raises(IndexError):
        dataset.load(-1)
    with pytest.raises(FileNotFoundError, match="LiDAR"):
        dataset.load(0, include_lidar=True)


def test_lidar_drops_ego_body_and_adds_sweeps_as_free_space_only(tmp_path):
    metadata = tmp_path / "v1.0-mini"
    metadata.mkdir()
    Image.new("RGB", (32, 24)).save(tmp_path / "camera.png")
    sensors, calibrations, rows = [], [], []
    for channel in (*visualization.CAMERAS, "LIDAR_TOP"):
        sensors.append({"token": channel, "channel": channel})
        calibrations.append({
            "token": channel, "sensor_token": channel,
            "translation": [0.0, 0.0, 1.0], "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [[10.0, 0.0, 16.0], [0.0, 10.0, 12.0],
                                 [0.0, 0.0, 1.0]],
        })
        rows.append({
            "token": channel, "sample_token": "sample", "is_key_frame": True,
            "calibrated_sensor_token": channel, "ego_pose_token": "pose",
            "filename": "camera.png", "prev": "", "next": "",
        })
    rows[-1].update(filename="key.bin", prev="sweep")
    rows.append({
        "token": "sweep", "sample_token": "sample", "is_key_frame": False,
        "calibrated_sensor_token": "LIDAR_TOP", "ego_pose_token": "earlier",
        "filename": "sweep.bin", "prev": "", "next": "LIDAR_TOP",
    })
    # Sensor-frame returns: a far wall, and one on the ego vehicle's roof.
    np.array([[10, 0, 0, 0, 0], [1, 0, 0, 0, 0]], np.float32).tofile(
        tmp_path / "key.bin")
    np.array([[5, 4, 0, 0, 0]], np.float32).tofile(tmp_path / "sweep.bin")
    tables = {
        "sample": [{"token": "sample", "scene_token": "scene"}],
        "sample_data": rows, "sensor": sensors, "calibrated_sensor": calibrations,
        "ego_pose": [
            {"token": "pose", "translation": [0.0, 0.0, 0.0],
             "rotation": [1.0, 0.0, 0.0, 0.0]},
            {"token": "earlier", "translation": [-2.0, 0.0, 0.0],
             "rotation": [1.0, 0.0, 0.0, 0.0]},
        ],
    }
    for name, values in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")
    dataset = visualization.NuScenesMini(tmp_path, image_size=(16, 12))

    batch = dataset.load(0, lidar_sweeps=0)
    torch.testing.assert_close(batch["points"], torch.tensor([[10.0, 0.0, 1.0]]))
    assert "sweep_points" not in batch
    batch = dataset.load(0, lidar_sweeps=3)
    torch.testing.assert_close(batch["sweep_points"], torch.tensor([[3.0, 4.0, 1.0]]))
    torch.testing.assert_close(batch["sweep_origins"], torch.tensor([[-2.0, 0.0, 1.0]]))

    from riemannengine.camera_occupancy import GridSpec

    reference = GridSpec((8, 4, 1), (-4.0, -1.0, 0.0), (2.0, 2.0, 2.0)).reference()
    keyframe_only = visualization.build_targets(
        dataset.load(0, lidar_sweeps=0), reference)
    merged = visualization.build_targets(batch, reference)
    assert torch.equal(merged.occupied, keyframe_only.occupied)
    assert merged.occupied.any()
    assert (merged.free & ~keyframe_only.free).any()
    assert torch.equal(merged.observed, merged.free | merged.occupied)
    assert not (merged.free & merged.occupied).any()
