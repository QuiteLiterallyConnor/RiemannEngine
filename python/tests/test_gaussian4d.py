"""Dynamic Gaussian bindings and D-NeRF input contracts."""

import json
import math
import struct
import zlib

import pytest
import torch

import riemannengine as re_


pytestmark = pytest.mark.skipif(re_.gaussian is None,
                                reason="Gaussian support disabled")


def test_time_conditioning_snapshot_and_checkpoint(tmp_path):
    g = re_.gaussian
    model = g.Gaussian4DModel.from_points(
        torch.zeros(1, 3), torch.full((1, 3), 0.5), sh_degree=0)
    model.time_center.fill_(0.5)
    model.velocity.fill_(0.2)
    assert torch.allclose(model.positions(1.0), torch.full((1, 3), 0.1))
    camera = g.Camera.look_at([0, 0, -3], [0, 0, 0], [0, -1, 0],
                             16, 16, 1.0)
    dynamic = g.render(model, camera, time=0.7)
    snapshot = g.render(model.at_time(0.7), camera)
    assert torch.allclose(dynamic.image, snapshot.image, atol=1e-6)
    path = tmp_path / "dynamic.pt"
    torch.save([p.detach().cpu() for p in model.parameters()], path)
    restored = g.Gaussian4DModel(*torch.load(path, weights_only=True))
    assert torch.equal(restored.velocity, model.velocity)
    assert torch.equal(g.render(restored, camera, time=0.7).image, dynamic.image)


def test_trainer_configuration_and_temporal_gradients():
    g = re_.gaussian
    model = g.Gaussian4DModel.from_points(
        torch.zeros(1, 3), torch.full((1, 3), 0.5), sh_degree=0)
    config = g.Gaussian4DTrainerConfig()
    assert isinstance(config, g.GaussianTrainerConfig)
    config.densify = False
    config.lambda_ssim = 0.0
    trainer = g.Gaussian4DTrainer(model, config)
    camera = g.Camera.look_at([0, 0, -3], [0, 0, 0], [0, -1, 0],
                             16, 16, 1.0)
    result = trainer.step(camera, time=0.7, target=torch.zeros(16, 16, 3))
    assert result.iteration == 1
    assert math.isfinite(result.loss)
    for parameter in (model.velocity, model.time_center, model.time_scale_raw):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_trainer_checkpoint_continuation(tmp_path):
    g = re_.gaussian
    model = g.Gaussian4DModel.from_points(
        torch.zeros(1, 3), torch.full((1, 3), 0.5), sh_degree=0)
    config = g.Gaussian4DTrainerConfig()
    config.densify = False
    config.lambda_ssim = 0.0
    camera = g.Camera.look_at([0, 0, -3], [0, 0, 0], [0, -1, 0],
                             16, 16, 1.0)
    target = torch.zeros(16, 16, 3)
    trainer = g.Gaussian4DTrainer(model, config)
    trainer.step(camera, 0.7, target)
    path = str(tmp_path / "training.pt")
    trainer.save_checkpoint(path)
    expected = trainer.step(camera, 0.7, target)
    restored_model = model.clone()
    restored = g.Gaussian4DTrainer(restored_model, config)
    restored.load_checkpoint(path)
    assert restored.iteration == 1
    actual = restored.step(camera, 0.7, target)
    assert actual.loss == pytest.approx(expected.loss)
    for left, right in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(left, right)


def _png_chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _dataset(root):
    # A non-square 3x2 RGBA image with constant half-transparent red pixels.
    header = struct.pack(">IIBBBBB", 3, 2, 8, 6, 0, 0, 0)
    scanlines = (b"\x00" + bytes([255, 0, 0, 128]) * 3) * 2
    (root / "frame.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b""))
    document = {
        "camera_angle_x": 1.0,
        "frames": [{
            "file_path": "./frame.png",
            "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                 [0, 0, 1, 3], [0, 0, 0, 1]],
        }],
    }
    (root / "transforms_train.json").write_text(json.dumps(document))
    return document


def test_dnerf_alpha_camera_extension_and_nondivisible_resize(tmp_path):
    _dataset(tmp_path)
    g = re_.gaussian
    options = g.DNerfLoadOptions()
    options.downscale = 2
    view, = g.load_dnerf_views(str(tmp_path), "train", options)
    assert view.time == 0.0
    assert view.image.shape == (1, 1, 3)
    assert torch.allclose(view.image[0, 0],
                          torch.tensor([1, 127 / 255, 127 / 255]), atol=1e-6)
    assert view.camera.fx == pytest.approx(0.5 / math.tan(0.5))
    assert view.camera.fy == pytest.approx(0.75 / math.tan(0.5))
    assert torch.equal(view.camera.position(), torch.tensor([0.0, 0.0, 3.0]))
    assert torch.equal(view.camera.translation(), torch.tensor([0.0, 0.0, 3.0]))


def test_dnerf_reports_invalid_metadata_and_missing_images(tmp_path):
    document = _dataset(tmp_path)
    g = re_.gaussian
    path = tmp_path / "transforms_train.json"
    path.write_text("{invalid json")
    with pytest.raises(RuntimeError, match="parse"):
        g.load_dnerf_views(str(tmp_path), "train")
    path.write_text(json.dumps(document))
    options = g.DNerfLoadOptions()
    options.downscale = 4
    with pytest.raises(RuntimeError, match="downscale"):
        g.load_dnerf_views(str(tmp_path), "train", options)
    (tmp_path / "frame.png").unlink()
    with pytest.raises(RuntimeError, match="failed to load PNG"):
        g.load_dnerf_views(str(tmp_path), "train")
