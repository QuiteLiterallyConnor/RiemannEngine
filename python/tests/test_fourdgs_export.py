"""Native export contracts and optional avala-ai/fourdgs interoperability."""

import math
import struct

import numpy as np
import pytest
import torch

import riemannengine as re_


pytestmark = pytest.mark.skipif(re_.gaussian is None,
                                reason="Gaussian support disabled")


def _model(degree=3, count=7):
    torch.manual_seed(19)
    model = re_.gaussian.Gaussian4DModel.from_points(
        torch.randn(count, 3), torch.rand(count, 3), sh_degree=degree)
    model.scale_raw.copy_(torch.randn(count, 3) - 2)
    model.rotation_raw.copy_(torch.randn(count, 4))
    model.velocity.copy_(torch.randn(count, 3))
    model.time_center.copy_(torch.linspace(2.2, 2.8, count).reshape(-1, 1))
    model.time_scale_raw.copy_(
        torch.linspace(math.log(0.01), math.log(0.5), count).reshape(-1, 1))
    model.f_rest.copy_(torch.rand_like(model.f_rest) * 7 - 3.5)
    return model


def test_export_framing_immutability_clipping_and_overwrite(tmp_path):
    g = re_.gaussian
    model = _model()
    model.f_dc[0, 0] = 100
    model.f_dc[0, 1] = -100
    model.f_rest[0, 0, 0] = 5
    before = [p.clone() for p in model.parameters()]
    path = tmp_path / "scene.4dgs"
    result = g.save_four_dgs(model, str(path))
    data = path.read_bytes()
    magic = b"\x894DGS1\r\n"
    assert data.startswith(magic) and data.endswith(magic)
    assert result.gaussians == 7 and result.bytes == len(data)
    assert result.clamped_rgb_components == 2
    assert result.clamped_sh_coefficients == 1
    assert struct.unpack_from("<BQ", data, len(data) - 37) == (2, 20)
    for left, right in zip(model.parameters(), before):
        assert torch.equal(left, right)
    with pytest.raises(RuntimeError, match="overwrite"):
        g.save_four_dgs(model, str(path))
    assert path.read_bytes() == data
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize("field,value", [
    ("time_start", float("nan")), ("time_end", 0),
    ("seconds_per_time_unit", 0), ("seconds_per_time_unit", float("inf")),
    ("cutoff", 0), ("cutoff", 1), ("chunk_size", 0), ("chunk_size", 65537),
])
def test_export_rejects_invalid_options(tmp_path, field, value):
    options = re_.gaussian.FourDGSExportOptions()
    setattr(options, field, value)
    with pytest.raises(RuntimeError):
        re_.gaussian.save_four_dgs(_model(), str(tmp_path / "bad.4dgs"), options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value,message", [
    ("rotation_raw", 0, "quaternion"),
    ("rotation_raw", 1e30, "quaternion"),
    ("velocity", float("nan"), "finite"),
    ("velocity", 1e30, "32-bit"),
    ("scale_raw", 100, "finite"),
])
def test_export_rejects_invalid_models_and_cleans_temporary(
        tmp_path, field, value, message):
    model = _model()
    getattr(model, field).fill_(value)
    with pytest.raises(RuntimeError, match=message):
        re_.gaussian.save_four_dgs(model, str(tmp_path / "bad.4dgs"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_reference_stream_index_sh_and_time_roundtrip(tmp_path, degree):
    fourdgs = pytest.importorskip("fourdgs")
    from fourdgs.indexed_reader import open_indexed, read_chunk
    from fourdgs.readable import BytesReadable
    from fourdgs.validate import validate

    model = _model(degree)
    options = re_.gaussian.FourDGSExportOptions()
    options.time_start = 2
    options.time_end = 3
    options.seconds_per_time_unit = 2.5
    options.chunk_size = 3
    path = tmp_path / "scene.4dgs"
    result = re_.gaussian.save_four_dgs(model, str(path), options)
    report = validate(path.read_bytes())
    assert report.ok, report.findings
    scene = fourdgs.read(str(path))
    decoded = scene.gaussians
    assert decoded.count == 7 and scene.header.sh_degree == degree
    assert scene.header.duration_sec == 2.5
    assert scene.header.attributes["color_space"] == "srgb-rec709-display"

    def check(actual, expected, field, bound):
        error = np.max(np.abs(np.asarray(actual, dtype=np.float64) - expected))
        assert error <= getattr(result, field) + 1e-12
        assert error <= float(scene.quantization.bounds[bound]) + 1e-12

    check(decoded.positions, model.xyz.numpy(), "max_position_error", "pos")
    check(np.log(decoded.scales.astype(np.float64)), model.scale_raw.numpy(),
          "max_scale_log_error", "scale_rel")
    check(decoded.motions, model.velocity.numpy().astype(np.float64) / 2.5,
          "max_velocity_error", "motion")
    check(decoded.mu_t, (model.time_center.numpy().ravel().astype(np.float64) - 2) * 2.5,
          "max_time_error", "time")
    check(np.log(decoded.sigma_t.astype(np.float64)),
          np.log(model.time_scales().numpy().ravel().astype(np.float64)) + math.log(2.5),
          "max_sigma_log_error", "sigma_rel")
    check(decoded.colors[:, :3], (0.5 + 0.28209479177387814 * model.f_dc).numpy(),
          "max_rgb_error", "rgb")
    check(decoded.colors[:, 3], model.opacities().numpy().ravel(),
          "max_opacity_error", "alpha")
    expected_q = model.rotations().numpy()[:, [1, 2, 3, 0]]
    signs = np.where(np.sum(decoded.rotations * expected_q, axis=1) < 0, -1, 1)
    check(decoded.rotations * signs[:, None], expected_q, "max_rotation_error", "rot")
    if degree:
        expected_sh = model.f_rest.numpy().transpose(0, 2, 1).reshape(7, -1)
        actual_sh = -4 + decoded.sh.astype(np.float64) * 8 / 255
        assert np.max(np.abs(actual_sh - expected_sh)) <= result.max_sh_error + 1e-12
    else:
        assert decoded.sh is None

    source = BytesReadable(path.read_bytes())
    indexed = open_indexed(source)
    assert indexed.summary_crc_ok and len(indexed.index) == 3
    chunks = [read_chunk(source, indexed, entry, max_sh_band=3)
              for entry in indexed.index]
    np.testing.assert_array_equal(
        np.concatenate([chunk["positions"] for chunk in chunks]).astype(np.float32),
        decoded.positions)
    begin = 0
    for chunk, entry in zip(chunks, indexed.index):
        assert set(chunk["sh"]) == set(range(1, degree + 1))
        assert read_chunk(source, indexed, entry)["sh"] == {}
        for band in range(1, degree + 1):
            first, last = band * band - 1, (band + 1) ** 2 - 1
            expected = decoded.sh[begin:begin + entry.gaussian_count].reshape(
                entry.gaussian_count, 3, -1)[:, :, first:last]
            np.testing.assert_array_equal(
                np.asarray(chunk["sh"][band]).reshape(expected.shape), expected)
        begin += entry.gaussian_count
    for time in (0, 0.3, 1.25, 2.49):
        state = scene.state_at(time)
        indices = state["indices"]
        model_time = 2 + time / 2.5
        expected_xyz = model.positions(model_time).numpy()[indices]
        expected_alpha = model.opacities(model_time).numpy().ravel()[indices]
        # Bound motion propagation, temporal-center shift, and opacity sensitivity.
        center_bound = (result.max_position_error + 2.5 * result.max_velocity_error
                        + np.max(np.abs(decoded.motions)) * result.max_time_error)
        np.testing.assert_allclose(state["centers"], expected_xyz,
                                   atol=center_bound + 1e-6, rtol=0)
        np.testing.assert_allclose(state["opacity"], expected_alpha, atol=0.025, rtol=0)
    assert len(scene.state_at(2.5)["indices"]) == 0
    assert indexed.chunks_for_time(2.5) == []


def test_reference_empty_scene_and_effective_minimum_sigma(tmp_path):
    fourdgs = pytest.importorskip("fourdgs")
    from fourdgs.validate import validate

    model = _model(0)
    empty = model.select(torch.empty(0, dtype=torch.int64))
    path = tmp_path / "empty.4dgs"
    result = re_.gaussian.save_four_dgs(empty, str(path))
    assert result.gaussians == 0
    assert validate(path.read_bytes()).ok
    assert fourdgs.read(str(path)).gaussians.count == 0
    model.time_scale_raw.fill_(-100)
    path = tmp_path / "tiny.4dgs"
    re_.gaussian.save_four_dgs(model, str(path))
    scene = fourdgs.read(str(path))
    np.testing.assert_allclose(scene.gaussians.sigma_t, 1e-6, rtol=0.006)
