"""Loop timing, interactive camera controls, decoded rendering and GIF playback."""

import importlib.util
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import riemannengine as re_

pytest.importorskip("PIL.Image")
from PIL import Image, ImageSequence


_PATH = Path(__file__).resolve().parents[2] / "examples" / "view_fourdgs.py"
_SPEC = importlib.util.spec_from_file_location("view_fourdgs", _PATH)
viewer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = viewer
_SPEC.loader.exec_module(viewer)


def test_ping_pong_has_continuous_endpoints_and_half_open_time():
    duration = 2.5
    assert viewer.loop_time(0, duration) == 0
    assert viewer.loop_time(1, duration) == 0
    assert viewer.loop_time(0.5, duration) < duration
    assert viewer.loop_time(0.5, duration) == math.nextafter(duration, 0)
    for phase in np.linspace(0, 1, 101):
        assert 0 <= viewer.loop_time(phase, duration) < duration
        assert viewer.loop_time(phase, duration) == pytest.approx(
            viewer.loop_time(1 - phase, duration), abs=1e-14)
    epsilon = 1e-5
    assert viewer.loop_time(epsilon, duration) / epsilon < 0.001
    assert viewer.loop_time(1 - epsilon, duration) / epsilon < 0.001
    assert (duration - viewer.loop_time(0.5 - epsilon, duration)) / epsilon < 0.001


@pytest.mark.parametrize("axis", ["y", "z"])
def test_orbit_pan_zoom_and_pole_limits(axis):
    camera = viewer.OrbitCamera((0, 0, 0), 3, up_axis=axis)
    np.testing.assert_array_equal(camera.eye(0), camera.eye(1))
    for phase in np.linspace(0, 1, 21):
        assert np.linalg.norm(camera.eye(phase) - camera.target) == pytest.approx(3)
    camera.rotate(10, 1e6)
    assert math.degrees(camera.pitch) == pytest.approx(85)
    camera.pan(20, -10, 512)
    assert np.isfinite(camera.target).all() and np.linalg.norm(camera.target) > 0
    camera.zoom(1)
    assert camera.distance < 3
    camera.zoom(-1e10)
    assert math.isfinite(camera.distance) and camera.distance <= 1e8


@pytest.mark.parametrize("arguments", [
    ["--seconds", "nan"], ["--seconds", "0"], ["--seconds", "0.001"], ["--fps", "101"],
    ["--fps", "0"], ["--width", "0"], ["--pitch", "90"],
    ["--fov", "180"], ["--distance", "-1"], ["--target", "0", "nan", "0"],
])
def test_cli_rejects_invalid_options(arguments):
    with pytest.raises(SystemExit):
        viewer.parse_args(["scene.4dgs", *arguments])


class FakeRenderer:
    duration = 1.0
    device = "cpu"

    def __init__(self):
        self.phases = []

    def render(self, phase, camera, width, height, orbit=True):
        self.phases.append(phase)
        return Image.new("RGB", (width, height),
                         (round(phase * 240), round(viewer.loop_time(phase, 1) * 240), 128))


def test_gif_loop_timing_frame_count_and_overwrite(tmp_path):
    renderer = FakeRenderer()
    camera = viewer.OrbitCamera((0, 0, 0), 3)
    path = tmp_path / "loop.gif"
    viewer.export_gif(renderer, camera, path, 32, 24, 1, 20)
    assert renderer.phases == [index / 20 for index in range(20)]
    with Image.open(path) as image:
        assert image.info["loop"] == 0
        assert image.size == (32, 24) and image.n_frames == 20
        assert sum(frame.info["duration"] for frame in ImageSequence.Iterator(image)) == 1000
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        viewer.export_gif(renderer, camera, path, 32, 24, 1, 20)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("tmp*"))


def test_gif_failure_does_not_publish_or_leave_temporary(tmp_path, monkeypatch):
    renderer = FakeRenderer()
    camera = viewer.OrbitCamera((0, 0, 0), 3)

    def fail(*args, **kwargs):
        raise OSError("test disk write failure")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="disk write"):
        viewer.export_gif(renderer, camera, tmp_path / "bad.gif", 32, 24, 1, 2)
    assert not list(tmp_path.iterdir())


def test_reference_decoded_render_matches_reference_state(tmp_path):
    pytest.importorskip("fourdgs")
    if re_.gaussian is None:
        pytest.skip("Gaussian support disabled")
    g = re_.gaussian
    model = g.Gaussian4DModel.from_points(
        torch.tensor([[-0.2, 0.1, 0], [0.2, 0.2, 0.3]]),
        torch.tensor([[0.8, 0.2, 0.1], [0.2, 0.3, 0.7]]), sh_degree=3)
    model.velocity.fill_(0.1)
    model.scale_raw.fill_(math.log(0.15))
    model.time_scale_raw.fill_(math.log(0.4))
    model.opacity_raw.fill_(2)
    model.time_center.fill_(0.5)
    model.f_rest.copy_(torch.linspace(-0.2, 0.2, model.f_rest.numel()).reshape_as(model.f_rest))
    path = tmp_path / "test.4dgs"
    g.save_four_dgs(model, str(path))
    renderer = viewer.SceneRenderer(path, re_, "cpu", "white")
    camera = viewer.OrbitCamera((0, 0, 0), 2)
    first = np.array(renderer.render(0, camera, 48, 48))
    np.testing.assert_array_equal(first, np.array(renderer.render(1, camera, 48, 48)))
    np.testing.assert_allclose(first, np.array(renderer.render(1 - 1e-7, camera, 48, 48)),
                               atol=1, rtol=0)
    assert np.min(first) < 250
    assert np.isfinite(renderer.fitted_camera(width=128, height=64).eye()).all()
    assert renderer.fitted_camera(width=128, height=64).distance > renderer.fitted_camera().distance
    phase = 0.25
    image = np.array(renderer.render(phase, camera, 48, 48)).astype(float) / 255
    options = g.RenderOptions()
    options.background = torch.ones(3)
    native_camera = g.Camera.look_at(
        camera.eye(phase).tolist(), list(camera.target), camera.up.tolist(),
        48, 48, camera.fov)
    data = renderer.scene.gaussians
    state = renderer.scene.state_at(viewer.loop_time(phase, 1))
    indices = state["indices"]
    reference = g.GaussianModel(
        torch.tensor(state["centers"], dtype=torch.float32),
        (torch.tensor(data.colors[indices, :3]) - 0.5) / viewer.SH_C0,
        torch.tensor((data.sh[indices].astype(np.float64) * 8 / 255 - 4)
                     .reshape(-1, 3, 15).transpose(0, 2, 1), dtype=torch.float32),
        torch.logit(torch.tensor(state["opacity"][:, None], dtype=torch.float32)),
        torch.tensor(data.scales[indices]).log(),
        torch.tensor(state["orientations"][:, [3, 0, 1, 2]], dtype=torch.float32))
    expected = g.render(reference, native_camera, options).image.numpy()
    np.testing.assert_allclose(image, expected, atol=1 / 255, rtol=0)
    # Respect the encoded window rather than playing a raw 4D tensor beyond it.
    renderer.scene.gaussians.win_lo[:] = 0.8
    assert np.all(np.array(renderer.render(0, camera, 48, 48)) == 255)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            viewer.SceneRenderer(path, re_, "cuda", "white")
    damaged = bytearray(path.read_bytes())
    damaged[-12] ^= 1
    bad_path = tmp_path / "bad-crc.4dgs"
    bad_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="CRC"):
        viewer.SceneRenderer(bad_path, re_, "cpu", "white")


def test_interactive_window_controls_and_responsive_background_render(monkeypatch):
    tk = pytest.importorskip("tkinter")
    from tkinter import ttk

    original_tk = tk.Tk
    events = []
    renderer = FakeRenderer()
    real_render = renderer.render

    def slow_render(*args, **kwargs):
        time.sleep(0.15)
        return real_render(*args, **kwargs)

    renderer.render = slow_render

    def create_window():
        try:
            root = original_tk()
        except tk.TclError as error:
            pytest.skip(f"Tk display unavailable: {error}")

        def manipulate():
            canvas = next(child for child in root.winfo_children() if isinstance(child, tk.Canvas))
            canvas.event_generate("<ButtonPress-1>", x=10, y=10)
            canvas.event_generate("<B1-Motion>", x=20, y=20)
            canvas.event_generate("<ButtonPress-3>", x=20, y=20)
            canvas.event_generate("<B3-Motion>", x=30, y=25)
            canvas.event_generate("<MouseWheel>", delta=120)
            controls = next(child for child in root.winfo_children() if isinstance(child, ttk.Frame))
            for child in controls.winfo_children():
                if isinstance(child, ttk.Checkbutton):
                    child.invoke()
            slider = next(child for child in root.winfo_children() if isinstance(child, ttk.Scale))
            slider.set(0.3)
            for child in controls.winfo_children():
                if isinstance(child, ttk.Button):
                    child.invoke()
            events.append("responsive")

        root.after(60, manipulate)
        root.after(500, root.destroy)
        return root

    monkeypatch.setattr(tk, "Tk", create_window)
    viewer.show_viewer(renderer, viewer.OrbitCamera((0, 0, 0), 3),
                       SimpleNamespace(scene=Path("test.4dgs"), no_orbit=False,
                                       width=64, height=64, seconds=6, fps=20))
    assert events == ["responsive"]
    assert len(renderer.phases) >= 1
