#!/usr/bin/env python
"""Play a .4dgs scene interactively or render a seamless orbit GIF."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import importlib.util
import logging
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch


LOG = logging.getLogger("view_fourdgs")
SH_C0 = 0.28209479177387814


def load_engine(extension: Path | None = None):
    if extension is not None:
        extension = extension.resolve(strict=True)
        if "riemannengine" in sys.modules:
            raise RuntimeError("--extension must be selected before importing riemannengine")
        spec = importlib.util.spec_from_file_location("riemannengine._C", extension)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Python extension: {extension}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["riemannengine._C"] = module
        spec.loader.exec_module(module)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import riemannengine

    if riemannengine.gaussian is None:
        raise RuntimeError("Rebuild RiemannEngine with Gaussian support enabled")
    return riemannengine


def loop_time(phase: float, duration: float) -> float:
    """Cosine ping-pong: zero temporal speed at both turns, half-open file time."""
    phase %= 1.0
    value = duration * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
    return min(value, math.nextafter(duration, 0.0))


@dataclass
class OrbitCamera:
    target: tuple[float, float, float]
    distance: float
    yaw: float = -math.pi / 2
    pitch: float = math.radians(20)
    fov: float = math.radians(50)
    up_axis: str = "z"

    @property
    def up(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0] if self.up_axis == "z" else [0.0, 1.0, 0.0])

    def eye(self, phase: float = 0, orbit: bool = True) -> np.ndarray:
        yaw = self.yaw + (2 * math.pi * (phase % 1) if orbit else 0)
        horizontal = math.cos(self.pitch)
        offset = np.array([horizontal * math.cos(yaw),
                           horizontal * math.sin(yaw), math.sin(self.pitch)])
        if self.up_axis == "y":
            offset = offset[[0, 2, 1]]
        return np.asarray(self.target) + self.distance * offset

    def rotate(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.008
        limit = math.radians(85)
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -limit, limit))

    def pan(self, dx: float, dy: float, width: int,
            phase: float = 0, orbit: bool = True) -> None:
        forward = np.asarray(self.target) - self.eye(phase, orbit)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = 2 * self.distance * math.tan(self.fov / 2) / width
        self.target = tuple(np.asarray(self.target) + (-dx * right + dy * up) * scale)

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(
            self.distance * math.exp(float(np.clip(-steps * 0.12, -20, 20))),
            1e-4, 1e8))


class SceneRenderer:
    def __init__(self, path: Path, engine, device: str, background: str):
        try:
            import fourdgs
        except ModuleNotFoundError as error:
            if error.name != "fourdgs":
                raise
            raise RuntimeError(
                "Install the decoder: python -m pip install -r "
                "examples\\requirements-fourdgs-viewer.txt") from error

        self.scene = fourdgs.read(str(path))
        if self.scene.truncated:
            raise ValueError("The .4dgs file is truncated")
        if self.scene.summary_crc_ok is False:
            raise ValueError("The .4dgs summary CRC does not match")
        self.duration = self.scene.duration_sec
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("The scene must have a finite positive duration")
        if self.scene.header.temporal_model != "gaussian-birth":
            raise ValueError("This viewer supports gaussian-birth .4dgs scenes")
        self.g = engine.gaussian
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() and engine.cuda_enabled() else "cpu"
        if device == "cuda" and not (torch.cuda.is_available() and engine.cuda_enabled()):
            raise RuntimeError("CUDA requires both CUDA PyTorch and a CUDA RiemannEngine extension")
        self.device = torch.device(device)
        self.background = 1 if background == "white" else 0
        self.options = self.g.RenderOptions()
        self.options.background = torch.full((3,), float(self.background), device=self.device)
        data = self.scene.gaussians
        if data.count == 0:
            raise ValueError("The scene contains no Gaussians")
        color_space = self.scene.header.attributes.get("color_space")
        if color_space is None:
            LOG.warning("No color_space metadata; treating colors as display-space RGB")
        elif color_space != "srgb-rec709-display":
            raise ValueError(f"Unsupported color_space {color_space!r}; expected srgb-rec709-display")

        def tensor(array):
            return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(self.device)

        self.dc = (tensor(data.colors[:, :3]) - 0.5) / SH_C0
        coefficients = (data.sh_degree + 1) ** 2 - 1
        if coefficients and data.sh is None:
            raise ValueError("Scene declares SH coefficients but has no SH data")
        self.rest = tensor(
            (data.sh.astype(np.float32) * (8 / 255) - 4)
            .reshape(data.count, 3, coefficients).transpose(0, 2, 1)
            if coefficients else np.empty((data.count, 0, 3)))
        self.scales = tensor(data.scales).log()
        if not torch.isfinite(self.scales).all():
            raise ValueError("Spatial scales must be finite and positive")
        LOG.info("Loaded %s Gaussians, SH degree %s, %.3fs on %s",
                 data.count, data.sh_degree, self.duration, self.device)
        if self.device.type == "cpu":
            LOG.warning("CPU rendering may be below real time; lower --width/--height for interaction")

    def fitted_camera(self, up_axis: str = "z", width: int = 512,
                      height: int = 512, fov: float = math.radians(50)) -> OrbitCamera:
        data = self.scene.gaussians
        samples = []
        for t in (0, self.duration / 2, math.nextafter(self.duration, 0)):
            state = self.scene.state_at(t)
            samples.append(state["centers"][state["opacity"] >= 1 / 255])
        points = np.concatenate(samples)
        if not len(points):
            points = data.positions
        if not np.isfinite(points).all():
            raise ValueError("Cannot fit a camera to nonfinite scene positions")
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) / 2
        radius = max(float(np.linalg.norm(upper - lower) / 2), 0.1)
        half_fov = min(fov / 2, math.atan(math.tan(fov / 2) * height / width))
        return OrbitCamera(tuple(center), radius / math.sin(half_fov) * 1.15,
                           fov=fov, up_axis=up_axis)

    @torch.inference_mode()
    def render(self, phase: float, camera: OrbitCamera, width: int, height: int,
               orbit: bool = True) -> Image.Image:
        state = self.scene.state_at(loop_time(phase, self.duration))
        keep = np.asarray(state["opacity"]) > 0
        indices = torch.as_tensor(state["indices"][keep], dtype=torch.long, device=self.device)
        if not len(indices):
            return Image.new("RGB", (width, height),
                             (255 * self.background,) * 3)

        def tensor(values):
            return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(self.device)

        opacity = tensor(state["opacity"][keep]).reshape(-1, 1).clamp(1e-12, 1 - 1e-7)
        model = self.g.GaussianModel(
            tensor(state["centers"][keep]), self.dc[indices], self.rest[indices],
            torch.logit(opacity), self.scales[indices],
            tensor(state["orientations"][keep][:, [3, 0, 1, 2]]))
        view = self.g.Camera.look_at(
            camera.eye(phase, orbit).tolist(), list(camera.target), camera.up.tolist(),
            width, height, camera.fov)
        image = self.g.render(model, view, self.options).image
        if not torch.isfinite(image).all():
            raise RuntimeError("Renderer returned nonfinite pixels")
        pixels = image.clamp(0, 1).mul(255).round().to(dtype=torch.uint8, device="cpu").numpy()
        return Image.fromarray(pixels)


def export_gif(renderer: SceneRenderer, camera: OrbitCamera, path: Path,
               width: int, height: int, seconds: float, fps: int,
               orbit: bool = True) -> None:
    if path.suffix.lower() != ".gif":
        raise ValueError("Output must have a .gif extension")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    count = max(2, round(seconds * fps))
    frames = []
    for index in range(count):
        # Never duplicate the endpoint: frame count / fps is one whole orbit.
        frames.append(renderer.render(index / count, camera, width, height, orbit))
        if index % fps == 0 or index + 1 == count:
            LOG.info("Rendering frame %s/%s", index + 1, count)
    # One palette for the whole clip avoids independent per-frame palette flicker.
    sheet = Image.new("RGB", (64 * min(count, 20), 64))
    for slot, index in enumerate(np.linspace(0, count - 1, min(count, 20), dtype=int)):
        sheet.paste(frames[index].resize((64, 64)), (slot * 64, 0))
    palette = sheet.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [10 * (round((i + 1) * 100 / fps) - round(i * 100 / fps))
                 for i in range(count)]
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".gif", delete=False) as file:
        temporary = Path(file.name)
    try:
        frames[0].save(temporary, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, disposal=2, optimize=False)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("Saved %s (%s frames, %.2fs, infinite loop)",
             path, count, sum(durations) / 1000)


def show_viewer(renderer: SceneRenderer, camera: OrbitCamera, args) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title(f"4DGS - {args.scene.name}")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=args.width, height=args.height, highlightthickness=0)
    canvas.pack()
    image_item = canvas.create_image(0, 0, anchor="nw")
    controls = ttk.Frame(root)
    controls.pack(fill="x")
    playing = tk.BooleanVar(value=True)
    orbiting = tk.BooleanVar(value=not args.no_orbit)
    ttk.Checkbutton(controls, text="Play [Space]", variable=playing).pack(side="left")
    orbit_button = ttk.Checkbutton(controls, text="Auto orbit [O]", variable=orbiting)
    orbit_button.pack(side="left")
    status = tk.StringVar(value="Rendering first frame...")
    ttk.Label(root, textvariable=status).pack()
    ttk.Label(root, text="Left drag: orbit | Right/middle drag: pan | Wheel: zoom | R: reset").pack()
    ttk.Label(root, text="Loop phase: forward, then backward").pack()
    slider = ttk.Scale(root, from_=0, to=1, orient="horizontal")
    slider.pack(fill="x")
    initial_camera = replace(camera)
    phase = 0.0
    last_tick = time.perf_counter()
    last_mouse = (0, 0)
    future = None
    last_request = None
    photo = None
    failure = None
    updating_slider = False
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="4dgs-render")

    def reset(_event=None):
        nonlocal camera
        camera = replace(initial_camera)

    def seek(value):
        nonlocal phase, last_tick
        if not updating_slider:
            phase = float(value) % 1
            last_tick = time.perf_counter()

    slider.configure(command=seek)
    ttk.Button(controls, text="Reset [R]", command=reset).pack(side="right")

    def press(event):
        nonlocal last_mouse
        last_mouse = (event.x, event.y)

    def drag(event, pan=False):
        nonlocal last_mouse
        dx, dy = event.x - last_mouse[0], event.y - last_mouse[1]
        if pan:
            camera.pan(dx, dy, args.width, phase, orbiting.get())
        else:
            camera.rotate(dx, dy)
        last_mouse = (event.x, event.y)

    def toggle(variable):
        variable.set(not variable.get())

    def orbit_changed():
        camera.yaw += (-1 if orbiting.get() else 1) * 2 * math.pi * phase

    def toggle_orbit(_event=None):
        toggle(orbiting)
        orbit_changed()

    orbit_button.configure(command=orbit_changed)
    for button in (1, 2, 3):
        canvas.bind(f"<ButtonPress-{button}>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<B2-Motion>", lambda event: drag(event, pan=True))
    canvas.bind("<B3-Motion>", lambda event: drag(event, pan=True))
    canvas.bind("<MouseWheel>", lambda event: camera.zoom(event.delta / 120))
    canvas.bind("<Button-4>", lambda event: camera.zoom(1))
    canvas.bind("<Button-5>", lambda event: camera.zoom(-1))
    root.bind("<space>", lambda event: toggle(playing))
    root.bind("o", toggle_orbit)
    root.bind("r", reset)
    root.bind("<Escape>", lambda event: root.destroy())

    def tick():
        nonlocal phase, last_tick, future, last_request, photo, failure, updating_slider
        now = time.perf_counter()
        if playing.get():
            phase = (phase + (now - last_tick) / args.seconds) % 1
        last_tick = now
        updating_slider = True
        slider.set(phase)
        updating_slider = False
        if future is not None and future.done():
            try:
                image, rendered_phase, elapsed = future.result()
            except Exception as error:
                LOG.exception("Interactive render failed")
                failure = error
                messagebox.showerror("4DGS rendering failed", str(error))
                root.destroy()
                return
            photo = ImageTk.PhotoImage(image)
            canvas.itemconfigure(image_item, image=photo)
            status.set(f"Scene {loop_time(rendered_phase, renderer.duration):.3f}s | "
                       f"{1 / max(elapsed, 1e-6):.1f} render fps | {renderer.device}")
            future = None
        request = (phase, camera.target, camera.distance, camera.yaw, camera.pitch, orbiting.get())
        if future is None and request != last_request:
            snapshot, requested_phase, auto_orbit = replace(camera), phase, orbiting.get()

            def render():
                start = time.perf_counter()
                image = renderer.render(requested_phase, snapshot, args.width, args.height, auto_orbit)
                return image, requested_phase, time.perf_counter() - start

            future = executor.submit(render)
            last_request = request
        root.after(max(1, round(1000 / args.fps)), tick)

    root.after(0, tick)
    try:
        root.mainloop()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failure is not None:
        raise RuntimeError("Interactive rendering failed") from failure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="avala-ai version-1 .4dgs scene")
    parser.add_argument("--output", type=Path, help="Render an infinite-loop GIF instead of opening the viewer")
    parser.add_argument("--extension", type=Path, help="Use a CMake-built _C.pyd/_C.so instead of the installed extension")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=6, help="Seconds per forward/backward cycle and camera orbit")
    parser.add_argument("--fps", type=int, default=20, help="Viewer refresh cap / exported GIF frame rate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--up", choices=("z", "y"), default="z", help="World up axis (D-NeRF uses z)")
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--distance", type=float)
    parser.add_argument("--yaw", type=float, default=-90, help="Initial orbit azimuth in degrees")
    parser.add_argument("--pitch", type=float, default=20, help="Elevation in degrees")
    parser.add_argument("--fov", type=float, default=50, help="Horizontal field of view in degrees")
    parser.add_argument("--no-orbit", action="store_true", help="Disable automatic orbit (manual controls remain available)")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.threads <= 0:
        parser.error("width, height and threads must be positive")
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.fps <= 100:
        parser.error("seconds must be finite and positive; GIF fps must be in [1,100]")
    if not math.isfinite(args.seconds * args.fps) or args.seconds * args.fps < 2:
        parser.error("seconds * fps must be finite and allow at least two frames")
    if not 1 <= args.fov <= 150 or not -85 <= args.pitch <= 85 or not math.isfinite(args.yaw):
        parser.error("fov must be in [1,150], pitch in [-85,85], and yaw finite")
    if args.distance is not None and (not math.isfinite(args.distance) or args.distance <= 0):
        parser.error("distance must be finite and positive")
    if args.target is not None and not all(math.isfinite(value) for value in args.target):
        parser.error("target coordinates must be finite")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(args.threads)
    engine = load_engine(args.extension)
    renderer = SceneRenderer(args.scene.resolve(strict=True), engine, args.device, args.background)
    camera = renderer.fitted_camera(args.up, args.width, args.height, math.radians(args.fov))
    camera.yaw, camera.pitch, camera.fov = map(math.radians, (args.yaw, args.pitch, args.fov))
    if args.target is not None:
        camera.target = tuple(args.target)
    if args.distance is not None:
        camera.distance = args.distance
    if args.output is not None:
        export_gif(renderer, camera, args.output, args.width, args.height,
                   args.seconds, args.fps, not args.no_orbit)
    else:
        show_viewer(renderer, camera, args)


if __name__ == "__main__":
    main()
