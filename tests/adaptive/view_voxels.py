"""Side-by-side viewer for the fixed and adaptive volume voxelizations written
by test_voxelize_obj.cpp / voxelize_obj_bench (the *.json voxel files).

Usage:
    python view_voxels.py                         # newest *_fixed_r*/*_adaptive_r* in cwd
    python view_voxels.py --dir build-cpu-compat/voxel_out
    python view_voxels.py FIXED.json ADAPTIVE.json
    python view_voxels.py --cut z                 # cutaway: hide voxels above mid-height
    python view_voxels.py --cut x:0.3             # cut along x at 30% of the extent
    python view_voxels.py --edges                 # draw cube outlines
    python view_voxels.py --save shot.png         # render to an image instead of a window

Fixed voxels are blue (lighter = surface, darker = solid interior); adaptive
cells are coloured by refinement level (coarse = blue ... fine = red). Use
--cut to see the filled interior. Requires numpy and open3d.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

# Unit cube corners and its 12 triangles / 12 edges (indices into the corners).
CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64) - 0.5
TRIANGLES = np.array(
    [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
EDGES = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
     [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int32)

FIXED_SURFACE_COLOR = np.array([0.55, 0.68, 0.88])
FIXED_INTERIOR_COLOR = np.array([0.18, 0.25, 0.45])
LEVEL_COLORS = np.array(
    [[0.17, 0.40, 0.85], [0.20, 0.70, 0.75], [0.35, 0.80, 0.35],
     [0.95, 0.80, 0.20], [0.95, 0.50, 0.15], [0.85, 0.20, 0.20]])


def load_voxels(path):
    with open(path, "r") as f:
        doc = json.load(f)
    voxels = doc["voxels"]
    centers = np.array([v["position_m"] for v in voxels], dtype=np.float64).reshape(-1, 3)
    sizes = np.array([v["size_m"] for v in voxels], dtype=np.float64)
    levels = np.array([v.get("level", 0) for v in voxels], dtype=np.int32)
    surface = np.array([v.get("surface", 1.0) for v in voxels], dtype=np.float64)
    return centers, sizes, levels, surface


def parse_cut(spec):
    axis, _, frac = spec.partition(":")
    if axis not in ("x", "y", "z"):
        raise argparse.ArgumentTypeError("--cut must be x, y or z, optionally ':FRACTION'")
    return "xyz".index(axis), float(frac) if frac else 0.5


def level_colors(levels, max_level):
    if max_level <= 0:
        return np.tile(LEVEL_COLORS[-1], (len(levels), 1))
    # Spread the palette across the levels actually used.
    t = levels / max_level * (len(LEVEL_COLORS) - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(LEVEL_COLORS) - 1)
    w = (t - lo)[:, None]
    return LEVEL_COLORS[lo] * (1 - w) + LEVEL_COLORS[hi] * w


def cubes_mesh(centers, sizes, colors, offset):
    n = len(centers)
    # Shrink slightly so neighbouring cubes stay visually distinct.
    verts = (centers[:, None, :] + offset) + CORNERS[None] * (sizes[:, None, None] * 0.96)
    tris = TRIANGLES[None] + (np.arange(n, dtype=np.int32) * 8)[:, None, None]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.reshape(-1, 3)),
        o3d.utility.Vector3iVector(tris.reshape(-1, 3)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(colors, 8, axis=0))
    mesh.compute_vertex_normals()
    return mesh


def cubes_edges(centers, sizes, offset):
    n = len(centers)
    verts = (centers[:, None, :] + offset) + CORNERS[None] * sizes[:, None, None]
    lines = EDGES[None] + (np.arange(n, dtype=np.int32) * 8)[:, None, None]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(verts.reshape(-1, 3)),
        o3d.utility.Vector2iVector(lines.reshape(-1, 2)))
    ls.paint_uniform_color([0.1, 0.1, 0.1])
    return ls


def newest(pattern):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fixed", nargs="?", help="fixed-resolution voxel JSON")
    ap.add_argument("adaptive", nargs="?", help="adaptive-resolution voxel JSON")
    ap.add_argument("--dir", default=".", help="where to look when files are omitted")
    ap.add_argument("--edges", action="store_true", help="draw cube outlines")
    ap.add_argument("--cut", type=parse_cut, metavar="AXIS[:FRACTION]",
                    help="cutaway: keep only voxels whose centre lies below this plane")
    ap.add_argument("--save", metavar="PNG", help="render offscreen to this image and exit")
    args = ap.parse_args()

    fixed = args.fixed or newest(os.path.join(args.dir, "*_fixed_r*.json"))
    adaptive = args.adaptive or newest(os.path.join(args.dir, "*_adaptive_r*.json"))
    if not fixed or not adaptive:
        sys.exit(f"could not find fixed/adaptive voxel JSON files in {args.dir!r}; "
                 "pass them explicitly or use --dir")

    fc, fs, _, fsurf = load_voxels(fixed)
    ac, asz, al, _ = load_voxels(adaptive)
    print(f"fixed:    {fixed}  ({len(fc)} voxels, {int((fsurf > 0).sum())} surface, "
          f"{int((fsurf == 0).sum())} interior, volume {np.sum(fs ** 3):.6g})")
    print(f"adaptive: {adaptive}  ({len(ac)} voxels, volume {np.sum(asz ** 3):.6g})")
    max_level = int(al.max()) if len(al) else 0
    for lvl in range(max_level + 1):
        print(f"  level {lvl}: {int((al == lvl).sum())} voxels")

    # Place the adaptive model to the right of the fixed one with a small gap.
    lo = np.minimum((fc - fs[:, None] / 2).min(0), (ac - asz[:, None] / 2).min(0))
    hi = np.maximum((fc + fs[:, None] / 2).max(0), (ac + asz[:, None] / 2).max(0))
    offset = np.array([(hi[0] - lo[0]) * 1.25, 0.0, 0.0])

    fixed_colors = np.where((fsurf > 0)[:, None], FIXED_SURFACE_COLOR, FIXED_INTERIOR_COLOR)
    adaptive_colors = level_colors(al, max_level)
    if args.cut:
        axis, frac = args.cut
        plane = lo[axis] + frac * (hi[axis] - lo[axis])
        fk, ak = fc[:, axis] < plane, ac[:, axis] < plane
        fc, fs, fixed_colors = fc[fk], fs[fk], fixed_colors[fk]
        ac, asz, adaptive_colors = ac[ak], asz[ak], adaptive_colors[ak]

    geoms = [
        cubes_mesh(fc, fs, fixed_colors, np.zeros(3)),
        cubes_mesh(ac, asz, adaptive_colors, offset),
    ]
    if args.edges:
        geoms += [cubes_edges(fc, fs, np.zeros(3)), cubes_edges(ac, asz, offset)]

    title = f"Fixed ({len(fc)} voxels)  |  Adaptive ({len(ac)} voxels, by level)"
    if args.save:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1600, height=900, visible=False)
        for g in geoms:
            vis.add_geometry(g)
        vis.get_render_option().mesh_show_back_face = True
        vis.get_view_control().set_zoom(0.6)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save, do_render=True)
        vis.destroy_window()
        print(f"saved {args.save}")
    else:
        o3d.visualization.draw_geometries(geoms, window_name=title, width=1600, height=900,
                                          mesh_show_back_face=True)


if __name__ == "__main__":
    main()
