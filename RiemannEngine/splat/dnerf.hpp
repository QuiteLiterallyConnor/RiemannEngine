#ifndef RIEMANN_SPLAT_DNERF_HPP
#define RIEMANN_SPLAT_DNERF_HPP

#include "splat/camera.hpp"

#include <torch/torch.h>

#include <array>
#include <string>
#include <vector>

namespace riemann {
namespace gaussian {

// A posed, timed training view from a D-NeRF / Blender-style dataset.
struct TimedView {
  Camera camera;
  // Normalized capture time, usually in [0, 1] (0 for static datasets).
  double time = 0.0;
  // [H, W, 3] float32 image in [0, 1], alpha pre-blended over background.
  at::Tensor image;
  // Absolute path of the source PNG.
  std::string image_path;
};

struct DNerfLoadOptions {
  // Keep every stride-th frame (>= 1).
  int64_t stride = 1;
  // Maximum number of frames to load after striding (0 = all).
  int64_t max_frames = 0;
  // Downscale factor applied to both image and camera (1 = native size).
  int64_t downscale = 1;
  // Background blended behind transparent pixels (D-NeRF renders use white).
  std::array<double, 3> background = {1.0, 1.0, 1.0};
};

// Loads a split ("train", "val" or "test") of a D-NeRF / NeRF-synthetic
// dataset directory containing transforms_<split>.json and per-frame PNGs.
// Frames without a "time" field load with time 0, so static NeRF-synthetic
// scenes work too. Camera poses are converted from the Blender/OpenGL
// camera-to-world convention to the engine's OpenCV world-to-camera
// convention. Malformed JSON, missing files and invalid inputs raise errors.
std::vector<TimedView> load_dnerf_views(const std::string &dataset_dir,
                                        const std::string &split,
                                        const DNerfLoadOptions &options = {});

} // namespace gaussian
} // namespace riemann

#endif
