#include "splat/dnerf.hpp"

#include "json.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace riemann {
namespace gaussian {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

at::Tensor world_to_camera_from_blender(const Json &matrix) {
  TORCH_CHECK(matrix.is_array() && matrix.size() == 4,
              "transform_matrix must have 4 rows");
  auto c2w = at::zeros({4, 4}, at::kDouble);
  auto access = c2w.accessor<double, 2>();
  for (int row = 0; row < 4; ++row) {
    const auto &values = matrix.at(row);
    TORCH_CHECK(values.is_array() && values.size() == 4,
                "transform_matrix rows must have 4 entries");
    for (int col = 0; col < 4; ++col) {
      const auto value = values.at(col).get<double>();
      TORCH_CHECK(std::isfinite(value), "camera transform must be finite");
      access[row][col] = value;
    }
  }
  TORCH_CHECK(at::allclose(c2w[3], at::tensor({0., 0., 0., 1.}, at::kDouble)),
              "camera transform must be affine");
  // OpenGL looks along -z with +y up; OpenCV looks along +z with +y down.
  const auto flip = at::diag(at::tensor({1., -1., -1.}, at::kDouble));
  const auto rotation = at::matmul(c2w.slice(0, 0, 3).slice(1, 0, 3), flip);
  TORCH_CHECK(at::allclose(at::matmul(rotation.transpose(0, 1), rotation),
                           at::eye(3, at::kDouble), 1e-4, 1e-5),
              "camera rotation must be orthonormal");
  const auto translation = c2w.slice(0, 0, 3).select(1, 3);
  auto w2c = at::eye(4, at::kDouble);
  w2c.slice(0, 0, 3).slice(1, 0, 3).copy_(rotation.transpose(0, 1));
  w2c.slice(0, 0, 3).select(1, 3).copy_(
      -at::matmul(rotation.transpose(0, 1), translation));
  return w2c;
}

at::Tensor load_png_rgb(const fs::path &path,
                        const std::array<double, 3> &background) {
  int width = 0, height = 0, channels = 0;
  using Pixels = std::unique_ptr<stbi_uc, void (*)(void *)>;
  Pixels pixels(stbi_load(path.string().c_str(), &width, &height, &channels, 4),
                &stbi_image_free);
  if (!pixels)
    throw std::runtime_error("failed to load PNG " + path.string() + ": " +
                             stbi_failure_reason());
  const auto rgba = at::from_blob(pixels.get(), {height, width, 4}, at::kByte)
                        .to(at::kFloat).div_(255.0);
  const auto alpha = rgba.slice(2, 3, 4);
  const auto bg = at::tensor({background[0], background[1], background[2]},
                             at::kFloat);
  return (rgba.slice(2, 0, 3) * alpha + bg * (1.0 - alpha)).contiguous();
}

} // namespace

std::vector<TimedView> load_dnerf_views(const std::string &dataset_dir,
                                        const std::string &split,
                                        const DNerfLoadOptions &options) {
  TORCH_CHECK(split == "train" || split == "val" || split == "test",
              "split must be train, val or test");
  TORCH_CHECK(options.stride >= 1, "stride must be at least 1");
  TORCH_CHECK(options.downscale >= 1, "downscale must be at least 1");
  TORCH_CHECK(options.max_frames >= 0, "max_frames must be nonnegative");
  for (const auto value : options.background)
    TORCH_CHECK(std::isfinite(value) && value >= 0.0 && value <= 1.0,
                "background must be finite and in [0, 1]");
  const auto root = fs::absolute(dataset_dir);
  const auto transforms_path = root / ("transforms_" + split + ".json");
  std::ifstream file(transforms_path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open " + transforms_path.string());
  const auto document = Json::parse(file);
  const double fov_x = document.at("camera_angle_x").get<double>();
  TORCH_CHECK(std::isfinite(fov_x) && fov_x > 0.0 && fov_x < std::acos(-1.0),
              "camera_angle_x must be finite and in (0, pi)");
  const auto &frames = document.at("frames");
  TORCH_CHECK(frames.is_array() && !frames.empty(),
              "frames must be a nonempty array");

  std::vector<TimedView> views;
  for (size_t index = 0; index < frames.size();
       index += static_cast<size_t>(options.stride)) {
    if (options.max_frames > 0 &&
        views.size() >= static_cast<size_t>(options.max_frames))
      break;
    const auto &frame = frames.at(index);
    auto relative_path = fs::path(frame.at("file_path").get<std::string>());
    if (!relative_path.has_extension())
      relative_path += ".png";
    const auto image_path = root / relative_path;
    TimedView view;
    view.image_path = image_path.string();
    view.time = frame.value("time", 0.0);
    TORCH_CHECK(std::isfinite(view.time), "frame time must be finite");
    view.image = load_png_rgb(image_path, options.background);
    const auto height = view.image.size(0), width = view.image.size(1);
    TORCH_CHECK(height >= options.downscale && width >= options.downscale,
                "downscale must not exceed the image dimensions");
    view.camera = Camera::from_fov(
        width, height, fov_x,
        world_to_camera_from_blender(frame.at("transform_matrix")));
    if (options.downscale > 1) {
      const int64_t h = height / options.downscale;
      const int64_t w = width / options.downscale;
      view.image = at::adaptive_avg_pool2d(
          view.image.permute({2, 0, 1}).unsqueeze(0), {h, w})
                       .squeeze(0).permute({1, 2, 0}).contiguous();
      const double sx = static_cast<double>(w) / width;
      const double sy = static_cast<double>(h) / height;
      view.camera.width = w;
      view.camera.height = h;
      view.camera.fx *= sx;
      view.camera.fy *= sy;
      view.camera.cx *= sx;
      view.camera.cy *= sy;
    }
    views.push_back(std::move(view));
  }
  return views;
}

} // namespace gaussian
} // namespace riemann
