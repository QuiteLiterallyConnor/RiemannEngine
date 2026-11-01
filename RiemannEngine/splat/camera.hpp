#ifndef RIEMANN_SPLAT_CAMERA_HPP
#define RIEMANN_SPLAT_CAMERA_HPP

#include <torch/torch.h>

#include <array>

namespace riemann {
namespace gaussian {

// Pinhole camera using the OpenCV/COLMAP convention: +x right, +y down,
// +z forward. Pixel (i, j) has its center at (i + 0.5, j + 0.5).
struct Camera {
  int64_t width = 0;
  int64_t height = 0;
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  // [4, 4] float32 world-to-camera transform.
  at::Tensor world_to_camera;
  double near_plane = 0.01;

  at::Tensor rotation() const {
    return world_to_camera.slice(0, 0, 3).slice(1, 0, 3);
  }
  at::Tensor translation() const {
    return world_to_camera.slice(0, 0, 3).select(1, 3);
  }
  // Camera center in world coordinates.
  at::Tensor position() const {
    return -at::matmul(rotation().transpose(0, 1), translation());
  }

  static Camera from_fov(int64_t width, int64_t height, double fov_x_radians,
                         at::Tensor world_to_camera);

  static Camera look_at(const std::array<double, 3> &eye,
                        const std::array<double, 3> &target,
                        const std::array<double, 3> &up, int64_t width,
                        int64_t height, double fov_x_radians);
};

} // namespace gaussian
} // namespace riemann

#endif
