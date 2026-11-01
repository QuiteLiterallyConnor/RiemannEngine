#include "splat/camera.hpp"

#include <cmath>

namespace riemann {
namespace gaussian {
namespace {

using V3 = std::array<double, 3>;

V3 difference(const V3 &a, const V3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

V3 cross_product(const V3 &a, const V3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

V3 unit(const V3 &a) {
  const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  TORCH_CHECK(n > 1e-12, "look_at received a degenerate direction");
  return {a[0] / n, a[1] / n, a[2] / n};
}

double dot(const V3 &a, const V3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // namespace

Camera Camera::from_fov(int64_t width, int64_t height, double fov_x_radians,
                        at::Tensor world_to_camera) {
  TORCH_CHECK(width > 0 && height > 0, "camera size must be positive");
  TORCH_CHECK(world_to_camera.sizes() == at::IntArrayRef({4, 4}),
              "world_to_camera must have shape [4, 4]");
  Camera camera;
  camera.width = width;
  camera.height = height;
  camera.fx = 0.5 * static_cast<double>(width) / std::tan(0.5 * fov_x_radians);
  camera.fy = camera.fx;
  camera.cx = 0.5 * static_cast<double>(width);
  camera.cy = 0.5 * static_cast<double>(height);
  camera.world_to_camera =
      world_to_camera.to(at::kFloat).to(at::kCPU).contiguous();
  return camera;
}

Camera Camera::look_at(const std::array<double, 3> &eye,
                       const std::array<double, 3> &target,
                       const std::array<double, 3> &up, int64_t width,
                       int64_t height, double fov_x_radians) {
  const V3 forward = unit(difference(target, eye));
  const V3 right = unit(cross_product(V3{-up[0], -up[1], -up[2]}, forward));
  const V3 down = cross_product(forward, right);

  auto matrix = at::zeros({4, 4}, at::kDouble);
  auto access = matrix.accessor<double, 2>();
  const V3 *axes[3] = {&right, &down, &forward};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col)
      access[row][col] = (*axes[row])[col];
    access[row][3] = -dot(*axes[row], eye);
  }
  access[3][3] = 1.0;
  return from_fov(width, height, fov_x_radians, matrix);
}

} // namespace gaussian
} // namespace riemann
