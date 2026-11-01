#include "sphere.hpp"

#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

namespace riemann {
namespace adaptive {

namespace sphere_detail {

void validate_args(double radius, double target_accuracy_percent, int64_t max_voxels,
                   uint32_t max_depth) {
  if (!(radius > 0.0) || !std::isfinite(radius))
    throw std::invalid_argument("voxelize_sphere: radius must be positive and finite");
  if (!(target_accuracy_percent > 0.0 && target_accuracy_percent < 100.0))
    throw std::invalid_argument("voxelize_sphere: target accuracy must be in (0, 100)");
  if (max_voxels < 1 || max_depth > AdaptiveVoxelConfig::kMaxSupportedDepth)
    throw std::invalid_argument("voxelize_sphere: invalid max_voxels or max_depth");
}

AdaptiveVoxelConfig make_config(double radius, const std::array<double, 3> &center,
                                uint32_t max_depth) {
  const double side = 2.0 * radius;
  const double finest = std::ldexp(side, -static_cast<int>(max_depth));
  AdaptiveVoxelConfig cfg;
  cfg.base_voxel_size = {side, side, side};
  cfg.min_voxel_size = {finest, finest, finest};
  cfg.origin = {center[0] - radius, center[1] - radius, center[2] - radius};
  cfg.max_depth = max_depth;
  return cfg;
}

} // namespace sphere_detail

namespace {

using sphere_detail::SphereFrame;

// Cell in the sphere's bounding-cube frame.
struct Cube {
  int64_t x, y, z;
  int level;
};

Cube nth_child(const Cube &p, int n) {
  return {2 * p.x + (n & 1), 2 * p.y + ((n >> 1) & 1), 2 * p.z + ((n >> 2) & 1), p.level + 1};
}

bool straddles(const SphereFrame &s, const Cube &c) {
  return s.is_boundary(c.x, c.y, c.z, c.level);
}
double contributes(const SphereFrame &s, const Cube &c) {
  return s.contribution(c.x, c.y, c.z, c.level);
}

} // namespace

AdaptiveSparseTensor voxelize_sphere(double radius, double target_accuracy_percent,
                                     std::array<double, 3> center, int64_t max_voxels,
                                     uint32_t max_depth, double *achieved_accuracy_percent) {
  sphere_detail::validate_args(radius, target_accuracy_percent, max_voxels, max_depth);
  const SphereFrame sphere{radius};

  // Split boundary cells in FIFO order (whole levels before deeper ones),
  // tracking the estimate incrementally; stop at the first configuration that
  // reaches the target.
  std::vector<Cube> settled;
  std::deque<Cube> frontier;
  const Cube root{0, 0, 0, 0};
  double volume = contributes(sphere, root);
  int64_t count = 1;
  if (straddles(sphere, root))
    frontier.push_back(root);
  else
    settled.push_back(root);

  double achieved = sphere.accuracy(volume);
  while (achieved < target_accuracy_percent) {
    if (frontier.empty() || count + 7 > max_voxels ||
        static_cast<uint32_t>(frontier.front().level) >= max_depth)
      throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");
    const Cube parent = frontier.front();
    frontier.pop_front();
    volume -= contributes(sphere, parent);
    count += 7;
    for (int n = 0; n < 8; ++n) {
      const Cube child = nth_child(parent, n);
      volume += contributes(sphere, child);
      if (straddles(sphere, child))
        frontier.push_back(child);
      else
        settled.push_back(child);
    }
    achieved = sphere.accuracy(volume);
  }
  settled.insert(settled.end(), frontier.begin(), frontier.end());
  if (achieved_accuracy_percent)
    *achieved_accuracy_percent = achieved;

  const int64_t n = static_cast<int64_t>(settled.size());
  auto coordinates = torch::zeros({n, 3}, torch::kInt64);
  auto levels = torch::zeros({n}, torch::kInt32);
  auto features = torch::zeros({n, 1}, torch::kFloat32);
  auto ca = coordinates.accessor<int64_t, 2>();
  auto la = levels.accessor<int32_t, 1>();
  auto fa = features.accessor<float, 2>();
  for (int64_t i = 0; i < n; ++i) {
    const Cube &c = settled[static_cast<size_t>(i)];
    ca[i][0] = c.x;
    ca[i][1] = c.y;
    ca[i][2] = c.z;
    la[i] = static_cast<int32_t>(c.level);
    fa[i][0] = sphere.occupied(c.x, c.y, c.z, c.level) ? 1.0f : 0.0f;
  }

  return AdaptiveSparseTensor::from_voxels(coordinates, levels, features,
                                           sphere_detail::make_config(radius, center, max_depth));
}

double sphere_volume_estimate(const AdaptiveSparseTensor &sphere) {
  const auto volumes = sphere.volumes().contiguous();
  const auto occupied = sphere.features().to(torch::kCPU).to(torch::kFloat64).reshape({-1});
  return (volumes * (occupied > 0.5).to(torch::kFloat64)).sum().item<double>();
}

#ifdef CPU_ONLY
AdaptiveSparseTensor voxelize_sphere_cuda(double, double, std::array<double, 3>, int64_t,
                                          uint32_t, double *) {
  throw std::runtime_error("voxelize_sphere_cuda: CUDA not available in CPU_ONLY build");
}
#endif

} // namespace adaptive
} // namespace riemann
