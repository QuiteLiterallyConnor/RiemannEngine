#include "sphere.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace riemann {
namespace adaptive {
namespace {

using sphere_detail::SphereFrame;

constexpr int kThreads = 256;

int grid_for(int64_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

// One thread per frontier cell: writes its 8 children, flags which children are
// still boundary cells, and the volume-estimate change if this cell is split.
__global__ void split_frontier(SphereFrame s, const int64_t *coords, int64_t n, int level,
                               int64_t *child_coords, bool *child_boundary, double *delta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int64_t px = coords[i * 3], py = coords[i * 3 + 1], pz = coords[i * 3 + 2];
  double change = -s.contribution(px, py, pz, level);
  for (int c = 0; c < 8; ++c) {
    const int64_t cx = 2 * px + (c & 1);
    const int64_t cy = 2 * py + ((c >> 1) & 1);
    const int64_t cz = 2 * pz + ((c >> 2) & 1);
    const int64_t o = i * 8 + c;
    child_coords[o * 3] = cx;
    child_coords[o * 3 + 1] = cy;
    child_coords[o * 3 + 2] = cz;
    child_boundary[o] = s.is_boundary(cx, cy, cz, level + 1);
    change += s.contribution(cx, cy, cz, level + 1);
  }
  delta[i] = change;
}

__global__ void mark_occupied(SphereFrame s, const int64_t *coords, const int32_t *levels,
                              int64_t n, float *features) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  features[i] =
      s.occupied(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2], levels[i]) ? 1.f : 0.f;
}

} // namespace

AdaptiveSparseTensor voxelize_sphere_cuda(double radius, double target_accuracy_percent,
                                          std::array<double, 3> center, int64_t max_voxels,
                                          uint32_t max_depth, double *achieved_accuracy_percent) {
  sphere_detail::validate_args(radius, target_accuracy_percent, max_voxels, max_depth);
  if (!torch::cuda::is_available())
    throw std::runtime_error("voxelize_sphere_cuda: CUDA is not available");

  const SphereFrame sphere{radius};
  const torch::Device device(torch::kCUDA, c10::cuda::current_device());
  const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  const auto boolean = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  std::vector<at::Tensor> leaf_coords, leaf_levels;
  const auto settle = [&](const at::Tensor &coords, int level) {
    if (coords.size(0) == 0)
      return;
    leaf_coords.push_back(coords);
    leaf_levels.push_back(torch::full({coords.size(0)}, level, i32));
  };

  // Mirrors the host refinement: a level's cells split in order (children of
  // earlier parents first) until the estimate first reaches the target. Per
  // level the running estimate is a prefix sum over the parents' volume
  // deltas, so the first sufficient split is located without a per-split loop.
  double volume = sphere.contribution(0, 0, 0, 0);
  int64_t count = 1;
  int level = 0;
  at::Tensor frontier = torch::zeros({0, 3}, i64);
  if (sphere.is_boundary(0, 0, 0, 0))
    frontier = torch::zeros({1, 3}, i64);
  else
    settle(torch::zeros({1, 3}, i64), 0);

  double achieved = sphere.accuracy(volume);
  while (achieved < target_accuracy_percent) {
    const int64_t n = frontier.size(0);
    const int64_t budget = n == 0 ? 0 : std::min<int64_t>(n, (max_voxels - count) / 7);
    if (n == 0 || budget == 0 || static_cast<uint32_t>(level) >= max_depth)
      throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");

    auto child_coords = torch::empty({n * 8, 3}, i64);
    auto child_boundary = torch::empty({n * 8}, boolean);
    auto delta = torch::empty({n}, f64);
    split_frontier<<<grid_for(n), kThreads, 0, stream>>>(
        sphere, frontier.data_ptr<int64_t>(), n, level, child_coords.data_ptr<int64_t>(),
        child_boundary.data_ptr<bool>(), delta.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const double exact = sphere.exact_volume();
    const auto estimate = delta.cumsum(0) + volume;
    const auto accuracy = 100.0 * (1.0 - (estimate - exact).abs() / exact);
    const auto reached = accuracy.slice(0, 0, budget) >= target_accuracy_percent;
    const bool any_reached = budget > 0 && reached.any().item<bool>();

    if (!any_reached) {
      if (budget < n)
        throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");
      // Split the whole level and continue with its boundary children.
      settle(child_coords.index({~child_boundary}), level + 1);
      frontier = child_coords.index({child_boundary});
      volume = estimate[n - 1].item<double>();
      achieved = accuracy[n - 1].item<double>();
      count += 7 * n;
      ++level;
      continue;
    }

    // Only the first k parents split; the rest of the level stays leaves.
    const int64_t k = reached.to(torch::kInt32).argmax().item<int64_t>() + 1;
    const auto split_children = child_coords.slice(0, 0, 8 * k);
    const auto split_boundary = child_boundary.slice(0, 0, 8 * k);
    settle(split_children.index({~split_boundary}), level + 1);
    settle(split_children.index({split_boundary}), level + 1);
    frontier = frontier.slice(0, k, n);
    achieved = accuracy[k - 1].item<double>();
    break;
  }
  settle(frontier, level);
  if (achieved_accuracy_percent)
    *achieved_accuracy_percent = achieved;

  const auto coordinates = torch::cat(leaf_coords, 0).contiguous();
  const auto levels = torch::cat(leaf_levels, 0).contiguous();
  const int64_t total = coordinates.size(0);
  auto features = torch::empty({total, 1}, f32);
  mark_occupied<<<grid_for(total), kThreads, 0, stream>>>(
      sphere, coordinates.data_ptr<int64_t>(), levels.data_ptr<int32_t>(), total,
      features.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return AdaptiveSparseTensor::from_voxels(coordinates, levels, features,
                                           sphere_detail::make_config(radius, center, max_depth));
}

} // namespace adaptive
} // namespace riemann
