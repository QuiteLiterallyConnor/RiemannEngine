#include "gaussian_voxel/cuda.hpp"
#include "octree/tree.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace riemann::gaussian_voxel {
namespace {

constexpr int kThreads = 128;

int blocks(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count + kThreads - 1) / kThreads, 4096));
}

struct Grid {
  const int64_t *coordinates;
  const int64_t *offsets;
  const double *geometry; // origin[3], base extent[3]
  int levels;
};

__device__ int64_t find_cell(Grid grid, int level, const int64_t *key) {
  int64_t lo = grid.offsets[level], hi = grid.offsets[level + 1];
  const int64_t end = hi;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    const int64_t *probe = grid.coordinates + 3 * mid;
    bool before = false;
    for (int d = 0; d < 3; ++d) {
      if (probe[d] != key[d]) {
        before = probe[d] < key[d];
        break;
      }
    }
    if (before)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == end)
    return -1;
  for (int d = 0; d < 3; ++d)
    if (grid.coordinates[lo * 3 + d] != key[d])
      return -1;
  return lo;
}

// Two passes avoid allocating max_associations entries for small inputs.
template <bool Emit>
__global__ void association_kernel(
    const double *positions, const double *covariance, int64_t count,
    Grid grid, double radius, int64_t limit, int64_t *counts,
    const int64_t *prefix, int64_t *gaussian_ids, int64_t *voxel_ids,
    int *error) {
  for (int64_t g = blockIdx.x * blockDim.x + threadIdx.x; g < count;
       g += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    double lower[3], upper[3];
    bool invalid = false;
    for (int d = 0; d < 3; ++d) {
      const double mean = positions[g * 3 + d];
      const double variance = covariance[g * 9 + d * 3 + d];
      const double reach = radius * sqrt(variance);
      lower[d] = mean - reach;
      upper[d] = mean + reach;
      invalid |= !isfinite(mean) || !isfinite(variance) || variance <= 0 ||
                 !isfinite(lower[d]) || !isfinite(upper[d]);
    }
    int64_t emitted = 0, examined = 0;
    for (int level = 0; level < grid.levels && !invalid; ++level) {
      if (grid.offsets[level] == grid.offsets[level + 1])
        continue;
      int64_t first[3], last[3], cells = 1;
      double extent[3];
      for (int d = 0; d < 3; ++d) {
        extent[d] = ldexp(grid.geometry[3 + d], -level);
        const double a = (lower[d] - grid.geometry[d]) / extent[d];
        const double b = (upper[d] - grid.geometry[d]) / extent[d];
        if (!isfinite(a) || !isfinite(b) || fabs(a) > 4e18 || fabs(b) > 4e18) {
          invalid = true;
          break;
        }
        first[d] = static_cast<int64_t>(floor(a + 1e-9));
        last[d] = static_cast<int64_t>(ceil(b - 1e-9)) - 1;
        if (last[d] < first[d]) {
          cells = 0;
          break;
        }
        const int64_t axis = last[d] - first[d] + 1;
        if (axis > limit / cells) {
          invalid = true;
          break;
        }
        cells *= axis;
      }
      if (invalid || cells > limit - examined) {
        invalid = true;
        break;
      }
      examined += cells;
      if (!cells)
        continue;
      for (int64_t x = first[0]; x <= last[0]; ++x)
        for (int64_t y = first[1]; y <= last[1]; ++y)
          for (int64_t z = first[2]; z <= last[2]; ++z) {
            const int64_t key[3] = {x, y, z};
            const int64_t row = find_cell(grid, level, key);
            if (row < 0)
              continue;
            bool overlap = true;
            for (int d = 0; d < 3; ++d) {
              const double lo = grid.geometry[d] + key[d] * extent[d];
              const double segment = fmin(upper[d], lo + extent[d]) -
                                     fmax(lower[d], lo);
              overlap &= segment > 1e-9 * extent[d];
            }
            if (!overlap)
              continue;
            if constexpr (Emit) {
              const int64_t output = (g ? prefix[g - 1] : 0) + emitted;
              gaussian_ids[output] = g;
              voxel_ids[output] = row;
            }
            ++emitted;
          }
    }
    if constexpr (!Emit)
      counts[g] = invalid ? 0 : emitted;
    if (invalid)
      atomicExch(error, 1);
  }
}

template <typename T>
__device__ T pair_values(
    const T *positions, const T *precision, const T *norm, const T *amplitude,
    const T *centers, const T *volumes, int64_t g, int64_t v,
    T *delta, T *quadratic) {
  for (int d = 0; d < 3; ++d)
    delta[d] = centers[v * 3 + d] - positions[g * 3 + d];
  *quadratic = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      *quadratic += delta[i] * precision[g * 9 + i * 3 + j] * delta[j];
  return exp(T(-0.5) * (*quadratic > 0 ? *quadratic : T(0))) * volumes[v];
}

template <typename T, bool Backward>
__global__ void accumulate_kernel(
    const T *positions, const T *precision, const T *norm, const T *amplitude,
    const T *features, const int64_t *gaussian_ids, const int64_t *voxel_ids,
    const T *centers, const T *volumes, int64_t pairs, int64_t channels,
    T *density, T *feature_sum, const T *grad_density, const T *grad_sum,
    T *grad_positions, T *grad_precision, T *grad_norm,
    T *grad_amplitude, T *grad_features) {
  for (int64_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pairs;
       pair += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t g = gaussian_ids[pair], v = voxel_ids[pair];
    T delta[3], q;
    const T base = pair_values(positions, precision, norm, amplitude,
                              centers, volumes, g, v, delta, &q);
    const T weight = amplitude[g] * norm[g] * base;
    if constexpr (!Backward) {
      atomicAdd(density + v, weight);
      for (int64_t c = 0; c < channels; ++c)
        atomicAdd(feature_sum + v * channels + c,
                  weight * features[g * channels + c]);
    } else {
      T grad_weight = grad_density[v];
      for (int64_t c = 0; c < channels; ++c) {
        const T upstream = grad_sum[v * channels + c];
        grad_weight += upstream * features[g * channels + c];
        atomicAdd(grad_features + g * channels + c, upstream * weight);
      }
      atomicAdd(grad_norm + g, grad_weight * amplitude[g] * base);
      atomicAdd(grad_amplitude + g, grad_weight * norm[g] * base);
      if (q >= 0) {
        const T factor = grad_weight * weight;
        for (int i = 0; i < 3; ++i) {
          T symmetric_product = 0;
          for (int j = 0; j < 3; ++j) {
            symmetric_product += T(0.5) *
                (precision[g * 9 + i * 3 + j] +
                 precision[g * 9 + j * 3 + i]) * delta[j];
            atomicAdd(grad_precision + g * 9 + i * 3 + j,
                      T(-0.5) * factor * delta[i] * delta[j]);
          }
          atomicAdd(grad_positions + g * 3 + i, factor * symmetric_product);
        }
      }
    }
  }
}

class Accumulate : public torch::autograd::Function<Accumulate> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext *ctx, at::Tensor positions,
      at::Tensor precision, at::Tensor norm, at::Tensor amplitude,
      at::Tensor features, at::Tensor gaussian_ids, at::Tensor voxel_ids,
      at::Tensor centers, at::Tensor volumes) {
    const c10::cuda::CUDAGuard guard(positions.device());
    positions = positions.contiguous();
    precision = precision.contiguous();
    norm = norm.contiguous();
    amplitude = amplitude.contiguous();
    features = features.contiguous();
    centers = centers.contiguous();
    volumes = volumes.contiguous();
    ctx->save_for_backward({positions, precision, norm, amplitude, features,
                           gaussian_ids, voxel_ids, centers, volumes});
    auto density = at::zeros({centers.size(0), 1}, positions.options());
    auto sums = at::zeros({centers.size(0), features.size(1)}, positions.options());
    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "voxel_forward", [&] {
      accumulate_kernel<scalar_t, false><<<blocks(gaussian_ids.numel()), kThreads,
          0, at::cuda::getCurrentCUDAStream()>>>(
          positions.data_ptr<scalar_t>(), precision.data_ptr<scalar_t>(),
          norm.data_ptr<scalar_t>(), amplitude.data_ptr<scalar_t>(),
          features.data_ptr<scalar_t>(), gaussian_ids.data_ptr<int64_t>(),
          voxel_ids.data_ptr<int64_t>(), centers.data_ptr<scalar_t>(),
          volumes.data_ptr<scalar_t>(), gaussian_ids.numel(), features.size(1),
          density.data_ptr<scalar_t>(), sums.data_ptr<scalar_t>(),
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {density, sums};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list gradients) {
    TORCH_CHECK(!at::GradMode::is_enabled(),
                "CUDA Gaussian voxel accumulation supports first-order gradients only");
    const auto saved = ctx->get_saved_variables();
    const c10::cuda::CUDAGuard guard(saved[0].device());
    std::vector<at::Tensor> grads;
    for (int i = 0; i < 5; ++i)
      grads.push_back(at::zeros_like(saved[i]));
    const auto gd = gradients[0].contiguous(), gs = gradients[1].contiguous();
    AT_DISPATCH_FLOATING_TYPES(saved[0].scalar_type(), "voxel_backward", [&] {
      accumulate_kernel<scalar_t, true><<<blocks(saved[5].numel()), kThreads,
          0, at::cuda::getCurrentCUDAStream()>>>(
          saved[0].data_ptr<scalar_t>(), saved[1].data_ptr<scalar_t>(),
          saved[2].data_ptr<scalar_t>(), saved[3].data_ptr<scalar_t>(),
          saved[4].data_ptr<scalar_t>(), saved[5].data_ptr<int64_t>(),
          saved[6].data_ptr<int64_t>(), saved[7].data_ptr<scalar_t>(),
          saved[8].data_ptr<scalar_t>(), saved[5].numel(), saved[4].size(1),
          nullptr, nullptr, gd.data_ptr<scalar_t>(), gs.data_ptr<scalar_t>(),
          grads[0].data_ptr<scalar_t>(), grads[1].data_ptr<scalar_t>(),
          grads[2].data_ptr<scalar_t>(), grads[3].data_ptr<scalar_t>(),
          grads[4].data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grads[0], grads[1], grads[2], grads[3], grads[4], {}, {}, {}, {}};
  }
};

} // namespace

std::pair<at::Tensor, at::Tensor> cuda_associations(
    const at::Tensor &positions, const at::Tensor &covariances,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options) {
  const c10::cuda::CUDAGuard guard(positions.device());
  const auto device = positions.device();
  const auto integer_options = positions.options().dtype(at::kLong);
  if (!positions.size(0) || !reference.size())
    return {at::empty({0}, integer_options), at::empty({0}, integer_options)};
  const auto &topology = *reference.topology();
  const auto coordinates = reference.coordinates().to(device).contiguous();
  const auto offsets = at::tensor(topology.level_offsets, at::kLong).to(device);
  const auto geometry = at::tensor(
      {topology.origin[0], topology.origin[1], topology.origin[2],
       topology.base[0], topology.base[1], topology.base[2]}, at::kDouble).to(device);
  const auto means = positions.detach().to(at::kDouble).contiguous();
  const auto covariance = covariances.detach().to(at::kDouble).contiguous();
  auto counts = at::zeros({positions.size(0)}, integer_options);
  auto error = at::zeros({}, integer_options.dtype(at::kInt));
  const Grid grid{coordinates.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                  geometry.data_ptr<double>(), static_cast<int>(topology.max_depth + 1)};
  association_kernel<false><<<blocks(positions.size(0)), kThreads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      means.data_ptr<double>(), covariance.data_ptr<double>(), positions.size(0),
      grid, options.support_radius, options.max_associations,
      counts.data_ptr<int64_t>(), nullptr, nullptr, nullptr, error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(!error.item<int>(),
              "Gaussian support is invalid, out of range, or exceeds max_associations");
  auto prefix = counts.cumsum(0);
  const int64_t total = prefix[-1].item<int64_t>();
  TORCH_CHECK(total <= options.max_associations,
              "Gaussian-to-voxel associations exceed max_associations");
  auto gaussian_ids = at::empty({total}, integer_options);
  auto voxel_ids = at::empty({total}, integer_options);
  if (total) {
    association_kernel<true><<<blocks(positions.size(0)), kThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<double>(), covariance.data_ptr<double>(), positions.size(0),
        grid, options.support_radius, options.max_associations,
        counts.data_ptr<int64_t>(), prefix.data_ptr<int64_t>(),
        gaussian_ids.data_ptr<int64_t>(), voxel_ids.data_ptr<int64_t>(),
        error.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gaussian_ids, voxel_ids};
}

std::pair<at::Tensor, at::Tensor> cuda_accumulate(
    const at::Tensor &positions, const at::Tensor &precision,
    const at::Tensor &normalization, const at::Tensor &amplitudes,
    const at::Tensor &features, const at::Tensor &gaussian_ids,
    const at::Tensor &voxel_ids, const at::Tensor &centers,
    const at::Tensor &volumes) {
  auto result = Accumulate::apply(positions, precision, normalization, amplitudes,
                                 features, gaussian_ids, voxel_ids, centers, volumes);
  return {result[0], result[1]};
}

} // namespace riemann::gaussian_voxel
