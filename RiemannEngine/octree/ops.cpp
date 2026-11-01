#include "ops.hpp"
#include "tree.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace riemann {
namespace adaptive {

using detail::DVec;
using detail::Topology;

namespace {

void require_weight(const at::Tensor &weight, const AdaptiveKernelMap &m, int64_t cin) {
  if (weight.dim() != 3 || weight.size(0) != m.num_kernels || weight.size(1) != cin)
    throw std::invalid_argument(
        "weight must have shape [kernel_size^D, in_channels, out_channels]");
}

// Gather -> optional per-pair weight -> per-slot GEMM -> scatter-add. The
// gather/scatter index tensors are parameters so the transposed convolution
// can reuse the same map with the direction swapped.
at::Tensor sweep_kernel_map(const at::Tensor &x, const at::Tensor &weight,
                            const AdaptiveKernelMap &m, const at::Tensor &gather_idx,
                            const at::Tensor &scatter_idx, int64_t n_out, bool weighted) {
  const auto dev = x.device();
  const int64_t cout = weight.size(2);
  auto gather_d = gather_idx.to(dev);
  auto scatter_d = scatter_idx.to(dev);
  at::Tensor pair_w;
  if (weighted)
    pair_w = m.weight.to(dev).to(x.scalar_type());
  auto wt = weight.to(x.scalar_type());
  auto y = torch::zeros({n_out, cout}, x.options());
  for (int64_t k = 0; k < m.num_kernels; ++k) {
    const int64_t b = m.kernel_offsets[k], e = m.kernel_offsets[k + 1];
    if (b == e)
      continue;
    auto xs = x.index_select(0, gather_d.slice(0, b, e));
    if (weighted)
      xs = xs * pair_w.slice(0, b, e).unsqueeze(1);
    y.index_add_(0, scatter_d.slice(0, b, e), xs.matmul(wt[k]));
  }
  return y;
}

} // namespace

AdaptiveSparseTensor adaptive_convolution(const AdaptiveSparseTensor &input,
                                          const at::Tensor &weight,
                                          const AdaptiveKernelConfig &config) {
  auto entry = KernelMapCache::instance().get_or_build(input, config);
  require_weight(weight, entry.map, input.channels());
  auto y = sweep_kernel_map(input.features(), weight, entry.map, entry.map.in_index,
                            entry.map.out_index, entry.out_topology->size(),
                            config.mode != AdaptiveConvMode::CenterSampled);
  return AdaptiveSparseTensor::from_topology(entry.out_topology, y);
}

AdaptiveSparseTensor adaptive_convolution_transpose(const AdaptiveSparseTensor &input,
                                                    const at::Tensor &weight,
                                                    const AdaptiveKernelConfig &config,
                                                    const AdaptiveSparseTensor &reference) {
  // The forward convolution applied to `reference` defines the pair list.
  auto entry = KernelMapCache::instance().get_or_build(reference, config);
  if (!detail::same_topology(*entry.out_topology, *input.topology()))
    throw std::invalid_argument(
        "transposed convolution input does not match the forward convolution output topology");
  require_weight(weight, entry.map, input.channels());
  auto y = sweep_kernel_map(input.features(), weight, entry.map, entry.map.out_index,
                            entry.map.in_index, reference.size(),
                            config.mode != AdaptiveConvMode::CenterSampled);
  return AdaptiveSparseTensor::from_topology(reference.topology(), y);
}

at::Tensor adaptive_global_avg_pool(const AdaptiveSparseTensor &input, bool volume_weighted) {
  const auto &f = input.features();
  if (input.size() == 0)
    return torch::zeros({f.size(1)}, f.options());
  if (!volume_weighted)
    return f.mean(0);
  auto v = input.volumes().to(f.device()).to(f.scalar_type());
  return (f * v.unsqueeze(1)).sum(0) / v.sum();
}

AdaptiveSparseTensor adaptive_pool(const AdaptiveSparseTensor &input, uint32_t target_level,
                                   Reduction reduction) {
  AdaptiveSparseTensor out = input;
  out.coarsen(AxisAlignedRegion::everything(static_cast<size_t>(input.dimension())),
              target_level, reduction);
  return out;
}

at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode) {
  const Topology &t = *input.topology();
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != t.D)
    throw std::invalid_argument("points must have shape [M, D]");
  const int64_t M = pts.size(0);
  const double *pd = pts.data_ptr<double>();

  std::vector<int64_t> point_idx, leaf_idx;
  std::vector<double> weights;
  std::vector<std::pair<int64_t, double>> touched;
  for (int64_t m = 0; m < M; ++m) {
    DVec q{};
    for (int d = 0; d < t.D; ++d)
      q[d] = pd[m * t.D + d];
    const int64_t own = t.find_containing(q);
    if (own < 0)
      continue;
    if (mode == InterpMode::Nearest) {
      point_idx.push_back(m);
      leaf_idx.push_back(own);
      weights.push_back(1.0);
      continue;
    }
    // A box of the owning leaf's size centered on the point; overlap fractions
    // reproduce multilinear interpolation between leaf centers.
    const DVec ext = t.extent_at(t.keys[own].level);
    DVec lo{}, hi{};
    for (int d = 0; d < t.D; ++d) {
      lo[d] = q[d] - 0.5 * ext[d];
      hi[d] = q[d] + 0.5 * ext[d];
    }
    touched.clear();
    double total = 0.0;
    t.for_each_overlap(lo, hi, [&](int64_t leaf, double overlap) {
      touched.emplace_back(leaf, overlap);
      total += overlap;
    });
    for (const auto &[leaf, overlap] : touched) {
      point_idx.push_back(m);
      leaf_idx.push_back(leaf);
      weights.push_back(overlap / total);
    }
  }

  const auto &x = input.features();
  auto y = torch::zeros({M, x.size(1)}, x.options());
  if (point_idx.empty())
    return y;
  const int64_t n = static_cast<int64_t>(point_idx.size());
  auto pt_t = torch::from_blob(point_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto lf_t = torch::from_blob(leaf_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto w_t = torch::from_blob(weights.data(), {n}, torch::kFloat64)
                 .clone()
                 .to(x.scalar_type())
                 .to(x.device())
                 .unsqueeze(1);
  y.index_add_(0, pt_t, x.index_select(0, lf_t) * w_t);
  return y;
}

} // namespace adaptive
} // namespace riemann
