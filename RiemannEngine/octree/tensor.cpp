#include "tensor.hpp"
#include "tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace riemann {
namespace adaptive {

using detail::DVec;
using detail::Key;
using detail::Topology;

namespace {

void require_matrix(const at::Tensor &f) {
  TORCH_CHECK(f.dim() == 2, "adaptive features must be 2-D [N, C]");
}

at::Tensor host_index(const std::vector<int64_t> &v, const torch::Device &dev) {
  return torch::from_blob(const_cast<int64_t *>(v.data()),
                          {static_cast<int64_t>(v.size())}, torch::kInt64)
      .clone()
      .to(dev);
}

// out[dst[m]] (reduce)= scale[m] * f[src[m]], expressed with differentiable
// torch ops so gradients reach f. Destination rows without entries stay zero.
at::Tensor reduce_rows(const at::Tensor &f, const std::vector<int64_t> &src,
                       const std::vector<int64_t> &dst, const std::vector<double> *scale,
                       int64_t rows, Reduction reduction) {
  require_matrix(f);
  const int64_t C = f.size(1);
  auto out = torch::zeros({rows, C}, f.options());
  const int64_t M = static_cast<int64_t>(src.size());
  if (M == 0)
    return out;
  const auto dev = f.device();
  auto src_t = host_index(src, dev);
  auto dst_t = host_index(dst, dev);
  auto g = f.index_select(0, src_t);
  if (scale) {
    auto s = torch::from_blob(const_cast<double *>(scale->data()), {M}, torch::kFloat64)
                 .clone()
                 .to(f.scalar_type())
                 .to(dev)
                 .unsqueeze(1);
    g = g * s;
  }
  switch (reduction) {
  case Reduction::Sum:
    return out.index_add_(0, dst_t, g);
  case Reduction::Mean: {
    std::vector<double> tally(static_cast<size_t>(rows), 0.0);
    for (int64_t d : dst)
      tally[static_cast<size_t>(d)] += 1.0;
    for (double &c : tally)
      c = std::max(c, 1.0);
    auto cnt = torch::from_blob(tally.data(), {rows}, torch::kFloat64)
                   .clone()
                   .to(f.scalar_type())
                   .to(dev)
                   .unsqueeze(1);
    return out.index_add_(0, dst_t, g) / cnt;
  }
  case Reduction::Max:
    return torch::scatter_reduce(out, 0, dst_t.unsqueeze(1).expand({M, C}), g, "amax",
                                 /*include_self=*/false);
  }
  return out;
}

// Proposed next leaf set: one (cell, source row, scale) record per contribution.
struct LeafPlan {
  std::vector<Key> cell;
  std::vector<int64_t> from;
  std::vector<double> scale;

  void add(const Key &k, int64_t src, double s) {
    cell.push_back(k);
    from.push_back(src);
    scale.push_back(s);
  }
};

struct Rebuilt {
  std::shared_ptr<const Topology> topology;
  at::Tensor features;
};

// Sorts the plan into canonical order, merges records targeting the same cell
// into one leaf, and produces the matching feature rows.
Rebuilt materialize(const AdaptiveVoxelConfig &cfg, const LeafPlan &plan,
                    const at::Tensor &features, Reduction reduction, bool scaled) {
  const size_t M = plan.cell.size();
  std::vector<int64_t> order(M);
  std::iota(order.begin(), order.end(), int64_t(0));
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return detail::key_before(plan.cell[a], plan.cell[b]);
  });

  std::vector<Key> leaves;
  std::vector<int64_t> src(M), dst(M);
  std::vector<double> scale(M);
  leaves.reserve(M);
  for (size_t m = 0; m < M; ++m) {
    const int64_t i = order[m];
    if (leaves.empty() || !(leaves.back() == plan.cell[i]))
      leaves.push_back(plan.cell[i]);
    src[m] = plan.from[i];
    dst[m] = static_cast<int64_t>(leaves.size()) - 1;
    scale[m] = plan.scale[i];
  }
  const int64_t rows = static_cast<int64_t>(leaves.size());
  auto feats = reduce_rows(features, src, dst, scaled ? &scale : nullptr, rows, reduction);
  return {Topology::make(cfg, std::move(leaves)), feats};
}

void check_region(const Topology &t, const AxisAlignedRegion &r) {
  if (r.lo.size() != static_cast<size_t>(t.D) || r.hi.size() != static_cast<size_t>(t.D))
    throw std::invalid_argument("region dimension does not match the tensor dimension");
}

bool cell_touches(const Topology &t, const Key &k, const AxisAlignedRegion &r) {
  DVec lo, hi;
  t.cell_box(k, lo, hi);
  for (int d = 0; d < t.D; ++d)
    if (!(lo[d] < r.hi[d] && hi[d] > r.lo[d]))
      return false;
  return true;
}

bool cell_inside(const Topology &t, const Key &k, const AxisAlignedRegion &r) {
  DVec lo, hi;
  t.cell_box(k, lo, hi);
  for (int d = 0; d < t.D; ++d)
    if (!(lo[d] >= r.lo[d] && hi[d] <= r.hi[d]))
      return false;
  return true;
}

// Replaces each selected leaf by its 2^D children (one level only).
Rebuilt split_selected(const Topology &t, const at::Tensor &f, const std::vector<char> &picked,
                       RefinePolicy policy) {
  LeafPlan plan;
  std::vector<Key> kids;
  const double share = policy == RefinePolicy::Split ? 1.0 / double(1u << t.D) : 1.0;
  for (int64_t i = 0; i < t.size(); ++i) {
    if (!picked[static_cast<size_t>(i)]) {
      plan.add(t.keys[i], i, 1.0);
      continue;
    }
    kids.clear();
    detail::append_children(t.keys[i], t.D, kids);
    for (const Key &c : kids)
      plan.add(c, i, share);
  }
  return materialize(t.config, plan, f, Reduction::Sum, policy == RefinePolicy::Split);
}

// Merges leaves at exactly `level` into their parents when the parent box lies
// inside the region. Returns an empty Rebuilt when nothing merged.
Rebuilt merge_level(const Topology &t, const at::Tensor &f, const AxisAlignedRegion &region,
                    uint32_t level, Reduction reduction, bool &changed) {
  LeafPlan plan;
  changed = false;
  for (int64_t i = 0; i < t.size(); ++i) {
    const Key &k = t.keys[i];
    if (k.level == level && level >= 1) {
      const Key p = detail::parent_of(k);
      if (cell_inside(t, p, region)) {
        plan.add(p, i, 1.0);
        changed = true;
        continue;
      }
    }
    plan.add(k, i, 1.0);
  }
  if (!changed)
    return {};
  return materialize(t.config, plan, f, reduction, false);
}

// Containing-leaf row per point, or -1.
std::vector<int64_t> assign_points(const Topology &t, const at::Tensor &points) {
  const int64_t P = points.size(0);
  const double *p = points.data_ptr<double>();
  std::vector<int64_t> rows(static_cast<size_t>(P), -1);
  for (int64_t i = 0; i < P; ++i) {
    DVec q{};
    for (int d = 0; d < t.D; ++d)
      q[d] = p[i * t.D + d];
    rows[static_cast<size_t>(i)] = t.find_containing(q);
  }
  return rows;
}

} // namespace

AdaptiveSparseTensor
AdaptiveSparseTensor::from_topology(std::shared_ptr<const Topology> topology,
                                    at::Tensor features,
                                    std::shared_ptr<const SourcePoints> source) {
  require_matrix(features);
  if (features.size(0) != topology->size())
    throw std::invalid_argument("feature rows (" + std::to_string(features.size(0)) +
                                ") do not match the number of voxels (" +
                                std::to_string(topology->size()) + ")");
  AdaptiveSparseTensor t;
  t.topology_ = std::move(topology);
  t.features_ = std::move(features);
  t.source_ = std::move(source);
  return t;
}

AdaptiveSparseTensor AdaptiveSparseTensor::from_voxels(const at::Tensor &coordinates,
                                                       const at::Tensor &levels,
                                                       const at::Tensor &features,
                                                       const AdaptiveVoxelConfig &config) {
  config.validate();
  require_matrix(features);
  const int D = static_cast<int>(config.dimension());
  auto coords = coordinates.to(torch::kInt64).cpu().contiguous();
  auto lev = levels.to(torch::kInt32).cpu().contiguous();
  if (coords.dim() != 2 || coords.size(1) != D)
    throw std::invalid_argument("coordinates must have shape [N, D]");
  const int64_t N = coords.size(0);
  if (lev.numel() != N || features.size(0) != N)
    throw std::invalid_argument("coordinates, levels and features must have the same length");

  std::vector<Key> keys(static_cast<size_t>(N));
  const auto *cd = coords.data_ptr<int64_t>();
  const auto *ld = lev.data_ptr<int32_t>();
  detail::parallel_blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      if (ld[i] < 0)
        throw std::invalid_argument("negative voxel level");
      Key &k = keys[i];
      k = Key{};
      k.level = static_cast<uint32_t>(ld[i]);
      for (int d = 0; d < D; ++d)
        k.coordinates[d] = cd[i * D + d];
    }
  });

  std::vector<int64_t> perm;
  detail::sort_keys(keys, perm);
  auto feats =
      perm.empty() ? features : features.index_select(0, host_index(perm, features.device()));
  return from_topology(Topology::make(config, std::move(keys)), feats);
}

AdaptiveSparseTensor AdaptiveSparseTensor::from_points(const at::Tensor &points,
                                                       const at::Tensor &features,
                                                       const AdaptiveVoxelConfig &config) {
  config.validate();
  require_matrix(features);
  const int D = static_cast<int>(config.dimension());
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != D)
    throw std::invalid_argument("points must have shape [P, D]");
  const int64_t P = pts.size(0);
  if (features.size(0) != P)
    throw std::invalid_argument("points and features must have the same length");

  AdaptiveVoxelConfig cfg = config;
  if (cfg.origin.empty())
    cfg.origin.assign(D, 0.0);

  // Quantize every point to its base-level cell, grouping duplicates.
  const double *pd = pts.data_ptr<double>();
  std::unordered_map<Key, int64_t, detail::KeyHash> groups;
  std::vector<Key> group_keys;
  std::vector<int64_t> group_of(static_cast<size_t>(P));
  for (int64_t i = 0; i < P; ++i) {
    Key k{};
    for (int d = 0; d < D; ++d) {
      const double q = std::floor((pd[i * D + d] - cfg.origin[d]) / cfg.base_voxel_size[d]);
      if (!std::isfinite(q) || std::fabs(q) > 4e18)
        throw std::invalid_argument("point coordinate is not finite or out of range");
      k.coordinates[d] = static_cast<int64_t>(q);
    }
    auto ins = groups.emplace(k, static_cast<int64_t>(group_keys.size()));
    if (ins.second)
      group_keys.push_back(k);
    group_of[i] = ins.first->second;
  }

  const auto order = detail::sorted_order(group_keys);
  std::vector<Key> leaves(group_keys.size());
  std::vector<int64_t> slot(group_keys.size());
  for (size_t i = 0; i < order.size(); ++i) {
    leaves[i] = group_keys[order[i]];
    slot[order[i]] = static_cast<int64_t>(i);
  }
  std::vector<int64_t> src(static_cast<size_t>(P)), dst(static_cast<size_t>(P));
  for (int64_t i = 0; i < P; ++i) {
    src[i] = i;
    dst[i] = slot[group_of[i]];
  }
  auto feats = reduce_rows(features, src, dst, nullptr, static_cast<int64_t>(leaves.size()),
                           Reduction::Mean);
  auto source = std::make_shared<SourcePoints>();
  source->points = pts;
  source->features = features.detach();
  return from_topology(Topology::make(cfg, std::move(leaves)), feats, source);
}

int64_t AdaptiveSparseTensor::size() const { return topology_->size(); }
int AdaptiveSparseTensor::dimension() const { return topology_->D; }
int64_t AdaptiveSparseTensor::channels() const { return features_.size(1); }
const AdaptiveVoxelConfig &AdaptiveSparseTensor::config() const { return topology_->config; }
at::Tensor AdaptiveSparseTensor::coordinates() const { return topology_->coords; }
at::Tensor AdaptiveSparseTensor::levels() const { return topology_->levels; }
at::Tensor AdaptiveSparseTensor::volumes() const { return topology_->volumes; }
at::Tensor AdaptiveSparseTensor::voxel_sizes() const { return topology_->cell_extents(); }
at::Tensor AdaptiveSparseTensor::centers() const { return topology_->cell_centers(); }
std::pair<at::Tensor, at::Tensor> AdaptiveSparseTensor::bounds() const {
  return topology_->cell_boxes();
}
std::vector<int64_t> AdaptiveSparseTensor::counts_by_level() const { return topology_->counts; }
uint64_t AdaptiveSparseTensor::topology_version() const { return topology_->id; }
bool AdaptiveSparseTensor::has_overlapping_leaves() const { return topology_->has_overlap(); }

void AdaptiveSparseTensor::set_features(at::Tensor features) {
  require_matrix(features);
  if (features.size(0) != size())
    throw std::invalid_argument("feature rows do not match the number of voxels");
  features_ = std::move(features);
}

int64_t AdaptiveSparseTensor::index_of(std::array<int64_t, 3> coordinates,
                                       uint32_t level) const {
  return index_of_key(std::vector<int64_t>(coordinates.begin(), coordinates.end()), level);
}

int64_t AdaptiveSparseTensor::index_of_key(const std::vector<int64_t> &coordinates,
                                           uint32_t level) const {
  if (coordinates.size() != static_cast<size_t>(topology_->D))
    throw std::invalid_argument("coordinate dimension does not match the tensor dimension");
  Key k{};
  k.level = level;
  for (size_t d = 0; d < coordinates.size(); ++d)
    k.coordinates[d] = coordinates[d];
  return topology_->find(k);
}

void AdaptiveSparseTensor::refine(const AxisAlignedRegion &region, uint32_t target_level,
                                  RefinePolicy policy) {
  check_region(*topology_, region);
  if (target_level > topology_->max_depth)
    throw std::invalid_argument("refine target_level exceeds max_depth");
  for (;;) {
    const Topology &t = *topology_;
    std::vector<char> picked(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i)
      if (t.keys[i].level < target_level && cell_touches(t, t.keys[i], region)) {
        picked[i] = 1;
        any = true;
      }
    if (!any)
      break;
    auto next = split_selected(t, features_, picked, policy);
    topology_ = std::move(next.topology);
    features_ = std::move(next.features);
  }
}

void AdaptiveSparseTensor::coarsen(const AxisAlignedRegion &region, uint32_t target_level,
                                   Reduction reduction) {
  check_region(*topology_, region);
  for (int64_t level = static_cast<int64_t>(topology_->max_depth);
       level > static_cast<int64_t>(target_level); --level) {
    bool changed = false;
    auto next = merge_level(*topology_, features_, region, static_cast<uint32_t>(level),
                            reduction, changed);
    if (changed) {
      topology_ = std::move(next.topology);
      features_ = std::move(next.features);
    }
  }
}

void AdaptiveSparseTensor::refine_if(
    const std::function<bool(int64_t, const std::array<double, 4> &,
                             const std::array<double, 4> &, uint32_t)> &criterion) {
  for (;;) {
    const Topology &t = *topology_;
    std::vector<char> picked(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i) {
      if (t.keys[i].level >= t.max_depth)
        continue;
      DVec lo, hi;
      t.cell_box(t.keys[i], lo, hi);
      if (criterion(i, lo, hi, t.keys[i].level)) {
        picked[i] = 1;
        any = true;
      }
    }
    if (!any)
      break;
    auto next = split_selected(t, features_, picked, RefinePolicy::Replicate);
    topology_ = std::move(next.topology);
    features_ = std::move(next.features);
  }
}

void AdaptiveSparseTensor::refine_by_density(int64_t points_per_leaf_threshold) {
  if (!source_)
    throw std::invalid_argument("refine_by_density requires source points (use from_points)");
  for (;;) {
    const Topology &t = *topology_;
    std::vector<int64_t> tally(static_cast<size_t>(t.size()), 0);
    for (int64_t row : assign_points(t, source_->points))
      if (row >= 0)
        ++tally[row];
    std::vector<char> picked(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i)
      if (tally[i] > points_per_leaf_threshold && t.keys[i].level < t.max_depth) {
        picked[i] = 1;
        any = true;
      }
    if (!any)
      break;
    auto next = split_selected(t, features_, picked, RefinePolicy::Replicate);
    topology_ = std::move(next.topology);
    features_ = std::move(next.features);
  }
}

void AdaptiveSparseTensor::refine_by_variance(double feature_variance_threshold) {
  if (!source_)
    throw std::invalid_argument("refine_by_variance requires source points (use from_points)");
  for (;;) {
    const Topology &t = *topology_;
    const auto rows = assign_points(t, source_->points);
    std::vector<int64_t> point_idx, leaf_idx;
    for (size_t p = 0; p < rows.size(); ++p)
      if (rows[p] >= 0) {
        point_idx.push_back(static_cast<int64_t>(p));
        leaf_idx.push_back(rows[p]);
      }
    std::vector<char> picked(static_cast<size_t>(t.size()), 0);
    bool any = false;
    if (!point_idx.empty()) {
      auto x = source_->features
                   .index_select(0, host_index(point_idx, source_->features.device()))
                   .to(torch::kFloat64)
                   .cpu();
      auto li = host_index(leaf_idx, torch::kCPU);
      const int64_t C = x.size(1);
      const int64_t M = static_cast<int64_t>(leaf_idx.size());
      auto cnt = torch::zeros({t.size()}, torch::kFloat64)
                     .index_add_(0, li, torch::ones({M}, torch::kFloat64));
      auto s1 = torch::zeros({t.size(), C}, torch::kFloat64).index_add_(0, li, x);
      auto s2 = torch::zeros({t.size(), C}, torch::kFloat64).index_add_(0, li, x * x);
      auto n = cnt.clamp_min(1.0).unsqueeze(1);
      auto var = (s2 / n - (s1 / n) * (s1 / n)).clamp_min(0.0).mean(1);
      const double *v = var.data_ptr<double>();
      for (int64_t i = 0; i < t.size(); ++i)
        if (v[i] > feature_variance_threshold && t.keys[i].level < t.max_depth) {
          picked[i] = 1;
          any = true;
        }
    }
    if (!any)
      break;
    auto next = split_selected(t, features_, picked, RefinePolicy::Replicate);
    topology_ = std::move(next.topology);
    features_ = std::move(next.features);
  }
}

void AdaptiveSparseTensor::revoxelize_from_points() {
  if (!source_)
    throw std::invalid_argument("revoxelize_from_points requires source points");
  const Topology &t = *topology_;
  const auto rows = assign_points(t, source_->points);
  std::vector<int64_t> src, dst;
  for (size_t p = 0; p < rows.size(); ++p)
    if (rows[p] >= 0) {
      src.push_back(static_cast<int64_t>(p));
      dst.push_back(rows[p]);
    }
  features_ = reduce_rows(source_->features, src, dst, nullptr, t.size(), Reduction::Mean);
}

AdaptiveSparseTensor AdaptiveSparseTensor::with_features(at::Tensor features) const {
  return from_topology(topology_, std::move(features), source_);
}

AdaptiveSparseTensor AdaptiveSparseTensor::to(torch::Device device) const {
  return from_topology(topology_, features_.to(device), source_);
}

AdaptiveSparseTensor AdaptiveSparseTensor::resample_uniform(uint32_t level,
                                                            Reduction reduction) const {
  if (level > topology_->max_depth)
    throw std::invalid_argument("resample level exceeds max_depth");
  AdaptiveSparseTensor out = *this;
  out.source_ = nullptr;
  const auto all = AxisAlignedRegion::everything(static_cast<size_t>(topology_->D));
  out.refine(all, level, RefinePolicy::Replicate);
  out.coarsen(all, level, reduction);
  return out;
}

AdaptiveSparseTensor AdaptiveSparseTensor::resample_to(const AdaptiveSparseTensor &reference,
                                                       Reduction reduction) const {
  const Topology &s = *topology_;
  const Topology &r = *reference.topology_;
  if (s.D != r.D)
    throw std::invalid_argument("resample_to: dimension mismatch");
  std::vector<int64_t> src, dst;
  // Reference leaves covered by a same-level or coarser source leaf replicate it.
  for (int64_t j = 0; j < r.size(); ++j) {
    const Key &rk = r.keys[j];
    for (uint32_t l = 0; l <= rk.level; ++l) {
      const int64_t i = s.find(detail::coarse_ancestor(rk, l));
      if (i >= 0) {
        src.push_back(i);
        dst.push_back(j);
        break;
      }
    }
  }
  // Strictly finer source leaves reduce into the reference leaf containing them.
  for (int64_t i = 0; i < s.size(); ++i) {
    const Key &sk = s.keys[i];
    for (uint32_t l = 0; l < sk.level; ++l) {
      const int64_t j = r.find(detail::coarse_ancestor(sk, l));
      if (j >= 0) {
        src.push_back(i);
        dst.push_back(j);
        break;
      }
    }
  }
  auto feats = reduce_rows(features_, src, dst, nullptr, r.size(), reduction);
  return from_topology(reference.topology_, feats);
}

AdaptiveSparseTensor AdaptiveSparseTensor::operator+(const AdaptiveSparseTensor &other) const {
  if (!detail::same_topology(*topology_, *other.topology_))
    throw std::invalid_argument(
        "adaptive tensors have different topologies; align with resample_to() first");
  return from_topology(topology_, features_ + other.features_, source_);
}

AdaptiveSparseTensor concatenate(const AdaptiveSparseTensor &a, const AdaptiveSparseTensor &b) {
  if (!detail::same_topology(*a.topology(), *b.topology()))
    throw std::invalid_argument(
        "adaptive tensors have different topologies; align with resample_to() first");
  return AdaptiveSparseTensor::from_topology(a.topology(),
                                             torch::cat({a.features(), b.features()}, 1));
}

} // namespace adaptive
} // namespace riemann
