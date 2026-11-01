#include "stencil.hpp"
#include "tree.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace riemann {
namespace adaptive {

using detail::DVec;
using detail::Key;
using detail::Topology;

int64_t kernel_index(std::array<int, 3> offset, int kernel_size) {
  const int r = (kernel_size - 1) / 2;
  int64_t slot = offset[0] + r;
  slot = slot * kernel_size + offset[1] + r;
  slot = slot * kernel_size + offset[2] + r;
  return slot;
}

namespace {

int64_t slot_count(int kernel_size, int dim) {
  int64_t n = 1;
  for (int d = 0; d < dim; ++d)
    n *= kernel_size;
  return n;
}

// Refinement replaces each leaf with all descendants at the shifted level.
// Coarsening drops lifted keys nested under another lifted key.
std::shared_ptr<const Topology>
shifted_outputs(const std::shared_ptr<const Topology> &in, int shift) {
  if (shift == 0)
    return in;
  if (shift > 0) {
    if (static_cast<uint32_t>(shift) > in->max_depth)
      throw std::invalid_argument("output refinement exceeds max_depth");
    const int64_t scale = int64_t(1) << shift;
    for (const Key &k : in->keys) {
      if (static_cast<int64_t>(k.level) + shift > in->max_depth)
        throw std::invalid_argument("output refinement exceeds max_depth");
      for (int d = 0; d < in->D; ++d)
        if (k.coordinates[d] < std::numeric_limits<int64_t>::min() / scale ||
            k.coordinates[d] > std::numeric_limits<int64_t>::max() / scale)
          throw std::invalid_argument("output refinement overflows int64 coordinates");
    }

    std::vector<Key> leaves = in->keys;
    const size_t fanout = size_t(1) << in->D;
    const size_t limit = std::min(leaves.max_size(),
                                 static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    size_t count = leaves.size();
    for (int l = 0; l < shift; ++l) {
      if (count > limit / fanout)
        throw std::length_error("output refinement has too many leaves");
      count *= fanout;
    }
    for (int l = 0; l < shift; ++l) {
      std::vector<Key> children;
      children.reserve(leaves.size() * fanout);
      for (const Key &k : leaves)
        detail::append_children(k, in->D, children);
      leaves.swap(children);
    }
    // Mixed-level inputs and child enumeration need canonical row ordering.
    std::sort(leaves.begin(), leaves.end(), detail::key_before);
    return Topology::make(in->config, std::move(leaves));
  }

  std::vector<Key> lifted;
  lifted.reserve(in->keys.size());
  for (const Key &k : in->keys) {
    const int64_t target = std::max<int64_t>(static_cast<int64_t>(k.level) + shift, 0);
    lifted.push_back(detail::coarse_ancestor(k, static_cast<uint32_t>(target)));
  }
  std::sort(lifted.begin(), lifted.end(), detail::key_before);
  lifted.erase(std::unique(lifted.begin(), lifted.end()), lifted.end());

  const auto present = [&](const Key &a) {
    return std::binary_search(lifted.begin(), lifted.end(), a, detail::key_before);
  };
  std::vector<Key> kept;
  for (const Key &k : lifted) {
    bool covered = false;
    for (uint32_t l = 0; l < k.level && !covered; ++l)
      covered = present(detail::coarse_ancestor(k, l));
    if (!covered)
      kept.push_back(k);
  }
  return Topology::make(in->config, std::move(kept));
}

AdaptiveKernelMap collect_pairs(const Topology &out, const Topology &in,
                                const AdaptiveKernelConfig &kc) {
  const int K = kc.kernel_size;
  if (K < 1 || K % 2 == 0)
    throw std::invalid_argument("kernel_size must be a positive odd number");
  if (out.D != in.D)
    throw std::invalid_argument("dimension mismatch between input and output topology");
  const int D = out.D;
  const int r = (K - 1) / 2;
  const int64_t nk = slot_count(K, D);

  // Offset of each slot, first dimension slowest.
  std::vector<std::array<int, detail::kMaxDim>> taps(static_cast<size_t>(nk));
  for (int64_t slot = 0; slot < nk; ++slot) {
    int64_t rem = slot;
    for (int d = D - 1; d >= 0; --d) {
      taps[slot][d] = static_cast<int>(rem % K) - r;
      rem /= K;
    }
  }

  std::vector<int64_t> slot_of, outs, ins;
  std::vector<double> weights;
  const double *in_volume = in.volumes.data_ptr<double>();

  for (int64_t o = 0; o < out.size(); ++o) {
    const Key &ok = out.keys[o];
    DVec lo, hi;
    out.cell_box(ok, lo, hi);
    const DVec ext = out.extent_at(ok.level);
    double out_vol = 1.0;
    DVec mid{};
    for (int d = 0; d < D; ++d) {
      out_vol *= ext[d];
      mid[d] = lo[d] + 0.5 * ext[d];
    }
    for (int64_t slot = 0; slot < nk; ++slot) {
      DVec tap{};
      for (int d = 0; d < D; ++d)
        tap[d] = mid[d] + taps[slot][d] * ext[d];
      if (kc.mode == AdaptiveConvMode::OverlapWeighted) {
        DVec blo{}, bhi{};
        for (int d = 0; d < D; ++d) {
          blo[d] = tap[d] - 0.5 * ext[d];
          bhi[d] = tap[d] + 0.5 * ext[d];
        }
        in.for_each_overlap(blo, bhi, [&](int64_t row, double overlap) {
          slot_of.push_back(slot);
          outs.push_back(o);
          ins.push_back(row);
          weights.push_back(overlap / out_vol);
        });
      } else {
        const int64_t row = in.find_containing(tap);
        if (row < 0)
          continue;
        slot_of.push_back(slot);
        outs.push_back(o);
        ins.push_back(row);
        weights.push_back(kc.mode == AdaptiveConvMode::VolumeWeighted
                              ? in_volume[row] / out_vol
                              : 1.0);
      }
    }
  }

  // Counting sort by slot keeps the within-slot output order stable.
  const int64_t P = static_cast<int64_t>(slot_of.size());
  AdaptiveKernelMap m;
  m.num_kernels = nk;
  m.kernel_offsets.assign(static_cast<size_t>(nk) + 1, 0);
  for (int64_t s : slot_of)
    ++m.kernel_offsets[static_cast<size_t>(s) + 1];
  for (int64_t s = 0; s < nk; ++s)
    m.kernel_offsets[s + 1] += m.kernel_offsets[s];

  std::vector<int64_t> in_sorted(P), out_sorted(P), slot_sorted(P);
  std::vector<double> w_sorted(P);
  std::vector<int64_t> cursor(m.kernel_offsets.begin(), m.kernel_offsets.end() - 1);
  for (int64_t p = 0; p < P; ++p) {
    const int64_t pos = cursor[slot_of[p]]++;
    in_sorted[pos] = ins[p];
    out_sorted[pos] = outs[p];
    slot_sorted[pos] = slot_of[p];
    w_sorted[pos] = weights[p];
  }

  const auto as_i64 = [&](std::vector<int64_t> &v) {
    return torch::from_blob(v.data(), {P}, torch::kInt64).clone();
  };
  m.in_index = as_i64(in_sorted);
  m.out_index = as_i64(out_sorted);
  m.kernel_index = as_i64(slot_sorted);
  m.weight = torch::from_blob(w_sorted.data(), {P}, torch::kFloat64).clone();
  return m;
}

using CacheKey = std::tuple<uint64_t, int, int, int>;

} // namespace

struct KernelMapCache::Impl {
  mutable std::mutex lock;
  std::map<CacheKey, KernelMapEntry> entries;
  std::deque<CacheKey> arrival;
  size_t capacity = 64;
  size_t hits = 0;
  size_t misses = 0;

  void evict_over_capacity() {
    while (entries.size() > capacity && !arrival.empty()) {
      entries.erase(arrival.front());
      arrival.pop_front();
    }
  }
};

KernelMapCache::KernelMapCache() : impl_(new Impl) {}
KernelMapCache::~KernelMapCache() = default;

KernelMapCache &KernelMapCache::instance() {
  static KernelMapCache cache;
  return cache;
}

KernelMapEntry KernelMapCache::get_or_build(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config) {
  const CacheKey key{input.topology()->id, config.kernel_size, static_cast<int>(config.mode),
                     config.output_level_shift};
  {
    std::lock_guard<std::mutex> hold(impl_->lock);
    auto it = impl_->entries.find(key);
    if (it != impl_->entries.end()) {
      ++impl_->hits;
      return it->second;
    }
    ++impl_->misses;
  }
  KernelMapEntry entry;
  entry.out_topology = shifted_outputs(input.topology(), config.output_level_shift);
  entry.map = collect_pairs(*entry.out_topology, *input.topology(), config);
  {
    std::lock_guard<std::mutex> hold(impl_->lock);
    if (impl_->entries.emplace(key, entry).second)
      impl_->arrival.push_back(key);
    impl_->evict_over_capacity();
  }
  return entry;
}

void KernelMapCache::clear() {
  std::lock_guard<std::mutex> hold(impl_->lock);
  impl_->entries.clear();
  impl_->arrival.clear();
  impl_->hits = 0;
  impl_->misses = 0;
}

KernelMapCache::Stats KernelMapCache::stats() const {
  std::lock_guard<std::mutex> hold(impl_->lock);
  return {impl_->hits, impl_->misses, impl_->entries.size()};
}

void KernelMapCache::set_capacity(size_t max_entries) {
  std::lock_guard<std::mutex> hold(impl_->lock);
  impl_->capacity = std::max<size_t>(max_entries, 1);
  impl_->evict_over_capacity();
}

AdaptiveKernelMap build_adaptive_kernel_map(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config) {
  return KernelMapCache::instance().get_or_build(input, config).map;
}

} // namespace adaptive
} // namespace riemann
