#include "tree.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace riemann {
namespace adaptive {
namespace detail {

namespace {

// Below this a device round trip costs more than the host work it replaces.
constexpr int64_t kDeviceThreshold = int64_t(1) << 14;
// Below this a single thread wins; above it, roughly one thread per 8K items.
constexpr int64_t kThreadThreshold = 32768;

// std::thread on purpose: OpenMP (and at::parallel_for, which may use it) can
// collide with LibTorch's bundled OpenMP runtime on Windows.
int worker_count(int64_t n) {
  if (n < kThreadThreshold)
    return 1;
  static const int hw =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  return static_cast<int>(std::min<int64_t>(hw, n / 8192));
}

// Runs job(t) for t in [0, tasks); the calling thread takes task 0. The first
// exception thrown by any task is rethrown after everything joined.
template <class Job> void fan_out(int tasks, Job &&job) {
  if (tasks <= 1) {
    job(0);
    return;
  }
  std::exception_ptr first_error;
  std::mutex error_lock;
  auto shielded = [&](int t) {
    try {
      job(t);
    } catch (...) {
      std::lock_guard<std::mutex> hold(error_lock);
      if (!first_error)
        first_error = std::current_exception();
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(tasks) - 1);
  for (int t = 1; t < tasks; ++t)
    pool.emplace_back(shielded, t);
  shielded(0);
  for (auto &w : pool)
    w.join();
  if (first_error)
    std::rethrow_exception(first_error);
}

template <class Body> void blocks(int64_t n, Body &&body) {
  const int tasks = worker_count(n);
  fan_out(tasks, [&](int t) { body(n * t / tasks, n * (t + 1) / tasks); });
}

// Checks whether any leaf has an active strict ancestor. Keys are sorted, so
// each coarser present level is a contiguous slice searched by bisection;
// consecutive keys usually share ancestors, which the memo exploits.
bool host_overlap_scan(const std::vector<Key> &keys, const std::vector<uint32_t> &present,
                       const std::vector<int64_t> &level_offsets) {
  const Key *data = keys.data();
  const auto slice_contains = [&](uint32_t level, const Key &a) {
    const Key *lo = data + level_offsets[level];
    const Key *hi = data + level_offsets[level + 1];
    const Key *it = std::lower_bound(
        lo, hi, a, [](const Key &x, const Key &y) { return x.coordinates < y.coordinates; });
    return it != hi && it->coordinates == a.coordinates;
  };

  const size_t np = present.size();
  const int64_t first = level_offsets[present[1]];
  const int64_t total = static_cast<int64_t>(keys.size());
  const int tasks = worker_count(total - first);
  std::atomic<bool> hit{false};
  fan_out(tasks, [&](int t) {
    const int64_t begin = first + (total - first) * t / tasks;
    const int64_t end = first + (total - first) * (t + 1) / tasks;
    std::vector<Key> memo(np);
    std::vector<char> memo_set(np, 0);
    for (int64_t i = begin; i < end && !hit.load(std::memory_order_relaxed); ++i) {
      const Key &k = data[i];
      for (size_t p = 0; p + 1 < np && present[p] < k.level; ++p) {
        const Key a = coarse_ancestor(k, present[p]);
        if (memo_set[p] && memo[p] == a)
          continue;
        memo[p] = a;
        memo_set[p] = 1;
        if (slice_contains(present[p], a)) {
          hit.store(true, std::memory_order_relaxed);
          break;
        }
      }
    }
  });
  return hit.load();
}

bool overlap_exists(const std::vector<Key> &keys, const std::vector<uint32_t> &present,
                    const std::vector<int64_t> &level_offsets) {
  if (present.size() < 2)
    return false;
#ifdef USE_CUDA
  if (static_cast<int64_t>(keys.size()) >= kDeviceThreshold && cuda_topology_available())
    return cuda_has_active_ancestor(keys, level_offsets, level_offsets[present[1]]);
#endif
  return host_overlap_scan(keys, present, level_offsets);
}

} // namespace

void parallel_blocks(int64_t n, const std::function<void(int64_t, int64_t)> &body) {
  blocks(n, body);
}

void sort_keys(std::vector<Key> &keys, std::vector<int64_t> &perm) {
  perm.clear();
  const int64_t n = static_cast<int64_t>(keys.size());

  std::atomic<bool> ordered{true};
  blocks(n, [&](int64_t b, int64_t e) {
    for (int64_t i = std::max<int64_t>(b, 1); i < e; ++i)
      if (key_before(keys[i], keys[i - 1])) {
        ordered.store(false, std::memory_order_relaxed);
        return;
      }
  });
  if (ordered.load())
    return;

#ifdef USE_CUDA
  if (n >= kDeviceThreshold && cuda_topology_available()) {
    cuda_sort_keys(keys, perm);
    return;
  }
#endif

  // Host path: sort a permutation (ties keep input order), then gather keys.
  std::vector<int64_t> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), int64_t(0));
  const Key *data = keys.data();
  const auto before = [data](int64_t a, int64_t b) {
    if (key_before(data[a], data[b]))
      return true;
    if (key_before(data[b], data[a]))
      return false;
    return a < b;
  };

  int pieces = 1;
  const int workers = worker_count(n);
  while (pieces * 2 <= workers)
    pieces *= 2;
  if (pieces < 2) {
    std::sort(idx.begin(), idx.end(), before);
  } else {
    std::vector<int64_t> cut(static_cast<size_t>(pieces) + 1);
    for (int i = 0; i <= pieces; ++i)
      cut[i] = n * i / pieces;
    fan_out(pieces, [&](int i) {
      std::sort(idx.begin() + cut[i], idx.begin() + cut[i + 1], before);
    });
    for (int span = 1; span < pieces; span *= 2) {
      fan_out(pieces / (2 * span), [&](int m) {
        const int i = m * 2 * span;
        std::inplace_merge(idx.begin() + cut[i], idx.begin() + cut[i + span],
                           idx.begin() + cut[i + 2 * span], before);
      });
    }
  }

  std::vector<Key> gathered(static_cast<size_t>(n));
  blocks(n, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i)
      gathered[i] = data[idx[i]];
  });
  keys.swap(gathered);
  perm = std::move(idx);
}

std::shared_ptr<const Topology> Topology::make(const AdaptiveVoxelConfig &cfg,
                                               std::vector<Key> sorted_keys) {
  cfg.validate();
  static std::atomic<uint64_t> next_id{0};

  auto topo = std::make_shared<Topology>();
  topo->id = ++next_id;
  topo->D = static_cast<int>(cfg.dimension());
  topo->max_depth = cfg.max_depth;
  topo->config = cfg;
  if (topo->config.origin.empty())
    topo->config.origin.assign(topo->D, 0.0);
  for (int d = 0; d < topo->D; ++d) {
    topo->origin[d] = topo->config.origin[d];
    topo->base[d] = cfg.base_voxel_size[d];
  }
  topo->keys = std::move(sorted_keys);

  const int64_t N = topo->size();
  const int D = topo->D;
  const uint32_t depth = topo->max_depth;
  const Key *data = topo->keys.data();

  topo->counts.assign(depth + 1, 0);
  {
    std::mutex merge_lock;
    blocks(N, [&](int64_t b, int64_t e) {
      std::vector<int64_t> local(depth + 1, 0);
      for (int64_t i = b; i < e; ++i) {
        const Key &k = data[i];
        if (k.level > depth)
          throw std::invalid_argument("adaptive voxel level exceeds max_depth");
        if (i > 0 && !key_before(data[i - 1], k))
          throw std::invalid_argument("duplicate or unsorted adaptive voxel keys");
        ++local[k.level];
      }
      std::lock_guard<std::mutex> hold(merge_lock);
      for (uint32_t l = 0; l <= depth; ++l)
        topo->counts[l] += local[l];
    });
  }

  topo->level_offsets.assign(depth + 2, 0);
  for (uint32_t l = 0; l <= depth; ++l) {
    topo->level_offsets[l + 1] = topo->level_offsets[l] + topo->counts[l];
    if (topo->counts[l] > 0)
      topo->present_levels.push_back(l);
  }

  if (overlap_exists(topo->keys, topo->present_levels, topo->level_offsets))
    throw std::invalid_argument(
        "overlapping active voxels: a leaf and one of its ancestors are both active");

  topo->coords = torch::empty({N, D}, torch::kInt64);
  topo->levels = torch::empty({N}, torch::kInt32);
  topo->volumes = torch::empty({N}, torch::kFloat64);
  auto *cd = topo->coords.data_ptr<int64_t>();
  auto *ld = topo->levels.data_ptr<int32_t>();
  auto *vd = topo->volumes.data_ptr<double>();

  std::vector<double> volume_of_level(depth + 1);
  for (uint32_t l = 0; l <= depth; ++l) {
    const DVec e = topo->extent_at(l);
    double v = 1.0;
    for (int d = 0; d < D; ++d)
      v *= e[d];
    volume_of_level[l] = v;
  }
  blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const Key &k = data[i];
      for (int d = 0; d < D; ++d)
        cd[i * D + d] = k.coordinates[d];
      ld[i] = static_cast<int32_t>(k.level);
      vd[i] = volume_of_level[k.level];
    }
  });
  return topo;
}

namespace {

// Fills [N, D] doubles with per-leaf geometry derived from lo and extent.
template <class Emit>
at::Tensor leaf_geometry(const Topology &t, Emit &&emit) {
  const int64_t N = t.size();
  const int D = t.D;
  auto out = torch::empty({N, D}, torch::kFloat64);
  auto *od = out.data_ptr<double>();
  std::vector<DVec> extent(t.max_depth + 1);
  for (uint32_t l = 0; l <= t.max_depth; ++l)
    extent[l] = t.extent_at(l);
  const Key *data = t.keys.data();
  blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const Key &k = data[i];
      const DVec &s = extent[k.level];
      for (int d = 0; d < D; ++d) {
        const double lo = t.origin[d] + static_cast<double>(k.coordinates[d]) * s[d];
        od[i * D + d] = emit(lo, s[d]);
      }
    }
  });
  return out;
}

} // namespace

at::Tensor Topology::cell_extents() const {
  return leaf_geometry(*this, [](double, double s) { return s; });
}

at::Tensor Topology::cell_centers() const {
  return leaf_geometry(*this, [](double lo, double s) { return lo + 0.5 * s; });
}

std::pair<at::Tensor, at::Tensor> Topology::cell_boxes() const {
  return {leaf_geometry(*this, [](double lo, double) { return lo; }),
          leaf_geometry(*this, [](double lo, double s) { return lo + s; })};
}

} // namespace detail
} // namespace adaptive
} // namespace riemann
