// Device implementations of the expensive topology-build steps: canonical key
// sorting and active-ancestor detection. Keys live on the host inside Topology;
// each entry point uploads them once and downloads only the required results.
#include "tree.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <stdexcept>
#include <string>

namespace riemann {
namespace adaptive {
namespace detail {
namespace {

// Device-safe mirror of Key (std::array is unusable in kernels). The explicit
// pad keeps the byte layout identical so raw memcpy transfers are valid.
struct DeviceKey {
  int64_t c[kMaxDim];
  uint32_t level;
  uint32_t pad;
};
static_assert(sizeof(DeviceKey) == sizeof(Key), "DeviceKey must mirror Key's layout");

__host__ __device__ inline bool device_key_before(const DeviceKey &a, const DeviceKey &b) {
  if (a.level != b.level)
    return a.level < b.level;
  for (int d = 0; d < kMaxDim; ++d)
    if (a.c[d] != b.c[d])
      return a.c[d] < b.c[d];
  return false;
}

// Orders a permutation by the keys it points at; index ties keep input order.
struct PermBefore {
  const DeviceKey *keys;
  __host__ __device__ bool operator()(int64_t a, int64_t b) const {
    if (device_key_before(keys[a], keys[b]))
      return true;
    if (device_key_before(keys[b], keys[a]))
      return false;
    return a < b;
  }
};

void require(cudaError_t status, const char *step) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(step) + ": " + cudaGetErrorString(status));
}

// One thread per fine key: bisect every coarser level's slice for an ancestor.
__global__ void ancestor_probe(const DeviceKey *keys, int64_t first, int64_t n,
                               const int64_t *level_offsets, uint32_t top_level, int *hit) {
  const int64_t i = first + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n || atomicCAS(hit, 0, 0) != 0)
    return;
  const DeviceKey k = keys[i];
  for (uint32_t l = 0; l < k.level && l <= top_level; ++l) {
    const int64_t begin = level_offsets[l], end = level_offsets[l + 1];
    if (begin == end)
      continue;
    DeviceKey a;
    a.level = l;
    a.pad = 0;
    const uint32_t gap = k.level - l;
    for (int d = 0; d < kMaxDim; ++d)
      a.c[d] = gap >= 63 ? (k.c[d] < 0 ? -1 : 0) : (k.c[d] >> gap);
    int64_t lo = begin, hi = end;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (device_key_before(keys[mid], a))
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo < end && !device_key_before(a, keys[lo])) {
      atomicCAS(hit, 0, 1);
      return;
    }
  }
}

} // namespace

bool cuda_topology_available() {
  static const bool ready = [] {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
  }();
  return ready;
}

void cuda_sort_keys(std::vector<Key> &keys, std::vector<int64_t> &perm) {
  perm.clear();
  const int64_t n = static_cast<int64_t>(keys.size());
  if (n < 2)
    return;

  thrust::device_vector<DeviceKey> d_keys(static_cast<size_t>(n));
  require(cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys.data(),
                     static_cast<size_t>(n) * sizeof(DeviceKey), cudaMemcpyHostToDevice),
          "octree key upload");

  thrust::device_vector<int64_t> d_perm(static_cast<size_t>(n));
  thrust::sequence(thrust::device, d_perm.begin(), d_perm.end());
  thrust::sort(thrust::device, d_perm.begin(), d_perm.end(),
               PermBefore{thrust::raw_pointer_cast(d_keys.data())});

  thrust::device_vector<DeviceKey> d_sorted(static_cast<size_t>(n));
  thrust::gather(thrust::device, d_perm.begin(), d_perm.end(), d_keys.begin(), d_sorted.begin());

  perm.resize(static_cast<size_t>(n));
  require(cudaMemcpy(keys.data(), thrust::raw_pointer_cast(d_sorted.data()),
                     static_cast<size_t>(n) * sizeof(DeviceKey), cudaMemcpyDeviceToHost),
          "octree key download");
  require(cudaMemcpy(perm.data(), thrust::raw_pointer_cast(d_perm.data()),
                     static_cast<size_t>(n) * sizeof(int64_t), cudaMemcpyDeviceToHost),
          "octree permutation download");
}

bool cuda_has_active_ancestor(const std::vector<Key> &keys,
                              const std::vector<int64_t> &level_offsets, int64_t first) {
  const int64_t n = static_cast<int64_t>(keys.size());
  if (n - first <= 0)
    return false;

  thrust::device_vector<DeviceKey> d_keys(static_cast<size_t>(n));
  require(cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys.data(),
                     static_cast<size_t>(n) * sizeof(DeviceKey), cudaMemcpyHostToDevice),
          "octree key upload");
  thrust::device_vector<int64_t> d_offsets(level_offsets.begin(), level_offsets.end());
  thrust::device_vector<int> d_hit(1, 0);

  constexpr int kThreads = 256;
  const int64_t span = n - first;
  const int grid = static_cast<int>((span + kThreads - 1) / kThreads);
  ancestor_probe<<<grid, kThreads>>>(thrust::raw_pointer_cast(d_keys.data()), first, n,
                                     thrust::raw_pointer_cast(d_offsets.data()),
                                     static_cast<uint32_t>(level_offsets.size() - 2),
                                     thrust::raw_pointer_cast(d_hit.data()));
  require(cudaGetLastError(), "octree ancestor probe launch");
  require(cudaDeviceSynchronize(), "octree ancestor probe");
  return d_hit[0] != 0;
}

} // namespace detail
} // namespace adaptive
} // namespace riemann

#endif // USE_CUDA
