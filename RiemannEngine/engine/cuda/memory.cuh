#ifndef RIEMANN_ENGINE_CUDA_MEMORY_CUH
#define RIEMANN_ENGINE_CUDA_MEMORY_CUH

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "engine/cuda/runtime.cuh"
#include "engine/foundation/types.hpp"

namespace riemann {

namespace detail {

// Raw cudaMalloc/cudaFree allocator. On out-of-memory it empties the Torch
// caching allocator once and retries.
template <class T> struct default_allocator {
  typedef T value_type;

  default_allocator() = default;

  template <class U>
  constexpr default_allocator(const default_allocator<U> &) noexcept {}

  T *allocate(std::size_t n, cudaStream_t stream = 0) const {
    T *d_tmp;
    cudaError_t error = cudaMalloc((void **)&d_tmp, n * sizeof(T));
    if (error != cudaSuccess) {
      cudaGetLastError(); // clear the sticky error before retry
      c10::cuda::CUDACachingAllocator::emptyCache();
      LOG_DEBUG("Automatically called empty cache");
      CUDA_CHECK(cudaMalloc((void **)&d_tmp, n * sizeof(T)));
    }
    return d_tmp;
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const {
    cudaFree(p);
  }
};

// Allocator backed by the PyTorch CUDA caching allocator.
template <class T> struct c10_allocator {
  typedef T value_type;

  c10_allocator() = default;

  template <class U>
  constexpr c10_allocator(const c10_allocator<U> &) noexcept {}

  T *allocate(std::size_t n, cudaStream_t stream = 0) const {
    return reinterpret_cast<T *>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n * sizeof(T)));
  }

  std::shared_ptr<T[]> shared_allocate(std::size_t n,
                                       cudaStream_t stream = 0) const {
    T *d_ptr = reinterpret_cast<T *>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n * sizeof(T)));

    auto deleter = [](T *p) {
      c10::cuda::CUDACachingAllocator::raw_delete((void *)p);
    };

    return std::shared_ptr<T[]>{d_ptr,
                                std::bind(deleter, std::placeholders::_1)};
  }

  void deallocate(T *p, std::size_t n, cudaStream_t stream = 0) const {
    c10::cuda::CUDACachingAllocator::raw_delete((void *)p);
  }
};

// Size-keyed reuse pool over raw cudaMalloc blocks.
template <typename T = char> class cached_allocator {
public:
  using value_type = T;
  using free_blocks_type = std::multimap<std::ptrdiff_t, T *>;
  using allocated_blocks_type = std::map<T *, std::ptrdiff_t>;
  using iterator = typename free_blocks_type::iterator;

public:
  cached_allocator() {}
  ~cached_allocator() {
#ifndef __CUDACC__
    free_all();
#endif
  }

  T *allocate(std::ptrdiff_t num_values, cudaStream_t stream = 0) {
    T *result = 0;

    auto free_block = free_blocks.find(num_values * sizeof(value_type));

    if (free_block != free_blocks.end()) {
      LOG_DEBUG("using preallocated", num_values, "of", sizeof(value_type));
      result = free_block->second;
      free_blocks.erase(free_block);
    } else {
      LOG_DEBUG("allocating", num_values, "of", sizeof(value_type));
      CUDA_CHECK(
          cudaMalloc((void **)&result, num_values * sizeof(value_type)));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    allocated_blocks.insert(std::make_pair(result, num_values));

    return result;
  }

  void deallocate(T *ptr, size_t n, cudaStream_t stream = 0) {
    auto iter = allocated_blocks.find(ptr);
    std::ptrdiff_t num_values = iter->second;
    allocated_blocks.erase(iter);

    free_blocks.insert(std::make_pair(num_values, reinterpret_cast<T *>(ptr)));
  }

private:
  free_blocks_type free_blocks;
  allocated_blocks_type allocated_blocks;

  void free_all() {
    for (auto i = free_blocks.begin(); i != free_blocks.end(); i++) {
      cudaFree(i->second);
    }

    for (auto i = allocated_blocks.begin(); i != allocated_blocks.end(); i++) {
      cudaFree(i->first);
    }
  }
};

// Shared-ownership wrapper so a cached_allocator can be copied into functors
// and kernel launch parameter packs without duplicating the pool.
template <typename T = char> class shared_allocator {
public:
  using self_type = shared_allocator<T>;

public:
  __host__ __device__ shared_allocator() {
    m_p_alloc = std::make_shared<cached_allocator<T>>(cached_allocator<T>());
  }
  __host__ __device__ shared_allocator(self_type const &other) {
    m_p_alloc = other.m_p_alloc;
  }
  __host__ __device__ ~shared_allocator() {}

  __host__ T *allocate(std::ptrdiff_t num_values,
                       cudaStream_t stream = 0) const {
    return m_p_alloc->allocate(num_values, stream);
  }

  __host__ void deallocate(T *ptr, size_t n, cudaStream_t stream = 0) const {
    return m_p_alloc->deallocate(ptr, n, stream);
  }

private:
  std::shared_ptr<cached_allocator<T>> m_p_alloc;
};

} // namespace detail

} // namespace riemann

#endif // RIEMANN_ENGINE_CUDA_MEMORY_CUH
