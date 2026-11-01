#ifndef RIEMANN_ENGINE_CUDA_RUNTIME_CUH
#define RIEMANN_ENGINE_CUDA_RUNTIME_CUH

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <driver_types.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>

#include "engine/foundation/errors.hpp"

// atomicAdd(double*) shim for pre-Pascal device code
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static inline __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace riemann {

#define CUDA_CHECK(condition)                                                  \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(riemann::Formatter()                            \
                               << " " << cudaGetErrorString(error) << " at "   \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

#ifdef DEBUG
#define CUDA_CHECK_DEBUG(condition) CUDA_CHECK(condition)
#else
#define CUDA_CHECK_DEBUG(...) (void)0
#endif

#define CUDA_CHECK_ARGS(condition, ...)                                        \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      riemann::Formatter formatter;                                            \
      formatter << " " << cudaGetErrorString(error) << " at ";                 \
      formatter << __FILE__ << ":" << __LINE__;                                \
      formatter.append(__VA_ARGS__);                                           \
      throw std::runtime_error(formatter.str());                               \
    }                                                                          \
  }

#define CUBLAS_CHECK(condition)                                                \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(riemann::Formatter()                            \
                               << riemann::cublasGetErrorString(status)        \
                               << " at " << __FILE__ << ":" << __LINE__);      \
    }                                                                          \
  }

#define CUSPARSE_CHECK(call)                                                   \
  {                                                                            \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                           \
      throw std::runtime_error(riemann::Formatter()                            \
                               << riemann::cusparseGetErrorString(err)         \
                               << " at " << __FILE__ << ":" << __LINE__);      \
    }                                                                          \
  }

#define CURAND_CHECK(condition)                                                \
  {                                                                            \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(riemann::Formatter()                            \
                               << "CURAND error " << status << " at "          \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

// grid-stride loop
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK                                                 \
  {                                                                            \
    cudaError_t status = cudaPeekAtLastError();                                \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(riemann::Formatter()                            \
                               << " " << cudaGetErrorString(status) << " at "  \
                               << __FILE__ << ":" << __LINE__);                \
    }                                                                          \
  }

#ifndef THRUST_CHECK
#define THRUST_CHECK(condition)                                                \
  try {                                                                        \
    condition;                                                                 \
  } catch (thrust::system_error e) {                                           \
    throw std::runtime_error(riemann::Formatter()                              \
                             << "Thrust error: " << e.what() << " at "         \
                             << __FILE__ << ":" << __LINE__);                  \
  }
#endif

#define THRUST_CATCH                                                           \
  catch (thrust::system_error e) {                                             \
    throw std::runtime_error(riemann::Formatter()                              \
                             << "Thrust error: " << e.what() << " at "         \
                             << __FILE__ << ":" << __LINE__);                  \
  }

const char *cublasGetErrorString(cublasStatus_t error);

const char *cusparseGetErrorString(cusparseStatus_t error);

cusparseHandle_t getCurrentCUDASparseHandle();

constexpr uint32_t CUDA_NUM_THREADS = 128;

constexpr uint32_t SHARED_BLOCK_SIZE = 32;

constexpr uint32_t MAX_GRID = 65535;

inline int GET_BLOCKS(const uint32_t N, const uint32_t THREADS) {
  return std::max((N + THREADS - 1) / THREADS, uint32_t(1));
}

// (total, free) device memory in bytes
std::pair<size_t, size_t> get_memory_info();

} // namespace riemann

#endif // RIEMANN_ENGINE_CUDA_RUNTIME_CUH
