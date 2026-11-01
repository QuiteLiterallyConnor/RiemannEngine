#include "engine/cuda/runtime.cuh"

#include <cstdio>
#include <iomanip>
#include <iostream>

namespace riemann {

const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char *cusparseGetErrorString(cusparseStatus_t error) {
  switch (error) {
  case CUSPARSE_STATUS_SUCCESS:
    return "The operation completed successfully.";
  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";
  case CUSPARSE_STATUS_ALLOC_FAILED:
    return "CUSPARSE_STATUS_ALLOC_FAILED";
  case CUSPARSE_STATUS_INVALID_VALUE:
    return "CUSPARSE_STATUS_INVALID_VALUE";
  case CUSPARSE_STATUS_ARCH_MISMATCH:
    return "CUSPARSE_STATUS_ARCH_MISMATCH";
  case CUSPARSE_STATUS_MAPPING_ERROR:
    return "CUSPARSE_STATUS_MAPPING_ERROR";
  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";
  case CUSPARSE_STATUS_INTERNAL_ERROR:
    return "CUSPARSE_STATUS_INTERNAL_ERROR";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  default:
    break;
  }
  return "<unknown>";
}

cusparseHandle_t getCurrentCUDASparseHandle() {
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  return handle;
}

std::pair<size_t, size_t> get_memory_info() {
  size_t device_free;
  size_t device_total;
  CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
  return std::make_pair(device_total, device_free);
}

} // namespace riemann
