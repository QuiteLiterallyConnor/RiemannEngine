#ifndef RIEMANN_ENGINE_CUDA_SCRATCH_CUH
#define RIEMANN_ENGINE_CUDA_SCRATCH_CUH

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "engine/cuda/runtime.cuh"
#include "engine/foundation/errors.hpp"

// Typed access to dynamically sized shared memory. The extern array must be
// declared with a distinct name per element type to avoid duplicate symbol
// definitions, hence one specialization per supported type.
template <typename T> struct SharedMemory {
  __device__ T *getPointer() {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

template <> struct SharedMemory<int> {
  __device__ int *getPointer() {
    extern __shared__ int s_int[];
    return s_int;
  }
};

template <> struct SharedMemory<unsigned int> {
  __device__ unsigned int *getPointer() {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

template <> struct SharedMemory<char> {
  __device__ char *getPointer() {
    extern __shared__ char s_char[];
    return s_char;
  }
};

template <> struct SharedMemory<unsigned char> {
  __device__ unsigned char *getPointer() {
    extern __shared__ unsigned char s_uchar[];
    return s_uchar;
  }
};

template <> struct SharedMemory<short> {
  __device__ short *getPointer() {
    extern __shared__ short s_short[];
    return s_short;
  }
};

template <> struct SharedMemory<unsigned short> {
  __device__ unsigned short *getPointer() {
    extern __shared__ unsigned short s_ushort[];
    return s_ushort;
  }
};

template <> struct SharedMemory<long> {
  __device__ long *getPointer() {
    extern __shared__ long s_long[];
    return s_long;
  }
};

template <> struct SharedMemory<unsigned long> {
  __device__ unsigned long *getPointer() {
    extern __shared__ unsigned long s_ulong[];
    return s_ulong;
  }
};

template <> struct SharedMemory<bool> {
  __device__ bool *getPointer() {
    extern __shared__ bool s_bool[];
    return s_bool;
  }
};

template <> struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <> struct SharedMemory<double> {
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

namespace riemann {

// Owning device buffer parameterized by a byte allocator. Serves as the
// temporary storage used by coordinate-map generation kernels.
template <typename Dtype, typename ByteAllocator> class gpu_storage {
public:
  using data_type = Dtype;
  using byte_allocator_type = ByteAllocator;
  using self_type = gpu_storage<data_type, byte_allocator_type>;

  gpu_storage() : m_data(nullptr), m_num_elements(0) {}
  gpu_storage(uint64_t const num_elements) { allocate(num_elements); }
  gpu_storage(self_type const &other_storage) {
    LOG_DEBUG("copy storage constructor");
    if (other_storage.size() == 0)
      return;

    allocate(other_storage.size());
    CUDA_CHECK(cudaMemcpy(m_data, other_storage.cdata(),
                          other_storage.size() * sizeof(data_type),
                          cudaMemcpyDeviceToDevice));
  }
  gpu_storage(self_type &&other_storage) {
    LOG_DEBUG("move storage constructor from", other_storage.m_data,
              "with size", other_storage.m_num_elements);
    if (other_storage.size() == 0)
      return;
    m_num_elements = other_storage.size();
    m_data = other_storage.data();

    other_storage.m_data = nullptr;
    other_storage.m_num_elements = 0;
  }
  gpu_storage(std::vector<Dtype> const &vec) {
    LOG_DEBUG("vector storage constructor");
    from_vector(vec);
  }

  ~gpu_storage() { deallocate(); }

  data_type *allocate(uint64_t const num_elements) {
    if (num_elements == 0)
      return nullptr;
    m_num_elements = num_elements;
    m_data =
        (data_type *)m_allocator.allocate(m_num_elements * sizeof(data_type));
    LOG_DEBUG("Allocating", num_elements, "gpu storage at", m_data);
    return m_data;
  }

  void deallocate() {
    LOG_DEBUG("Deallocating", m_num_elements, "gpu storage at", m_data);
    if (m_num_elements > 0) {
      m_allocator.deallocate((char *)m_data,
                             m_num_elements * sizeof(data_type));
    }
  }

  void from_vector(std::vector<Dtype> const &vec) {
    resize(vec.size());
    if (m_num_elements > 0) {
      CUDA_CHECK(cudaMemcpy(m_data, vec.data(),
                            m_num_elements * sizeof(data_type),
                            cudaMemcpyHostToDevice));
    }
  }

  data_type *data() {
    check_pointer("data");
    return m_data;
  }
  data_type const *cdata() const {
    check_pointer("cdata");
    return m_data;
  }
  data_type *begin() {
    check_pointer("begin");
    return m_data;
  }
  data_type *end() {
    check_pointer("end");
    return m_data + m_num_elements;
  }
  data_type const *cbegin() const {
    check_pointer("cbegin");
    return m_data;
  }
  data_type const *cend() const {
    check_pointer("cend");
    return m_data + m_num_elements;
  }

  std::vector<data_type> to_vector() { return to_vector(size()); }
  std::vector<data_type> to_vector(uint64_t const num_elements) {
    std::vector<data_type> cpu_storage(num_elements);
    if (num_elements > 0)
      CUDA_CHECK(cudaMemcpy(cpu_storage.data(), m_data,
                            num_elements * sizeof(data_type),
                            cudaMemcpyDeviceToHost));
    return cpu_storage;
  }

  uint64_t size() const { return m_num_elements; }

  void resize(uint64_t const new_num_elements) {
    LOG_DEBUG("resizing from", m_num_elements, "to", new_num_elements);
    if (new_num_elements == m_num_elements)
      return;

    data_type *new_data = (data_type *)m_allocator.allocate(
        new_num_elements * sizeof(data_type));

    if (m_num_elements > 0) {
      CUDA_CHECK(cudaMemcpy(new_data, m_data,
                            new_num_elements * sizeof(data_type),
                            cudaMemcpyDeviceToDevice));
      m_allocator.deallocate((char *)m_data,
                             m_num_elements * sizeof(data_type));
    }
    m_data = new_data;
    m_num_elements = new_num_elements;
  }

  void print_by_vector(uint64_t const num_vec, uint64_t const vec_size) {
    auto const print_n = std::min(num_vec, size() / vec_size);
    auto const cpu_storage = to_vector(vec_size * print_n);
    for (uint64_t i = 0; i < print_n; ++i) {
      std::cout << PtrToString(&cpu_storage[i * vec_size], vec_size) << "\n";
    }
  }

private:
  void check_pointer(std::string const &fn) const {
#ifdef DEBUG
    if (m_data == nullptr) {
      throw std::runtime_error("scratch.cuh: m_data == nullptr on " + fn);
    } else if (m_num_elements == 0) {
      throw std::runtime_error("scratch.cuh: m_num_elements == 0 on " + fn);
    }
#endif
  }
  byte_allocator_type m_allocator;
  data_type *m_data = nullptr;
  uint64_t m_num_elements = 0;
};

template <typename Dtype, typename ByteAllocator>
void print(const gpu_storage<Dtype, ByteAllocator> &v) {
  auto cpu_storage = const_cast<gpu_storage<Dtype, ByteAllocator> &>(v).to_vector();
  for (size_t i = 0; i < cpu_storage.size(); i++)
    std::cout << " " << std::fixed << std::setprecision(3) << cpu_storage[i];
  std::cout << "\n";
}

} // namespace riemann

#endif // RIEMANN_ENGINE_CUDA_SCRATCH_CUH
