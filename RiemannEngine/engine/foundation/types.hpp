#ifndef RIEMANN_ENGINE_FOUNDATION_TYPES_HPP
#define RIEMANN_ENGINE_FOUNDATION_TYPES_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <robin_hood.h>

namespace riemann {

// Bundles the integral/floating scalar choices used across the engine so a
// single template argument swap retargets every dependent alias.
template <typename uint_type, typename int_type, typename float_type>
struct type_wrapper {
  using tensor_order_type        = uint_type;
  using index_type               = uint_type;
  using stride_type              = std::vector<uint_type>;
  using size_type                = uint_type;
  using dcoordinate_type         = int_type;
  using ccoordinate_type         = float_type;
  using feature_type             = float_type;
  using coordinate_map_hash_type = uint64_t;
  using index_vector_type        = std::vector<index_type>;
};

using default_types = type_wrapper<uint32_t, int32_t, float>;

using size_type = default_types::size_type;
using index_type = default_types::index_type;
using stride_type = default_types::stride_type;

template <default_types::tensor_order_type D,
          typename int_type = default_types::index_type>
using dim_array = std::array<int_type, D>;

template <typename data_type, typename size_type = default_types::size_type>
struct ptr_vector {
  ptr_vector(data_type *ptr, size_type size) : ptr_(ptr), size_(size) {}
  size_type size() const { return size_; };
  data_type *data() { return ptr_; };
  const data_type *data() const { return ptr_; };

  data_type *ptr_;
  default_types::size_type size_;
};

// A coordinate map is identified by its tensor stride plus a free-form label.
using coordinate_map_key_type =
    std::pair<default_types::stride_type, std::string>;

struct coordinate_map_key_hasher {
  using result_type = size_t;

  result_type operator()(coordinate_map_key_type const &key) const {
    auto hash = robin_hood::hash_bytes(
        key.first.data(), sizeof(default_types::size_type) * key.first.size());
    hash ^= std::hash<std::string>{}(key.second);
    return hash;
  }
};

struct coordinate_map_key_comparator {
  bool operator()(coordinate_map_key_type const &lhs,
                  coordinate_map_key_type const &rhs) const {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    return std::lexicographical_compare(lhs.second.begin(), lhs.second.end(),
                                        rhs.second.begin(), rhs.second.end());
  }
};

template <typename vector_type>
std::vector<vector_type>
initialize_maps(default_types::size_type number_of_vectors,
                default_types::size_type vector_size) {
  std::vector<vector_type> vectors;
  vectors.reserve(number_of_vectors);
  for (default_types::size_type i = 0; i < number_of_vectors; ++i)
    vectors.emplace_back(vector_size);
  return vectors;
}

namespace SparseTensorQuantizationMode {
enum Type {
  RANDOM_SUBSAMPLE = 0,
  UNWEIGHTED_AVERAGE = 1,
  UNWEIGHTED_SUM = 2,
  MAX_POOL = 3,
  NO_QUANTIZATION = 4
};
}

namespace GPUMemoryAllocatorBackend {
enum Type { PYTORCH = 0, CUDA = 1 };
}

namespace MapManagerType {
enum Type { CPU = 0, CUDA = 1, C10 = 2 };
}

namespace CUDAKernelMapMode {
enum Mode { MEMORY_EFFICIENT = 0, SPEED_OPTIMIZED = 1 };
}

namespace RiemannAlgorithm {
enum Mode { DEFAULT = 0, MEMORY_EFFICIENT = 1, SPEED_OPTIMIZED = 2 };
}

namespace CoordinateMapBackend {
enum Type { CPU = 0, CUDA = 1 };
}

namespace RegionType {
enum Type { HYPER_CUBE, HYPER_CROSS, CUSTOM };
}

namespace PoolingMode {
enum Type {
  LOCAL_SUM_POOLING,
  LOCAL_AVG_POOLING,
  LOCAL_MAX_POOLING,
  GLOBAL_SUM_POOLING_DEFAULT,
  GLOBAL_AVG_POOLING_DEFAULT,
  GLOBAL_MAX_POOLING_DEFAULT,
  GLOBAL_SUM_POOLING_KERNEL,
  GLOBAL_AVG_POOLING_KERNEL,
  GLOBAL_MAX_POOLING_KERNEL,
  GLOBAL_SUM_POOLING_PYTORCH_INDEX,
  GLOBAL_AVG_POOLING_PYTORCH_INDEX,
  GLOBAL_MAX_POOLING_PYTORCH_INDEX
};
}

namespace BroadcastMode {
enum Type {
  ELEMENTWISE_ADDITON,
  ELEMENTWISE_MULTIPLICATION,
};
}

namespace ConvolutionMode {
enum Type {
  DEFAULT,
  DIRECT_GEMM,
  COPY_GEMM,
};
}

// Cache identity of one generated kernel map:
// (in key, out key, kernel size, kernel stride, dilation, region type,
//  transpose flag, pooling flag).
using kernel_map_key_type =
    std::tuple<coordinate_map_key_type,    // in
               coordinate_map_key_type,    // out
               default_types::stride_type, // kernel size
               default_types::stride_type, // kernel stride
               default_types::stride_type, // kernel dilation
               RegionType::Type,           // kernel region type
               bool,                       // is transpose
               bool                        // is pool
               >;

// FNV64-1a over the elements of any iterable integer sequence.
template <typename T> uint64_t hash_vec(T p) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (auto x : p) {
    hash ^= x;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

template <typename hasher = coordinate_map_key_hasher>
struct kernel_map_key_hasher {
  using stride_type = default_types::stride_type;
  using result_type = size_t;

  result_type hash_stride(stride_type const &stride) const {
    return robin_hood::hash_bytes(
        stride.data(), sizeof(default_types::size_type) * stride.size());
  }

  result_type operator()(kernel_map_key_type const &key) const {
    result_type hash = hasher{}(std::get<0>(key));
    hash ^= hasher{}(std::get<1>(key));
    hash ^= hash_stride(std::get<2>(key));
    hash ^= hash_stride(std::get<3>(key));
    hash ^= hash_stride(std::get<4>(key));
    hash ^= (result_type)std::get<5>(key);
    hash ^= (result_type)std::get<6>(key);
    hash ^= (result_type)std::get<7>(key);
    return hash;
  }
};

template <typename hasher = coordinate_map_key_hasher>
struct field_to_sparse_map_key_hasher {
  using result_type = size_t;

  result_type operator()(std::pair<coordinate_map_key_type,
                                   coordinate_map_key_type> const &key) const {
    result_type hash = hasher{}(key.first);
    hash ^= hasher{}(key.second);
    return hash;
  }
};

} // namespace riemann

#endif // RIEMANN_ENGINE_FOUNDATION_TYPES_HPP
