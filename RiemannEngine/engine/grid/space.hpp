#ifndef RIEMANN_ENGINE_GRID_SPACE_HPP
#define RIEMANN_ENGINE_GRID_SPACE_HPP

#include "engine/foundation/errors.hpp"
#include "engine/foundation/types.hpp"
#include "engine/grid/key.hpp"
#include "engine/grid/lattice_cpu.hpp"
#include "engine/grid/maps.hpp"
#include "engine/grid/stencil.hpp"

#if !defined(CPU_ONLY) && defined(__CUDACC__)
// full GPU map types are only visible to CUDA translation units
#include "engine/grid/lattice_gpu.cuh"
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <robin_hood.h>
#include <torch/torch.h>

#ifndef RIEMANN_NO_PYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace riemann {

namespace detail {

template <template <typename T, template <typename Q> class A>
          class CoordinateMapType>
struct is_cpu_coordinate_map : std::false_type {};

template <> struct is_cpu_coordinate_map<CoordinateMapCPU> : std::true_type {};

template <typename T1, typename T2> void copy_types(const T1 &src, T2 &dst) {
  size_t curr_it = 0;
  for (const auto s : src)
    dst[curr_it++] = s;
}

template <default_types::index_type V>
default_types::stride_type _fill_vec(size_t const len) {
  default_types::stride_type vec(len);
  std::for_each(vec.begin(), vec.end(), [](auto &i) { i = V; });
  return vec;
}

} // namespace detail

// GPU map types forward declarations: host-compiled translation units may
// name the GPU manager aliases without ever instantiating them.
#if !defined(CPU_ONLY) && !defined(__CUDACC__)
namespace detail {
template <class T> struct default_allocator;
template <class T> struct c10_allocator;
} // namespace detail

template <typename coordinate_field_type, typename coordinate_type,
          template <typename C> class TemplatedAllocator>
class CoordinateFieldMapGPU;

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
class CoordinateMapGPU;

template <typename index_type, typename ByteAllocator> class gpu_kernel_map;
#endif

/*
 * Owns every coordinate map, field map, and cached kernel map of one sparse
 * tensor universe. One instantiation per (coordinate scalar, allocator,
 * map backend) combination.
 */
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
class CoordinateMapManager {
public:
  using size_type = default_types::size_type;
  using index_type = default_types::index_type;
  using stride_type = default_types::stride_type;
  using map_type = CoordinateMapType<coordinate_type, TemplatedAllocator>;
#if !defined(CPU_ONLY) && defined(__CUDACC__)
  using field_map_type = typename std::conditional<
      detail::is_cpu_coordinate_map<CoordinateMapType>::value,
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>,
      CoordinateFieldMapGPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>>::type;
#else
  using field_map_type =
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                            TemplatedAllocator>;
#endif
  using self_type = CoordinateMapManager<coordinate_type, coordinate_field_type,
                                         TemplatedAllocator, CoordinateMapType>;
  using map_collection_type = std::map<coordinate_map_key_type, map_type,
                                       coordinate_map_key_comparator>;
  using kernel_map_type =
#if !defined(CPU_ONLY) && defined(__CUDACC__)
      typename std::conditional<
          detail::is_cpu_coordinate_map<CoordinateMapType>::value,
          cpu_kernel_map,
          gpu_kernel_map<index_type, TemplatedAllocator<char>>>::type;
#else
      cpu_kernel_map;
#endif
  using kernel_map_reference_type =
#if !defined(CPU_ONLY) && defined(__CUDACC__)
      typename std::conditional<
          detail::is_cpu_coordinate_map<CoordinateMapType>::value,
          cpu_kernel_map,
          gpu_kernel_map<index_type, TemplatedAllocator<char>>>::type;
#else
      cpu_kernel_map_reference;
#endif

public:
  // allocator backend is ignored when the coordinate map backend is CPU
  CoordinateMapManager(RiemannAlgorithm::Mode algo = RiemannAlgorithm::DEFAULT,
                       size_type num_threads = 0)
      : m_algorithm(algo) {
    // CPU parallelism is handled with std::thread blocks sized from the
    // input; an explicit thread-count request has no effect.
    (void)num_threads;
    switch (m_algorithm) {
    case RiemannAlgorithm::DEFAULT: {
      m_kernel_map_mode = CUDAKernelMapMode::SPEED_OPTIMIZED;
      m_gpu_default_occupancy = 25;
      break;
    }
    case RiemannAlgorithm::MEMORY_EFFICIENT: {
      m_kernel_map_mode = CUDAKernelMapMode::MEMORY_EFFICIENT;
      m_gpu_default_occupancy = 50;
      break;
    }
    case RiemannAlgorithm::SPEED_OPTIMIZED: {
      m_kernel_map_mode = CUDAKernelMapMode::SPEED_OPTIMIZED;
      m_gpu_default_occupancy = 25;
      break;
    }
    }
  }
  ~CoordinateMapManager() {}

  /****************************************************************************
   * Coordinate generation, modification, and initialization entry functions
   ****************************************************************************/

  CoordinateMapKey insert_field_cpp(at::Tensor const &th_coordinate,
                                    stride_type const tensor_stride,
                                    std::string const string_id = "");

#ifndef RIEMANN_NO_PYBIND
  py::object insert_field(at::Tensor const &th_coordinate,
                          stride_type const tensor_stride,
                          std::string const string_id = "");
#endif

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map_cpp(CoordinateMapKey const *p_in_field_map_key,
                                     stride_type const sparse_tensor_stride,
                                     std::string const sparse_string_id = "");

#ifndef RIEMANN_NO_PYBIND
  std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map(CoordinateMapKey const *p_in_field_map_key,
                                 stride_type const sparse_tensor_stride,
                                 std::string const sparse_string_id = "");
#endif

  std::pair<at::Tensor, at::Tensor>
  field_to_sparse_map(CoordinateMapKey const *p_in_field_map_key,
                      CoordinateMapKey const *p_out_sparse_map_key);

  /*
   * Insert coordinates and return (key, (mapping, inverse_mapping)).
   */
  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  insert_and_map_cpp(at::Tensor const &th_coordinate,
                     stride_type const tensor_stride,
                     std::string const string_id = "");

#ifndef RIEMANN_NO_PYBIND
  std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
  insert_and_map(at::Tensor const &th_coordinate,
                 stride_type const tensor_stride,
                 std::string const string_id = "");
#endif

  /*
   * Generate a strided coordinate map if it does not yet exist.
   * Returns (out map key, true iff a new map was created).
   */
  std::pair<coordinate_map_key_type, bool>
  stride(coordinate_map_key_type const &in_map_key,
         stride_type const &kernel_stride, std::string const string_id = "");

#ifndef RIEMANN_NO_PYBIND
  py::object py_stride(CoordinateMapKey const *in_map_key,
                       stride_type const &kernel_stride,
                       std::string const string_id = "") {
    auto key =
        std::get<0>(stride(in_map_key->get_key(), kernel_stride, string_id));
    return py::cast(new CoordinateMapKey(key.first.size() + 1, key));
  }
#endif

  // stride region: new coordinate generation
  std::pair<coordinate_map_key_type, bool>
  stride_region(coordinate_map_key_type const &in_map_key,
                cpu_kernel_region<coordinate_type> &kernel,
                stride_type const &out_tensor_stride,
                bool const expand_coordinates);

  // origin coordinate map creation
  std::pair<coordinate_map_key_type, bool> origin();
  std::pair<coordinate_map_key_type, bool> origin_field();

  // pruning
  coordinate_map_key_type prune(coordinate_map_key_type const &in_key,
                                bool const *keep_begin, bool const *keep_end);

#ifndef RIEMANN_NO_PYBIND
  py::object py_origin() {
    auto map_key_bool = origin();
    LOG_DEBUG("Return origin map key");
    return py::cast(new CoordinateMapKey(map_key_bool.first.first.size() + 1,
                                         map_key_bool.first));
  }

  py::object py_origin_field() {
    auto map_key_bool = origin_field();
    LOG_DEBUG("Return origin map key");
    return py::cast(new CoordinateMapKey(map_key_bool.first.first.size() + 1,
                                         map_key_bool.first));
  }
#endif

  // Merge
  coordinate_map_key_type
  merge(std::vector<coordinate_map_key_type> const &map_keys);
  std::pair<coordinate_map_key_type, std::vector<at::Tensor>>
  union_map(std::vector<coordinate_map_key_type> const &map_keys);
  std::vector<at::Tensor>
  union_map_th(std::vector<CoordinateMapKey *> const &map_keys,
               CoordinateMapKey *p_out_key);

  /****************************************************************************
   * Tensor field related operations
   ****************************************************************************/

#ifndef RIEMANN_NO_PYBIND
  py::object insert(at::Tensor const &coordinates);
#endif

  /****************************************************************************
   * Coordinate management helper functions
   ****************************************************************************/
  bool insert(coordinate_map_key_type map_key, map_type &map) {
    LOG_DEBUG("insert map with tensor_stride", map_key.first);
    auto result = m_coordinate_maps.insert(
        std::make_pair<coordinate_map_key_type, map_type>(std::move(map_key),
                                                          std::move(map)));
    LOG_DEBUG("map insertion", result.second);
    return result.second;
  }

  bool insert_field_map(coordinate_map_key_type map_key, field_map_type &map) {
    LOG_DEBUG("insert map with tensor_stride", map_key.first);
    auto result = m_field_coordinates.insert(
        std::make_pair<coordinate_map_key_type, field_map_type>(
            std::move(map_key), std::move(map)));
    LOG_DEBUG("map insertion", result.second);
    return result.second;
  }

  typename map_collection_type::iterator
  find(coordinate_map_key_type const &map_key) {
    return m_coordinate_maps.find(map_key);
  }

  typename map_collection_type::const_iterator map_end() const {
    return m_coordinate_maps.cend();
  }

  inline bool exists(coordinate_map_key_type const &key) const noexcept {
    return m_coordinate_maps.find(key) != m_coordinate_maps.end();
  }

  inline bool exists_field(coordinate_map_key_type const &key) const noexcept {
    return m_field_coordinates.find(key) != m_field_coordinates.end();
  }

  inline bool exists_field_to_sparse(
      coordinate_map_key_type const &field_key,
      coordinate_map_key_type const &sparse_key) const noexcept {
    auto key = std::pair<coordinate_map_key_type, coordinate_map_key_type>{
        field_key, sparse_key};
    return m_field_to_sparse_maps.find(key) != m_field_to_sparse_maps.end();
  }

  std::vector<CoordinateMapKey>
  field_to_sparse_keys_cpp(coordinate_map_key_type const &field_key) const {
    std::vector<CoordinateMapKey> return_keys;
    for (auto const &elem : m_field_to_sparse_maps) {
      if (elem.first.first == field_key) {
        auto const &tensor_key = elem.first.second;
        return_keys.push_back(
            CoordinateMapKey(tensor_key.first.size() + 1, tensor_key));
      }
    }
    return return_keys;
  }

#ifndef RIEMANN_NO_PYBIND
  std::vector<py::object>
  field_to_sparse_keys(coordinate_map_key_type const &field_key) const {
    std::vector<py::object> return_keys;
    for (auto const &elem : m_field_to_sparse_maps) {
      if (elem.first.first == field_key) {
        auto const &tensor_key = elem.first.second;
        return_keys.push_back(py::cast(
            new CoordinateMapKey(tensor_key.first.size() + 1, tensor_key)));
      }
    }
    return return_keys;
  }
#endif

  inline bool exists(CoordinateMapKey const *p_key) const {
    return exists(p_key->get_key());
  }

  inline bool exists_field(CoordinateMapKey const *p_key) const {
    return exists_field(p_key->get_key());
  }

  inline bool
  exists_field_to_sparse(CoordinateMapKey const *p_field_key,
                         CoordinateMapKey const *p_sparse_key) const {
    return exists_field_to_sparse(p_field_key->get_key(),
                                  p_sparse_key->get_key());
  }

  inline size_type size(coordinate_map_key_type const &key) const {
    auto const it = m_coordinate_maps.find(key);
    auto const field_it = m_field_coordinates.find(key);
    ASSERT(it != m_coordinate_maps.end() ||
               field_it != m_field_coordinates.end(),
           ERROR_MAP_NOT_FOUND);
    if (it != m_coordinate_maps.end())
      return it->second.size();
    else
      return field_it->second.size();
  }

  inline size_type size(CoordinateMapKey const *p_key) const {
    return size(p_key->get_key());
  }

  inline size_type capacity(coordinate_map_key_type const &key) const {
    auto it = m_coordinate_maps.find(key);
    ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
    return it->second.capacity();
  }

  at::Tensor get_coordinates(CoordinateMapKey const *p_key) const;

  at::Tensor get_coordinate_field(CoordinateMapKey const *p_key) const;

  std::pair<at::Tensor, at::Tensor>
  get_field_to_sparse_map(CoordinateMapKey const *p_field_key,
                          CoordinateMapKey const *p_sparse_key) const;

  std::vector<CoordinateMapKey>
  get_coordinate_map_keys_cpp(stride_type const tensor_stride) const {
    std::vector<CoordinateMapKey> keys;
    for (auto it = m_coordinate_maps.begin(); it != m_coordinate_maps.end();
         ++it) {
      coordinate_map_key_type const &key = it->first;
      if (key.first == tensor_stride) {
        keys.push_back(CoordinateMapKey(key.first.size() + 1, key));
      }
    }
    return keys;
  }

#ifndef RIEMANN_NO_PYBIND
  std::vector<py::object>
  get_coordinate_map_keys(stride_type const tensor_stride) const {
    std::vector<py::object> keys;
    for (auto it = m_coordinate_maps.begin(); it != m_coordinate_maps.end();
         ++it) {
      coordinate_map_key_type const &key = it->first;
      if (key.first == tensor_stride) {
        keys.push_back(
            py::cast(new CoordinateMapKey(key.first.size() + 1, key)));
      }
    }
    return keys;
  }
#endif

  std::string print_key(coordinate_map_key_type const &key) const {
    Formatter out;
    out << ArrToString(key.first);
    if (key.second.length() > 0)
      out << "-" << key.second;
    return out.str();
  }

  std::string to_string(CoordinateMapKey const *p_key) const {
    auto it = m_coordinate_maps.find(p_key->get_key());
    ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
    return print_key(it->first) + " : " + it->second.to_string();
  }

  std::string to_string() const {
    Formatter o;
    for (auto const &kv : m_coordinate_maps) {
      o << "\t" << print_key(kv.first) << ":\t" << kv.second.to_string()
        << "\n";
    }

    if (m_field_coordinates.size() > 0) {
      for (auto const &kv : m_field_coordinates) {
        o << "\tTensorField " << print_key(kv.first) << ":\t"
          << kv.second.to_string() << "\n";
      }
    }

    for (auto const &kv : m_kernel_maps) {
      o << "\t" << print_key(std::get<0>(kv.first)) << "->"
        << print_key(std::get<1>(kv.first)) << ":\t" << kv.second << "\n";
    }
    return o.str();
  }

  RiemannAlgorithm::Mode algorithm() const { return m_algorithm; }

  /****************************************************************************
   * Kernel map related functions
   ****************************************************************************/

  // Returns the cached (or newly generated) kernel map. CPU: {in maps, out
  // maps}; GPU: contiguous device arrays decomposed per kernel offset.
  kernel_map_type const &
  kernel_map(CoordinateMapKey const *py_in_coords_key,  //
             CoordinateMapKey const *py_out_coords_key, //
             stride_type const &kernel_size,            //
             stride_type const &kernel_stride,          //
             stride_type const &kernel_dilation,        //
             RegionType::Type const region_type,        //
             at::Tensor const &offsets, bool is_transpose, bool is_pool);

  // identity-stencil shortcut
  kernel_map_type const &kernel_map(CoordinateMapKey const *py_in_coords_key,
                                    CoordinateMapKey const *py_out_coords_key);

  kernel_map_type const &origin_map(CoordinateMapKey const *py_out_coords_key);
  kernel_map_type const &
  origin_field_map(CoordinateMapKey const *py_out_coords_key);

  std::unordered_map<int64_t, at::Tensor>
  kernel_map_th(CoordinateMapKey const *py_in_coords_key,  //
                CoordinateMapKey const *py_out_coords_key, //
                stride_type const &kernel_size,            //
                stride_type const &kernel_stride,          //
                stride_type const &kernel_dilation,        //
                RegionType::Type const region_type,        //
                at::Tensor const &offsets, bool is_transpose, bool is_pool);

  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield,
                           CoordinateMapKey const *py_in_coords_key);

  std::pair<at::Tensor, std::vector<at::Tensor>>
  origin_map_th(CoordinateMapKey const *py_out_coords_key);

  std::pair<at::Tensor, std::vector<at::Tensor>>
  origin_field_map_th(CoordinateMapKey const *py_out_coords_key);

  std::pair<at::Tensor, at::Tensor>
  stride_map_th(CoordinateMapKey const *p_in_map_key,
                CoordinateMapKey const *p_strided_map_key);

  size_t origin_map_size() {
    ASSERT(m_coordinate_maps.size() > 0 || m_field_coordinates.size() > 0,
           "No coordinate map found.");
    if (m_coordinate_maps.size() > 0) {
      auto const key = origin().first;
      return m_coordinate_maps.find(key)->second.size();
    } else {
      auto const key = origin_field().first;
      return m_coordinate_maps.find(key)->second.size();
    }
  }

  coordinate_map_key_type
  get_random_string_id(stride_type const &tensor_stride,
                       std::string string_id) {
    coordinate_map_key_type key = std::make_pair(
        tensor_stride, string_id.size() > 0 ? string_id + '-' + random_string(5)
                                            : random_string(5));
    while (m_coordinate_maps.find(key) != m_coordinate_maps.end()) {
      key =
          std::make_pair(tensor_stride, string_id.size() > 0
                                            ? string_id + '-' + random_string(5)
                                            : random_string(5));
    }
    return key;
  }

private:
  void coordinate_map_key_check(CoordinateMapKey const *p_map_key) const {
    ASSERT(p_map_key != nullptr, "Input coordinate map key not defined.");
    ASSERT(p_map_key->is_key_set(), "Key not defined.");
    ASSERT(exists(p_map_key->get_key()), "Key does not exist.");
  }

  std::string random_string(size_t length) {
    auto randchar = []() -> char {
      const char charset[] = "0123456789"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      return charset[rand() % max_index];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
  }

  kernel_map_key_type
  origin_map_key(coordinate_map_key_type const &in_key) const {
    map_type const &random_map = m_coordinate_maps.begin()->second;
    stride_type zero_vec(random_map.coordinate_size() - 1);
    std::for_each(zero_vec.begin(), zero_vec.end(), [](auto &i) { i = 0; });
    coordinate_map_key_type origin_key = std::make_pair(zero_vec, "");

    return std::make_tuple(in_key, origin_key,           // maps
                           zero_vec, zero_vec, zero_vec, // kernels
                           RegionType::HYPER_CUBE, false, false);
  }

public:
  size_t m_gpu_default_occupancy;
#ifndef CPU_ONLY
  void *allocate(size_type n) { return m_allocator.allocate(n); }

  void deallocate(void *p, size_type n) {
    m_allocator.deallocate((char *)p, n);
  }
#endif
private:
  // the manager owns the coordinate maps
  std::map<coordinate_map_key_type, map_type, coordinate_map_key_comparator>
      m_coordinate_maps;

  std::map<coordinate_map_key_type, field_map_type,
           coordinate_map_key_comparator>
      m_field_coordinates;

  // generated kernel maps, cached by full kernel identity
  std::unordered_map<kernel_map_key_type, kernel_map_type,
                     kernel_map_key_hasher<coordinate_map_key_hasher>>
      m_kernel_maps;

  // Preserve ordered offset values, not tensor addresses, in custom identity.
  std::map<std::pair<kernel_map_key_type, std::vector<coordinate_type>>,
           kernel_map_type>
      m_custom_kernel_maps;

  std::unordered_map<kernel_map_key_type, kernel_map_type,
                     kernel_map_key_hasher<coordinate_map_key_hasher>>
      m_field_kernel_maps;

  std::unordered_map<
      const std::pair<coordinate_map_key_type, coordinate_map_key_type>,
      const std::pair<at::Tensor, at::Tensor>,
      field_to_sparse_map_key_hasher<coordinate_map_key_hasher>>
      m_field_to_sparse_maps;

#ifndef CPU_ONLY
  TemplatedAllocator<char> m_allocator;
#endif
  CUDAKernelMapMode::Mode m_kernel_map_mode;

  RiemannAlgorithm::Mode m_algorithm;

}; // CoordinateMapManager

namespace detail {

/*
 * Backend-dispatch functor templates. Each coordinate-map backend provides
 * partial specializations; the generic templates are never defined.
 */
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename field_map_type>
struct insert_field_functor {
  void operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapType> &manager);
};

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
struct insert_and_map_functor {
  std::pair<at::Tensor, at::Tensor> operator()(
      coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
      CoordinateMapManager<coordinate_type, coordinate_field_type,
                           TemplatedAllocator, CoordinateMapType> &manager);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct kernel_map_functor {
  kernel_map_type operator()(
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &out_map,
      CUDAKernelMapMode::Mode kernel_map_mode,
      cpu_kernel_region<coordinate_type> &kernel);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct stride_map_functor {
  using stride_type = default_types::stride_type;

  kernel_map_type operator()(
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &in_map,
      CoordinateMapType<coordinate_type, TemplatedAllocator> const &out_map,
      stride_type const &kernel);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct empty_map_functor {
  using stride_type = default_types::stride_type;

  kernel_map_type operator()();
};

template <typename kernel_map_type> struct swap_in_out_map_functor {
  kernel_map_type operator()(kernel_map_type const &kernel_map);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct origin_map_functor {
  std::pair<at::Tensor, std::vector<at::Tensor>>
  operator()(CoordinateMapType<coordinate_type, TemplatedAllocator> const
                 &origin_coordinate_map,
             kernel_map_type const &origin_map);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct stride_map2tensor_functor {
  std::pair<at::Tensor, at::Tensor>
  operator()(kernel_map_type const &origin_map);
};

template <typename coordinate_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType,
          typename kernel_map_type>
struct kernel_map_to_tensors {
  using index_type = default_types::index_type;

  std::unordered_map<int64_t, at::Tensor>
  operator()(kernel_map_type const &kernel_map);
};

/*
 * CPU backend specializations (declarations; bodies live in space.cpp).
 */
template <typename coordinate_type, typename coordinate_field_type>
struct insert_and_map_functor<coordinate_type, coordinate_field_type,
                              std::allocator, CoordinateMapCPU> {
  std::pair<at::Tensor, at::Tensor>
  operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
             CoordinateMapManager<coordinate_type, coordinate_field_type,
                                  std::allocator, CoordinateMapCPU> &manager);
};

template <typename coordinate_type, typename coordinate_field_type>
struct insert_field_functor<
    coordinate_type, coordinate_field_type, std::allocator, CoordinateMapCPU,
    CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                          std::allocator>> {
  void
  operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
             CoordinateMapManager<coordinate_type, coordinate_field_type,
                                  std::allocator, CoordinateMapCPU> &manager);
};

template <typename coordinate_type>
struct kernel_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                          cpu_kernel_map> {
  cpu_kernel_map
  operator()(CoordinateMapCPU<coordinate_type, std::allocator> const &in_map,
             CoordinateMapCPU<coordinate_type, std::allocator> const &out_map,
             CUDAKernelMapMode::Mode kernel_map_mode,
             cpu_kernel_region<coordinate_type> &kernel);
};

template <typename coordinate_type>
struct stride_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                          cpu_kernel_map> {
  cpu_kernel_map
  operator()(CoordinateMapCPU<coordinate_type, std::allocator> const &in_map,
             CoordinateMapCPU<coordinate_type, std::allocator> const &out_map,
             default_types::stride_type const &out_tensor_stride);
};

template <> struct swap_in_out_map_functor<cpu_kernel_map> {
  cpu_kernel_map operator()(cpu_kernel_map const &kernel_map);
};

template <typename coordinate_type>
struct empty_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                         cpu_kernel_map> {
  cpu_kernel_map operator()();
};

template <typename coordinate_type>
struct origin_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                          cpu_kernel_map> {
  std::pair<at::Tensor, std::vector<at::Tensor>>
  operator()(CoordinateMapCPU<coordinate_type, std::allocator> const
                 &origin_coordinate_map,
             cpu_kernel_map const &origin_map);
};

template <typename coordinate_type>
struct stride_map2tensor_functor<coordinate_type, std::allocator,
                                 CoordinateMapCPU, cpu_kernel_map> {
  std::pair<at::Tensor, at::Tensor>
  operator()(cpu_kernel_map const &stride_kernel_map);
};

template <typename coordinate_type>
struct kernel_map_to_tensors<coordinate_type, std::allocator, CoordinateMapCPU,
                             cpu_kernel_map> {
  std::unordered_map<int64_t, at::Tensor>
  operator()(cpu_kernel_map const &kernel_map);
};

} // namespace detail

// manager aliases
template <typename coordinate_type>
using cpu_manager_type =
    CoordinateMapManager<coordinate_type, default_types::ccoordinate_type,
                         std::allocator, CoordinateMapCPU>;

#ifndef CPU_ONLY
template <typename coordinate_type,
          template <typename C> class TemplatedAllocator>
using gpu_manager_type =
    CoordinateMapManager<coordinate_type, default_types::ccoordinate_type,
                         TemplatedAllocator, CoordinateMapGPU>;

template <typename coordinate_type>
using gpu_default_manager_type =
    gpu_manager_type<coordinate_type, detail::default_allocator>;

template <typename coordinate_type>
using gpu_c10_manager_type =
    gpu_manager_type<coordinate_type, detail::c10_allocator>;

#endif

/*
 * User-facing coordinate manager. Dispatches to the backend-specific
 * CoordinateMapManager instance stored in a variant.
 */
class CoordinateManager {
public:
  CoordinateManager(
      int D,
      CoordinateMapBackend::Type map_type = CoordinateMapBackend::CUDA,
      GPUMemoryAllocatorBackend::Type alloc_type =
          GPUMemoryAllocatorBackend::PYTORCH,
      RiemannAlgorithm::Mode algo = RiemannAlgorithm::DEFAULT);

  ~CoordinateManager();

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  insert_and_map(const at::Tensor &coordinates, stride_type tensor_stride,
                 std::string string_id = "");

  CoordinateMapKey stride(const CoordinateMapKey &in_key,
                          const stride_type &kernel_stride,
                          std::string string_id = "");

  std::pair<CoordinateMapKey, bool> origin();

  at::Tensor get_coordinates(const CoordinateMapKey &key) const;
  uint32_t size(const CoordinateMapKey &key) const;

  template <typename ManagerType>
  auto const &kernel_map(const CoordinateMapKey &in_key,
                         const CoordinateMapKey &out_key,
                         const stride_type &kernel_size,
                         const stride_type &kernel_stride,
                         const stride_type &kernel_dilation,
                         RegionType::Type region_type,
                         const at::Tensor &offsets, bool is_transpose,
                         bool is_pool) const {
    auto *self = const_cast<CoordinateManager *>(this);
    return self->manager_ptr<ManagerType>()->kernel_map(
        &in_key, &out_key, kernel_size, kernel_stride, kernel_dilation,
        region_type, offsets, is_transpose, is_pool);
  }

  template <typename ManagerType>
  auto const &origin_map(const CoordinateMapKey &key) const {
    auto *self = const_cast<CoordinateManager *>(this);
    return self->manager_ptr<ManagerType>()->origin_map(&key);
  }

  std::vector<at::Tensor> union_map(const std::vector<CoordinateMapKey> &in_keys,
                                    CoordinateMapKey &out_key);

  std::vector<at::Tensor> interpolation_map_weight(const at::Tensor &tfield,
                                                   const CoordinateMapKey &key);

  CoordinateMapKey insert_field(const at::Tensor &coordinates,
                                stride_type tensor_stride,
                                std::string string_id = "");

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map(const CoordinateMapKey &field_key,
                                 stride_type sparse_stride,
                                 std::string string_id = "");

  template <typename ManagerType> ManagerType *manager_ptr() {
    return std::get<std::unique_ptr<ManagerType>>(manager_).get();
  }

#ifdef CPU_ONLY
  auto &get_manager_variant() { return manager_; }
  const auto &get_manager_variant() const { return manager_; }
#else
  auto &get_manager_variant() { return manager_; }
  const auto &get_manager_variant() const { return manager_; }
#endif

  int D() const { return D_; }
  CoordinateMapBackend::Type backend() const { return map_type_; }

private:
  int D_;
  CoordinateMapBackend::Type map_type_;
  GPUMemoryAllocatorBackend::Type alloc_type_;
  RiemannAlgorithm::Mode algo_;

#ifdef CPU_ONLY
  std::variant<std::unique_ptr<cpu_manager_type<int32_t>>> manager_;
#else
  std::variant<std::unique_ptr<cpu_manager_type<int32_t>>,
               std::unique_ptr<gpu_default_manager_type<int32_t>>,
               std::unique_ptr<gpu_c10_manager_type<int32_t>>>
      manager_;
#endif
};

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_SPACE_HPP
