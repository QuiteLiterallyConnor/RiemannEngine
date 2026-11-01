#ifndef RIEMANN_ENGINE_GRID_SPACE_IMPL_HPP
#define RIEMANN_ENGINE_GRID_SPACE_IMPL_HPP

// Out-of-line definitions for the CoordinateMapManager template. Included
// only by the translation units that explicitly instantiate a backend
// (space.cpp for the CPU manager, space.cu for the GPU managers).

#include "engine/grid/space.hpp"

#include <limits>
#include <string>
#include <unordered_map>

#ifndef CPU_ONLY
#include <ATen/cuda/CUDAContext.h>
#endif

namespace riemann {

namespace detail {

inline default_types::stride_type zeros(size_t const len) {
  return _fill_vec<0>(len);
}

inline default_types::stride_type ones(size_t const len) {
  return _fill_vec<1>(len);
}

} // namespace detail

/*******************************
 * Field map insertion
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
CoordinateMapKey CoordinateMapManager<coordinate_type, coordinate_field_type,
                                      TemplatedAllocator, CoordinateMapType>::
    insert_field_cpp(at::Tensor const &coordinates,
                     default_types::stride_type const tensor_stride,
                     std::string const string_id) {
  torch::TensorArg arg_coordinate(coordinates, "coordinates", 0);
  torch::CheckedFrom c = "initialize";
  torch::checkContiguous(c, arg_coordinate);
  torch::checkScalarType(c, arg_coordinate, torch::kFloat);
  torch::checkBackend(c, arg_coordinate.tensor,
                      detail::is_cpu_coordinate_map<CoordinateMapType>::value
                          ? torch::Backend::CPU
                          : torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinate, 2);

  auto const coordinate_size = (index_type)coordinates.size(1);

  ASSERT(coordinate_size - 1 == tensor_stride.size(),
         "The coordinate dimension (coordinate_size - 1):", coordinate_size - 1,
         " must match the size of tensor stride:", ArrToString(tensor_stride));

  coordinate_map_key_type map_key = std::make_pair(tensor_stride, string_id);
  if (m_field_coordinates.find(map_key) != m_field_coordinates.end()) {
    LOG_DEBUG("CoordinateMapKey collision detected:", map_key,
              "generating new string id.");
    map_key = get_random_string_id(tensor_stride, string_id);
  }

  LOG_DEBUG("initializing a map with tensor stride:", map_key.first,
            "string id:", map_key.second);
  detail::insert_field_functor<coordinate_type, coordinate_field_type,
                               TemplatedAllocator, CoordinateMapType,
                               field_map_type>()(map_key, coordinates, *this);

  return CoordinateMapKey(coordinate_size, map_key);
}

#ifndef RIEMANN_NO_PYBIND
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
py::object CoordinateMapManager<coordinate_type, coordinate_field_type,
                                TemplatedAllocator, CoordinateMapType>::
    insert_field(at::Tensor const &coordinates,
                 default_types::stride_type const tensor_stride,
                 std::string const string_id) {
  auto key = insert_field_cpp(coordinates, tensor_stride, string_id);
  return py::cast(new CoordinateMapKey(key));
}
#endif

/*******************************
 * Field quantization
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    field_to_sparse_insert_and_map_cpp(
        CoordinateMapKey const *p_in_field_map_key,
        default_types::stride_type const sparse_tensor_stride,
        std::string const sparse_tensor_string_id) {
  auto const coordinate_size = p_in_field_map_key->get_coordinate_size();
  ASSERT(coordinate_size - 1 == sparse_tensor_stride.size(),
         "The coordinate dimension (coordinate_size - 1):", coordinate_size - 1,
         " must match the size of tensor stride:",
         ArrToString(sparse_tensor_stride));

  auto const it = m_field_coordinates.find(p_in_field_map_key->get_key());
  ASSERT(it != m_field_coordinates.end(), ERROR_MAP_NOT_FOUND);
  auto const &field_map = it->second;

  auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);

  if (!detail::is_cpu_coordinate_map<CoordinateMapType>::value) {
#ifndef CPU_ONLY
    auto device_id = at::cuda::current_device();
    options = options.device(torch::kCUDA, device_id);
#else
    ASSERT(false, ERROR_CPU_ONLY);
#endif
  }

  coordinate_map_key_type map_key =
      std::make_pair(sparse_tensor_stride, sparse_tensor_string_id);
  if (m_coordinate_maps.find(map_key) != m_coordinate_maps.end()) {
    LOG_DEBUG("CoordinateMapKey collision detected:", map_key,
              "generating new string id.");
    map_key =
        get_random_string_id(sparse_tensor_stride, sparse_tensor_string_id);
  }

  LOG_DEBUG("initializing a field map with tensor stride:", map_key.first,
            "string id:", map_key.second);

  // Quantize the field with the requested stride; the integer buffer is
  // wrapped in a tensor so it can flow through the standard insertion path.
  at::Tensor int_coordinates =
      at::empty({field_map.size(), coordinate_size}, options);
  field_map.quantize_coordinates(int_coordinates.data_ptr<coordinate_type>(),
                                 sparse_tensor_stride);

  auto const map_inverse_map =
      detail::insert_and_map_functor<coordinate_type, coordinate_field_type,
                                     TemplatedAllocator, CoordinateMapType>()(
          map_key, int_coordinates, *this);

  auto const field_to_sparse_map_key =
      std::pair<coordinate_map_key_type, coordinate_map_key_type>{
          p_in_field_map_key->get_key(), map_key};

  auto result = m_field_to_sparse_maps.insert(
      std::pair<
          const std::pair<coordinate_map_key_type, coordinate_map_key_type>,
          const std::pair<at::Tensor, at::Tensor>>{field_to_sparse_map_key,
                                                   map_inverse_map});
  LOG_DEBUG("field to sparse tensor map insertion", result.second);

  return std::make_pair(CoordinateMapKey(coordinate_size, map_key),
                        map_inverse_map);
}

#ifndef RIEMANN_NO_PYBIND
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    field_to_sparse_insert_and_map(
        CoordinateMapKey const *p_in_field_map_key,
        default_types::stride_type const sparse_tensor_stride,
        std::string const sparse_tensor_string_id) {
  auto result = field_to_sparse_insert_and_map_cpp(
      p_in_field_map_key, sparse_tensor_stride, sparse_tensor_string_id);

  return std::make_pair(py::cast(new CoordinateMapKey(result.first)),
                        result.second);
}
#endif

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<at::Tensor, at::Tensor>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    get_field_to_sparse_map(CoordinateMapKey const *p_field_key,
                            CoordinateMapKey const *p_sparse_key) const {
  auto key = std::pair<coordinate_map_key_type, coordinate_map_key_type>{
      p_field_key->get_key(), p_sparse_key->get_key()};
  auto it = m_field_to_sparse_maps.find(key);
  ASSERT(it != m_field_to_sparse_maps.end(),
         "Field To Sparse Map doesn't exist");
  return it->second;
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<at::Tensor, at::Tensor>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    field_to_sparse_map(CoordinateMapKey const *p_in_field_map_key,
                        CoordinateMapKey const *p_out_sparse_map_key) {
  auto const coordinate_size = p_in_field_map_key->get_coordinate_size();
  ASSERT(coordinate_size == p_out_sparse_map_key->get_coordinate_size(),
         "The coordinate dimension mismatch.", coordinate_size,
         "!=", p_out_sparse_map_key->get_coordinate_size());

  auto const it_field = m_field_coordinates.find(p_in_field_map_key->get_key());
  ASSERT(it_field != m_field_coordinates.end(), ERROR_MAP_NOT_FOUND);
  auto const &field_map = it_field->second;
  auto const it_sparse =
      m_coordinate_maps.find(p_out_sparse_map_key->get_key());
  ASSERT(it_sparse != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
  auto const &sparse_map = it_sparse->second;

  auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);

  if (!detail::is_cpu_coordinate_map<CoordinateMapType>::value) {
#ifndef CPU_ONLY
    auto device_id = at::cuda::current_device();
    options = options.device(torch::kCUDA, device_id);
#else
    ASSERT(false, ERROR_CPU_ONLY);
#endif
  }

  auto const map_inverse_map =
      sparse_map.field_map(field_map.const_coordinate_data(), field_map.size());

  auto const field_to_sparse_map_key =
      std::pair<coordinate_map_key_type, coordinate_map_key_type>{
          p_in_field_map_key->get_key(), p_out_sparse_map_key->get_key()};

  auto result = m_field_to_sparse_maps.insert(
      std::pair<
          const std::pair<coordinate_map_key_type, coordinate_map_key_type>,
          const std::pair<at::Tensor, at::Tensor>>{field_to_sparse_map_key,
                                                   map_inverse_map});
  LOG_DEBUG("field to sparse tensor map insertion", result.second);

  return map_inverse_map;
}

/*******************************
 * Sparse coordinate insertion
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    insert_and_map_cpp(at::Tensor const &coordinate,
                       default_types::stride_type const tensor_stride,
                       std::string const string_id) {
  torch::TensorArg arg_coordinate(coordinate, "coordinates", 0);
  torch::CheckedFrom c = "initialize";
  torch::checkContiguous(c, arg_coordinate);
  torch::checkScalarType(c, arg_coordinate, torch::kInt);
  torch::checkBackend(c, arg_coordinate.tensor,
                      detail::is_cpu_coordinate_map<CoordinateMapType>::value
                          ? torch::Backend::CPU
                          : torch::Backend::CUDA);
  torch::checkDim(c, arg_coordinate, 2);

  auto const coordinate_size = (index_type)coordinate.size(1);

  ASSERT(coordinate_size - 1 == tensor_stride.size(),
         "The coordinate dimension (coordinate_size - 1):", coordinate_size - 1,
         " must match the size of tensor stride:", ArrToString(tensor_stride));

  coordinate_map_key_type map_key = std::make_pair(tensor_stride, string_id);
  if (m_coordinate_maps.find(map_key) != m_coordinate_maps.end()) {
    LOG_DEBUG("CoordinateMapKey collision detected:", map_key,
              "generating new string id.");
    map_key = get_random_string_id(tensor_stride, string_id);
  }

  LOG_DEBUG("initializing a map with tensor stride:", map_key.first,
            "string id:", map_key.second);
  auto const map_inverse_map =
      detail::insert_and_map_functor<coordinate_type, coordinate_field_type,
                                     TemplatedAllocator, CoordinateMapType>()(
          map_key, coordinate, *this);

  LOG_DEBUG("map_inverse_map initialized");

  return std::make_pair(CoordinateMapKey(coordinate_size, map_key),
                        map_inverse_map);
}

#ifndef RIEMANN_NO_PYBIND
template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<py::object, std::pair<at::Tensor, at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    insert_and_map(at::Tensor const &coordinate,
                   default_types::stride_type const tensor_stride,
                   std::string const string_id) {
  auto result = insert_and_map_cpp(coordinate, tensor_stride, string_id);
  LOG_DEBUG("py key initialized");

  return std::make_pair(py::cast(new CoordinateMapKey(result.first)),
                        result.second);
}
#endif

/*******************************
 * Stride / origin / prune
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<coordinate_map_key_type, bool> CoordinateMapManager<
    coordinate_type, coordinate_field_type, TemplatedAllocator,
    CoordinateMapType>::stride(coordinate_map_key_type const &in_map_key,
                               stride_type const &kernel_stride,
                               std::string const string_id) {
  ASSERT(exists(in_map_key), ERROR_MAP_NOT_FOUND);
  LOG_DEBUG("In tensor stride:", in_map_key.first,
            "kernel stride:", kernel_stride);
  coordinate_map_key_type out_map_key(
      detail::stride_tensor_stride(in_map_key.first, kernel_stride, false),
      string_id == "" ? in_map_key.second : string_id);
  LOG_DEBUG("Out stride map key:", out_map_key);
  bool const exists_out_map = exists(out_map_key);
  if (!exists_out_map) {
    map_type const &in_map = m_coordinate_maps.find(in_map_key)->second;
    map_type out_map = in_map.stride(kernel_stride);
    insert(out_map_key, out_map);
  }
  return std::make_pair(out_map_key, !exists_out_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<coordinate_map_key_type, bool>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    stride_region(coordinate_map_key_type const &in_map_key,
                  cpu_kernel_region<coordinate_type> &kernel,
                  stride_type const &out_tensor_stride,
                  bool const expand_coordinates) {
  ASSERT(exists(in_map_key), ERROR_MAP_NOT_FOUND);
  LOG_DEBUG("stride_region");

  coordinate_map_key_type out_map_key(out_tensor_stride, "");
  bool const exists_out_map = exists(out_map_key);
  if (!exists_out_map || expand_coordinates) {
    LOG_DEBUG("Create a new stride region map for tensor_stride:",
              out_tensor_stride);
    map_type const &in_map = m_coordinate_maps.find(in_map_key)->second;
    map_type out_map = in_map.stride_region(kernel, out_tensor_stride);
    if (exists_out_map) {
      LOG_DEBUG("coordinate map exists for tensor_stride:", out_tensor_stride);
      out_map_key = get_random_string_id(out_tensor_stride, "");
      LOG_DEBUG("created a random key:", out_map_key);
    }
    insert(out_map_key, out_map);
  }
  return std::make_pair(out_map_key, !exists_out_map || expand_coordinates);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<coordinate_map_key_type, bool>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::origin() {
  ASSERT(m_coordinate_maps.size() > 0, "No coordinate map found");
  map_type const &random_map = m_coordinate_maps.begin()->second;
  stride_type origin_tensor_stride(random_map.coordinate_size() - 1);
  std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                [](auto &i) { i = 0; });
  LOG_DEBUG("origin tensor stride:", origin_tensor_stride);

  coordinate_map_key_type origin_map_key(origin_tensor_stride, "");
  bool const exists_origin_map = exists(origin_map_key);
  if (!exists_origin_map) {
    LOG_DEBUG("origin coordinate map not found");
    map_type const *p_seed_map{nullptr};
    size_type seed_size = std::numeric_limits<size_type>::max();
    for (auto map_it = m_coordinate_maps.begin();
         map_it != m_coordinate_maps.end(); ++map_it) {
      if (seed_size > map_it->second.size()) {
        p_seed_map = &(map_it->second);
      }
    }

    if (p_seed_map != nullptr) {
      map_type origin_map = p_seed_map->origin();
      LOG_DEBUG("origin map with size:", origin_map.size(), " inserted");
      insert(origin_map_key, origin_map);
    } else {
      ASSERT(false, "Invalid origin map");
    }
  }

  LOG_DEBUG("return origin()");
  return std::make_pair(origin_map_key, !exists_origin_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<coordinate_map_key_type, bool>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::origin_field() {
  ASSERT(m_field_coordinates.size() > 0, "No coordinate map found");
  field_map_type const &random_map = m_field_coordinates.begin()->second;
  stride_type origin_tensor_stride(random_map.coordinate_size() - 1);
  std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                [](auto &i) { i = 0; });
  LOG_DEBUG("origin tensor stride:", origin_tensor_stride);

  coordinate_map_key_type origin_map_key(origin_tensor_stride, "");
  bool const exists_origin_map = exists(origin_map_key);

  if (!exists_origin_map) {
    LOG_DEBUG("origin coordinate map not found");
    field_map_type const *p_seed_map{nullptr};
    size_type seed_size = std::numeric_limits<size_type>::max();
    for (auto map_it = m_field_coordinates.begin();
         map_it != m_field_coordinates.end(); ++map_it) {
      if (seed_size > map_it->second.size()) {
        p_seed_map = &(map_it->second);
      }
    }

    if (p_seed_map != nullptr) {
      map_type origin_map = p_seed_map->origin();
      LOG_DEBUG("origin map with size:", origin_map.size(), " inserted");
      insert(origin_map_key, origin_map);
    } else {
      ASSERT(false, "Invalid origin map");
    }
  }

  return std::make_pair(origin_map_key, !exists_origin_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
coordinate_map_key_type
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::prune(coordinate_map_key_type const
                                                   &in_key,
                                               bool const *keep_begin,
                                               bool const *keep_end) {
  auto const map_it = m_coordinate_maps.find(in_key);
  ASSERT(map_it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);

  coordinate_map_key_type map_key = std::make_pair(in_key.first, "pruned");
  if (m_coordinate_maps.find(map_key) != m_coordinate_maps.end()) {
    map_key = get_random_string_id(map_key.first, map_key.second);
  }

  map_type pruned_map = map_it->second.prune(keep_begin, keep_end);
  LOG_DEBUG("pruned map with size:", pruned_map.size(), " inserted");
  insert(map_key, pruned_map);

  return map_key;
}

/*******************************
 * Kernel map generation
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
typename CoordinateMapManager<coordinate_type, coordinate_field_type,
                              TemplatedAllocator,
                              CoordinateMapType>::kernel_map_type const &
CoordinateMapManager<
    coordinate_type, coordinate_field_type, TemplatedAllocator,
    CoordinateMapType>::kernel_map(CoordinateMapKey const *p_in_map_key,
                                   CoordinateMapKey const *p_out_map_key) {
  // identity stencil (volume one)
  auto const &map_it = m_coordinate_maps.find(p_in_map_key->get_key());
  ASSERT(map_it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
  auto const coordinate_size = map_it->second.coordinate_size();
  auto const one_vec = detail::ones(coordinate_size - 1);
  auto const offset = torch::empty(
      {0}, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));

  return kernel_map(p_in_map_key, p_out_map_key, one_vec, one_vec, one_vec,
                    RegionType::HYPER_CUBE, offset, false, false);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
typename CoordinateMapManager<coordinate_type, coordinate_field_type,
                              TemplatedAllocator,
                              CoordinateMapType>::kernel_map_type const &
CoordinateMapManager<
    coordinate_type, coordinate_field_type, TemplatedAllocator,
    CoordinateMapType>::kernel_map(CoordinateMapKey const *p_in_map_key,
                                   CoordinateMapKey const *p_out_map_key,
                                   stride_type const &kernel_size, //
                                   stride_type const &kernel_stride,
                                   stride_type const &kernel_dilation,
                                   RegionType::Type const region_type,
                                   at::Tensor const &offset, bool is_transpose,
                                   bool is_pool) {
  size_type kernel_dim = kernel_size.size();

  ASSERT(kernel_dim == kernel_stride.size(), "kernel size mismatch");
  ASSERT(kernel_dim == kernel_dilation.size(), "kernel size mismatch");

  at::Tensor host_offsets;
  std::vector<coordinate_type> offset_values;
  if (region_type == RegionType::CUSTOM) {
    host_offsets = detail::normalize_region_offsets(offset, kernel_dim);
    auto const *data = host_offsets.data_ptr<coordinate_type>();
    offset_values.assign(data, data + host_offsets.numel());
    for (size_type i = 0; i < kernel_dim; ++i)
      TORCH_CHECK(kernel_size[i] > 0 && kernel_stride[i] > 0 &&
                      kernel_dilation[i] > 0,
                  "CUSTOM kernel size, stride and dilation must be positive");
  }

  kernel_map_key_type const kernel_map_key =
      std::make_tuple(p_in_map_key->get_key(), p_out_map_key->get_key(), // maps
                      kernel_size, kernel_stride, kernel_dilation, // kernels
                      region_type, is_transpose, is_pool);

  auto find_cached_map = [&](kernel_map_key_type const &key)
      -> kernel_map_type const * {
    if (region_type == RegionType::CUSTOM) {
      auto it = m_custom_kernel_maps.find(std::make_pair(key, offset_values));
      return it == m_custom_kernel_maps.end() ? nullptr : &it->second;
    }
    auto it = m_kernel_maps.find(key);
    return it == m_kernel_maps.end() ? nullptr : &it->second;
  };
  auto cache_entry = [&](kernel_map_key_type const &key) -> kernel_map_type & {
    if (region_type == RegionType::CUSTOM)
      return m_custom_kernel_maps[std::make_pair(key, offset_values)];
    return m_kernel_maps[key];
  };
  auto const *cached_map = find_cached_map(kernel_map_key);
  LOG_DEBUG("set kernel map key for kernel map:", p_in_map_key->get_key(), "->",
            p_out_map_key->get_key());

  if (cached_map == nullptr) {
    auto const in_map_it = m_coordinate_maps.find(p_in_map_key->get_key());
    auto const out_map_it = m_coordinate_maps.find(p_out_map_key->get_key());

    ASSERT(in_map_it != m_coordinate_maps.end(), "in_map", ERROR_MAP_NOT_FOUND);
    ASSERT(out_map_it != m_coordinate_maps.end(), "out_map",
           ERROR_MAP_NOT_FOUND);

    auto const &in_map = in_map_it->second;
    auto const &out_map = out_map_it->second;

    LOG_DEBUG("coordinate_size:", in_map.coordinate_size(),
              "in tensor_stride:", in_map.get_tensor_stride(),
              "out tensor_stride:", out_map.get_tensor_stride());

    // +1 for the batch column
    ASSERT(kernel_dim + 1 == in_map.coordinate_size(), "kernel size mismatch");
    ASSERT(kernel_dim + 1 == out_map.coordinate_size(), "kernel size mismatch");

    if (region_type == RegionType::CUSTOM) {
      auto const &tensor_stride =
          is_transpose ? out_map.get_tensor_stride() : in_map.get_tensor_stride();
      for (size_type i = 0; i < kernel_dim; ++i) {
        TORCH_CHECK(tensor_stride[i] > 0, "CUSTOM tensor stride must be positive");
        for (size_t k = i; k < offset_values.size(); k += kernel_dim) {
          int64_t const value = offset_values[k];
          uint64_t const magnitude = value < 0 ? -value : value;
          uint64_t const limit =
              value < 0 ? uint64_t{1} << 31
                        : std::numeric_limits<coordinate_type>::max();
          TORCH_CHECK(magnitude <= limit / tensor_stride[i] / kernel_dilation[i],
                      "Scaled CUSTOM offset must fit the coordinate type");
        }
      }
    }

    if (in_map.size() == 0 || out_map.size() == 0) {
      auto empty_map =
          detail::empty_map_functor<coordinate_type, TemplatedAllocator,
                                    CoordinateMapType, kernel_map_type>()();
      cache_entry(kernel_map_key) = std::move(empty_map);
      return cache_entry(kernel_map_key);
    }

    if (!is_transpose) {
      if (region_type != RegionType::CUSTOM &&
          is_pool && (kernel_stride == kernel_size)) {
        LOG_DEBUG("generating stride_map");
        auto const stride_map =
            detail::stride_map_functor<coordinate_type, TemplatedAllocator,
                                       CoordinateMapType, kernel_map_type>()(
                in_map, out_map, out_map.get_tensor_stride());

        cache_entry(kernel_map_key) = std::move(stride_map);

      } else {
        LOG_DEBUG("generating kernel map");

        auto kernel_region = cpu_kernel_region<coordinate_type>(
            region_type,                       //
            in_map.coordinate_size(),          //
            in_map.get_tensor_stride().data(), //
            kernel_size.data(),                //
            kernel_dilation.data(),            //
            0, host_offsets.defined() ? host_offsets.data_ptr<coordinate_type>()
                                      : nullptr,
            host_offsets.defined() ? host_offsets.size(0) : 0);

        auto const kernel_map =
            detail::kernel_map_functor<coordinate_type, TemplatedAllocator,
                                       CoordinateMapType, kernel_map_type>()(
                in_map, out_map, m_kernel_map_mode, kernel_region);

        LOG_DEBUG("kernel_map done");
        cache_entry(kernel_map_key) = std::move(kernel_map);
        LOG_DEBUG("kernel_map saved");
      }
    } else { // transpose: try to reuse the forward-direction map
      kernel_map_key_type const swapped_kernel_map_key = std::make_tuple(
          p_out_map_key->get_key(), p_in_map_key->get_key(), // maps
          kernel_size, kernel_stride, kernel_dilation,       // kernels
          region_type, false, is_pool);

      if (auto const *swapped_map = find_cached_map(swapped_kernel_map_key)) {
        LOG_DEBUG("found existing kernel_map_key for transposed kernel map");
        cache_entry(kernel_map_key) =
            detail::swap_in_out_map_functor<kernel_map_type>()(*swapped_map);
      } else {
        LOG_DEBUG("No existing kernel_map_key for transposed kernel map");
        if (region_type != RegionType::CUSTOM &&
            is_pool && kernel_stride == kernel_size) {
          // the strided map runs out -> in; swap afterwards
          auto const stride_map =
              detail::stride_map_functor<coordinate_type, TemplatedAllocator,
                                         CoordinateMapType, kernel_map_type>()(
                  out_map, in_map, in_map.get_tensor_stride());

          cache_entry(kernel_map_key) =
              detail::swap_in_out_map_functor<kernel_map_type>()(stride_map);
        } else {
          auto kernel_region = cpu_kernel_region<coordinate_type>(
              region_type,                        //
              out_map.coordinate_size(),          //
              out_map.get_tensor_stride().data(), //
              kernel_size.data(),                 //
              kernel_dilation.data(),             //
              0, host_offsets.defined() ? host_offsets.data_ptr<coordinate_type>()
                                        : nullptr,
              host_offsets.defined() ? host_offsets.size(0) : 0,
              true // is_transpose
          );

          // generated out -> in, then swapped
          auto const kernel_map =
              detail::kernel_map_functor<coordinate_type, TemplatedAllocator,
                                         CoordinateMapType, kernel_map_type>()(
                  out_map, in_map, m_kernel_map_mode, kernel_region);

          LOG_DEBUG("kernel_map done");
          cache_entry(kernel_map_key) =
              detail::swap_in_out_map_functor<kernel_map_type>()(
                  std::move(kernel_map));
          LOG_DEBUG("kernel_map saved");
        }
      }
    }
  }
#ifdef DEBUG
  else {
    LOG_DEBUG("kernel map found");
  }
#endif

  return cached_map ? *cached_map : cache_entry(kernel_map_key);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
typename CoordinateMapManager<coordinate_type, coordinate_field_type,
                              TemplatedAllocator,
                              CoordinateMapType>::kernel_map_type const &
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::origin_map(CoordinateMapKey const
                                                        *p_in_map_key) {
  ASSERT(exists(p_in_map_key), ERROR_MAP_NOT_FOUND);
  kernel_map_key_type const kernel_map_key =
      origin_map_key(p_in_map_key->get_key());

  if (m_kernel_maps.find(kernel_map_key) == m_kernel_maps.end()) {
    auto const key = origin().first;
    auto const &origin_coordinate_map = m_coordinate_maps.find(key)->second;
    auto origin_map = m_coordinate_maps.find(p_in_map_key->get_key())
                          ->second.origin_map(origin_coordinate_map);
    m_kernel_maps[kernel_map_key] = std::move(origin_map);
  }

  return m_kernel_maps[kernel_map_key];
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
typename CoordinateMapManager<coordinate_type, coordinate_field_type,
                              TemplatedAllocator,
                              CoordinateMapType>::kernel_map_type const &
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::origin_field_map(CoordinateMapKey const
                                                              *p_in_map_key) {
  ASSERT(exists_field(p_in_map_key), ERROR_MAP_NOT_FOUND);
  kernel_map_key_type const kernel_map_key =
      origin_map_key(p_in_map_key->get_key());

  if (m_field_kernel_maps.find(kernel_map_key) == m_field_kernel_maps.end()) {
    auto const key = origin_field().first;
    auto const &origin_coordinate_map = m_coordinate_maps.find(key)->second;
    auto origin_map = m_field_coordinates.find(p_in_map_key->get_key())
                          ->second.origin_map(origin_coordinate_map);
    m_field_kernel_maps[kernel_map_key] = std::move(origin_map);
  }

  return m_field_kernel_maps[kernel_map_key];
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<at::Tensor, at::Tensor>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::stride_map_th(CoordinateMapKey const
                                                           *p_in_map_key,
                                                       CoordinateMapKey const
                                                           *p_strided_map_key) {
  ASSERT(exists(p_in_map_key), ERROR_MAP_NOT_FOUND);
  ASSERT(exists(p_strided_map_key), ERROR_MAP_NOT_FOUND);

  map_type const &in_map =
      m_coordinate_maps.find(p_in_map_key->get_key())->second;
  map_type const &strided_map =
      m_coordinate_maps.find(p_strided_map_key->get_key())->second;

  auto const &in_map_stride = in_map.get_tensor_stride();
  auto const &strided_map_stride = strided_map.get_tensor_stride();

  stride_type kernel_stride(in_map_stride.size());
  for (index_type i = 0; i < kernel_stride.size(); ++i) {
    ASSERT(strided_map_stride[i] % in_map_stride[i] == 0,
           "The tensor stride of the strided map must be divisible by the "
           "tensor stride of the input map. strided_map_stride:",
           ArrToString(strided_map_stride),
           " in_map_stride:", ArrToString(in_map_stride));
    kernel_stride[i] = strided_map_stride[i] / in_map_stride[i];
  }

  auto const one_vec = detail::ones(in_map.coordinate_size() - 1);
  kernel_map_key_type const kernel_map_key = std::make_tuple(
      p_in_map_key->get_key(), p_strided_map_key->get_key(), // maps
      kernel_stride, kernel_stride, one_vec,                 // kernels
      RegionType::HYPER_CUBE /* region_type */, 0 /* is_transpose */,
      true /* is_pool */);

  if (m_kernel_maps.find(kernel_map_key) == m_kernel_maps.end()) {
    LOG_DEBUG("Creating stride kernel map with kernel size:",
              ArrToString(kernel_stride));
    auto const stride_map =
        detail::stride_map_functor<coordinate_type, TemplatedAllocator,
                                   CoordinateMapType, kernel_map_type>()(
            in_map, strided_map, strided_map.get_tensor_stride());

    m_kernel_maps[kernel_map_key] = std::move(stride_map);
  }

  return detail::stride_map2tensor_functor<coordinate_type, TemplatedAllocator,
                                           CoordinateMapType,
                                           kernel_map_type>()(
      m_kernel_maps[kernel_map_key]);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<at::Tensor, std::vector<at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::origin_map_th(CoordinateMapKey const
                                                           *p_in_map_key) {
  kernel_map_type const &kernel_map = origin_map(p_in_map_key);

  coordinate_map_key_type const origin_key = origin().first;
  map_type const &origin_map = m_coordinate_maps.find(origin_key)->second;

  return detail::origin_map_functor<coordinate_type, TemplatedAllocator,
                                    CoordinateMapType, kernel_map_type>()(
      origin_map, kernel_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<at::Tensor, std::vector<at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    origin_field_map_th(CoordinateMapKey const *p_in_map_key) {
  kernel_map_type const &kernel_map = origin_field_map(p_in_map_key);

  coordinate_map_key_type const origin_key = origin_field().first;
  map_type const &origin_map = m_coordinate_maps.find(origin_key)->second;

  return detail::origin_map_functor<coordinate_type, TemplatedAllocator,
                                    CoordinateMapType, kernel_map_type>()(
      origin_map, kernel_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::vector<at::Tensor>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    interpolation_map_weight(at::Tensor const &tfield,
                             CoordinateMapKey const *p_in_map_key) {
  ASSERT(exists(p_in_map_key), ERROR_MAP_NOT_FOUND);
  return m_coordinate_maps.find(p_in_map_key->get_key())
      ->second.interpolation_map_weight(tfield);
}

/*******************************
 * Merge / union
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
coordinate_map_key_type
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    merge(std::vector<coordinate_map_key_type> const &map_keys) {
  ASSERT(map_keys.size() > 1, "Got one or zero map. Merge at least 2 maps.");
  std::vector<std::reference_wrapper<map_type>> maps;
  auto const tensor_stride_size = map_keys[0].first.size();
  stride_type merged_map_tensor_stride{map_keys[0].first};
  for (const auto &key : map_keys) {
    ASSERT(exists(key), ERROR_MAP_NOT_FOUND);
    auto &map = m_coordinate_maps.find(key)->second;
    maps.push_back(map);
    for (size_t k = 0; k < tensor_stride_size; ++k) {
      merged_map_tensor_stride[k] =
          std::min(merged_map_tensor_stride[k], map.get_tensor_stride()[k]);
    }
  }

  coordinate_map_key_type merged_map_key =
      get_random_string_id(merged_map_tensor_stride, "merge");
  map_type const &map = m_coordinate_maps.find(map_keys[0])->second;
  map_type merged_map = map.merge(maps);
  insert(merged_map_key, merged_map);
  return merged_map_key;
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::pair<coordinate_map_key_type, std::vector<at::Tensor>>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    union_map(std::vector<coordinate_map_key_type> const &map_keys) {
  auto const merged_key = merge(map_keys);
  map_type const &merged_map = m_coordinate_maps.find(merged_key)->second;

  std::vector<std::reference_wrapper<map_type>> maps;
  for (const auto &key : map_keys) {
    ASSERT(exists(key), ERROR_MAP_NOT_FOUND);
    maps.push_back(std::ref(m_coordinate_maps.find(key)->second));
  }

  return std::make_pair(merged_key, merged_map.union_map(maps));
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::vector<at::Tensor>
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::
    union_map_th(std::vector<CoordinateMapKey *> const &p_map_keys,
                 CoordinateMapKey *p_out_key) {
  ASSERT(!p_out_key->is_key_set(),
         "Out coordinate map key should be uninitialized");

  std::vector<coordinate_map_key_type> map_keys;
  map_keys.reserve(p_map_keys.size());
  std::for_each(
      p_map_keys.begin(), p_map_keys.end(),
      [&](CoordinateMapKey *p_key) { map_keys.push_back(p_key->get_key()); });
  auto union_pair = union_map(map_keys);
  p_out_key->set_key(union_pair.first);
  return union_pair.second;
}

/*******************************
 * Accessors
 *******************************/

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
at::Tensor
CoordinateMapManager<coordinate_type, coordinate_field_type, TemplatedAllocator,
                     CoordinateMapType>::get_coordinates(CoordinateMapKey const
                                                             *p_key) const {
  ASSERT(exists(p_key), ERROR_MAP_NOT_FOUND);
  auto const it = m_coordinate_maps.find(p_key->get_key());
  ASSERT(it != m_coordinate_maps.end(), ERROR_MAP_NOT_FOUND);
  auto const &map = it->second;
  auto const nrows = map.size();
  auto const ncols = map.coordinate_size();
  LOG_DEBUG("coordinate map nrows:", nrows, "ncols:", ncols);

  auto options = torch::TensorOptions().dtype(torch::kInt).requires_grad(false);
  if (!detail::is_cpu_coordinate_map<CoordinateMapType>::value) {
#ifndef CPU_ONLY
    auto device_id = at::cuda::current_device();
    options = options.device(torch::kCUDA, device_id);
#else
    ASSERT(false, ERROR_CPU_ONLY);
#endif
  }
  at::Tensor coordinates =
      torch::empty({(int64_t)nrows, (int64_t)ncols}, options);

  LOG_DEBUG("Initialized coordinates");
  map.copy_coordinates(coordinates.template data_ptr<coordinate_type>());
  LOG_DEBUG("Copied coordinates");
  return coordinates;
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
std::unordered_map<int64_t, at::Tensor> CoordinateMapManager<
    coordinate_type, coordinate_field_type, TemplatedAllocator,
    CoordinateMapType>::kernel_map_th(CoordinateMapKey const *p_in_map_key,
                                      CoordinateMapKey const *p_out_map_key,
                                      stride_type const &kernel_size, //
                                      stride_type const &kernel_stride,
                                      stride_type const &kernel_dilation,
                                      RegionType::Type const region_type,
                                      at::Tensor const &offset,
                                      bool is_transpose, bool is_pool) {
  auto const &curr_kernel_map =
      kernel_map(p_in_map_key, p_out_map_key,                 // maps
                 kernel_size, kernel_stride, kernel_dilation, // kernels
                 region_type, offset, is_transpose, is_pool);

  return detail::kernel_map_to_tensors<coordinate_type, TemplatedAllocator,
                                       CoordinateMapType, kernel_map_type>()(
      curr_kernel_map);
}

template <typename coordinate_type, typename coordinate_field_type,
          template <typename C> class TemplatedAllocator,
          template <typename T, template <typename Q> class A>
          class CoordinateMapType>
at::Tensor CoordinateMapManager<coordinate_type, coordinate_field_type,
                                TemplatedAllocator, CoordinateMapType>::
    get_coordinate_field(CoordinateMapKey const *p_key) const {
  auto const it = m_field_coordinates.find(p_key->get_key());
  ASSERT(it != m_field_coordinates.end(), ERROR_MAP_NOT_FOUND);
  auto const &map = it->second;
  auto const nrows = map.size();
  auto const ncols = map.coordinate_size();

  auto options = torch::TensorOptions()
                     .dtype(std::is_same<float, coordinate_field_type>::value
                                ? torch::kFloat
                                : torch::kDouble)
                     .requires_grad(false);

  if (!detail::is_cpu_coordinate_map<CoordinateMapType>::value) {
#ifndef CPU_ONLY
    auto device_id = at::cuda::current_device();
    options = options.device(torch::kCUDA, device_id);
#else
    ASSERT(false, ERROR_CPU_ONLY);
#endif
  }
  at::Tensor coordinates =
      torch::empty({(int64_t)nrows, (int64_t)ncols}, options);

  map.copy_coordinates(coordinates.template data_ptr<coordinate_field_type>());
  return coordinates;
}

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_SPACE_IMPL_HPP
