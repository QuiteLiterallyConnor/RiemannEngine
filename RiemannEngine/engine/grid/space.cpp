#include "engine/grid/space_impl.hpp"

#include "engine/grid/lattice_cpu.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace riemann {

/*******************************
 * CPU backend functors
 *******************************/

namespace detail {

template <typename coordinate_type, typename coordinate_field_type>
std::pair<at::Tensor, at::Tensor>
insert_and_map_functor<coordinate_type, coordinate_field_type, std::allocator,
                       CoordinateMapCPU>::
operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
           CoordinateMapManager<coordinate_type, coordinate_field_type,
                                std::allocator, CoordinateMapCPU> &manager) {
  LOG_DEBUG("initialize_and_map");
  uint32_t const N = th_coordinate.size(0);
  uint32_t const coordinate_size = th_coordinate.size(1);
  coordinate_type *p_coordinate = th_coordinate.data_ptr<coordinate_type>();
  auto map = CoordinateMapCPU<coordinate_type, std::allocator>(
      N, coordinate_size, map_key.first);
  auto map_inverse_map = map.template insert_and_map<true>(
      p_coordinate, p_coordinate + N * coordinate_size);
  LOG_DEBUG("mapping size:", map_inverse_map.first.size());

  // insertion moves the map into the manager
  manager.insert(map_key, map);

  auto const &mapping = map_inverse_map.first;
  auto const &inverse_mapping = map_inverse_map.second;

  at::Tensor th_mapping = torch::empty(
      {(int64_t)mapping.size()},
      torch::TensorOptions().requires_grad(false).dtype(torch::kInt64));
  at::Tensor th_inverse_mapping = torch::empty(
      {(int64_t)inverse_mapping.size()},
      torch::TensorOptions().requires_grad(false).dtype(torch::kInt64));

  int64_t *p_mapping = th_mapping.data_ptr<int64_t>();
  for (default_types::index_type i = 0; i < mapping.size(); ++i) {
    p_mapping[i] = mapping[i];
  }

  int64_t *p_inverse_mapping = th_inverse_mapping.data_ptr<int64_t>();
  for (default_types::index_type i = 0; i < inverse_mapping.size(); ++i) {
    p_inverse_mapping[i] = inverse_mapping[i];
  }

  return std::make_pair(std::move(th_mapping), std::move(th_inverse_mapping));
}

template <typename coordinate_type, typename coordinate_field_type>
void insert_field_functor<
    coordinate_type, coordinate_field_type, std::allocator, CoordinateMapCPU,
    CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                          std::allocator>>::
operator()(coordinate_map_key_type &map_key, at::Tensor const &th_coordinate,
           CoordinateMapManager<coordinate_type, coordinate_field_type,
                                std::allocator, CoordinateMapCPU> &manager) {
  LOG_DEBUG("insert field");
  uint32_t const N = th_coordinate.size(0);
  uint32_t const coordinate_size = th_coordinate.size(1);
  coordinate_field_type *p_coordinate =
      th_coordinate.data_ptr<coordinate_field_type>();
  auto map = CoordinateFieldMapCPU<coordinate_field_type, coordinate_type,
                                   std::allocator>(N, coordinate_size,
                                                   map_key.first);
  map.insert(p_coordinate, p_coordinate + N * coordinate_size);

  LOG_DEBUG("insert map with tensor_stride", map_key.first);
  manager.insert_field_map(map_key, map);
}

template <typename coordinate_type>
cpu_kernel_map
kernel_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                   cpu_kernel_map>::
operator()(CoordinateMapCPU<coordinate_type, std::allocator> const &in_map,
           CoordinateMapCPU<coordinate_type, std::allocator> const &out_map,
           CUDAKernelMapMode::Mode kernel_map_mode,
           cpu_kernel_region<coordinate_type> &kernel) {
  (void)kernel_map_mode;
  return in_map.kernel_map(out_map, kernel);
}

template <typename coordinate_type>
cpu_kernel_map
stride_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                   cpu_kernel_map>::
operator()(CoordinateMapCPU<coordinate_type, std::allocator> const &in_map,
           CoordinateMapCPU<coordinate_type, std::allocator> const &out_map,
           default_types::stride_type const &out_tensor_stride) {
  return in_map.stride_map(out_map, out_tensor_stride);
}

cpu_kernel_map swap_in_out_map_functor<cpu_kernel_map>::operator()(
    cpu_kernel_map const &kernel_map) {
  return std::make_pair(kernel_map.second, kernel_map.first);
}

template <typename coordinate_type>
cpu_kernel_map empty_map_functor<coordinate_type, std::allocator,
                                 CoordinateMapCPU, cpu_kernel_map>::
operator()() {
  return cpu_kernel_map{};
}

template <typename coordinate_type>
std::pair<at::Tensor, std::vector<at::Tensor>>
origin_map_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                   cpu_kernel_map>::
operator()(CoordinateMapCPU<coordinate_type, std::allocator> const
               &origin_coordinate_map,
           cpu_kernel_map const &origin_map) {
  auto options =
      torch::TensorOptions().dtype(torch::kLong).requires_grad(false);
  auto const out_size = origin_coordinate_map.size();
  auto const coordinate_size = origin_coordinate_map.coordinate_size();

  at::Tensor batch_indices =
      torch::empty({origin_coordinate_map.size()}, options);
  int64_t *p_batch_indices = batch_indices.data_ptr<int64_t>();

  LOG_DEBUG("Copying", origin_coordinate_map.size(), "batch indices");
  for (default_types::index_type i = 0; i < out_size; ++i) {
    p_batch_indices[i] =
        origin_coordinate_map.const_coordinate_data()[i * coordinate_size];
  }

  // inclusive maximum batch index
  coordinate_type const max_batch_index =
      *std::max_element(p_batch_indices, p_batch_indices + out_size);

  std::vector<at::Tensor> in_maps;
  for (auto i = 0; i <= max_batch_index; ++i) {
    at::Tensor row_indices = torch::empty({0}, options);
    in_maps.push_back(std::move(row_indices));
  }

  ASSERT(origin_map.first.size() == origin_map.second.size(),
         "invalid kernel_map");
  LOG_DEBUG("Iterating over", origin_map.first.size(), "unique maps");
  for (uint32_t out_row_index = 0; out_row_index < origin_map.first.size();
       ++out_row_index) {
    auto const &in_map = origin_map.first[out_row_index];
    int32_t const curr_size = in_map.size();
    ASSERT(curr_size > 0, "invalid kernel map for index", out_row_index);
    auto const curr_batch_index = p_batch_indices[out_row_index];

    ASSERT(curr_batch_index <= max_batch_index, "invalid batch index");
    at::Tensor &row_indices = in_maps[curr_batch_index];
    row_indices.resize_({curr_size});
    int64_t *p_row_indices = row_indices.data_ptr<int64_t>();

    LOG_DEBUG("Copying", curr_size, "elements to batch index",
              curr_batch_index, "and row index", out_row_index);
    for (int32_t i = 0; i < curr_size; ++i) {
      p_row_indices[i] = in_map[i];
    }
  }

  return std::make_pair(batch_indices, in_maps);
}

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
stride_map2tensor_functor<coordinate_type, std::allocator, CoordinateMapCPU,
                          cpu_kernel_map>::
operator()(cpu_kernel_map const &stride_kernel_map) {
  ASSERT(stride_kernel_map.first.size() == 1, "Invalid kernel_map");
  ASSERT(stride_kernel_map.first.size() == stride_kernel_map.second.size(),
         "invalid kernel_map");

  auto const &in_map = stride_kernel_map.first[0];
  auto const &out_map = stride_kernel_map.second[0];

  auto options =
      torch::TensorOptions().dtype(torch::kLong).requires_grad(false);
  int64_t const out_size = (int64_t)in_map.size();

  at::Tensor th_in_map = torch::empty({out_size}, options);
  at::Tensor th_out_map = torch::empty({out_size}, options);
  int64_t *p_in_map = th_in_map.data_ptr<int64_t>();
  int64_t *p_out_map = th_out_map.data_ptr<int64_t>();

  for (int64_t i = 0; i < out_size; ++i)
    p_in_map[i] = in_map[i];
  for (int64_t i = 0; i < out_size; ++i)
    p_out_map[i] = out_map[i];

  return std::make_pair(std::move(th_in_map), std::move(th_out_map));
}

template <typename coordinate_type>
std::unordered_map<int64_t, at::Tensor>
kernel_map_to_tensors<coordinate_type, std::allocator, CoordinateMapCPU,
                      cpu_kernel_map>::
operator()(cpu_kernel_map const &kernel_map) {
  const auto &in_maps = kernel_map.first;
  const auto &out_maps = kernel_map.second;

  auto options =
      torch::TensorOptions().dtype(torch::kInt).requires_grad(false);

  std::unordered_map<int64_t, at::Tensor> th_kernel_maps;
  for (size_t k = 0; k < in_maps.size(); ++k) {
    const auto &in_map = in_maps[k];
    const auto &out_map = out_maps[k];
    const int64_t N = in_map.size();
    if (N > 0) {
      at::Tensor th_kernel_map = torch::empty({2, N}, options);
      int32_t *p_kernel_map = th_kernel_map.data_ptr<int32_t>();
      std::copy_n(&in_map[0], N, p_kernel_map);
      std::copy_n(&out_map[0], N, p_kernel_map + N);
      th_kernel_maps[k] = std::move(th_kernel_map);
    }
  }

  return th_kernel_maps;
}

// functor instantiations for the CPU backend
template struct insert_and_map_functor<default_types::dcoordinate_type,
                                       default_types::ccoordinate_type,
                                       std::allocator, CoordinateMapCPU>;
template struct insert_field_functor<
    default_types::dcoordinate_type, default_types::ccoordinate_type,
    std::allocator, CoordinateMapCPU,
    CoordinateFieldMapCPU<default_types::ccoordinate_type,
                          default_types::dcoordinate_type, std::allocator>>;
template struct kernel_map_functor<default_types::dcoordinate_type,
                                   std::allocator, CoordinateMapCPU,
                                   cpu_kernel_map>;
template struct stride_map_functor<default_types::dcoordinate_type,
                                   std::allocator, CoordinateMapCPU,
                                   cpu_kernel_map>;
template struct empty_map_functor<default_types::dcoordinate_type,
                                  std::allocator, CoordinateMapCPU,
                                  cpu_kernel_map>;
template struct origin_map_functor<default_types::dcoordinate_type,
                                   std::allocator, CoordinateMapCPU,
                                   cpu_kernel_map>;
template struct stride_map2tensor_functor<default_types::dcoordinate_type,
                                          std::allocator, CoordinateMapCPU,
                                          cpu_kernel_map>;
template struct kernel_map_to_tensors<default_types::dcoordinate_type,
                                      std::allocator, CoordinateMapCPU,
                                      cpu_kernel_map>;

} // namespace detail

template class CoordinateMapManager<default_types::dcoordinate_type,
                                    default_types::ccoordinate_type,
                                    std::allocator, CoordinateMapCPU>;

/*******************************
 * CoordinateManager (CPU-only builds)
 *******************************/

#ifdef CPU_ONLY

CoordinateManager::CoordinateManager(int D, CoordinateMapBackend::Type map_type,
                                     GPUMemoryAllocatorBackend::Type alloc_type,
                                     RiemannAlgorithm::Mode algo)
    : D_(D), map_type_(map_type), alloc_type_(alloc_type), algo_(algo) {
  TORCH_CHECK(D > 0, "Dimension D must be positive");

  if (map_type != CoordinateMapBackend::CPU) {
    throw std::runtime_error("GPU backends not available in CPU_ONLY build");
  }
  manager_ = std::make_unique<cpu_manager_type<int32_t>>(algo, -1);
}

CoordinateManager::~CoordinateManager() = default;

std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateManager::insert_and_map(const at::Tensor &coordinates,
                                  stride_type tensor_stride,
                                  std::string string_id) {
  return std::visit(
      [&](auto &manager_ptr) {
        return manager_ptr->insert_and_map_cpp(coordinates, tensor_stride,
                                               string_id);
      },
      manager_);
}

CoordinateMapKey CoordinateManager::stride(const CoordinateMapKey &in_key,
                                           const stride_type &kernel_stride,
                                           std::string string_id) {
  return std::visit(
      [&](auto &manager_ptr) {
        auto key_pair =
            manager_ptr->stride(in_key.get_key(), kernel_stride, string_id);
        return CoordinateMapKey(D_ + 1, key_pair.first);
      },
      manager_);
}

std::pair<CoordinateMapKey, bool> CoordinateManager::origin() {
  return std::visit(
      [&](auto &manager_ptr) {
        auto [key_pair, is_new] = manager_ptr->origin();
        return std::make_pair(CoordinateMapKey(D_ + 1, key_pair), is_new);
      },
      manager_);
}

at::Tensor
CoordinateManager::get_coordinates(const CoordinateMapKey &key) const {
  return std::visit(
      [&](const auto &manager_ptr) -> at::Tensor {
        return manager_ptr->get_coordinates(&key);
      },
      manager_);
}

uint32_t CoordinateManager::size(const CoordinateMapKey &key) const {
  return std::visit(
      [&](const auto &manager_ptr) { return manager_ptr->size(&key); },
      manager_);
}

std::vector<at::Tensor>
CoordinateManager::union_map(const std::vector<CoordinateMapKey> &in_keys,
                             CoordinateMapKey &out_key) {
  return std::visit(
      [&](auto &manager_ptr) {
        std::vector<coordinate_map_key_type> internal_keys;
        internal_keys.reserve(in_keys.size());
        for (const auto &key : in_keys) {
          internal_keys.push_back(key.get_key());
        }

        auto result = manager_ptr->union_map(internal_keys);
        auto out_internal_key = result.first;
        auto maps = result.second;

        out_key = CoordinateMapKey(D_ + 1, out_internal_key);

        return maps;
      },
      manager_);
}

std::vector<at::Tensor>
CoordinateManager::interpolation_map_weight(const at::Tensor &tfield,
                                            const CoordinateMapKey &key) {
  return std::visit(
      [&](auto &manager_ptr) {
        return manager_ptr->interpolation_map_weight(tfield, &key);
      },
      manager_);
}

CoordinateMapKey CoordinateManager::insert_field(const at::Tensor &coordinates,
                                                 stride_type tensor_stride,
                                                 std::string string_id) {
  return std::visit(
      [&](auto &manager_ptr) {
        return manager_ptr->insert_field_cpp(coordinates, tensor_stride,
                                             string_id);
      },
      manager_);
}

std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateManager::field_to_sparse_insert_and_map(
    const CoordinateMapKey &field_key, stride_type sparse_stride,
    std::string string_id) {
  return std::visit(
      [&](auto &manager_ptr) {
        return manager_ptr->field_to_sparse_insert_and_map_cpp(
            &field_key, sparse_stride, string_id);
      },
      manager_);
}

#endif // CPU_ONLY

} // namespace riemann
