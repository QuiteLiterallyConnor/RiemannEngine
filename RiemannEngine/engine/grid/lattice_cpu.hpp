#ifndef RIEMANN_ENGINE_GRID_LATTICE_CPU_HPP
#define RIEMANN_ENGINE_GRID_LATTICE_CPU_HPP

#include "engine/foundation/errors.hpp"
#include "engine/foundation/hashing.hpp"
#include "engine/foundation/types.hpp"
#include "engine/grid/maps.hpp"
#include "engine/grid/stencil.hpp"

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

#include <robin_hood.h>
#include <torch/torch.h>

namespace riemann {

namespace detail {

// Block-partitioned std::thread execution. Small inputs run inline; larger
// inputs are split so each worker owns a contiguous [begin, end) block.
// fn(block_begin, block_end) must be thread-safe across disjoint blocks.
template <typename Fn>
inline void parallel_blocks(size_t const n, Fn &&fn) {
  constexpr size_t parallel_threshold = 32768;
  constexpr size_t min_items_per_thread = 8192;
  if (n == 0)
    return;

  size_t num_threads = 1;
  if (n >= parallel_threshold) {
    size_t const hw = std::max<size_t>(std::thread::hardware_concurrency(), 1);
    num_threads = std::min(hw, n / min_items_per_thread);
    if (num_threads == 0)
      num_threads = 1;
  }

  if (num_threads <= 1) {
    fn(size_t(0), n);
    return;
  }

  size_t const chunk = (n + num_threads - 1) / num_threads;
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    size_t const begin = t * chunk;
    if (begin >= n)
      break;
    size_t const end = std::min(n, begin + chunk);
    workers.emplace_back([&fn, begin, end] { fn(begin, end); });
  }
  for (auto &w : workers)
    w.join();
}

template <typename coordinate_type,
          typename stride_type = default_types::stride_type>
bool is_coordinate_aligned(coordinate<coordinate_type> const &point,
                           stride_type const &stride) {
  for (uint32_t i = 0; i < stride.size(); ++i) {
    if (point[i + 1] % stride[i] != 0)
      return false;
  }
  return true;
}

// Quantize a continuous field row-by-row onto the sparse lattice and look it
// up in `in_map`; emits (sparse row, field row) pairs for the hits.
template <typename coordinate_type, typename Dtype, typename MapType>
std::pair<at::Tensor, at::Tensor>
field_map_kernel(uint32_t const num_tfield,      //
                 uint32_t const coordinate_size, //
                 Dtype const *const p_tfield,    //
                 MapType const &in_map,          //
                 default_types::stride_type const &tensor_stride) {
  cpu_in_maps in_maps = initialize_maps<cpu_in_map>(1, num_tfield);
  cpu_out_maps out_maps = initialize_maps<cpu_out_map>(1, num_tfield);

  std::atomic<uint32_t> num_used{0};

  parallel_blocks(num_tfield, [&](size_t block_begin, size_t block_end) {
    std::vector<coordinate_type> curr_vec(coordinate_size);
    coordinate<coordinate_type> curr_coordinate(curr_vec.data());

    for (size_t i = block_begin; i < block_end; ++i) {
      curr_vec[0] =
          (coordinate_type)std::lroundf((float)p_tfield[i * coordinate_size]);
      for (uint32_t j = 1; j < coordinate_size; ++j) {
        auto const curr_tensor_stride = tensor_stride[j - 1];
        curr_vec[j] =
            curr_tensor_stride *
            (coordinate_type)std::floor(p_tfield[coordinate_size * i + j] /
                                        curr_tensor_stride);
      }

      const auto iter_in = in_map.find(curr_coordinate);
      if (iter_in != in_map.end()) {
        uint32_t const slot = num_used.fetch_add(1);
        in_maps[0][slot] = iter_in->second;
        out_maps[0][slot] = (uint32_t)i;
      }
    }
  });

  uint32_t const total = num_used.load();
  auto final_in_map = torch::empty(
      {total},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_out_map = torch::empty(
      {total},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));

  std::copy_n(in_maps[0].data(), total, final_in_map.template data_ptr<int>());
  std::copy_n(out_maps[0].data(), total,
              final_out_map.template data_ptr<int>());

  return std::make_pair(final_in_map, final_out_map);
}

// For every field row test all 2^D surrounding lattice corners and emit
// (sparse row, field row, multilinear weight) triples for corners present in
// the sparse map. Results are grouped per corner index.
template <typename coordinate_type, typename Dtype, typename MapType>
std::vector<at::Tensor> interpolation_map_weight_kernel(
    uint32_t const num_tfield,      //
    uint32_t const coordinate_size, //
    Dtype const *const p_tfield,    //
    MapType const &in_map,          //
    default_types::stride_type const &tensor_stride) {
  constexpr bool is_float32 = std::is_same<Dtype, float>::value;
  at::ScalarType const float_type =
      is_float32 ? at::ScalarType::Float : at::ScalarType::Double;
  uint32_t const neighbor_volume = (uint32_t)std::pow(2, (coordinate_size - 1));
  LOG_DEBUG("neighbor_volume :", neighbor_volume, "num_tfield:", num_tfield);

  cpu_in_maps in_maps =
      initialize_maps<cpu_in_map>(neighbor_volume, num_tfield);
  cpu_out_maps out_maps =
      initialize_maps<cpu_out_map>(neighbor_volume, num_tfield);

  std::vector<std::vector<Dtype>> weights =
      initialize_maps<std::vector<Dtype>>(neighbor_volume, num_tfield);

  std::unique_ptr<std::atomic<uint32_t>[]> num_used(
      new std::atomic<uint32_t>[neighbor_volume]);
  for (uint32_t i = 0; i < neighbor_volume; ++i)
    num_used[i].store(0);

  parallel_blocks(num_tfield, [&](size_t block_begin, size_t block_end) {
    std::vector<coordinate_type> curr_vec(coordinate_size), lb(coordinate_size),
        ub(coordinate_size);
    coordinate<coordinate_type> curr_coordinate(curr_vec.data());

    for (size_t i = block_begin; i < block_end; ++i) {
      curr_vec[0] =
          (coordinate_type)std::lroundf((float)p_tfield[i * coordinate_size]);
      for (uint32_t j = 1; j < coordinate_size; ++j) {
        auto const curr_tensor_stride = tensor_stride[j - 1];
        lb[j] = curr_tensor_stride *
                (coordinate_type)std::floor(p_tfield[coordinate_size * i + j] /
                                            curr_tensor_stride);
        ub[j] = lb[j] + curr_tensor_stride;
        curr_vec[j] = lb[j];
      }

      for (uint32_t neighbor_ind = 0; neighbor_ind < neighbor_volume;
           ++neighbor_ind) {
        uint32_t mask = 1;
        for (uint32_t j = coordinate_size - 1; j > 0; --j) {
          if ((neighbor_ind & mask) == 0)
            curr_vec[j] = lb[j];
          else
            curr_vec[j] = ub[j];
          mask = mask << 1;
        }

        const auto iter_in = in_map.find(curr_coordinate);
        if (iter_in != in_map.end()) {
          uint32_t const slot = num_used[neighbor_ind].fetch_add(1);

          Dtype weight = 1.0;
          for (uint32_t j = 1; j < coordinate_size; ++j) {
            weight *=
                1 - std::abs(p_tfield[coordinate_size * i + j] - curr_vec[j]) /
                        tensor_stride[j - 1];
          }

          in_maps[neighbor_ind][slot] = iter_in->second;
          out_maps[neighbor_ind][slot] = (uint32_t)i;
          weights[neighbor_ind][slot] = weight;
        }
      }
    }
  });

  uint32_t total_num_used = 0;
  for (uint32_t i = 0; i < neighbor_volume; ++i)
    total_num_used += num_used[i].load();

  auto final_in_map = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_out_map = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
  auto final_weights = torch::empty(
      {total_num_used},
      torch::TensorOptions().dtype(float_type).requires_grad(false));

  uint32_t final_begin = 0;
  for (uint32_t i = 0; i < neighbor_volume; ++i) {
    uint32_t const max_num = num_used[i].load();
    LOG_DEBUG("kernel index", i, "size:", max_num);

    std::copy_n(in_maps[i].data(), max_num,
                &final_in_map.template data_ptr<int>()[final_begin]);
    std::copy_n(out_maps[i].data(), max_num,
                &final_out_map.template data_ptr<int>()[final_begin]);
    std::copy_n(weights[i].data(), max_num,
                final_weights.template data_ptr<Dtype>() + final_begin);

    final_begin += max_num;
  }
  return {final_in_map, final_out_map, final_weights};
}

} // namespace detail

/*
 * Shared storage layer for coordinate maps: an owned row-major coordinate
 * buffer plus the tensor stride metadata. Allocation strategy is supplied by
 * the templated byte allocator.
 */
// clang-format off
template <typename coordinate_type, template <typename T> class TemplatedAllocator>
class CoordinateMap {

public:
  using self_type   = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using index_type  = default_types::index_type;
  using size_type   = default_types::size_type;
  using stride_type = default_types::stride_type;

  using index_vector_type   = std::vector<default_types::index_type>;
  using index_set_type      = std::set<default_types::index_type>;

  using byte_allocator_type = TemplatedAllocator<char>;

  CoordinateMap() = delete;
  CoordinateMap(size_type const number_of_coordinates,
                size_type const coordinate_size,
                stride_type const &stride = {1},
                byte_allocator_type alloc = byte_allocator_type())
      : m_coordinate_size(coordinate_size),
        m_capacity(0), /* updated inside allocate() */
        m_tensor_stride(stride), m_byte_allocator(alloc) {
    allocate(number_of_coordinates);
    expand_tensor_stride();
    LOG_DEBUG("tensor stride:", m_tensor_stride);
  }

  template <typename key_iterator, typename mapped_iterator>
  void insert(key_iterator key_first, key_iterator key_last,
              mapped_iterator value_first, mapped_iterator value_last) {
    ASSERT(false, "Not implemented"); // no virtual members for a templated class
  }

  self_type stride(stride_type const &tensor_strides) const {
    ASSERT(false, "Not implemented"); // no virtual members for a templated class
  }

  // clang-format on

  coordinate_type *coordinate_data() { return m_coordinates.get(); }
  coordinate_type const *const_coordinate_data() const {
    return m_coordinates.get();
  }

  void reserve(size_type size) {
    if (m_capacity < size) {
      LOG_DEBUG("Reserve coordinates:", size, "current capacity:", m_capacity);
      allocate(size);
    }
  }

  std::string to_string() const;

  stride_type const &get_tensor_stride() const noexcept {
    return m_tensor_stride;
  }

  inline size_type capacity() const noexcept { return m_capacity; }

  inline size_type coordinate_size() const noexcept {
    return m_coordinate_size;
  }

protected:
  void allocate(size_type const number_of_coordinates) {
    if (m_capacity < number_of_coordinates) {
      LOG_DEBUG("Allocate", number_of_coordinates, "coordinates.");
      auto const size = number_of_coordinates * m_coordinate_size;
      m_coordinates = allocate_ptr(size);
      m_capacity = number_of_coordinates;
    }
  }

  std::shared_ptr<coordinate_type[]> allocate_ptr(size_type const size) {
    coordinate_type *ptr = reinterpret_cast<coordinate_type *>(
        m_byte_allocator.allocate(size * sizeof(coordinate_type)));

    auto deleter = [](coordinate_type *p, byte_allocator_type alloc,
                      std::size_t size) {
      alloc.deallocate(reinterpret_cast<char *>(p), size);
    };

    return std::shared_ptr<coordinate_type[]>{
        ptr, std::bind(deleter, std::placeholders::_1, m_byte_allocator,
                       size * sizeof(coordinate_type))};
  }

private:
  // a scalar tensor stride broadcasts to every spatial axis
  void expand_tensor_stride() {
    if (m_tensor_stride.size() == 1) {
      for (size_type i = 0; i < m_coordinate_size - 2; ++i) {
        m_tensor_stride.push_back(m_tensor_stride[0]);
      }
    }
    ASSERT(m_tensor_stride.size() == m_coordinate_size - 1,
           "Invalid tensor stride", m_tensor_stride);
  }

protected:
  size_type m_number_of_coordinates;
  size_type m_coordinate_size;
  size_type m_capacity;
  stride_type m_tensor_stride;

  byte_allocator_type m_byte_allocator;
  std::shared_ptr<coordinate_type[]> m_coordinates;
};

/*
 * Host coordinate map backed by a robin-hood flat table keyed on coordinate
 * views into the owned coordinate buffer.
 */
// clang-format off
template <typename coordinate_type,
          template <typename T> class TemplatedAllocator = std::allocator>
class CoordinateMapCPU : public CoordinateMap<coordinate_type, TemplatedAllocator> {
public:
  using base_type                 = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using self_type                 = CoordinateMapCPU<coordinate_type, TemplatedAllocator>;
  using size_type                 = typename base_type::size_type;
  using index_type                = typename base_type::index_type;
  using stride_type               = typename base_type::stride_type;

  using key_type       = coordinate<coordinate_type>;
  using mapped_type    = default_types::index_type;
  using hasher         = detail::coordinate_murmur3<coordinate_type>;
  using key_equal      = detail::coordinate_equal_to<coordinate_type>;
  using map_type       =
      robin_hood::unordered_flat_map<key_type,    // key
                                     mapped_type, // mapped_type
                                     hasher,      // hasher
                                     key_equal    // equality
                                     >;

  using value_type                = typename map_type::value_type;
  using iterator                  = typename map_type::iterator;
  using const_iterator            = typename map_type::const_iterator;

  using index_vector_type         = typename base_type::index_vector_type;
  using byte_allocator_type       = TemplatedAllocator<char>;
  // clang-format on

public:
  CoordinateMapCPU() = delete;
  CoordinateMapCPU(size_type const number_of_coordinates,
                   size_type const coordinate_size,
                   stride_type const &stride = {1},
                   byte_allocator_type alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, alloc),
        m_map(
            map_type{0, hasher{coordinate_size}, key_equal{coordinate_size}}) {
    m_map.reserve(number_of_coordinates);
  }

  /*
   * Insert a range of packed coordinates; duplicates collapse onto the first
   * inserted row.
   */
  void insert(coordinate_type const *coordinate_begin,
              coordinate_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;
    base_type::allocate(N);
    index_type value = 0;
    for (coordinate_type const *key = coordinate_begin; key != coordinate_end;
         key += m_coordinate_size, ++value) {
      insert(key_type(key), value);
    }
  }

  /*
   * Insert all rows and report both directions of the dedup mapping:
   *
   * >>> unique_coordinates = input_coordinates[mapping]
   * >>> reconstructed = unique_coordinates[inverse_mapping]
   * >>> torch.all(reconstructed == input_coordinates)
   */
  template <bool remap>
  std::pair<std::vector<int64_t>, std::vector<int64_t>>
  insert_and_map(coordinate_type const *coordinate_begin,
                 coordinate_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;

    std::vector<int64_t> mapping, inverse_mapping;
    base_type::allocate(N);
    mapping.reserve(N);
    inverse_mapping.reserve(N);

    index_type value{0}, row_index{0};
    for (coordinate_type const *key = coordinate_begin; key != coordinate_end;
         key += m_coordinate_size, row_index += 1) {
      auto const result = insert(key_type(key), value);
      if (result.second) {
        mapping.push_back(row_index);
        inverse_mapping.push_back(value);
      } else {
        inverse_mapping.push_back(result.first->second);
      }
      value += remap ? result.second : 1;
    }

    return std::make_pair(std::move(mapping), std::move(inverse_mapping));
  }

  /*
   * Query a key range; returns (valid query positions, found unique indices).
   */
  template <typename key_iterator>
  std::pair<index_vector_type, index_vector_type> find(key_iterator key_first,
                                                       key_iterator key_last) {
    size_type N = key_last - key_first;
    ASSERT(N <= base_type::m_capacity,
           "Invalid search range. Current capacity:", base_type::m_capacity,
           ", search range:", N);

    index_vector_type valid_query_index, query_result;
    valid_query_index.reserve(N);
    query_result.reserve(N);

    key_iterator key_curr{key_first};
    for (; key_curr != key_last; ++key_curr) {
      auto const query_iter = m_map.find(*key_curr);
      if (query_iter != m_map.end()) {
        valid_query_index.push_back(key_curr - key_first);
        query_result.push_back(query_iter->second);
      }
    }
    return std::make_pair(valid_query_index, query_result);
  }

  /*
   * Floor-align every coordinate to the strided lattice and dedup.
   */
  self_type stride(stride_type const &stride) const {
    ASSERT(stride.size() == m_coordinate_size - 1, "Invalid stride", stride);
    self_type stride_map(
        size(), m_coordinate_size,
        detail::stride_tensor_stride(base_type::m_tensor_stride, stride),
        base_type::m_byte_allocator);

    index_type c = 0;
    std::vector<coordinate_type> dst(m_coordinate_size);
    coordinate<coordinate_type> strided_coordinate(&dst[0]);
    for (auto const &kv : m_map) {
      detail::stride_coordinate<coordinate_type>(kv.first, dst,
                                                 stride_map.m_tensor_stride);
      auto result = stride_map.insert(strided_coordinate, c);
      c += result.second;
    }

    return stride_map;
  }

  /*
   * Expand each coordinate through the kernel region and keep the generated
   * points (transpose keeps all, forward keeps lattice-aligned ones).
   */
  self_type stride_region(cpu_kernel_region<coordinate_type> const &kernel,
                          stride_type const &out_tensor_stride) const {
    ASSERT(kernel.coordinate_size() == m_coordinate_size, "Invalid kernel");
    self_type stride_map(size() * kernel.volume(), m_coordinate_size,
                         out_tensor_stride, base_type::m_byte_allocator);

    auto ckernel = cpu_kernel_region<coordinate_type>(kernel);
    std::vector<coordinate_type> tmp(m_coordinate_size);
    coordinate<coordinate_type> point(tmp.data());

    index_type num_used{0};
    if (kernel.is_transpose()) {
      for (auto iter_in = m_map.begin(); iter_in != m_map.end(); ++iter_in) {
        for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
             ++kernel_ind) {
          ckernel.coordinate_at(kernel_ind, iter_in->first.data(), tmp.data());
          auto const result = stride_map.insert(point, num_used);
          num_used += result.second;
        }
      }
    } else {
      LOG_DEBUG("stride_region with no transpose");
      for (auto iter_in = m_map.begin(); iter_in != m_map.end(); ++iter_in) {
        for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
             ++kernel_ind) {
          ckernel.coordinate_at(kernel_ind, iter_in->first.data(), tmp.data());
          if (detail::is_coordinate_aligned<coordinate_type, stride_type>(
                  point, out_tensor_stride)) {
            auto const result = stride_map.insert(point, num_used);
            num_used += result.second;
          }
        }
      }
    }
    return stride_map;
  }

  /*
   * One output row per unique batch index; all spatial components zero.
   */
  self_type origin() const {
    stride_type origin_tensor_stride(m_coordinate_size - 1);
    std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                  [](auto &i) { i = 0; });

    self_type origin_map(size(), m_coordinate_size, origin_tensor_stride,
                         base_type::m_byte_allocator);

    index_type c = 0;
    std::vector<coordinate_type> dst(m_coordinate_size);
    std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });

    coordinate<coordinate_type> tmp_coordinate(&dst[0]);
    for (auto const &kv : m_map) {
      dst[0] = kv.first[0];
      auto result = origin_map.insert(tmp_coordinate, c);
      c += result.second;
    }
    return origin_map;
  }

  self_type prune(bool const *keep_begin, bool const *keep_end) const {
    ASSERT(keep_end - keep_begin == size(), "Invalid range for pruning");

    self_type pruned_map(size(), m_coordinate_size, base_type::m_tensor_stride,
                         base_type::m_byte_allocator);

    index_type c = 0;
    for (auto const &kv : m_map) {
      if (keep_begin[kv.second]) {
        auto result = pruned_map.insert(kv.first, c);
        c += result.second;
      }
    }
    LOG_DEBUG("size:", pruned_map.size(), "capacity:", pruned_map.capacity());
    return pruned_map;
  }

  self_type merge(const self_type &other) const {
    std::vector<std::reference_wrapper<self_type>> maps{*this, other};
    return merge(maps);
  }

  self_type
  merge(const std::vector<std::reference_wrapper<self_type>> &maps) const {
    size_t all_size = std::accumulate(
        maps.begin(), maps.end(), (size_t)0,
        [](size_t sum, const self_type &map) { return sum + map.size(); });
    self_type merged_map(all_size, m_coordinate_size,
                         base_type::m_tensor_stride,
                         base_type::m_byte_allocator);
    index_type c = 0;
    for (self_type const &map : maps) {
      for (auto const &kv : map.m_map) {
        auto result = merged_map.insert(kv.first, c);
        c += result.second;
      }
    }

    return merged_map;
  }

  /*
   * Enumerate the kernel region around each output coordinate and record the
   * (input row, output row) pairs that exist in this map, grouped by kernel
   * offset index.
   */
  cpu_kernel_map
  kernel_map(self_type const &out_coordinate_map,
             cpu_kernel_region<coordinate_type> const &kernel) const {
    size_type out_size = out_coordinate_map.size();
    size_type kernel_volume = kernel.volume();
    LOG_DEBUG("kernel volume:", kernel_volume, "out_size:", out_size);

    cpu_in_maps in_maps = initialize_maps<cpu_in_map>(kernel_volume, out_size);
    cpu_out_maps out_maps =
        initialize_maps<cpu_out_map>(kernel_volume, out_size);

    const auto &out_mmap = out_coordinate_map.m_map;
    const size_t out_map_num_elements = out_mmap.capacity();

    if (kernel.region_type() != RegionType::CUSTOM && kernel_volume == 1) {
      // identity stencil: single sequential pass, no region iteration
      index_type curr_index_begin = 0;
      for (auto iter_out = out_mmap.begin(); iter_out != out_mmap.end();
           ++iter_out) {
        const auto iter_in = m_map.find(iter_out->first);
        if (iter_in != m_map.end()) {
          in_maps[0][curr_index_begin] = iter_in->second;
          out_maps[0][curr_index_begin] = iter_out->second;
          ++curr_index_begin;
        }
      }
      in_maps[0].resize(curr_index_begin);
      out_maps[0].resize(curr_index_begin);
      return std::make_pair(in_maps, out_maps);
    }

    std::unique_ptr<std::atomic<size_type>[]> num_used(
        new std::atomic<size_type>[kernel_volume]);
    for (size_type i = 0; i < kernel_volume; ++i)
      num_used[i].store(0);

    detail::parallel_blocks(
        out_map_num_elements, [&](size_t block_begin, size_t block_end) {
          auto ckernel = cpu_kernel_region<coordinate_type>(kernel);
          std::vector<coordinate_type> tmp(m_coordinate_size);
          coordinate<coordinate_type> curr_kernel_coordinate(tmp.data());

          size_t const block_size = block_end - block_begin;
          for (auto iter_out = out_mmap.begin(block_begin);
               iter_out.num_steps() < block_size; ++iter_out) {
            for (uint32_t kernel_ind = 0; kernel_ind < ckernel.volume();
                 ++kernel_ind) {
              ckernel.coordinate_at(kernel_ind, iter_out->first.data(),
                                    tmp.data());
              const auto iter_in = m_map.find(curr_kernel_coordinate);
              if (iter_in != m_map.end()) {
                size_type const slot = num_used[kernel_ind].fetch_add(1);
                in_maps[kernel_ind][slot] = iter_in->second;
                out_maps[kernel_ind][slot] = iter_out->second;
              }
            }
          }
        });

    for (index_type i = 0; i < kernel_volume; ++i) {
      index_type max_num = num_used[i].load();
      LOG_DEBUG("kernel index", i, "size:", max_num);
      in_maps[i].resize(max_num);
      out_maps[i].resize(max_num);
    }

    return std::make_pair(in_maps, out_maps);
  }

  /*
   * Map every input row onto its floor-aligned voxel in the strided output
   * map. Every input row must resolve.
   */
  cpu_kernel_map stride_map(self_type const &out_coordinate_map,
                            stride_type const &out_tensor_stride) const {
    size_type in_size = size();
    LOG_DEBUG("Generate stride_map with in NNZ:", in_size,
              "out NNZ:", out_coordinate_map.size(),
              "out_tensor_stride:", out_tensor_stride);
    cpu_in_maps in_maps = initialize_maps<cpu_in_map>(1, in_size);
    cpu_out_maps out_maps = initialize_maps<cpu_out_map>(1, in_size);

    const size_t in_map_num_elements = m_map.capacity();

    std::atomic<index_type> num_used{0};
    detail::parallel_blocks(
        in_map_num_elements, [&](size_t block_begin, size_t block_end) {
          std::vector<coordinate_type> dst(m_coordinate_size);
          size_t const block_size = block_end - block_begin;
          for (auto iter_in = m_map.begin(block_begin);
               iter_in.num_steps() < block_size; ++iter_in) {
            detail::stride_coordinate<coordinate_type>(iter_in->first, dst,
                                                       out_tensor_stride);
            const auto iter_out = out_coordinate_map.find(
                coordinate<coordinate_type>(dst.data()));
            ASSERT(iter_out != out_coordinate_map.m_map.cend(),
                   "Invalid out_coordinate_map");
            index_type const slot = num_used.fetch_add(1);
            in_maps[0][slot] = iter_in->second;
            out_maps[0][slot] = iter_out->second;
          }
        });

    return std::make_pair(std::move(in_maps), std::move(out_maps));
  }

  /*
   * Group all rows by batch; origin_coordinate_map supplies the batch rows.
   */
  cpu_kernel_map origin_map(self_type const &origin_coordinate_map) const {
    ASSERT(std::all_of(origin_coordinate_map.get_tensor_stride().begin(),
                       origin_coordinate_map.get_tensor_stride().end(),
                       [](auto const &i) { return i == 0; }),
           "Invalid origin tensor stride",
           origin_coordinate_map.get_tensor_stride());

    size_type const in_size = size();
    size_type const out_size = origin_coordinate_map.size();
    LOG_DEBUG("Generate origin_map with in NNZ:", in_size,
              "out NNZ:", out_size);
    ASSERT(in_size > out_size, "Invalid out_coordinate_map");

    std::vector<std::pair<index_type, index_type>> in_out(in_size);

    size_t const in_map_num_elements = m_map.capacity();

    std::atomic<size_type> num_used{0};
    detail::parallel_blocks(
        in_map_num_elements, [&](size_t block_begin, size_t block_end) {
          std::vector<coordinate_type> dst(m_coordinate_size);
          std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });

          size_t const block_size = block_end - block_begin;
          for (auto iter_in = m_map.begin(block_begin);
               iter_in.num_steps() < block_size; ++iter_in) {
            dst[0] = iter_in->first[0];
            const auto iter_origin = origin_coordinate_map.find(
                coordinate<coordinate_type>(dst.data()));
            ASSERT(iter_origin != origin_coordinate_map.m_map.cend(),
                   "Invalid origin_coordinate_map");
            index_type origin_row_index = iter_origin->second;

            size_type const slot = num_used.fetch_add(1);
            in_out[slot] =
                std::make_pair(iter_in->second, origin_row_index);
          }
        });

    auto batch_indices = origin_coordinate_map.batch_indices();
    return cpu_kernel_map(in_out, batch_indices);
  }

  /*
   * Multilinear interpolation maps/weights for a continuous tensor field.
   */
  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield) const {
    ASSERT(tfield.dim() == 2, "Invalid tfield dimension");
    ASSERT(tfield.size(1) == m_coordinate_size, "Invalid tfield size");

    switch (tfield.scalar_type()) {
    case at::ScalarType::Double:
      return detail::interpolation_map_weight_kernel<coordinate_type, double,
                                                     map_type>(
          tfield.size(0),                     //
          m_coordinate_size,                  //
          tfield.template data_ptr<double>(), //
          m_map,                              //
          base_type::m_tensor_stride);
    case at::ScalarType::Float:
      return detail::interpolation_map_weight_kernel<coordinate_type, float,
                                                     map_type>(
          tfield.size(0),                    //
          m_coordinate_size,                 //
          tfield.template data_ptr<float>(), //
          m_map,                             //
          base_type::m_tensor_stride);
    default:
      ASSERT(false, "Unsupported float type");
    }
    return {};
  }

  template <typename coordinate_field_type>
  std::pair<at::Tensor, at::Tensor>
  field_map(coordinate_field_type const *p_tfield,
            size_type const num_tfield) const {
    return detail::field_map_kernel<coordinate_type, coordinate_field_type,
                                    map_type>(num_tfield,        //
                                              m_coordinate_size, //
                                              p_tfield,          //
                                              m_map,             //
                                              base_type::m_tensor_stride);
  }

  /*
   * One 2xN int64 tensor per input map, mapping input rows to rows in this
   * (union) map. Assumes every input coordinate exists in the union map.
   */
  std::vector<at::Tensor> union_map(
      std::vector<std::reference_wrapper<self_type>> const &in_maps) const {
    std::vector<at::Tensor> in_out_maps;

    for (self_type const &in_map : in_maps) {
      size_type const N = in_map.size();
      size_type const capacity = in_map.m_map.capacity();
      auto curr_map = torch::empty(
          {2, N},
          torch::TensorOptions().dtype(torch::kInt64).requires_grad(false));
      int64_t *p_in_rows = curr_map.template data_ptr<int64_t>();
      std::fill_n(p_in_rows, 2 * N, -1);
      int64_t *p_union_rows = p_in_rows + N;

      index_type offset{0};
      for (auto iter_in = in_map.m_map.begin(); iter_in.num_steps() < capacity;
           ++iter_in) {
        p_in_rows[offset] = iter_in->second;
        p_union_rows[offset] = find(iter_in->first)->second;
        LOG_DEBUG("offset:", offset, " in:", p_in_rows[offset],
                  " out:", p_union_rows[offset]);
        offset++;
      }

      in_out_maps.push_back(std::move(curr_map));
    }

    return in_out_maps;
  }

  inline size_type size() const noexcept { return m_map.size(); }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateMapCPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

  using base_type::capacity;
  using base_type::coordinate_size;
  using base_type::get_tensor_stride;

  inline void reserve(size_type c) {
    base_type::reserve(c);
    m_map.reserve(c);
  }

  void copy_coordinates(coordinate_type *dst_coordinate) const {
    if (m_map.size() == 0)
      return;
    size_t const capacity = m_map.capacity();

    detail::parallel_blocks(
        capacity, [&](size_t block_begin, size_t block_end) {
          size_t const block_size = block_end - block_begin;
          for (auto it = m_map.begin(block_begin);
               it.num_steps() < block_size; ++it) {
            std::copy_n(it->first.data(), m_coordinate_size,
                        dst_coordinate + m_coordinate_size * it->second);
          }
        });
  }

  std::vector<coordinate_type> batch_indices() const {
    std::vector<coordinate_type> indices(size());
    for (auto it = m_map.begin(); it != m_map.end(); ++it) {
      indices[it->second] = it->first.data()[0];
    }
    return indices;
  }

  std::pair<iterator, bool> insert(key_type const &key,
                                   mapped_type const &val) {
    ASSERT(val < base_type::m_capacity, "Invalid mapped value: ", val,
           ", current capacity: ", base_type::m_capacity);
    coordinate_type *ptr = &base_type::m_coordinates[val * m_coordinate_size];
    std::copy_n(key.data(), m_coordinate_size, ptr);
    return m_map.insert(value_type(coordinate<coordinate_type>{ptr}, val));
  }

  inline iterator find(key_type const &key) { return m_map.find(key); }
  inline const_iterator find(key_type const &key) const {
    return m_map.find(key);
  }
  inline const_iterator cend() const { return m_map.cend(); }

private:
  using base_type::m_coordinate_size;
  map_type m_map;
};

/*
 * Continuous (floating point) coordinate storage. Rows are kept verbatim —
 * no deduplication — and quantized on demand.
 */
template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator = std::allocator>
class CoordinateFieldMapCPU
    : public CoordinateMap<coordinate_field_type, TemplatedAllocator> {
public:
  using base_type = CoordinateMap<coordinate_field_type, TemplatedAllocator>;
  using coordinate_map_type =
      CoordinateMapCPU<coordinate_int_type, TemplatedAllocator>;
  using self_type =
      CoordinateFieldMapCPU<coordinate_field_type, coordinate_int_type,
                            TemplatedAllocator>;
  using size_type = typename base_type::size_type;
  using index_type = typename base_type::index_type;
  using stride_type = typename base_type::stride_type;
  using byte_allocator_type = TemplatedAllocator<char>;

public:
  CoordinateFieldMapCPU() = delete;
  CoordinateFieldMapCPU(size_type const number_of_coordinates,
                        size_type const coordinate_size,
                        stride_type const &stride = {1},
                        byte_allocator_type alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, alloc),
        m_size(number_of_coordinates) {
    base_type::reserve(number_of_coordinates);
  }

  void insert(coordinate_field_type const *coordinate_begin,
              coordinate_field_type const *coordinate_end) {
    size_type N = (coordinate_end - coordinate_begin) / m_coordinate_size;
    base_type::allocate(N);
    std::copy_n(coordinate_begin, N * m_coordinate_size,
                base_type::coordinate_data());
  }

  using base_type::const_coordinate_data;
  using base_type::coordinate_data;

  void copy_coordinates(coordinate_field_type *dst_coordinate) const {
    std::copy_n(base_type::const_coordinate_data(), size() * m_coordinate_size,
                dst_coordinate);
  }

  // Floor-quantize rows onto the given stride lattice; batch rounds to the
  // nearest integer.
  void quantize_coordinates(coordinate_int_type *p_dst_coordinates,
                            stride_type const &tensor_stride) const {
    coordinate_field_type const *const p_tfield = const_coordinate_data();
    int64_t const stride_prod = std::accumulate(
        tensor_stride.begin(), tensor_stride.end(), (int64_t)1,
        std::multiplies<>());
    ASSERT(stride_prod > 0, "Invalid stride");

    if (stride_prod == 1) {
      detail::parallel_blocks(size(), [&](size_t block_begin,
                                          size_t block_end) {
        for (size_t i = block_begin; i < block_end; ++i) {
          coordinate_int_type *p_curr_dst =
              &p_dst_coordinates[i * m_coordinate_size];
          p_curr_dst[0] = (coordinate_int_type)std::lroundf(
              (float)p_tfield[i * m_coordinate_size]);
          for (uint32_t j = 1; j < m_coordinate_size; ++j) {
            p_curr_dst[j] = (coordinate_int_type)std::floor(
                p_tfield[m_coordinate_size * i + j]);
          }
        }
      });
    } else {
      detail::parallel_blocks(size(), [&](size_t block_begin,
                                          size_t block_end) {
        for (size_t i = block_begin; i < block_end; ++i) {
          coordinate_int_type *p_curr_dst =
              &p_dst_coordinates[i * m_coordinate_size];
          p_curr_dst[0] = (coordinate_int_type)std::lroundf(
              (float)p_tfield[i * m_coordinate_size]);
          for (uint32_t j = 1; j < m_coordinate_size; ++j) {
            auto const curr_tensor_stride = tensor_stride[j - 1];
            p_curr_dst[j] =
                curr_tensor_stride *
                (coordinate_int_type)std::floor(
                    p_tfield[m_coordinate_size * i + j] / curr_tensor_stride);
          }
        }
      });
    }
  }

  coordinate_map_type origin() const {
    stride_type origin_tensor_stride(m_coordinate_size - 1);
    std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                  [](auto &i) { i = 0; });

    CoordinateMapCPU<coordinate_int_type, TemplatedAllocator> origin_map(
        size(), m_coordinate_size, origin_tensor_stride,
        base_type::m_byte_allocator);

    coordinate_field_type const *const p_tfield = const_coordinate_data();

    std::vector<coordinate_int_type> dst(m_coordinate_size);
    std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });
    coordinate<coordinate_int_type> tmp_coordinate(&dst[0]);

    index_type c = 0;
    for (size_t i = 0; i < size(); ++i) {
      dst[0] = coordinate_int_type(p_tfield[i * m_coordinate_size]);
      auto result = origin_map.insert(tmp_coordinate, c);
      c += result.second;
    }
    return origin_map;
  }

  cpu_kernel_map
  origin_map(coordinate_map_type const &origin_coordinate_map) const {
    ASSERT(std::all_of(origin_coordinate_map.get_tensor_stride().begin(),
                       origin_coordinate_map.get_tensor_stride().end(),
                       [](auto const &i) { return i == 0; }),
           "Invalid origin tensor stride",
           origin_coordinate_map.get_tensor_stride());

    size_type const in_size = size();
    size_type const out_size = origin_coordinate_map.size();
    LOG_DEBUG("Generate origin_map with in NNZ:", in_size,
              "out NNZ:", out_size);
    ASSERT(in_size > out_size, "Invalid out_coordinate_map");

    std::vector<std::pair<index_type, index_type>> in_out(in_size);

    coordinate_field_type const *const p_tfield = const_coordinate_data();
    std::atomic<size_type> num_used{0};
    detail::parallel_blocks(in_size, [&](size_t block_begin,
                                         size_t block_end) {
      std::vector<coordinate_int_type> dst(m_coordinate_size);
      std::for_each(dst.begin(), dst.end(), [](auto &i) { i = 0; });

      for (size_t i = block_begin; i < block_end; ++i) {
        dst[0] = (coordinate_int_type)p_tfield[i * m_coordinate_size];
        const auto iter_origin = origin_coordinate_map.find(
            coordinate<coordinate_int_type>(dst.data()));
        ASSERT(iter_origin != origin_coordinate_map.cend(),
               "Invalid origin_coordinate_map");
        index_type origin_row_index = iter_origin->second;

        size_type const slot = num_used.fetch_add(1);
        in_out[slot] = std::make_pair((index_type)i, origin_row_index);
      }
    });

    auto batch_indices = origin_coordinate_map.batch_indices();
    return cpu_kernel_map(in_out, batch_indices);
  }

  inline size_type size() const noexcept { return m_size; }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateFieldMapCPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

private:
  using base_type::m_coordinate_size;
  size_type m_size;
};

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_LATTICE_CPU_HPP
