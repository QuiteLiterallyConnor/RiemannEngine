#ifndef RIEMANN_ENGINE_GRID_LATTICE_GPU_CUH
#define RIEMANN_ENGINE_GRID_LATTICE_GPU_CUH

#include <concurrent_unordered_map.cuh>
#include <hash/helper_functions.cuh>

#include "engine/cuda/memory.cuh"
#include "engine/cuda/runtime.cuh"
#include "engine/cuda/scratch.cuh"
#include "engine/foundation/hashing.hpp"
#include "engine/foundation/types.hpp"
#include "engine/grid/lattice_cpu.hpp"
#include "engine/grid/stencil.hpp"

#include <functional>
#include <map>
#include <memory>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <torch/torch.h>

namespace riemann {

namespace detail {

/*
 * Device-side functors shared by the GPU hash-map kernels.
 */
template <typename Key, typename Element, typename Equality> struct is_used {
  using value_type = thrust::pair<Key, Element>;

  is_used(Key &&unused, Equality &&equal)
      : m_unused_key(unused), m_equal(equal) {}

  is_used(Key const &unused, Equality const &equal)
      : m_unused_key(unused), m_equal(equal) {}

  __host__ __device__ bool operator()(value_type const &x) {
    return !m_equal(x.first, m_unused_key);
  }

  Key const m_unused_key;
  Equality const m_equal;
};

template <typename coordinate_type, typename map_type> struct get_element {
  using value_type = thrust::pair<coordinate<coordinate_type>, uint32_t>;

  get_element(map_type &map) : m_map(map) {}

  __device__ uint32_t operator()(value_type const &x) { return x.second; }

  map_type const m_map;
};

// Insert row i of a packed coordinate buffer; records both the row index and
// the bucket offset of the inserted/found entry.
template <typename coordinate_type, typename map_type,
          typename mapped_iterator>
struct insert_coordinate {
  using value_type = typename map_type::value_type;
  using mapped_type = typename map_type::mapped_type;

  insert_coordinate(map_type &map,                       //
                    coordinate_type const *p_coordinate, //
                    mapped_iterator row_iter, mapped_iterator map_iter,
                    uint32_t const coordinate_size)
      : m_coordinate_size{coordinate_size}, m_coordinates{p_coordinate},
        m_row_iter{row_iter}, m_map_iter{map_iter}, m_map{map} {}

  __device__ bool operator()(uint32_t i) {
    auto coord =
        coordinate<coordinate_type>{&m_coordinates[i * m_coordinate_size]};
    value_type pair = thrust::make_pair(coord, i);
    auto result = m_map.insert(pair);

    m_row_iter[i] = i;
    m_map_iter[i] = result.first.offset();

    return result.second;
  }

  size_t const m_coordinate_size;
  coordinate_type const *m_coordinates;
  mapped_iterator m_row_iter;
  mapped_iterator m_map_iter;
  map_type m_map;
};

template <typename coordinate_type, typename map_type>
struct find_coordinate {
  using mapped_type = typename map_type::mapped_type;
  using return_type = thrust::pair<mapped_type, mapped_type>;
  using argument_type = uint32_t;
  using result_type = return_type;

  find_coordinate(map_type const &_map, coordinate_type const *_d_ptr,
                  mapped_type const unused_element, size_t const _size)
      : m_coordinate_size{_size},
        m_unused_element(unused_element), m_coordinates{_d_ptr}, m_map{_map} {}

  __device__ mapped_type operator()(uint32_t i) {
    auto coord =
        coordinate<coordinate_type>{&m_coordinates[i * m_coordinate_size]};
    auto result = m_map.find(coord);
    if (result == m_map.end()) {
      return m_unused_element;
    }
    return result->second;
  }

  size_t const m_coordinate_size;
  mapped_type const m_unused_element;
  coordinate_type const *m_coordinates;
  map_type const m_map;
};

template <typename coordinate_type, typename map_type> struct update_value {
  using mapped_type = typename map_type::mapped_type;

  update_value(map_type &_map, coordinate_type const *coordinates,
               uint32_t const *valid_index, size_t const _size)
      : m_coordinate_size{_size}, m_coordinates{coordinates},
        m_valid_index(valid_index), m_map{_map} {}

  __device__ void operator()(uint32_t i) {
    auto coord = coordinate<coordinate_type>{
        &m_coordinates[m_valid_index[i] * m_coordinate_size]};
    auto result = m_map.find(coord);
    result->second = i;
  }

  size_t const m_coordinate_size;
  coordinate_type const *m_coordinates;
  uint32_t const *m_valid_index;
  map_type m_map;
};

// Renumber the mapped values of surviving entries to their compacted order.
template <typename index_type, typename map_type>
struct update_value_with_offset {
  update_value_with_offset(map_type &_map, index_type const *valid_map_offset)
      : m_valid_map_offset(valid_map_offset), m_map{_map} {}

  __device__ void operator()(index_type i) {
    auto &result = m_map.data()[m_valid_map_offset[i]];
    result.second = i;
  }

  index_type const *m_valid_map_offset;
  map_type m_map;
};

template <typename T> struct is_first {
  is_first(T value) : m_value(value) {}

  template <typename Tuple>
  inline __device__ bool operator()(Tuple const &item) const {
    return thrust::get<0>(item) == m_value;
  }

  T m_value;
};

template <typename coordinate_type, typename mapped_type>
struct is_unused_pair {
  is_unused_pair(mapped_type const unused_element)
      : m_unused_element(unused_element) {}

  template <typename Tuple>
  inline __device__ bool operator()(Tuple const &item) const {
    return thrust::get<1>(item) == m_unused_element;
  }

  mapped_type const m_unused_element;
};

template <typename T, typename pair_type, typename pair_iterator>
struct split_functor {
  split_functor(pair_iterator begin, T *firsts, T *seconds)
      : m_begin(begin), m_firsts(firsts), m_seconds(seconds) {}

  __device__ void operator()(uint32_t i) {
    pair_type const item = *(m_begin + i);
    m_firsts[i] = item.first;
    m_seconds[i] = item.second;
  }

  pair_iterator m_begin;
  T *m_firsts;
  T *m_seconds;
};

// reduce_by_key combiner: (min offset, summed count)
template <typename T> struct min_size_functor {
  using min_size_type = thrust::tuple<T, T>;

  __host__ __device__ min_size_type operator()(const min_size_type &lhs,
                                               const min_size_type &rhs) {
    return thrust::make_tuple(min(thrust::get<0>(lhs), thrust::get<0>(rhs)),
                              thrust::get<1>(lhs) + thrust::get<1>(rhs));
  }
};

} // namespace detail

/*
 * Device kernel map: three parallel index arrays (kernel offset, in row,
 * out row) in contiguous device memory, decomposed per kernel offset after
 * a sort by offset.
 */
template <typename index_type, typename ByteAllocator> class gpu_kernel_map {
public:
  using size_type = default_types::size_type;
  using byte_allocator_type = ByteAllocator;
  using self_type = gpu_kernel_map<index_type, ByteAllocator>;

  class contiguous_memory {
  public:
    using self_type = contiguous_memory;

  public:
    contiguous_memory() = delete;
    contiguous_memory(gpu_kernel_map &kernel_map) : m_kernel_map{kernel_map} {}

    self_type &operator=(self_type const &other) {
      m_data = other.m_data;
      m_kernel_map = other.m_kernel_map;
      return *this;
    }

    inline typename std::map<index_type, index_type>::const_iterator
    key_cbegin() const {
      return m_kernel_map.m_kernel_offset_map.cbegin();
    }
    inline typename std::map<index_type, index_type>::const_iterator
    key_cend() const {
      return m_kernel_map.m_kernel_offset_map.cend();
    }

    inline index_type *data() { return m_data; }
    inline index_type const *cdata() const { return m_data; }
    inline void data(index_type *p_data) { m_data = p_data; }

    inline index_type *begin() const { return m_data; }
    inline index_type *begin(index_type kernel_index) const {
      auto const &offset_map = m_kernel_map.m_kernel_offset_map;
      auto const iter = offset_map.find(kernel_index);
      if (iter == offset_map.end()) {
        return m_data;
      } else {
        return m_data + iter->second;
      }
    }

    inline index_type *end() const { return m_data + m_kernel_map.m_capacity; }
    inline index_type *end(index_type kernel_index) const {
      auto const &offset_map = m_kernel_map.m_kernel_offset_map;
      auto const iter = offset_map.find(kernel_index);
      if (iter == offset_map.end()) {
        return m_data + m_kernel_map.m_capacity;
      } else {
        return m_data + iter->second +
               m_kernel_map.m_kernel_size_map[kernel_index];
      }
    }

    size_type size(index_type kernel_index) const {
      auto const &size_map = m_kernel_map.m_kernel_size_map;
      auto const iter = size_map.find(kernel_index);
      if (iter == size_map.end())
        return 0;
      return iter->second;
    }

  private:
    index_type *m_data;
    gpu_kernel_map &m_kernel_map;
  }; // contiguous_memory

public:
  gpu_kernel_map()
      : m_memory_size_byte{0},
        m_capacity{0}, kernels{*this}, in_maps{*this}, out_maps{*this} {
    LOG_DEBUG("Initialized gpu_kernel_map");
  }
  gpu_kernel_map(self_type const &other)
      : m_decomposed(other.m_decomposed),                       //
        m_requires_kernel_index(other.m_requires_kernel_index), //
        m_memory_size_byte(other.m_memory_size_byte),           //
        m_capacity{other.m_capacity},                           //
        m_in_map_memory{other.m_in_map_memory},                 //
        m_out_map_memory{other.m_out_map_memory},               //
        m_allocator{other.m_allocator},                         //
        m_kernel_size_map{other.m_kernel_size_map},             //
        m_kernel_offset_map{other.m_kernel_offset_map},         //
        kernels{*this},                                         //
        in_maps{*this},                                         //
        out_maps{*this} {
    LOG_DEBUG("gpu_kernel_map copy constructor");
    in_maps.data(other.in_maps.begin());
    out_maps.data(other.out_maps.begin());
    if (m_requires_kernel_index) {
      m_kernel_index_memory = other.m_kernel_index_memory;
      kernels.data(other.kernels.begin());
    }
  }

  gpu_kernel_map(size_type capacity,
                 byte_allocator_type alloc = byte_allocator_type(),
                 bool requires_kernel_index = true)
      : m_requires_kernel_index(requires_kernel_index), m_capacity{capacity},
        m_allocator{alloc}, kernels{*this}, in_maps{*this}, out_maps{*this} {
    m_memory_size_byte = capacity * sizeof(index_type);
    index_type *ptr_in_map = reinterpret_cast<index_type *>(
        m_allocator.allocate(m_memory_size_byte));
    index_type *ptr_out_map = reinterpret_cast<index_type *>(
        m_allocator.allocate(m_memory_size_byte));
    index_type *ptr_kernel = nullptr;

    auto deleter = [](index_type *p, byte_allocator_type alloc,
                      size_type size) {
      alloc.deallocate(reinterpret_cast<char *>(p), size);
      LOG_DEBUG("Deallocate kernel map");
    };

    m_in_map_memory = std::shared_ptr<index_type[]>{
        ptr_in_map, std::bind(deleter, std::placeholders::_1, m_allocator,
                              m_memory_size_byte)};
    m_out_map_memory = std::shared_ptr<index_type[]>{
        ptr_out_map, std::bind(deleter, std::placeholders::_1, m_allocator,
                               m_memory_size_byte)};
    in_maps.data(m_in_map_memory.get());
    out_maps.data(m_out_map_memory.get());

    if (requires_kernel_index) {
      ptr_kernel = reinterpret_cast<index_type *>(
          m_allocator.allocate(m_memory_size_byte));
      m_kernel_index_memory = std::shared_ptr<index_type[]>{
          ptr_kernel, std::bind(deleter, std::placeholders::_1, m_allocator,
                                m_memory_size_byte)};
      kernels.data(m_kernel_index_memory.get());
    } else {
      // single implicit kernel offset covering the full capacity
      m_kernel_offset_map[0] = 0;
      m_kernel_size_map[0] = capacity;
      m_decomposed = true;
    }
  }

  self_type swap() const {
    self_type swapped_gpu_kernel_map(*this);
    swapped_gpu_kernel_map.in_maps.data(
        swapped_gpu_kernel_map.m_out_map_memory.get());
    swapped_gpu_kernel_map.out_maps.data(
        swapped_gpu_kernel_map.m_in_map_memory.get());
    return swapped_gpu_kernel_map;
  }

  self_type &operator=(self_type const &other) {
    m_decomposed = other.m_decomposed;
    m_requires_kernel_index = other.m_requires_kernel_index;

    m_memory_size_byte = other.m_memory_size_byte;
    m_capacity = other.m_capacity;

    m_kernel_index_memory = other.m_kernel_index_memory;
    m_in_map_memory = other.m_in_map_memory;
    m_out_map_memory = other.m_out_map_memory;
    m_allocator = other.m_allocator;

    m_kernel_size_map = other.m_kernel_size_map;
    m_kernel_offset_map = other.m_kernel_offset_map;

    in_maps.data(other.in_maps.begin());
    out_maps.data(other.out_maps.begin());
    kernels.data(other.kernels.begin());

    return *this;
  }

  inline typename std::map<index_type, index_type>::const_iterator
  key_cbegin() const {
    return m_kernel_offset_map.cbegin();
  }

  inline typename std::map<index_type, index_type>::const_iterator
  key_cend() const {
    return m_kernel_offset_map.cend();
  }

  size_type size() const {
    size_type nmap = 0;
    for (auto const &k : m_kernel_size_map) {
      nmap += k.second;
    }
    return nmap;
  }

  size_type volume() const { return m_kernel_size_map.size(); }

  size_type max_size() const {
    size_type nmap = 0;
    for (auto const &k : m_kernel_size_map) {
      if (k.second > nmap)
        nmap = k.second;
    }
    return nmap;
  }

  size_type size(index_type const kernel_index) const {
    auto const iter = m_kernel_size_map.find(kernel_index);
    if (iter == m_kernel_size_map.end())
      return 0;
    return iter->second;
  }

  std::string to_string() const {
    Formatter o;
    size_type map_size = 0;
    for (auto const &kv : m_kernel_size_map) {
      map_size += kv.second;
    }
    o << "gpu_kernel_map: number of unique maps:" << m_kernel_size_map.size()
      << ", kernel map size:" << map_size;
    return o.str();
  }

  // Sort triplets by kernel offset and build the per-offset (start, count)
  // directory on the host.
  void decompose() {
    LOG_DEBUG("Decomposing", kernels.end() - kernels.begin(), "elements");
    THRUST_CHECK(thrust::sort_by_key(thrust::device,            //
                                     kernels.begin(),           // key begin
                                     kernels.end(),             // key end
                                     thrust::make_zip_iterator( // value begin
                                         thrust::make_tuple(    //
                                             in_maps.begin(),   //
                                             out_maps.begin()   //
                                             )                  //
                                         )));

    thrust::counting_iterator<index_type> min_begin{0};
    thrust::constant_iterator<index_type> size_begin{1};

    gpu_storage<index_type, byte_allocator_type> out_key(m_capacity);
    gpu_storage<index_type, byte_allocator_type> out_key_min(m_capacity);
    gpu_storage<index_type, byte_allocator_type> out_key_size(m_capacity);

    size_type num_unique_keys;

    try {
      auto end = thrust::reduce_by_key(
          thrust::device,  //
          kernels.begin(), //
          kernels.end(),   //
          thrust::make_zip_iterator(
              thrust::make_tuple(min_begin, size_begin)),
          out_key.begin(),
          thrust::make_zip_iterator(thrust::make_tuple(
              out_key_min.begin(), out_key_size.begin())),
          thrust::equal_to<index_type>(),
          detail::min_size_functor<index_type>());
      num_unique_keys = end.first - out_key.begin();
      LOG_DEBUG(num_unique_keys, "unique kernel map keys found");
    }
    THRUST_CATCH;

    auto const cpu_out_keys = out_key.to_vector(num_unique_keys);
    auto const cpu_out_offset = out_key_min.to_vector(num_unique_keys);
    auto const cpu_out_size = out_key_size.to_vector(num_unique_keys);

    for (index_type i = 0; i < num_unique_keys; ++i) {
      m_kernel_offset_map[cpu_out_keys[i]] = cpu_out_offset[i];
      m_kernel_size_map[cpu_out_keys[i]] = cpu_out_size[i];
    }

    m_decomposed = true;
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  self_type const &kernel_map) {
    out << kernel_map.to_string();
    return out;
  }

private:
  bool m_decomposed{false};
  bool m_requires_kernel_index;
  size_type m_memory_size_byte, m_capacity;
  std::shared_ptr<index_type[]> m_kernel_index_memory;
  std::shared_ptr<index_type[]> m_in_map_memory;
  std::shared_ptr<index_type[]> m_out_map_memory;
  byte_allocator_type m_allocator;

  std::map<index_type, index_type> m_kernel_size_map;
  std::map<index_type, index_type> m_kernel_offset_map;

public:
  contiguous_memory kernels;
  contiguous_memory in_maps;
  contiguous_memory out_maps;
}; // gpu_kernel_map

/*
 * Device coordinate map built on a concurrent GPU hash table. Valid entries
 * are tracked with compacted row/bucket index arrays so the insertion-order
 * contract matches the host map behavior.
 */
template <typename coordinate_type, template <typename T>
                                    class TemplatedAllocator =
                                        detail::c10_allocator>
class CoordinateMapGPU
    : public CoordinateMap<coordinate_type, TemplatedAllocator> {
public:
  // clang-format off
  using base_type           = CoordinateMap<coordinate_type, TemplatedAllocator>;
  using self_type           = CoordinateMapGPU<coordinate_type, TemplatedAllocator>;
  using size_type           = typename base_type::size_type;
  using index_type          = typename base_type::index_type;
  using stride_type         = typename base_type::stride_type;

  using map_allocator_type  = TemplatedAllocator<thrust::pair<coordinate<coordinate_type>, index_type>>;
  using byte_allocator_type = TemplatedAllocator<char>;

  using key_type            = coordinate<coordinate_type>;
  using mapped_type         = index_type;
  using hasher_type         = detail::coordinate_murmur3<coordinate_type>;
  using key_equal_type      = detail::coordinate_equal_to<coordinate_type>;
  using map_type            = concurrent_unordered_map<key_type,
                                                       mapped_type,
                                                       hasher_type,
                                                       key_equal_type,
                                                       map_allocator_type>;
  using value_type          = typename map_type::value_type;

  using kernel_map_type     = gpu_kernel_map<index_type, byte_allocator_type>;

  using iterator            = typename map_type::iterator;
  using const_iterator      = typename map_type::const_iterator;

  using index_storage_type     = gpu_storage<default_types::index_type, byte_allocator_type>;
  using coordnate_storage_type = gpu_storage<coordinate_type, byte_allocator_type>;
  // clang-format on

public:
  CoordinateMapGPU() = delete;
  CoordinateMapGPU(size_type const number_of_coordinates,
                   size_type const coordinate_size,
                   size_type const hashtable_occupancy = 50,
                   stride_type const stride = {1},
                   map_allocator_type map_alloc = map_allocator_type(),
                   byte_allocator_type byte_alloc = byte_allocator_type())
      : base_type(number_of_coordinates, coordinate_size, stride, byte_alloc),
        m_hashtable_occupancy{hashtable_occupancy},
        m_capacity(0), // updated by reserve()
        m_hasher(hasher_type{coordinate_size}),
        m_equal(key_equal_type{coordinate_size}),
        m_unused_key(coordinate<coordinate_type>{nullptr}),
        m_unused_element(std::numeric_limits<coordinate_type>::max()),
        m_map_allocator(map_alloc) {
    reserve(number_of_coordinates);
    LOG_DEBUG("tensor_stride", base_type::m_tensor_stride);
    m_device_tensor_stride.from_vector(base_type::m_tensor_stride);
    LOG_DEBUG("device tensor_stride set");
    static_assert(
        sizeof(index_type) == sizeof(size_type),
        "kernel_map shared memory requires the type sizes to be the same");
    static_assert(
        sizeof(coordinate_type) == sizeof(size_type),
        "kernel_map shared memory requires the type sizes to be the same");
  }

  template <bool remap>
  void insert(coordinate_iterator<coordinate_type> key_first,
              coordinate_iterator<coordinate_type> key_last);

  template <bool remap>
  std::pair<index_storage_type, index_storage_type>
  insert_and_map(coordinate_iterator<coordinate_type> key_first,
                 coordinate_iterator<coordinate_type> key_last);

  std::pair<index_storage_type, index_storage_type>
  find(coordinate_iterator<coordinate_type> key_first,
       coordinate_iterator<coordinate_type> key_last) const;

  inline void reserve(size_type size) {
    if (size > m_capacity) {
      base_type::reserve(size);
      LOG_DEBUG("Reserve map of",
                compute_hash_table_size(size, m_hashtable_occupancy),
                "for concurrent_unordered_map of size", size);
      auto map_unique = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      // explicit conversion avoids __to_address ambiguity between cuda::std
      // and std
      m_map = std::shared_ptr<map_type>(map_unique.release(),
                                        map_unique.get_deleter());
      LOG_DEBUG("Done concurrent_unordered_map creation");
      CUDA_TRY(cudaStreamSynchronize(0));
      m_capacity = size;
      LOG_DEBUG("Reserved concurrent_unordered_map");
    }
  }

  self_type stride(stride_type const &stride) const;
  self_type stride_region(cpu_kernel_region<coordinate_type> &kernel,
                          stride_type const &out_tensor_stride) const;
  self_type origin() const;
  self_type prune(bool const *keep_begin, bool const *keep_end) const;
  self_type
  merge(std::vector<std::reference_wrapper<self_type>> const &maps) const;

  kernel_map_type kernel_map(self_type const &out_coordinate_map,
                             gpu_kernel_region<coordinate_type> const &kernel,
                             CUDAKernelMapMode::Mode kernel_map_mode,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  kernel_map_type stride_map(self_type const &out_coordinate_map,
                             stride_type const &out_tensor_stride,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  kernel_map_type origin_map(self_type const &origin_coordinate_map,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;
  std::vector<at::Tensor>
  interpolation_map_weight(at::Tensor const &tfield) const;

  template <typename coordinate_field_type>
  std::pair<at::Tensor, at::Tensor>
  field_map(coordinate_field_type const *p_tfield,
            size_type const num_tfield) const;

  std::vector<at::Tensor>
  union_map(std::vector<std::reference_wrapper<self_type>> const &maps,
            uint32_t thread_dim = CUDA_NUM_THREADS) const;

  inline size_type size() const { return m_size; }
  void copy_coordinates(coordinate_type *dst_coordinate) const;
  std::string to_string() const {
    Formatter o;
    o << "CoordinateMapGPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }
  inline map_type const const_hash_map() const { return *m_map.get(); };

  // register pre-deduplicated coordinates in the hash map
  void initialize_valid_indices(size_t const N_unique);

  using base_type::const_coordinate_data;
  using base_type::coordinate_data;
  using base_type::coordinate_size;

private:
  using base_type::m_byte_allocator;
  using base_type::m_coordinate_size;

  size_type m_hashtable_occupancy;
  size_type m_capacity;
  size_type m_size;
  hasher_type const m_hasher;
  key_equal_type const m_equal;
  key_type const m_unused_key;
  mapped_type const m_unused_element;
  index_storage_type m_valid_row_index;
  index_storage_type m_valid_map_index;
  index_storage_type m_inverse_row_index;
  index_storage_type m_device_tensor_stride;
  map_allocator_type m_map_allocator;
  std::shared_ptr<map_type> m_map;
};

/*
 * Device-resident continuous coordinate storage.
 */
template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator =
              detail::c10_allocator>
class CoordinateFieldMapGPU
    : public CoordinateMap<coordinate_field_type, TemplatedAllocator> {
public:
  using base_type = CoordinateMap<coordinate_field_type, TemplatedAllocator>;
  using coordinate_map_type =
      CoordinateMapGPU<coordinate_int_type, TemplatedAllocator>;
  using self_type =
      CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                            TemplatedAllocator>;
  using size_type = typename base_type::size_type;
  using index_type = typename base_type::index_type;
  using stride_type = typename base_type::stride_type;
  using byte_allocator_type = TemplatedAllocator<char>;
  using map_allocator_type = TemplatedAllocator<
      thrust::pair<coordinate<coordinate_int_type>, index_type>>;
  using kernel_map_type = gpu_kernel_map<index_type, byte_allocator_type>;

  using key_type = coordinate<coordinate_int_type>;
  using mapped_type = index_type;
  using hasher_type = detail::coordinate_murmur3<coordinate_int_type>;
  using key_equal_type = detail::coordinate_equal_to<coordinate_int_type>;
  using int_hash_map_type =
      concurrent_unordered_map<key_type,            //
                               mapped_type,         //
                               hasher_type,         //
                               key_equal_type,      //
                               map_allocator_type>;

public:
  CoordinateFieldMapGPU() = delete;
  CoordinateFieldMapGPU(size_type const number_of_coordinates,
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
    CUDA_CHECK(cudaMemcpy(base_type::coordinate_data(), coordinate_begin,
                          N * m_coordinate_size * sizeof(coordinate_field_type),
                          cudaMemcpyDeviceToDevice));
  }

  void copy_coordinates(coordinate_field_type *dst_coordinate) const {
    CUDA_CHECK(
        cudaMemcpy(dst_coordinate, base_type::const_coordinate_data(),
                   size() * m_coordinate_size * sizeof(coordinate_field_type),
                   cudaMemcpyDeviceToDevice));
  }

  void quantize_coordinates(coordinate_int_type *p_dst_coordinates,
                            stride_type const &tensor_stride) const;

  using base_type::const_coordinate_data;
  using base_type::coordinate_data;

  coordinate_map_type origin() const;

  kernel_map_type origin_map(coordinate_map_type const &origin_coordinate_map,
                             uint32_t thread_dim = CUDA_NUM_THREADS) const;

  inline size_type size() const noexcept { return m_size; }
  std::string to_string() const {
    Formatter o;
    o << "CoordinateFieldMapGPU:" << size() << "x" << m_coordinate_size;
    return o.str();
  }

private:
  using base_type::m_byte_allocator;
  using base_type::m_coordinate_size;
  size_type m_size;
};

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_LATTICE_GPU_CUH
