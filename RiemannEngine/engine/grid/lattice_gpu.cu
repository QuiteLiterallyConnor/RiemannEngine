#include "engine/grid/lattice_gpu.cuh"
#include "engine/grid/maps.hpp"
#include "engine/grid/stencil.hpp"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <numeric>

namespace riemann {

namespace detail {

/*
 * Hash insertion: one thread per input row. Successful (first) insertions
 * record their row and bucket offset; losers record the sentinel so a later
 * compaction pass can drop them.
 */
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
insert_and_map_kernel(map_type map,                                    //
                      coordinate_type const *__restrict__ coordinates, //
                      index_type *__restrict__ valid_map_index,        //
                      index_type *__restrict__ valid_row_index,        //
                      size_type const num_threads,                     //
                      size_type const coordinate_size,                 //
                      index_type const unused_key) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    auto const result = map.insert(thrust::make_pair(
        coordinate<coordinate_type>{&coordinates[x * coordinate_size]}, x));

    if (result.second) {
      valid_row_index[x] = x;
      valid_map_index[x] = result.first.offset();
    } else {
      valid_map_index[x] = unused_key;
    }
  }
}

// After remapping, rewrite the inverse map through fresh lookups.
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
remap_inverse_map(map_type map,                                    //
                  coordinate_type const *__restrict__ coordinates, //
                  index_type *__restrict__ inverse_map,            //
                  size_type const num_threads,                     //
                  size_type const coordinate_size) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    auto result = map.find(
        coordinate<coordinate_type>{&coordinates[x * coordinate_size]});
    inverse_map[x] = result->second;
  }
}

/*
 * Field quantization: round the batch column, floor-align spatial columns.
 */
template <typename coordinate_field_type, typename coordinate_int_type,
          typename index_type, bool stride_one>
__global__ void quantize_coordinates_kernel(
    coordinate_field_type const *__restrict__ p_tfield, //
    coordinate_int_type *__restrict__ p_stensor,        //
    index_type const *__restrict__ p_tensor_stride,     //
    index_type const num_threads, index_type const coordinate_size) {
  extern __shared__ index_type sh_tensor_stride[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  if (stride_one) {
    if (x < num_threads) {
      if (x % coordinate_size == 0)
        p_stensor[x] = lrint(p_tfield[x]);
      else
        p_stensor[x] = floor(p_tfield[x]);
    }
  } else {
    for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
      sh_tensor_stride[i] = p_tensor_stride[i];
    }

    __syncthreads();

    if (x < num_threads) {
      if (x % coordinate_size == 0)
        p_stensor[x] = lrint(p_tfield[x]);
      else {
        index_type curr_tensor_stride =
            sh_tensor_stride[((x - 1) % coordinate_size)];
        p_stensor[x] =
            floor(p_tfield[x] / curr_tensor_stride) * curr_tensor_stride;
      }
    }
  }
}

} // namespace detail

template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator>
void CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                           TemplatedAllocator>::
    quantize_coordinates(coordinate_int_type *d_dst_coordinates,
                         stride_type const &tensor_stride) const {
  int64_t const stride_prod = std::accumulate(
      tensor_stride.begin(), tensor_stride.end(), (int64_t)1,
      std::multiplies<>());

  index_type *d_tensor_stride = reinterpret_cast<index_type *>(
      m_byte_allocator.allocate(m_coordinate_size * sizeof(index_type)));
  CUDA_CHECK(cudaMemcpy(d_tensor_stride, tensor_stride.data(),
                        sizeof(index_type) * m_coordinate_size,
                        cudaMemcpyHostToDevice));

  size_type const num_threads = size() * m_coordinate_size;
  auto const num_blocks = GET_BLOCKS(num_threads, CUDA_NUM_THREADS);

  if (stride_prod == 1) {
    detail::quantize_coordinates_kernel<coordinate_field_type,
                                        coordinate_int_type, index_type, true>
        <<<num_blocks, CUDA_NUM_THREADS,
           m_coordinate_size * sizeof(index_type)>>>(
            const_coordinate_data(), d_dst_coordinates, d_tensor_stride,
            num_threads, m_coordinate_size);
  } else {
    detail::quantize_coordinates_kernel<coordinate_field_type,
                                        coordinate_int_type, index_type, false>
        <<<num_blocks, CUDA_NUM_THREADS,
           m_coordinate_size * sizeof(index_type)>>>(
            const_coordinate_data(), d_dst_coordinates, d_tensor_stride,
            num_threads, m_coordinate_size);
  }
}

/*
 * Insert a device coordinate range into the hash map, compacting the valid
 * row/bucket indices and optionally rebuilding the inverse map.
 */
template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
template <bool remap>
void CoordinateMapGPU<coordinate_type, TemplatedAllocator>::insert(
    coordinate_iterator<coordinate_type> key_first,
    coordinate_iterator<coordinate_type> key_last) {
  size_type const N = key_last - key_first;
  LOG_DEBUG("key iterator length", N);

  if (N == 0) {
    m_size = 0;
    return;
  }

  m_valid_row_index.allocate(N);
  m_valid_map_index.allocate(N);

  base_type::reserve(N);
  CUDA_CHECK(
      cudaMemcpy(coordinate_data(), key_first->data(),
                 sizeof(coordinate_type) * N * m_coordinate_size,
                 cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  LOG_DEBUG("Reserved and copied", N, "x", m_coordinate_size, "coordinates");

  size_type const num_threads = N;
  size_type const num_blocks = GET_BLOCKS(num_threads, CUDA_NUM_THREADS);
  index_type const unused_key = std::numeric_limits<index_type>::max();

  detail::insert_and_map_kernel<coordinate_type, size_type, index_type,
                                map_type><<<num_blocks, CUDA_NUM_THREADS>>>(
      *m_map,                   //
      const_coordinate_data(),  //
      m_valid_map_index.data(), //
      m_valid_row_index.data(), //
      num_threads, m_coordinate_size, unused_key);
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Map size:", m_map->size());

  // drop failed insertions
  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(m_valid_map_index.begin(), m_valid_row_index.begin()));

  size_type const number_of_valid =
      thrust::remove_if(thrust::device, valid_begin,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            m_valid_map_index.end(), m_valid_row_index.end())),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;

  m_valid_row_index.resize(number_of_valid);
  m_valid_map_index.resize(number_of_valid);
  m_size = number_of_valid;
  LOG_DEBUG("Number of successful insertion", m_size);

  if (remap && number_of_valid != N) {
    m_inverse_row_index.allocate(N);
    thrust::counting_iterator<uint32_t> count_begin{0};
    thrust::for_each(count_begin, count_begin + number_of_valid,
                     detail::update_value_with_offset<index_type, map_type>{
                         *m_map, m_valid_map_index.data()});

    size_type const remap_threads = N;
    auto const remap_blocks = GET_BLOCKS(remap_threads, CUDA_NUM_THREADS);

    detail::remap_inverse_map<coordinate_type, size_type, index_type, map_type>
        <<<remap_blocks, CUDA_NUM_THREADS>>>(*m_map,                     //
                                             const_coordinate_data(),    //
                                             m_inverse_row_index.data(), //
                                             remap_threads, m_coordinate_size);

    LOG_DEBUG("Remapping finished");
  }
}

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
template <bool remap>
std::pair<gpu_storage<default_types::index_type, TemplatedAllocator<char>>,
          gpu_storage<default_types::index_type, TemplatedAllocator<char>>>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::insert_and_map(
    coordinate_iterator<coordinate_type> key_first,
    coordinate_iterator<coordinate_type> key_last) {
  LOG_DEBUG("insert_and_map");
  insert<remap>(key_first, key_last);
  return std::make_pair(m_valid_row_index, m_inverse_row_index);
}

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
void CoordinateMapGPU<coordinate_type, TemplatedAllocator>::
    initialize_valid_indices(size_t const N_unique) {
  m_valid_row_index.resize(N_unique);
  m_valid_map_index.resize(N_unique);
  m_size = N_unique;

  auto insert = detail::insert_coordinate<coordinate_type, map_type,
                                          index_type *>{
      *m_map,                   //
      const_coordinate_data(),  //
      m_valid_row_index.data(), //
      m_valid_map_index.data(), //
      m_coordinate_size};

  thrust::counting_iterator<uint32_t> count_begin{0};
  thrust::for_each(thrust::device, count_begin, count_begin + N_unique, insert);
}

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
std::pair<gpu_storage<default_types::index_type, TemplatedAllocator<char>>,
          gpu_storage<default_types::index_type, TemplatedAllocator<char>>>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::find(
    coordinate_iterator<coordinate_type> key_first,
    coordinate_iterator<coordinate_type> key_last) const {
  size_type N = key_last - key_first;

  LOG_DEBUG(N, "queries for find.");
  auto const find_functor = detail::find_coordinate<coordinate_type, map_type>(
      *m_map, key_first->data(), m_unused_element, m_coordinate_size);
  auto const invalid_functor =
      detail::is_unused_pair<coordinate_type, mapped_type>(m_unused_element);

  thrust::counting_iterator<index_type> index{0};
  gpu_storage<index_type, byte_allocator_type> input_index(N);
  gpu_storage<index_type, byte_allocator_type> results(N);
  thrust::sequence(thrust::device, input_index.begin(), input_index.end());
  thrust::transform(thrust::device, index, index + N, results.begin(),
                    find_functor);

  size_type const number_of_valid =
      thrust::remove_if(thrust::device,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            input_index.begin(), results.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            input_index.end(), results.end())),
                        invalid_functor) -
      thrust::make_zip_iterator(
          thrust::make_tuple(input_index.begin(), results.begin()));
  LOG_DEBUG("Number of valid", number_of_valid);
  input_index.resize(number_of_valid);
  results.resize(number_of_valid);

  return std::make_pair(input_index, results);
}

namespace detail {

// floor-align every valid source row onto the strided lattice
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type>
__global__ void
stride_copy(coordinate_type const *__restrict__ src_coordinates, //
            index_type const *__restrict__ src_valid_row_index,  //
            index_type const *__restrict__ stride,               //
            coordinate_type *__restrict__ dst_coordinates,       //
            size_type const num_threads, size_type const coordinate_size) {
  extern __shared__ size_type sh_stride[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x)
    sh_stride[i] = stride[i];

  __syncthreads();

  if (x < num_threads) {
    const index_type src_start = src_valid_row_index[x] * coordinate_size;
    const index_type dst_start = x * coordinate_size;
    dst_coordinates[dst_start] = src_coordinates[src_start];
    for (index_type j = 1; j < coordinate_size; ++j) {
      dst_coordinates[dst_start + j] =
          (__float2int_rd(
              __fdiv_rd(src_coordinates[src_start + j], sh_stride[j - 1]))) *
          sh_stride[j - 1];
    }
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::stride(
    stride_type const &stride) const {
  size_type const N = size();
  LOG_DEBUG("Strided map with kernel stride:", stride);

  self_type stride_map(
      N, m_coordinate_size, m_hashtable_occupancy,
      detail::stride_tensor_stride(base_type::m_tensor_stride, stride),
      m_map_allocator, base_type::m_byte_allocator);

  index_storage_type out_device_tensor_stride(stride_map.get_tensor_stride());

  size_type const num_threads = N;
  auto const num_blocks = GET_BLOCKS(num_threads, CUDA_NUM_THREADS);

  detail::stride_copy<coordinate_type, size_type, index_type>
      <<<num_blocks, CUDA_NUM_THREADS,
         m_coordinate_size * sizeof(size_type)>>>(
          const_coordinate_data(),           //
          m_valid_row_index.cbegin(),        //
          out_device_tensor_stride.cbegin(), //
          stride_map.coordinate_data(),      //
          num_threads, m_coordinate_size);

  LOG_DEBUG("Stride copy done.");
  auto &stride_valid_row_index = stride_map.m_valid_row_index;
  auto &stride_valid_map_index = stride_map.m_valid_map_index;

  stride_valid_row_index.resize(N);
  stride_valid_map_index.resize(N);

  index_type const unused_key = std::numeric_limits<index_type>::max();

  detail::insert_and_map_kernel<coordinate_type, size_type, index_type,
                                map_type><<<num_blocks, CUDA_NUM_THREADS>>>(
      *stride_map.m_map,                  //
      stride_map.const_coordinate_data(), //
      stride_valid_map_index.data(),      //
      stride_valid_row_index.data(),      //
      num_threads, m_coordinate_size, unused_key);
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Stride map insertion complete");

  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(stride_valid_map_index.begin(), //
                         stride_valid_row_index.begin()));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, //
                        valid_begin,    //
                        thrust::make_zip_iterator(
                            thrust::make_tuple(stride_valid_map_index.end(), //
                                               stride_valid_row_index.end())),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;
  stride_valid_row_index.resize(number_of_valid);
  stride_valid_map_index.resize(number_of_valid);
  stride_map.m_size = number_of_valid;
  LOG_DEBUG("Reduced to", number_of_valid);

  // compact remapping of the surviving entries
  thrust::counting_iterator<uint32_t> count_begin{0};
  thrust::for_each(count_begin, count_begin + number_of_valid,
                   detail::update_value_with_offset<index_type, map_type>{
                       *stride_map.m_map, stride_map.m_valid_map_index.data()});

  LOG_DEBUG("Stride remap done");

  return stride_map;
}

namespace detail {

template <typename coordinate_type, typename index_type>
__device__ bool is_coordinate_aligned(coordinate_type *point,
                                      index_type *out_tensor_stride,
                                      uint32_t const size) {
  for (uint32_t i = 0; i < size - 1; ++i) {
    if (point[i + 1] % out_tensor_stride[i] != 0)
      return false;
  }
  return true;
}

/*
 * Generative expansion: every input point emits all kernel-region neighbors.
 * Forward (non-transpose) expansion keeps only lattice-aligned points.
 */
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void kernel_region_insert(
    size_type const num_threads,                                //
    map_type out_map,                                           //
    coordinate_type const *const __restrict__ p_in_coordinates, //
    index_type const *const __restrict__ in_valid_row_index,    //
    coordinate_type *__restrict__ p_out_coordinates,            //
    index_type *__restrict__ out_valid_row_index,               //
    index_type *__restrict__ out_valid_map_index,               //
    gpu_kernel_region<coordinate_type> kernel,                  //
    size_type const *const __restrict__ out_tensor_stride,      //
    index_type const unused_key) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  size_type const coordinate_size = kernel.coordinate_size();
  size_type const volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride     = sh_size;
  size_type *sh_kernel_size       = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation          = sh_kernel_size   + coordinate_size;
  size_type *sh_out_tensor_stride = sh_dilation      + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_out_tensor_stride + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  // clang-format on

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
    sh_out_tensor_stride[i] = out_tensor_stride[i];
  }

  __syncthreads();

  auto sh_kernel = gpu_kernel_region<coordinate_type>(
      kernel, sh_tensor_stride, sh_kernel_size, sh_dilation);

  coordinate<coordinate_type> curr_coordinate(sh_tmp);
  if (x < num_threads) {
    index_type out_index = x * volume;
    for (uint32_t kernel_ind = 0; kernel_ind < volume; ++kernel_ind) {
      sh_kernel.coordinate_at(
          kernel_ind,
          &p_in_coordinates[in_valid_row_index[x] * coordinate_size], sh_tmp);

      if (kernel.is_transpose()) {
        for (uint32_t i = 0; i < coordinate_size; ++i)
          p_out_coordinates[out_index * coordinate_size + i] =
              curr_coordinate[i];

        auto const result = out_map.insert(thrust::make_pair(
            coordinate<coordinate_type>{
                &p_out_coordinates[out_index * coordinate_size]},
            out_index));

        if (result.second) {
          out_valid_row_index[out_index] = out_index;
          out_valid_map_index[out_index] = result.first.offset();
        } else {
          out_valid_row_index[out_index] = unused_key;
        }
        ++out_index;
      } else {
        if (!is_coordinate_aligned(sh_tmp, sh_out_tensor_stride,
                                   coordinate_size)) {
          out_valid_row_index[out_index] = unused_key;
          ++out_index;
        } else {
          for (uint32_t i = 0; i < coordinate_size; ++i)
            p_out_coordinates[out_index * coordinate_size + i] =
                curr_coordinate[i];

          auto const result = out_map.insert(thrust::make_pair(
              coordinate<coordinate_type>{
                  &p_out_coordinates[out_index * coordinate_size]},
              out_index));

          if (result.second) {
            out_valid_row_index[out_index] = out_index;
            out_valid_map_index[out_index] = result.first.offset();
          } else {
            out_valid_row_index[out_index] = unused_key;
          }
          ++out_index;
        }
      }
    }
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::stride_region(
    cpu_kernel_region<coordinate_type> &kernel,
    stride_type const &out_tensor_stride) const {
  ASSERT(m_coordinate_size == kernel.coordinate_size(),
         "Invalid kernel coordinate_size");
  gpu_kernel_region<coordinate_type> gpu_kernel(kernel.to_gpu());
  size_type const N_in = size();
  size_type const N_out = N_in * kernel.volume();

  LOG_DEBUG("Stride region out tensor stride:", out_tensor_stride,
            "with capacity:", N_out);
  self_type stride_map(N_out, m_coordinate_size, m_hashtable_occupancy,
                       out_tensor_stride, m_map_allocator,
                       base_type::m_byte_allocator);

  index_storage_type d_out_tensor_stride(out_tensor_stride);

  auto &out_valid_row_index = stride_map.m_valid_row_index;
  auto &out_valid_map_index = stride_map.m_valid_map_index;

  out_valid_row_index.resize(N_out);
  out_valid_map_index.resize(N_out);

  index_type const unused_key = std::numeric_limits<index_type>::max();
  uint32_t const shared_memory_size_in_bytes =
      4 * m_coordinate_size * sizeof(index_type) +
      CUDA_NUM_THREADS * m_coordinate_size * sizeof(coordinate_type);

  detail::kernel_region_insert<coordinate_type, size_type, index_type,
                               map_type>
      <<<GET_BLOCKS(N_in, CUDA_NUM_THREADS), CUDA_NUM_THREADS,
         shared_memory_size_in_bytes>>>(N_in,                         //
                                        *stride_map.m_map,            //
                                        const_coordinate_data(),      //
                                        m_valid_row_index.cbegin(),   //
                                        stride_map.coordinate_data(), //
                                        out_valid_row_index.data(),   //
                                        out_valid_map_index.data(),   //
                                        gpu_kernel,                   //
                                        d_out_tensor_stride.cbegin(), //
                                        unused_key);                  //
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("kernel_region_insert done");

  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(out_valid_row_index.begin(), //
                         out_valid_map_index.begin()));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, //
                        valid_begin,    //
                        thrust::make_zip_iterator(
                            thrust::make_tuple(out_valid_row_index.end(), //
                                               out_valid_map_index.end())),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;
  out_valid_row_index.resize(number_of_valid);
  out_valid_map_index.resize(number_of_valid);
  stride_map.m_size = number_of_valid;
  LOG_DEBUG("Reduced to", number_of_valid);

  thrust::counting_iterator<index_type> count_begin{0};
  thrust::for_each(count_begin, count_begin + number_of_valid,
                   detail::update_value_with_offset<index_type, map_type>{
                       *stride_map.m_map, out_valid_map_index.data()});
  LOG_DEBUG("Stride remap done");
  return stride_map;
}

namespace detail {

template <typename dst_coordinate_type, typename src_coordinate_type,
          typename size_type, typename index_type, bool stride_src>
__global__ void copy_column_with_valid(
    dst_coordinate_type *__restrict__ dst_coordinates,       //
    size_type const num_threads,                             //
    src_coordinate_type const *__restrict__ src_coordinates, //
    index_type const *__restrict__ src_valid_row_index,      //
    size_type const coordinate_size) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    if (stride_src)
      dst_coordinates[x] =
          src_coordinates[src_valid_row_index[x] * coordinate_size];
    else
      dst_coordinates[x * coordinate_size] =
          src_coordinates[src_valid_row_index[x]];
  }
}

template <typename dst_coordinate_type, typename src_coordinate_type,
          typename size_type, bool stride_src>
__global__ void
copy_column(dst_coordinate_type *__restrict__ dst_coordinates,       //
            size_type const num_threads,                             //
            src_coordinate_type const *__restrict__ src_coordinates, //
            size_type const coordinate_size) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    if (stride_src)
      dst_coordinates[x] = src_coordinates[x * coordinate_size];
    else
      dst_coordinates[x * coordinate_size] = src_coordinates[x];
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::origin() const {
  size_type const N = size();
  LOG_DEBUG("Origin map from in map size:", N);

  stride_type origin_tensor_stride(m_coordinate_size - 1);
  std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                [](auto &i) { i = 0; });

  // collect, sort, and unique the batch indices
  coordinate_type *d_batch_indices = reinterpret_cast<coordinate_type *>(
      m_byte_allocator.allocate(N * sizeof(coordinate_type)));
  detail::copy_column_with_valid<coordinate_type, coordinate_type, size_type,
                                 index_type, true>
      <<<GET_BLOCKS(N, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          d_batch_indices, N, const_coordinate_data(),
          m_valid_row_index.cbegin(), m_coordinate_size);

  thrust::sort(thrust::device, d_batch_indices, d_batch_indices + N);
  auto d_batch_indices_end =
      thrust::unique(thrust::device, d_batch_indices, d_batch_indices + N);
  size_type const N_unique = d_batch_indices_end - d_batch_indices;

  LOG_DEBUG("Origin map with size:", N_unique,
            " tensor stride:", origin_tensor_stride);
  self_type origin_map(N_unique, m_coordinate_size, m_hashtable_occupancy,
                       origin_tensor_stride, m_map_allocator,
                       base_type::m_byte_allocator);
  CUDA_CHECK(
      cudaMemset(origin_map.coordinate_data(), 0,
                 N_unique * m_coordinate_size * sizeof(coordinate_type)));

  detail::copy_column<coordinate_type, coordinate_type, size_type, false>
      <<<GET_BLOCKS(N_unique, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          origin_map.coordinate_data(), N_unique, d_batch_indices,
          m_coordinate_size);

  auto &origin_valid_row_index = origin_map.m_valid_row_index;
  auto &origin_valid_map_index = origin_map.m_valid_map_index;

  origin_valid_row_index.resize(N_unique);
  origin_valid_map_index.resize(N_unique);
  origin_map.m_size = N_unique;

  auto insert = detail::insert_coordinate<coordinate_type, map_type,
                                          index_type *>{
      *origin_map.m_map,                  //
      origin_map.const_coordinate_data(), //
      origin_valid_row_index.data(),      //
      origin_valid_map_index.data(),      //
      m_coordinate_size};

  thrust::counting_iterator<uint32_t> count_begin{0};
  thrust::for_each(thrust::device, count_begin, count_begin + N_unique,
                   insert);

  m_byte_allocator.deallocate((char *)d_batch_indices,
                              N * sizeof(coordinate_type));

  return origin_map;
}

template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_int_type, TemplatedAllocator>
CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                      TemplatedAllocator>::origin() const {
  size_type const N = size();
  LOG_DEBUG("Origin map from in map size:", N);

  stride_type origin_tensor_stride(m_coordinate_size - 1);
  std::for_each(origin_tensor_stride.begin(), origin_tensor_stride.end(),
                [](auto &i) { i = 0; });

  coordinate_int_type *d_batch_indices =
      reinterpret_cast<coordinate_int_type *>(
          m_byte_allocator.allocate(N * sizeof(coordinate_int_type)));

  detail::copy_column<coordinate_int_type, coordinate_field_type, size_type,
                      true>
      <<<GET_BLOCKS(N, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          d_batch_indices, N, const_coordinate_data(), m_coordinate_size);

  thrust::sort(thrust::device, d_batch_indices, d_batch_indices + N);
  auto d_batch_indices_end =
      thrust::unique(thrust::device, d_batch_indices, d_batch_indices + N);
  size_type const N_unique = d_batch_indices_end - d_batch_indices;

  LOG_DEBUG("Origin map with size:", N_unique,
            " tensor stride:", origin_tensor_stride);
  CoordinateMapGPU<coordinate_int_type, TemplatedAllocator> origin_map(
      N_unique, m_coordinate_size, 50, origin_tensor_stride);

  CUDA_CHECK(
      cudaMemset(origin_map.coordinate_data(), 0,
                 N_unique * m_coordinate_size * sizeof(coordinate_int_type)));

  detail::copy_column<coordinate_int_type, coordinate_int_type, size_type,
                      false>
      <<<GET_BLOCKS(N_unique, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          origin_map.coordinate_data(), N_unique, d_batch_indices,
          m_coordinate_size);

  m_byte_allocator.deallocate((char *)d_batch_indices,
                              N * sizeof(coordinate_int_type));

  origin_map.initialize_valid_indices(N_unique);

  return origin_map;
}

namespace detail {

// map every field row to its batch row in the origin map
template <typename coordinate_field_type, //
          typename coordinate_int_type,   //
          typename size_type,             //
          typename index_type,            //
          typename map_type>
__global__ void origin_field_map_kernel(
    size_type const num_threads,                              //
    coordinate_field_type const *__restrict__ d_field_coords, //
    map_type const origin_map,                                //
    index_type *__restrict__ p_in_maps,                       //
    index_type *__restrict__ p_out_maps,                      //
    index_type *__restrict__ p_kernels,                       //
    size_type const coordinate_size) {
  extern __shared__ coordinate_int_type sh_int_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  coordinate_int_type *sh_tmp = sh_int_all + tx * coordinate_size;

  if (x < num_threads)
    for (index_type i = 0; i < coordinate_size; ++i)
      sh_tmp[i] = 0;

  __syncthreads();

  if (x < num_threads) {
    sh_tmp[0] =
        coordinate_int_type(lroundf(d_field_coords[x * coordinate_size]));
    auto origin_iter = origin_map.find(coordinate<coordinate_int_type>(sh_tmp));
    auto out_index = origin_iter->second;
    p_in_maps[x] = x;
    p_out_maps[x] = out_index;
    // kernel index doubles as the batch row for decompose()
    p_kernels[x] = out_index;
  }
}

} // namespace detail

template <typename coordinate_field_type, typename coordinate_int_type,
          template <typename T> class TemplatedAllocator>
typename CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                               TemplatedAllocator>::kernel_map_type
CoordinateFieldMapGPU<coordinate_field_type, coordinate_int_type,
                      TemplatedAllocator>::
    origin_map(CoordinateMapGPU<coordinate_int_type, TemplatedAllocator> const
                   &origin_map,
               uint32_t thread_dim) const {
  ASSERT(std::all_of(origin_map.get_tensor_stride().begin(),
                     origin_map.get_tensor_stride().end(),
                     [](auto const &i) { return i == 0; }),
         "Invalid origin tensor stride", origin_map.get_tensor_stride());

  size_type const in_size = size();
  LOG_DEBUG("in_map size:", in_size, "origin_map size:", origin_map.size());
  uint32_t const shared_memory_size_in_bytes =
      thread_dim * m_coordinate_size * sizeof(coordinate_int_type);
  size_type const num_threads = in_size;
  auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);

  kernel_map_type kernel_map(in_size, base_type::m_byte_allocator);
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Allocated kernel_map.");

  detail::origin_field_map_kernel<coordinate_field_type, coordinate_int_type,
                                  size_type, index_type, int_hash_map_type>
      <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
          num_threads,                 //
          const_coordinate_data(),     //
          origin_map.const_hash_map(), //
          kernel_map.in_maps.begin(),  //
          kernel_map.out_maps.begin(), //
          kernel_map.kernels.begin(),  //
          m_coordinate_size);

  CUDA_CHECK(cudaStreamSynchronize(0));
  THRUST_CHECK(kernel_map.decompose());
  LOG_DEBUG("origin map decomposed");

  return kernel_map;
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void prune_copy_and_insert(
    size_type const num_threads,                              //
    size_type const coordinate_size,                          //
    index_type const unused_map_offset,                       //
    index_type const *const __restrict__ in_valid_row_index,  //
    coordinate_type const *const __restrict__ in_coordinates, //
    bool const *const __restrict__ keep_begin,                //
    index_type const *const __restrict__ inclusive_scan_keep, //
    map_type out_map,                                         //
    coordinate_type *__restrict__ out_coordinates,            //
    index_type *__restrict__ out_valid_row_index,             //
    index_type *__restrict__ out_valid_map_offset             //
) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    if (!keep_begin[x]) {
      out_valid_map_offset[x] = unused_map_offset;
    } else {
      auto out_row_index = (x < 1) ? 0 : inclusive_scan_keep[x - 1];
      coordinate_type const *curr_in_coord =
          &in_coordinates[in_valid_row_index[x] * coordinate_size];
      coordinate_type *curr_out_coord =
          &out_coordinates[out_row_index * coordinate_size];
      for (index_type i = 0; i < coordinate_size; ++i)
        curr_out_coord[i] = curr_in_coord[i];

      auto coord = coordinate<coordinate_type>{curr_out_coord};
      // value remapped in the follow-up kernel
      auto result = out_map.insert(thrust::make_pair(coord, 0));
      out_valid_row_index[x] = out_row_index;
      if (result.second)
        out_valid_map_offset[x] = result.first.offset();
      else
        out_valid_map_offset[x] = unused_map_offset;
    }
  }
}

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void remap(size_type const num_threads,                  //
                      map_type const out_map,                       //
                      index_type *__restrict__ out_valid_map_offset //
) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    auto &pair = out_map.data()[out_valid_map_offset[x]];
    pair.second = x;
  }
}

template <typename Dtype, typename Stype>
__global__ void typed_copy(uint32_t const num_threads,   //
                           Dtype *__restrict__ dst,      //
                           Stype const *__restrict__ src //
) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    dst[x] = src[x];
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::prune(
    bool const *keep_begin, bool const *keep_end) const {
  size_type const N = size();
  ASSERT(N == keep_end - keep_begin, "Invalid keep size");
  LOG_DEBUG("Prune size:", N);

  // inclusive scan determines each surviving row's output slot
  auto const inclusive_scan_size = N * sizeof(index_type);
  index_type *d_inclusive_scan =
      (index_type *)m_byte_allocator.allocate(inclusive_scan_size);
  detail::typed_copy<<<GET_BLOCKS(N, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
      N, d_inclusive_scan, keep_begin);
  CUDA_CHECK(cudaStreamSynchronize(0));
  thrust::inclusive_scan(thrust::device, d_inclusive_scan, d_inclusive_scan + N,
                         d_inclusive_scan);
  index_type N_pruned;
  CUDA_CHECK(cudaMemcpy(&N_pruned, d_inclusive_scan + N - 1, sizeof(index_type),
                        cudaMemcpyDeviceToHost));
  LOG_DEBUG("Pruned N:", N_pruned);

  self_type pruned_map(N, m_coordinate_size, m_hashtable_occupancy,
                       base_type::m_tensor_stride, m_map_allocator,
                       base_type::m_byte_allocator);

  auto &out_valid_map_offset = pruned_map.m_valid_map_index;
  auto &out_valid_row_index = pruned_map.m_valid_row_index;
  out_valid_map_offset.resize(N);
  out_valid_row_index.resize(N);

  index_type const unused_map_offset = std::numeric_limits<index_type>::max();
  detail::prune_copy_and_insert<coordinate_type, size_type, index_type,
                                map_type>
      <<<GET_BLOCKS(N, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          N, m_coordinate_size, unused_map_offset, m_valid_row_index.cbegin(),
          const_coordinate_data(), keep_begin, d_inclusive_scan,
          *(pruned_map.m_map), pruned_map.coordinate_data(),
          out_valid_row_index.data(), out_valid_map_offset.data());
  CUDA_CHECK(cudaStreamSynchronize(0));

  LOG_DEBUG("Pruned hash map size:", pruned_map.size());
  auto valid_begin = thrust::make_zip_iterator(thrust::make_tuple(
      out_valid_map_offset.begin(), out_valid_row_index.begin()));
  size_type const number_of_valid =
      thrust::remove_if(
          thrust::device, valid_begin,
          thrust::make_zip_iterator(thrust::make_tuple(
              out_valid_map_offset.end(), out_valid_row_index.end())),
          detail::is_first<index_type>(unused_map_offset)) -
      valid_begin;

  LOG_DEBUG("number of valid rows:", number_of_valid);
  out_valid_map_offset.resize(number_of_valid);
  out_valid_row_index.resize(number_of_valid);
  pruned_map.m_size = number_of_valid;

  detail::remap<coordinate_type, size_type, index_type, map_type>
      <<<GET_BLOCKS(number_of_valid, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          number_of_valid, *(pruned_map.m_map), out_valid_map_offset.data());
  CUDA_CHECK(cudaStreamSynchronize(0));

  m_byte_allocator.deallocate((char *)d_inclusive_scan, inclusive_scan_size);

  return pruned_map;
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void copy_coordinates_by_offset(
    map_type map,                               //
    coordinate_type *__restrict__ coordinates,  //
    index_type const *__restrict__ map_offsets, //
    size_type const num_threads,                //
    size_type const coordinate_size             //
) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    typename map_type::value_type const *p_value = map.data() + map_offsets[x];
    coordinate_type *dst_coordinate =
        coordinates + p_value->second * coordinate_size;
    for (index_type i = 0; i < coordinate_size; ++i)
      dst_coordinate[i] = p_value->first[i];
  }
}

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void copy_coordinates_by_valid_row(
    coordinate_type const *__restrict__ in_coordinates, //
    coordinate_type *__restrict__ out_coordinates,      //
    index_type const *__restrict__ valid_row,           //
    size_type const num_threads,                        //
    size_type const coordinate_size                     //
) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    index_type const row_index = x / coordinate_size;
    index_type const col_index = x % coordinate_size;
    out_coordinates[row_index * coordinate_size + col_index] =
        in_coordinates[valid_row[row_index] * coordinate_size + col_index];
  }
}

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void insert_and_map_kernel_with_offset(
    map_type map,                                    //
    coordinate_type const *__restrict__ coordinates, //
    index_type const coordinate_row_offset,          //
    index_type *__restrict__ valid_map_index,        //
    index_type *__restrict__ valid_row_index,        //
    size_type const num_threads,                     //
    size_type const coordinate_size,                 //
    index_type const unused_key) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    auto const result = map.insert(thrust::make_pair(
        coordinate<coordinate_type>{&coordinates[x * coordinate_size]}, x));

    if (result.second) {
      valid_row_index[x] = x + coordinate_row_offset;
      valid_map_index[x] = result.first.offset();
    } else {
      valid_map_index[x] = unused_key;
    }
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::merge(
    std::vector<std::reference_wrapper<self_type>> const &maps) const {
  size_t all_size = std::accumulate(
      maps.begin(), maps.end(), (size_t)0,
      [](size_t sum, const self_type &map) { return sum + map.size(); });
  LOG_DEBUG("Out merge map capacity:", all_size);
  self_type merged_map(all_size, m_coordinate_size, m_hashtable_occupancy,
                       base_type::m_tensor_stride, m_map_allocator,
                       base_type::m_byte_allocator);

  merged_map.m_valid_row_index.resize(all_size);
  merged_map.m_valid_map_index.resize(all_size);

  coordinate_type *curr_coordinates = merged_map.coordinate_data();
  index_type *curr_valid_map_offset = merged_map.m_valid_map_index.data();
  index_type *curr_valid_row_index = merged_map.m_valid_row_index.data();
  index_type const unused_key = std::numeric_limits<index_type>::max();
  index_type row_offset{0};
  for (self_type const &map : maps) {
    size_type const num_threads = map.size();
    if (num_threads == 0)
      continue;
    size_type const num_blocks =
        GET_BLOCKS(num_threads * m_coordinate_size, CUDA_NUM_THREADS);
    LOG_DEBUG("Current merge map size:", num_threads);
    detail::copy_coordinates_by_valid_row<coordinate_type, size_type,
                                          index_type, map_type>
        <<<num_blocks, CUDA_NUM_THREADS>>>(map.const_coordinate_data(),     //
                                           curr_coordinates,                //
                                           map.m_valid_row_index.cdata(),   //
                                           num_threads * m_coordinate_size, //
                                           m_coordinate_size);

    detail::insert_and_map_kernel_with_offset<coordinate_type, size_type,
                                              index_type, map_type>
        <<<num_blocks, CUDA_NUM_THREADS>>>(*(merged_map.m_map),
                                           curr_coordinates,      //
                                           row_offset,            //
                                           curr_valid_map_offset, //
                                           curr_valid_row_index,  //
                                           num_threads, m_coordinate_size,
                                           unused_key);
    CUDA_CHECK(cudaStreamSynchronize(0));

    curr_coordinates += num_threads * m_coordinate_size;
    curr_valid_map_offset += num_threads;
    curr_valid_row_index += num_threads;
    row_offset += num_threads;
  }

  auto valid_begin = thrust::make_zip_iterator(
      thrust::make_tuple(merged_map.m_valid_map_index.begin(),
                         merged_map.m_valid_row_index.begin()));

  size_type const number_of_valid =
      thrust::remove_if(thrust::device, valid_begin,
                        thrust::make_zip_iterator(thrust::make_tuple(
                            merged_map.m_valid_map_index.end(),
                            merged_map.m_valid_row_index.end())),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;

  detail::remap<coordinate_type, size_type, index_type, map_type>
      <<<GET_BLOCKS(number_of_valid, CUDA_NUM_THREADS), CUDA_NUM_THREADS>>>(
          number_of_valid, *(merged_map.m_map),
          merged_map.m_valid_map_index.data());

  merged_map.m_valid_row_index.resize(number_of_valid);
  merged_map.m_valid_map_index.resize(number_of_valid);
  merged_map.m_size = number_of_valid;

  return merged_map;
}

namespace detail {

// memory-efficient pass 1: count hits per output row
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
count_kernel(map_type const in_map,                                     //
             map_type const out_map,                                    //
             index_type const *const __restrict__ out_valid_map_index,  //
             size_type const num_threads,                               //
             gpu_kernel_region<coordinate_type> kernel,                 //
             index_type *__restrict__ p_count_per_thread) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  size_type const coordinate_size = kernel.coordinate_size();
  size_type const volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride = sh_size;
  size_type *sh_kernel_size   = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation      = sh_kernel_size   + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_dilation + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  // clang-format on

  auto const equal = out_map.get_key_equal();

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
  }

  __syncthreads();

  auto sh_kernel = gpu_kernel_region<coordinate_type>(
      kernel, sh_tensor_stride, sh_kernel_size, sh_dilation);

  coordinate<coordinate_type> point(sh_tmp);
  auto const unused_key = out_map.get_unused_key();
  if (x < num_threads) {
    size_type count = 0;
    typename map_type::value_type const &out_value =
        out_map.data()[out_valid_map_index[x]];
    if (!equal(out_value.first, unused_key)) {
      for (uint32_t kernel_ind = 0; kernel_ind < volume; ++kernel_ind) {
        sh_kernel.coordinate_at(kernel_ind, out_value.first.data(), sh_tmp);
        if (in_map.find(point) != in_map.end()) {
          ++count;
        }
      }
    }
    p_count_per_thread[x] = count;
  }
}

// memory-efficient pass 2: fill the preallocated triplets
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void preallocated_kernel_map_iteration(
    map_type const in_map,                                                  //
    map_type const out_map,                                                 //
    index_type const *const __restrict__ out_valid_map_index,               //
    size_type const num_threads,                                            //
    gpu_kernel_region<coordinate_type> kernel,                              //
    index_type const *const __restrict__ inclusive_count_cumsum_per_thread, //
    index_type *__restrict__ p_kernels,                                     //
    index_type *__restrict__ p_in_maps,                                     //
    index_type *__restrict__ p_out_maps) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  size_type const coordinate_size = kernel.coordinate_size();
  size_type const volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride = sh_size;
  size_type *sh_kernel_size   = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation      = sh_kernel_size   + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_dilation + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  // clang-format on

  auto const equal = out_map.get_key_equal();

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
  }

  __syncthreads();

  auto sh_kernel = gpu_kernel_region<coordinate_type>(
      kernel, sh_tensor_stride, sh_kernel_size, sh_dilation);
  coordinate<coordinate_type> curr_coordinate(sh_tmp);
  auto const unused_key = out_map.get_unused_key();
  if (x < num_threads) {
    auto kernel_map_index =
        (x < 1) ? 0 : inclusive_count_cumsum_per_thread[x - 1];
    typename map_type::value_type const &out_value =
        out_map.data()[out_valid_map_index[x]];
    if (!equal(out_value.first, unused_key)) {
      for (uint32_t kernel_index = 0; kernel_index < volume; ++kernel_index) {
        sh_kernel.coordinate_at(kernel_index, out_value.first.data(), sh_tmp);
        auto const &in_result = in_map.find(curr_coordinate);
        if (in_result != in_map.end()) {
          p_kernels[kernel_map_index] = kernel_index;
          p_in_maps[kernel_map_index] = (*in_result).second;
          p_out_maps[kernel_map_index] = out_value.second;
          ++kernel_map_index;
        }
      }
    }
  }
}

// volume-1 fast path: a pure in/out correspondence
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
direct_in_out_map(size_type const num_threads,                               //
                  map_type const in_map,                                     //
                  map_type const out_map,                                    //
                  index_type const *const __restrict__ out_valid_map_offset, //
                  index_type *__restrict__ p_in_maps,                        //
                  index_type *__restrict__ p_out_maps,
                  index_type const unused_key) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    typename map_type::value_type const &out_value =
        out_map.data()[out_valid_map_offset[x]];
    auto const &result = in_map.find(out_value.first);
    if (result != in_map.end()) {
      p_in_maps[x] = (*result).second;
      p_out_maps[x] = out_value.second;
    } else {
      p_in_maps[x] = unused_key;
    }
  }
}

// speed-optimized: one thread per (output row, kernel offset) pair
template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
direct_kernel_map(map_type const in_map,                                    //
                  map_type const out_map,                                   //
                  index_type const *const __restrict__ out_valid_map_index, //
                  size_type const num_threads,                              //
                  gpu_kernel_region<coordinate_type> kernel,                //
                  index_type *__restrict__ p_kernels,                       //
                  index_type *__restrict__ p_in_maps,                       //
                  index_type *__restrict__ p_out_maps,
                  index_type const unused_map_value) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  size_type const coordinate_size = kernel.coordinate_size();
  size_type const volume = kernel.volume();

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_tensor_stride = sh_size;
  size_type *sh_kernel_size   = sh_tensor_stride + coordinate_size;
  size_type *sh_dilation      = sh_kernel_size   + coordinate_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_dilation + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  // clang-format on

  auto const equal = out_map.get_key_equal();

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = kernel.tensor_stride()[i];
    sh_kernel_size[i] = kernel.kernel_size()[i];
    sh_dilation[i] = kernel.dilation()[i];
  }

  __syncthreads();

  auto sh_kernel = gpu_kernel_region<coordinate_type>(
      kernel, sh_tensor_stride, sh_kernel_size, sh_dilation);

  auto const unused_key = out_map.get_unused_key();
  if (x < num_threads) {
    index_type kernel_index = x % volume;
    typename map_type::value_type const &out_value =
        out_map.data()[out_valid_map_index[x / volume]];
    if (!equal(out_value.first, unused_key)) {
      sh_kernel.coordinate_at(kernel_index, out_value.first.data(), sh_tmp);
      auto const &in_result = in_map.find(coordinate<coordinate_type>(sh_tmp));
      if (in_result != in_map.end()) {
        p_kernels[x] = kernel_index;
        p_in_maps[x] = (*in_result).second;
        p_out_maps[x] = out_value.second;
      } else {
        p_kernels[x] = unused_map_value;
      }
    }
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
typename CoordinateMapGPU<coordinate_type, TemplatedAllocator>::kernel_map_type
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::kernel_map(
    self_type const &out_map, gpu_kernel_region<coordinate_type> const &kernel,
    CUDAKernelMapMode::Mode kernel_map_mode, uint32_t thread_dim) const {
  size_type const out_size = out_map.size();
  size_type const kernel_volume = kernel.volume();
  ASSERT(kernel_volume > 0, "Invalid kernel");

  if (kernel_volume == 1 && kernel.region_type() != RegionType::CUSTOM) {
    auto const N = out_size;

    LOG_DEBUG("out_map size:", N);
    index_type *in_out_map = (index_type *)base_type::m_byte_allocator.allocate(
        2 * (N + 1) * sizeof(index_type));
    index_type *ins = in_out_map;
    // +1 gap avoids __restrict__ aliasing between the two slices
    index_type *outs = in_out_map + N + 1;

    index_type unused_key = std::numeric_limits<index_type>::max();
    detail::direct_in_out_map<coordinate_type, size_type, index_type, map_type>
        <<<GET_BLOCKS(N, thread_dim), thread_dim>>>(
            N, *m_map,                         //
            *(out_map.m_map),                  //
            out_map.m_valid_map_index.cdata(), //
            ins,                               //
            outs,                              //
            unused_key);

    LOG_DEBUG("Direct in out map copy done");
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(ins, outs));
    auto const valid_size =
        thrust::remove_if(
            thrust::device, begin,
            thrust::make_zip_iterator(thrust::make_tuple(ins + N, outs + N)),
            detail::is_first<index_type>(unused_key)) -
        begin;
    LOG_DEBUG("Valid size:", valid_size);

    kernel_map_type kernel_map(valid_size, base_type::m_byte_allocator, false);
    CUDA_CHECK(cudaMemcpy(kernel_map.in_maps.data(), ins,
                          valid_size * sizeof(index_type),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kernel_map.out_maps.data(), outs,
                          valid_size * sizeof(index_type),
                          cudaMemcpyDeviceToDevice));

    base_type::m_byte_allocator.deallocate((char *)in_out_map,
                                           2 * (N + 1) * sizeof(index_type));
    LOG_DEBUG("Cleaning up");
    return kernel_map;
  } else if (kernel_map_mode == CUDAKernelMapMode::MEMORY_EFFICIENT) {
    uint32_t const shared_memory_size_in_bytes =
        3 * m_coordinate_size * sizeof(index_type) +
        thread_dim * m_coordinate_size * sizeof(coordinate_type);
    size_type const num_threads = out_size;
    auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);

    LOG_DEBUG("num block", num_blocks);
    LOG_DEBUG("shared_memory size", shared_memory_size_in_bytes);

    index_type *d_p_count_per_thread =
        reinterpret_cast<index_type *>(base_type::m_byte_allocator.allocate(
            num_threads * sizeof(index_type)));

    detail::count_kernel<coordinate_type, size_type, index_type, map_type>
        <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
            *m_map,                             //
            *out_map.m_map,                     //
            out_map.m_valid_map_index.cbegin(), //
            num_threads,                        //
            kernel,                             //
            d_p_count_per_thread);
    CUDA_CHECK(cudaStreamSynchronize(0));
    LOG_DEBUG("count_kernel finished");

    thrust::inclusive_scan(thrust::device, d_p_count_per_thread,
                           d_p_count_per_thread + num_threads,
                           d_p_count_per_thread);

    index_type num_kernel_map;
    CUDA_CHECK(cudaMemcpy(&num_kernel_map,
                          d_p_count_per_thread + num_threads - 1,
                          sizeof(index_type), cudaMemcpyDeviceToHost));

    LOG_DEBUG("Found", num_kernel_map, "kernel map elements.");

    kernel_map_type kernel_map(num_kernel_map, base_type::m_byte_allocator);
    CUDA_CHECK(cudaStreamSynchronize(0));
    LOG_DEBUG("Allocated kernel_map.");

    detail::preallocated_kernel_map_iteration<coordinate_type, size_type,
                                              index_type, map_type>
        <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
            *m_map,                             //
            *out_map.m_map,                     //
            out_map.m_valid_map_index.cbegin(), //
            num_threads,                        //
            kernel,                             //
            d_p_count_per_thread,               //
            kernel_map.kernels.begin(),         //
            kernel_map.in_maps.begin(),         //
            kernel_map.out_maps.begin());

    CUDA_CHECK(cudaStreamSynchronize(0));
    LOG_DEBUG("Preallocated kernel map done");

    THRUST_CHECK(kernel_map.decompose());
    base_type::m_byte_allocator.deallocate(
        reinterpret_cast<char *>(d_p_count_per_thread),
        num_threads * sizeof(index_type));
    LOG_DEBUG("cudaFree");

    return kernel_map;
  } else if (kernel_map_mode == CUDAKernelMapMode::SPEED_OPTIMIZED) {
    uint32_t const shared_memory_size_in_bytes =
        3 * m_coordinate_size * sizeof(index_type) +
        (thread_dim + (thread_dim + kernel_volume - 1) / kernel_volume) *
            m_coordinate_size * sizeof(coordinate_type);
    size_type const num_threads = out_size * kernel_volume;
    auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);

    LOG_DEBUG("num block", num_blocks);
    LOG_DEBUG("kernel_volume", kernel_volume);
    LOG_DEBUG("shared_memory size", shared_memory_size_in_bytes);

    index_type unused_map_value = std::numeric_limits<index_type>::max();

    index_type *d_p_valid_in_index =
        reinterpret_cast<index_type *>(base_type::m_byte_allocator.allocate(
            3 * (num_threads + 1) * sizeof(index_type)));
    index_type *d_p_valid_out_index = d_p_valid_in_index + num_threads + 1;
    index_type *d_p_valid_kernel_index = d_p_valid_out_index + num_threads + 1;

    detail::direct_kernel_map<coordinate_type, size_type, index_type, map_type>
        <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
            *m_map,                             //
            *out_map.m_map,                     //
            out_map.m_valid_map_index.cbegin(), //
            num_threads,                        //
            kernel,                             //
            d_p_valid_kernel_index,             //
            d_p_valid_in_index,                 //
            d_p_valid_out_index,                //
            unused_map_value);
    CUDA_CHECK(cudaStreamSynchronize(0));
    LOG_DEBUG("direct_kernel_map finished");

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        d_p_valid_kernel_index, d_p_valid_in_index, d_p_valid_out_index));
    auto const valid_size =
        thrust::remove_if(thrust::device, begin,
                          thrust::make_zip_iterator(thrust::make_tuple(
                              d_p_valid_kernel_index + num_threads,
                              d_p_valid_in_index + num_threads,
                              d_p_valid_out_index + num_threads)),
                          detail::is_first<index_type>(unused_map_value)) -
        begin;
    LOG_DEBUG("Valid size:", valid_size);

    kernel_map_type kernel_map(valid_size, base_type::m_byte_allocator);
    CUDA_CHECK(cudaMemcpy(kernel_map.kernels.data(), d_p_valid_kernel_index,
                          valid_size * sizeof(index_type),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kernel_map.in_maps.data(), d_p_valid_in_index,
                          valid_size * sizeof(index_type),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kernel_map.out_maps.data(), d_p_valid_out_index,
                          valid_size * sizeof(index_type),
                          cudaMemcpyDeviceToDevice));
    THRUST_CHECK(kernel_map.decompose());

    base_type::m_byte_allocator.deallocate(
        reinterpret_cast<char *>(d_p_valid_in_index),
        3 * (num_threads + 1) * sizeof(index_type));
    LOG_DEBUG("cudaFree");

    return kernel_map;

  } else {
    ASSERT(false, "Not implemented");
  }
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
stride_map_kernel(map_type const in_map,                                   //
                  map_type const out_map,                                  //
                  index_type const *const __restrict__ in_valid_map_index, //
                  size_type const num_threads,                             //
                  index_type const *const __restrict__ stride,             //
                  index_type *__restrict__ p_in_maps,                      //
                  index_type *__restrict__ p_out_maps,
                  size_type const coordinate_size,
                  index_type const unused_key) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  // clang-format off
  size_type *sh_size = reinterpret_cast<size_type *>(sh_all);

  size_type *sh_stride = sh_size;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_size + coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  // clang-format on

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_stride[i] = stride[i];
  }

  __syncthreads();

  if (x >= num_threads)
    return;

  typename map_type::value_type const &in_value =
      in_map.data()[in_valid_map_index[x]];

  sh_tmp[0] = in_value.first[0];
  for (index_type j = 1; j < coordinate_size; ++j) {
    sh_tmp[j] =
        (__float2int_rd(__fdiv_rd(in_value.first[j], sh_stride[j - 1]))) *
        sh_stride[j - 1];
  }

  auto out_iter = out_map.find(coordinate<coordinate_type>(sh_tmp));
  if (out_iter == out_map.end()) {
    p_in_maps[x] = unused_key;
  } else {
    p_in_maps[x] = in_value.second;
    p_out_maps[x] = out_iter->second;
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
typename CoordinateMapGPU<coordinate_type, TemplatedAllocator>::kernel_map_type
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::stride_map(
    self_type const &out_map, stride_type const &out_tensor_stride,
    uint32_t thread_dim) const {
  LOG_DEBUG("generating stride_map from stride", base_type::m_tensor_stride,
            "to", out_map.get_tensor_stride());
  size_type const in_size = size();
  index_storage_type d_out_tensor_stride(out_tensor_stride);

  index_type unused_key = std::numeric_limits<index_type>::max();
  uint32_t const shared_memory_size_in_bytes =
      m_coordinate_size * sizeof(index_type) +
      thread_dim * m_coordinate_size * sizeof(coordinate_type);
  size_type const num_threads = in_size;
  auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);

  LOG_DEBUG("num block", num_blocks);
  LOG_DEBUG("shared_memory size", shared_memory_size_in_bytes);

  index_type *in_out_map = (index_type *)base_type::m_byte_allocator.allocate(
      2 * (in_size + 1) * sizeof(index_type));
  index_type *ins = in_out_map;
  index_type *outs = in_out_map + in_size + 1;

  LOG_DEBUG("Allocated temporary memory");
  detail::stride_map_kernel<coordinate_type, size_type, index_type, map_type>
      <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
          *m_map,                       //
          *out_map.m_map,               //
          m_valid_map_index.cbegin(),   //
          num_threads,                  //
          d_out_tensor_stride.cbegin(), //
          ins,                          //
          outs,                         //
          m_coordinate_size,            //
          unused_key);

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(ins, outs));
  auto const valid_size =
      thrust::remove_if(thrust::device, begin,
                        thrust::make_zip_iterator(
                            thrust::make_tuple(ins + in_size, outs + in_size)),
                        detail::is_first<index_type>(unused_key)) -
      begin;

  LOG_DEBUG("Valid size:", valid_size);
  kernel_map_type kernel_map(valid_size, base_type::m_byte_allocator, false);
  CUDA_CHECK(cudaMemcpy(kernel_map.in_maps.data(), ins,
                        valid_size * sizeof(index_type),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(kernel_map.out_maps.data(), outs,
                        valid_size * sizeof(index_type),
                        cudaMemcpyDeviceToDevice));
  base_type::m_byte_allocator.deallocate(
      (char *)in_out_map, 2 * (in_size + 1) * sizeof(index_type));

  return kernel_map;
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename map_type>
__global__ void
origin_map_kernel(map_type const in_map,                                   //
                  map_type const origin_map,                               //
                  index_type const *const __restrict__ in_valid_map_index, //
                  size_type const num_threads,                             //
                  index_type *__restrict__ p_in_maps,                      //
                  index_type *__restrict__ p_out_maps,
                  index_type *__restrict__ p_kernels,
                  size_type const coordinate_size) {
  extern __shared__ coordinate_type sh_all[];

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  coordinate_type *sh_tmp = sh_all + tx * coordinate_size;

  if (x < num_threads)
    for (index_type i = 0; i < coordinate_size; ++i)
      sh_tmp[i] = 0;

  __syncthreads();

  if (x < num_threads) {
    typename map_type::value_type const &in_value =
        in_map.data()[in_valid_map_index[x]];

    sh_tmp[0] = in_value.first[0];
    auto origin_iter = origin_map.find(coordinate<coordinate_type>(sh_tmp));

    p_in_maps[x] = in_value.second;
    p_out_maps[x] = origin_iter->second;
    // batch row doubles as the kernel index for decompose()
    p_kernels[x] = origin_iter->second;
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
typename CoordinateMapGPU<coordinate_type, TemplatedAllocator>::kernel_map_type
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::origin_map(
    self_type const &origin_map, uint32_t thread_dim) const {
  ASSERT(std::all_of(origin_map.get_tensor_stride().begin(),
                     origin_map.get_tensor_stride().end(),
                     [](auto const &i) { return i == 0; }),
         "Invalid origin tensor stride", origin_map.get_tensor_stride());

  size_type const in_size = size();
  LOG_DEBUG("in_map size:", in_size, "origin_map size:", origin_map.size());
  uint32_t const shared_memory_size_in_bytes =
      thread_dim * m_coordinate_size * sizeof(coordinate_type);
  size_type const num_threads = in_size;
  auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);

  kernel_map_type kernel_map(in_size, base_type::m_byte_allocator);
  CUDA_CHECK(cudaStreamSynchronize(0));
  LOG_DEBUG("Allocated kernel_map.");

  detail::origin_map_kernel<coordinate_type, size_type, index_type, map_type>
      <<<num_blocks, thread_dim, shared_memory_size_in_bytes>>>(
          *m_map,                      //
          *origin_map.m_map,           //
          m_valid_map_index.cbegin(),  //
          num_threads,                 //
          kernel_map.in_maps.begin(),  //
          kernel_map.out_maps.begin(), //
          kernel_map.kernels.begin(),  //
          m_coordinate_size);

  CUDA_CHECK(cudaStreamSynchronize(0));
  THRUST_CHECK(kernel_map.decompose());
  LOG_DEBUG("origin map decomposed");

  return kernel_map;
}

namespace detail {

/*
 * Interpolation: thread x handles neighbor (x % 2^D) of field row
 * (x / 2^D); corners present in the sparse map emit a multilinear weight.
 */
template <typename coordinate_type,
          typename index_type,  //
          typename stride_type, //
          typename float_type,  //
          typename map_type>
__global__ void
interpolation_kernel(map_type in_map,                                  //
                     index_type const num_threads,                     //
                     float_type const *__restrict__ p_tfield,          //
                     index_type *__restrict__ p_in_maps,               //
                     index_type *__restrict__ p_out_maps,              //
                     float_type *__restrict__ p_weights,               //
                     stride_type const *__restrict__ p_tensor_stride,  //
                     index_type const unused_map_value,
                     index_type const coordinate_size,
                     index_type const neighbor_volume) {
  SharedMemory<float_type> shared;
  float_type *sh_all = shared.getPointer();

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  float_type *sh_tfield = sh_all + tx * coordinate_size;
  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(
      sh_all + CUDA_NUM_THREADS * coordinate_size);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  index_type *sh_tensor_stride = reinterpret_cast<index_type *>(
      sh_coordinate + CUDA_NUM_THREADS * coordinate_size);

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = p_tensor_stride[i];
  }

  if (x < num_threads) {
    index_type const offset = coordinate_size * (x / neighbor_volume);
    for (index_type i = 0; i < coordinate_size; ++i) {
      sh_tfield[i] = p_tfield[offset + i];
    }
  }

  __syncthreads();

  if (x < num_threads) {
    uint32_t neighbor_ind = x % neighbor_volume;

    sh_tmp[0] = lrint(sh_tfield[0]);
    uint32_t mask = 1;
    for (uint32_t j = coordinate_size - 1; j > 0; --j) {
      index_type curr_tensor_stride = sh_tensor_stride[j - 1];
      if ((neighbor_ind & mask) == 0)
        sh_tmp[j] =
            floor(sh_tfield[j] / curr_tensor_stride) * curr_tensor_stride;
      else
        sh_tmp[j] =
            floor(sh_tfield[j] / curr_tensor_stride) * curr_tensor_stride +
            curr_tensor_stride;
      mask = mask << 1;
    }

    auto const &in_result = in_map.find(coordinate<coordinate_type>(sh_tmp));
    if (in_result != in_map.end()) {
      p_in_maps[x] = (*in_result).second;
      p_out_maps[x] = x / neighbor_volume;
      float_type weight = 1;
      for (uint32_t j = 1; j < coordinate_size; ++j) {
        weight *= 1 - abs(sh_tfield[j] - sh_tmp[j]) / sh_tensor_stride[j - 1];
      }
      p_weights[x] = weight;
    } else {
      p_in_maps[x] = unused_map_value;
    }
  }
}

template <typename coordinate_type,
          typename index_type,  //
          typename stride_type, //
          typename float_type,  //
          typename map_type>
__global__ void
field_map_kernel(map_type in_map,                                  //
                 index_type const num_threads,                     //
                 float_type const *__restrict__ p_tfield,          //
                 index_type *__restrict__ p_in_maps,               //
                 index_type *__restrict__ p_out_maps,              //
                 stride_type const *__restrict__ p_tensor_stride,  //
                 index_type const unused_map_value,
                 index_type const coordinate_size) {
  SharedMemory<float_type> shared;
  float_type *sh_all = shared.getPointer();

  auto const tx = threadIdx.x;
  auto const x = blockDim.x * blockIdx.x + tx;

  coordinate_type *sh_coordinate = reinterpret_cast<coordinate_type *>(sh_all);
  coordinate_type *sh_tmp = sh_coordinate + tx * coordinate_size;
  index_type *sh_tensor_stride = reinterpret_cast<index_type *>(
      sh_coordinate + CUDA_NUM_THREADS * coordinate_size);

  for (index_type i = tx; i < coordinate_size - 1; i += blockDim.x) {
    sh_tensor_stride[i] = p_tensor_stride[i];
  }

  __syncthreads();

  index_type const offset = coordinate_size * x;

  if (x < num_threads) {
    float_type const *curr_tfield = p_tfield + offset;

    sh_tmp[0] = lrint(curr_tfield[0]);
    for (uint32_t j = coordinate_size - 1; j > 0; --j) {
      index_type curr_tensor_stride = sh_tensor_stride[j - 1];
      sh_tmp[j] =
          floor(curr_tfield[j] / curr_tensor_stride) * curr_tensor_stride;
    }

    auto const &in_result = in_map.find(coordinate<coordinate_type>(sh_tmp));
    if (in_result != in_map.end()) {
      p_in_maps[x] = (*in_result).second;
      p_out_maps[x] = x;
    } else {
      p_in_maps[x] = unused_map_value;
    }
  }
}

template <typename coordinate_type, typename index_type, typename size_type,
          typename stride_type, typename field_type, typename map_type,
          typename ByteAllocatorType>
std::vector<at::Tensor> interpolation_map_weight_tfield_type(
    uint32_t const num_tfield,                //
    uint32_t const coordinate_size,           //
    index_type const unused_key,              //
    field_type const *const p_tfield,         //
    map_type &map,                            //
    stride_type const *const p_tensor_stride, //
    ByteAllocatorType const &byte_allocator,
    c10::TensorOptions tfield_options) {
  uint32_t const neighbor_volume = std::pow(2, (coordinate_size - 1));
  size_type num_threads = neighbor_volume * num_tfield;
  LOG_DEBUG("neighbor_volume:", neighbor_volume, "num_tfield:", num_tfield,
            "num_threads:", num_threads);

  index_type *d_in_map = reinterpret_cast<index_type *>(
      byte_allocator.allocate(num_threads * sizeof(index_type)));
  index_type *d_out_map = reinterpret_cast<index_type *>(
      byte_allocator.allocate(num_threads * sizeof(index_type)));
  field_type *d_weight = reinterpret_cast<field_type *>(
      byte_allocator.allocate(num_threads * sizeof(field_type)));

  size_type shared_memory_size_in_bytes =
      coordinate_size * CUDA_NUM_THREADS * sizeof(field_type) +
      coordinate_size * CUDA_NUM_THREADS * sizeof(coordinate_type) +
      coordinate_size * sizeof(index_type);
  LOG_DEBUG("Shared memory size:", shared_memory_size_in_bytes);
  interpolation_kernel<coordinate_type, index_type, stride_type, field_type,
                       map_type>
      <<<GET_BLOCKS(num_threads, CUDA_NUM_THREADS), CUDA_NUM_THREADS,
         shared_memory_size_in_bytes>>>(map,             //
                                        num_threads,     //
                                        p_tfield,        //
                                        d_in_map,        //
                                        d_out_map,       //
                                        d_weight,        //
                                        p_tensor_stride, //
                                        unused_key,      //
                                        coordinate_size, //
                                        neighbor_volume);

  auto valid_begin =
      thrust::make_zip_iterator(thrust::make_tuple(d_in_map, //
                                                   d_out_map, d_weight));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, //
                        valid_begin,    //
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_in_map + num_threads, //
                            d_out_map + num_threads, d_weight + num_threads)),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;
  LOG_DEBUG("number_of_valid:", number_of_valid);

  auto final_in_map =
      torch::empty({number_of_valid},
                   tfield_options.dtype(torch::kInt32).requires_grad(false));
  auto final_out_map =
      torch::empty({number_of_valid},
                   tfield_options.dtype(torch::kInt32).requires_grad(false));
  auto final_weights =
      torch::empty({number_of_valid}, tfield_options.requires_grad(false));

  if (number_of_valid > 0) {
    CUDA_CHECK(cudaMemcpy(final_in_map.template data_ptr<int32_t>(), d_in_map,
                          number_of_valid * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(final_out_map.template data_ptr<int32_t>(),
                          d_out_map, number_of_valid * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(final_weights.template data_ptr<field_type>(),
                          d_weight, number_of_valid * sizeof(field_type),
                          cudaMemcpyDeviceToDevice));
  }

  byte_allocator.deallocate((char *)d_in_map, num_threads * sizeof(index_type));
  byte_allocator.deallocate((char *)d_out_map,
                            num_threads * sizeof(index_type));
  byte_allocator.deallocate((char *)d_weight, num_threads * sizeof(field_type));

  return {final_in_map, final_out_map, final_weights};
}

template <typename coordinate_type, typename index_type, typename size_type,
          typename stride_type, typename field_type, typename map_type,
          typename ByteAllocatorType>
std::pair<at::Tensor, at::Tensor>
field_map_type(uint32_t const num_tfield,                //
               uint32_t const coordinate_size,           //
               index_type const unused_key,              //
               field_type const *const p_tfield,         //
               map_type &map,                            //
               stride_type const *const p_tensor_stride, //
               ByteAllocatorType const &byte_allocator) {
  size_type num_threads = num_tfield;
  LOG_DEBUG("num_threads:", num_threads);

  index_type *d_in_map = reinterpret_cast<index_type *>(
      byte_allocator.allocate(num_threads * sizeof(index_type)));
  index_type *d_out_map = reinterpret_cast<index_type *>(
      byte_allocator.allocate(num_threads * sizeof(index_type)));

  size_type shared_memory_size_in_bytes =
      coordinate_size * CUDA_NUM_THREADS * sizeof(coordinate_type) +
      coordinate_size * sizeof(index_type);
  LOG_DEBUG("Shared memory size:", shared_memory_size_in_bytes);
  field_map_kernel<coordinate_type, index_type, stride_type, field_type,
                   map_type>
      <<<GET_BLOCKS(num_threads, CUDA_NUM_THREADS), CUDA_NUM_THREADS,
         shared_memory_size_in_bytes>>>(map,             //
                                        num_threads,     //
                                        p_tfield,        //
                                        d_in_map,        //
                                        d_out_map,       //
                                        p_tensor_stride, //
                                        unused_key,      //
                                        coordinate_size);

  auto valid_begin =
      thrust::make_zip_iterator(thrust::make_tuple(d_in_map, d_out_map));
  size_type const number_of_valid =
      thrust::remove_if(thrust::device, //
                        valid_begin,    //
                        thrust::make_zip_iterator(
                            thrust::make_tuple(d_in_map + num_threads, //
                                               d_out_map + num_threads)),
                        detail::is_first<index_type>(unused_key)) -
      valid_begin;
  LOG_DEBUG("number_of_valid:", number_of_valid);

  auto curr_device = at::cuda::current_device();

  auto tfield_options = torch::TensorOptions({at::kCUDA, curr_device})
                            .dtype(torch::kInt32)
                            .requires_grad(false);

  auto final_in_map = torch::empty({number_of_valid}, tfield_options);
  auto final_out_map = torch::empty({number_of_valid}, tfield_options);

  if (number_of_valid > 0) {
    CUDA_CHECK(cudaMemcpy(final_in_map.template data_ptr<int32_t>(), d_in_map,
                          number_of_valid * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(final_out_map.template data_ptr<int32_t>(),
                          d_out_map, number_of_valid * sizeof(int32_t),
                          cudaMemcpyDeviceToDevice));
  }

  byte_allocator.deallocate((char *)d_in_map, num_threads * sizeof(index_type));
  byte_allocator.deallocate((char *)d_out_map,
                            num_threads * sizeof(index_type));

  return {final_in_map, final_out_map};
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
std::vector<at::Tensor>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::interpolation_map_weight(
    at::Tensor const &tfield) const {
  ASSERT(tfield.dim() == 2, "Invalid tfield dimension");
  ASSERT(tfield.size(1) == m_coordinate_size, "Invalid tfield size");

  size_type const num_tfield = tfield.size(0);
  index_type const unused_key = std::numeric_limits<index_type>::max();

  LOG_DEBUG("map size", m_size);

  switch (tfield.scalar_type()) {
  case at::ScalarType::Double:
    return detail::interpolation_map_weight_tfield_type<
        coordinate_type, index_type, size_type, index_type, double, map_type,
        TemplatedAllocator<char>>(num_tfield,                         //
                                  m_coordinate_size,                  //
                                  unused_key,                         //
                                  tfield.template data_ptr<double>(), //
                                  *m_map,                             //
                                  m_device_tensor_stride.cbegin(),    //
                                  m_byte_allocator,                   //
                                  tfield.options());
  case at::ScalarType::Float:
    return detail::interpolation_map_weight_tfield_type<
        coordinate_type, index_type, size_type, index_type, float, map_type,
        TemplatedAllocator<char>>(num_tfield,                        //
                                  m_coordinate_size,                 //
                                  unused_key,                        //
                                  tfield.template data_ptr<float>(), //
                                  *m_map,                            //
                                  m_device_tensor_stride.cbegin(),   //
                                  m_byte_allocator,                  //
                                  tfield.options());
  default:
    ASSERT(false, "Unsupported float type");
  }
}

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
template <typename coordinate_field_type>
std::pair<at::Tensor, at::Tensor>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::field_map(
    coordinate_field_type const *p_tfield, size_type const num_tfield) const {
  index_type const unused_key = std::numeric_limits<index_type>::max();

  LOG_DEBUG("map size", m_size);

  return detail::field_map_type<coordinate_type, index_type, size_type,
                                index_type, coordinate_field_type, map_type,
                                TemplatedAllocator<char>>(
      num_tfield,                      //
      m_coordinate_size,               //
      unused_key,                      //
      p_tfield,                        //
      *m_map,                          //
      m_device_tensor_stride.cbegin(), //
      m_byte_allocator);
}

namespace detail {

template <typename coordinate_type, //
          typename size_type,       //
          typename index_type,      //
          typename tensor_type,     //
          typename map_type>
__global__ void
union_map_kernel(size_type const num_threads,                             //
                 map_type const in_map,                                   //
                 map_type const union_map,                                //
                 index_type const *const __restrict__ in_valid_map_index, //
                 tensor_type *__restrict__ p_in_maps,                     //
                 tensor_type *__restrict__ p_union_maps,
                 size_type const coordinate_size) {
  auto const x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < num_threads) {
    typename map_type::value_type const &in_value =
        in_map.data()[in_valid_map_index[x]];

    auto union_iter = union_map.find(in_value.first);

    p_in_maps[x] = in_value.second;
    p_union_maps[x] = union_iter->second;
  }
}

} // namespace detail

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
std::vector<at::Tensor>
CoordinateMapGPU<coordinate_type, TemplatedAllocator>::union_map(
    std::vector<std::reference_wrapper<self_type>> const &in_maps,
    uint32_t thread_dim) const {
  auto options = torch::TensorOptions({at::kCUDA, at::cuda::current_device()})
                     .dtype(torch::kInt64)
                     .requires_grad(false);

  std::vector<at::Tensor> union_maps;
  for (self_type const &in_map : in_maps) {
    size_type const num_threads = in_map.m_valid_map_index.size();
    auto const num_blocks = GET_BLOCKS(num_threads, thread_dim);
    at::Tensor curr_map = torch::empty({2, num_threads}, options);
    LOG_DEBUG("in_map size", num_threads, ", num block", num_blocks,
              ", threads dim", thread_dim);

    int64_t *d_in_map = curr_map.template data_ptr<int64_t>();

    detail::union_map_kernel<coordinate_type, size_type, index_type, int64_t,
                             map_type>
        <<<num_blocks, thread_dim>>>(num_threads,                       //
                                     *in_map.m_map,                     //
                                     *m_map,                            //
                                     in_map.m_valid_map_index.cbegin(), //
                                     d_in_map,                          //
                                     d_in_map + num_threads,            //
                                     m_coordinate_size);

    CUDA_CHECK(cudaStreamSynchronize(0));
    union_maps.push_back(std::move(curr_map));
  }

  return union_maps;
}

template <typename coordinate_type,
          template <typename T> class TemplatedAllocator>
void CoordinateMapGPU<coordinate_type, TemplatedAllocator>::copy_coordinates(
    coordinate_type *dst_coordinate) const {
  size_type const num_threads = size();
  if (num_threads <= 0)
    return;

  size_type const num_blocks =
      GET_BLOCKS(num_threads * m_coordinate_size, CUDA_NUM_THREADS);
  detail::copy_coordinates_by_valid_row<coordinate_type, size_type, index_type,
                                        map_type>
      <<<num_blocks, CUDA_NUM_THREADS>>>(
          const_coordinate_data(),         //
          dst_coordinate,                  //
          m_valid_row_index.cbegin(),      //
          num_threads * m_coordinate_size, //
          m_coordinate_size);
}

/*
 * Explicit instantiations
 */
template class CoordinateFieldMapGPU<default_types::ccoordinate_type,
                                     default_types::dcoordinate_type,
                                     detail::default_allocator>;
template class CoordinateFieldMapGPU<default_types::ccoordinate_type,
                                     default_types::dcoordinate_type,
                                     detail::c10_allocator>;

template class CoordinateMapGPU<default_types::dcoordinate_type,
                                detail::default_allocator>;
template class CoordinateMapGPU<default_types::dcoordinate_type,
                                detail::c10_allocator>;

template std::pair<
    gpu_storage<default_types::index_type, detail::default_allocator<char>>,
    gpu_storage<default_types::index_type, detail::default_allocator<char>>>
CoordinateMapGPU<default_types::dcoordinate_type, detail::default_allocator>::
    insert_and_map<true>(
        coordinate_iterator<default_types::dcoordinate_type> key_first,
        coordinate_iterator<default_types::dcoordinate_type> key_last);

template std::pair<
    gpu_storage<default_types::index_type, detail::default_allocator<char>>,
    gpu_storage<default_types::index_type, detail::default_allocator<char>>>
CoordinateMapGPU<default_types::dcoordinate_type, detail::default_allocator>::
    insert_and_map<false>(
        coordinate_iterator<default_types::dcoordinate_type> key_first,
        coordinate_iterator<default_types::dcoordinate_type> key_last);

template std::pair<
    gpu_storage<default_types::index_type, detail::c10_allocator<char>>,
    gpu_storage<default_types::index_type, detail::c10_allocator<char>>>
CoordinateMapGPU<default_types::dcoordinate_type, detail::c10_allocator>::
    insert_and_map<true>(
        coordinate_iterator<default_types::dcoordinate_type> key_first,
        coordinate_iterator<default_types::dcoordinate_type> key_last);

template std::pair<
    gpu_storage<default_types::index_type, detail::c10_allocator<char>>,
    gpu_storage<default_types::index_type, detail::c10_allocator<char>>>
CoordinateMapGPU<default_types::dcoordinate_type, detail::c10_allocator>::
    insert_and_map<false>(
        coordinate_iterator<default_types::dcoordinate_type> key_first,
        coordinate_iterator<default_types::dcoordinate_type> key_last);

template std::pair<at::Tensor, at::Tensor>
CoordinateMapGPU<default_types::dcoordinate_type, detail::default_allocator>::
    field_map<float>(float const *p_tfield,
                     default_types::size_type const num_tfield) const;

template std::pair<at::Tensor, at::Tensor>
CoordinateMapGPU<default_types::dcoordinate_type, detail::c10_allocator>::
    field_map<float>(float const *p_tfield,
                     default_types::size_type const num_tfield) const;

} // namespace riemann
