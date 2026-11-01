#ifndef RIEMANN_ENGINE_GRID_STENCIL_HPP
#define RIEMANN_ENGINE_GRID_STENCIL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "engine/foundation/errors.hpp"
#include "engine/foundation/hashing.hpp"
#include "engine/foundation/types.hpp"

#ifndef CPU_ONLY
#include "engine/cuda/memory.cuh"
#include "engine/cuda/runtime.cuh"
#endif

namespace riemann {

namespace detail {

// Custom offsets stay in spatial lattice units; coordinate_at applies scaling.
at::Tensor normalize_region_offsets(const at::Tensor &offsets, int dimension);

/*
 * Stride arithmetic shared by every coordinate-map implementation. Spatial
 * components floor-align to the stride lattice; the batch column passes
 * through untouched.
 */
template <typename Itype>
inline void
stride_coordinate(const coordinate<Itype> &src, std::vector<Itype> &dst,
                  const default_types::stride_type &stride) noexcept {
  dst[0] = src[0];
  for (default_types::index_type i = 0; i < stride.size(); ++i) {
    dst[i + 1] = std::floor((float)src[i + 1] / stride[i]) * stride[i];
  }
}

template <typename Itype, typename stride_type>
inline void stride_coordinate(const coordinate<Itype> &src,
                              std::vector<Itype> &dst,
                              const stride_type stride) noexcept {
  dst[0] = src[0];
  for (default_types::index_type i = 0; i < dst.size() - 1; ++i) {
    dst[i + 1] = std::floor((float)src[i + 1] / stride[i]) * stride[i];
  }
}

// Forward stride multiplies; transpose divides and must divide exactly.
inline default_types::stride_type
stride_tensor_stride(const default_types::stride_type &tensor_stride,
                     const default_types::stride_type &stride,
                     bool is_transpose = false) {
  ASSERT(tensor_stride.size() == stride.size(), "stride size mismatch.");
  default_types::stride_type strided_tensor_stride{tensor_stride};
  if (is_transpose) {
    for (default_types::size_type i = 0; i < tensor_stride.size(); ++i) {
      ASSERT(strided_tensor_stride[i] % stride[i] == 0,
             "Invalid up stride on tensor stride:", tensor_stride,
             "kernel stride:", stride);
      strided_tensor_stride[i] /= stride[i];
    }
  } else {
    for (default_types::size_type i = 0; i < tensor_stride.size(); ++i)
      strided_tensor_stride[i] *= stride[i];
  }
  return strided_tensor_stride;
}

} // namespace detail

/*
 * Geometric description of a convolution/pooling neighborhood. Offsets are
 * enumerated on demand through coordinate_at; the object itself only points
 * at externally owned stride/size/dilation arrays.
 */
template <typename coordinate_type = default_types::dcoordinate_type>
class kernel_region {
public:
  using index_type = default_types::index_type;

public:
  class kernel_region_iterator {
  public:
    // clang-format off
    using self_type         = kernel_region_iterator;

    using iterator_category = std::forward_iterator_tag;
    using value_type        = coordinate<coordinate_type>;
    using difference_type   = int32_t;
    using pointer           = coordinate<coordinate_type>*;
    using reference         = coordinate<coordinate_type>&;
    // clang-format on

  public:
    kernel_region_iterator() = delete;
    // `tmp` provides scratch space for the dereferenced coordinate.
    RIEMANN_CUDA_HOST_DEVICE
    kernel_region_iterator(coordinate_type *tmp, kernel_region const &region,
                           index_type kernel_index = 0)
        : done(false), m_kernel_index(kernel_index), m_region{region},
          m_tmp{tmp}, m_coordinate(m_tmp) {
      if (m_tmp != nullptr) {
        for (index_type i = 0; i < m_region.m_coordinate_size; ++i) {
          m_tmp[i] = m_region.m_lb[i];
        }
      }
    }

    RIEMANN_CUDA_HOST_DEVICE inline reference operator*() noexcept {
      return m_coordinate;
    }
    RIEMANN_CUDA_HOST_DEVICE inline pointer operator->() noexcept {
      return &m_coordinate;
    }

    RIEMANN_CUDA_HOST_DEVICE inline self_type operator++() noexcept {
      // Odometer increment over spatial axes; axis 0 is the batch column and
      // never advances.
      for (index_type m_axis = 0;;) {
        m_tmp[m_axis + 1] +=
            m_region.m_dilation[m_axis] * m_region.m_tensor_stride[m_axis];
        if (m_tmp[m_axis + 1] <= m_region.m_ub[m_axis + 1]) {
          break;
        }
        m_tmp[m_axis + 1] = m_region.m_lb[m_axis + 1];
        ++m_axis;
        if (m_axis >= m_region.m_coordinate_size - 1) {
          done = true;
          break;
        }
      }
      return *this;
    }

    RIEMANN_CUDA_HOST_DEVICE inline bool
    operator!=(self_type const &other) const noexcept {
      return !done;
    }

  private:
    bool done;
    index_type m_kernel_index;
    kernel_region const &m_region;
    coordinate_type *m_tmp;
    coordinate<coordinate_type> m_coordinate;
  };

public:
  // clang-format off
  using size_type      = default_types::size_type;
  using stride_type    = default_types::stride_type;
  using iterator       = kernel_region_iterator;
  using const_iterator = const kernel_region_iterator;
  // clang-format on

public:
  kernel_region() = delete;
  RIEMANN_CUDA_HOST_DEVICE kernel_region(
      RegionType::Type region_type,
      size_type coordinate_size,      // spatial dimension + 1
      size_type const *tensor_stride, //
      size_type const *kernel_size,   //
      size_type const *dilation,      //
      size_type const volume = 0,     // 0 derives it from the region type
      coordinate_type const *p_offset = nullptr, // CUSTOM offsets
      uint32_t n_offset = 0,                     //
      bool is_transpose = false)
      : m_is_transpose{is_transpose}, m_region_type(region_type),
        m_coordinate_size{coordinate_size}, m_num_offset{n_offset},
        m_volume{volume}, m_tensor_stride{tensor_stride},
        m_kernel_size{kernel_size}, m_dilation{dilation}, m_offset{p_offset} {
    if (m_volume == 0)
      set_volume();
  }

  RIEMANN_CUDA_HOST_DEVICE iterator begin() { return iterator(m_tmp, *this); }
  RIEMANN_CUDA_HOST_DEVICE const_iterator cbegin() const {
    return const_iterator(m_tmp, *this);
  }
  RIEMANN_CUDA_HOST_DEVICE iterator end() { return iterator(nullptr, *this); }
  RIEMANN_CUDA_HOST_DEVICE kernel_region_iterator end() const {
    return const_iterator(nullptr, *this);
  }

  RIEMANN_CUDA_HOST_DEVICE RegionType::Type region_type() const {
    return m_region_type;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type volume() const { return m_volume; }
  RIEMANN_CUDA_HOST_DEVICE inline size_type coordinate_size() const {
    return m_coordinate_size;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type num_offset() const {
    return m_num_offset;
  }
  RIEMANN_CUDA_HOST_DEVICE inline coordinate_type const *offset() const {
    return m_offset;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *tensor_stride() const {
    return m_tensor_stride;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *kernel_size() const {
    return m_kernel_size;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *dilation() const {
    return m_dilation;
  }
  RIEMANN_CUDA_HOST_DEVICE inline bool is_transpose() const {
    return m_is_transpose;
  }

  /*
   * Write the kernel_index-th neighbor of src_coordinate into dst_coordinate.
   * Odd cube sizes center on zero; even sizes enumerate nonnegative offsets.
   * The cross region emits the center first, then one axis at a time.
   */
  RIEMANN_CUDA_HOST_DEVICE void
  coordinate_at(index_type kernel_index, coordinate_type const *src_coordinate,
                coordinate_type *dst_coordinate) const {
    dst_coordinate[0] = src_coordinate[0];
    switch (m_region_type) {
    case RegionType::HYPER_CUBE: {
      for (index_type i = 0; i < m_coordinate_size - 1; ++i) {
        auto const curr_kernel_size = m_kernel_size[i];
        auto const curr_kernel_index = kernel_index % curr_kernel_size;

        if (m_kernel_size[i] % 2 == 0) {
          dst_coordinate[i + 1] =
              src_coordinate[i + 1] +
              m_dilation[i] * m_tensor_stride[i] * curr_kernel_index;
        } else {
          dst_coordinate[i + 1] =
              src_coordinate[i + 1] +
              (curr_kernel_index - int(curr_kernel_size / 2)) * m_dilation[i] *
                  m_tensor_stride[i];
        }
        kernel_index /= curr_kernel_size;
      }
    } break;

    case RegionType::HYPER_CROSS: {
      for (index_type i = 1; i < m_coordinate_size; ++i) {
        dst_coordinate[i] = src_coordinate[i];
      }
      if (kernel_index == 0)
        break;
      auto ind = kernel_index - 1;
      index_type axis = 0;
      while (axis < m_coordinate_size - 1) {
        if (ind < m_kernel_size[axis] - 1)
          break;
        ind -= m_kernel_size[axis] - 1;
        axis += 1;
      }
      auto const r = (m_kernel_size[axis] - 1) / 2;
      coordinate_type coord_offset =
          (ind < r) ? (coordinate_type)(ind + 1) : (coordinate_type)(ind - 2 * r);
      dst_coordinate[axis + 1] +=
          coord_offset * m_dilation[axis] * m_tensor_stride[axis];
    } break;

    case RegionType::CUSTOM: {
      auto const *offset =
          m_offset + static_cast<size_t>(kernel_index) * (m_coordinate_size - 1);
      for (index_type i = 0; i < m_coordinate_size - 1; ++i) {
        dst_coordinate[i + 1] = static_cast<coordinate_type>(
            static_cast<int64_t>(src_coordinate[i + 1]) +
            static_cast<int64_t>(offset[i]) * m_tensor_stride[i] * m_dilation[i]);
      }
    } break;
    }
  }

private:
  RIEMANN_CUDA_HOST_DEVICE void set_volume() {
#ifndef __CUDA_ARCH__
    switch (m_region_type) {
    case RegionType::HYPER_CUBE:
      m_volume = 1;
      for (index_type i = 0; i < m_coordinate_size - 1; ++i)
        m_volume *= m_kernel_size[i];
      break;
    case RegionType::HYPER_CROSS:
      m_volume = 1;
      for (index_type i = 0; i < m_coordinate_size - 1; ++i)
        m_volume += (m_kernel_size[i] - 1);
      break;
    case RegionType::CUSTOM:
      m_volume = m_num_offset;
      break;
    };
#else
    // device-side copies always receive an initialized volume
#endif
  }

protected:
  bool m_is_transpose{false};

  RegionType::Type const m_region_type;
  size_type const m_coordinate_size;
  size_type m_num_offset, m_volume{0};

  size_type const *m_tensor_stride;
  size_type const *m_kernel_size;
  size_type const *m_dilation;

  coordinate_type const *m_offset;
  coordinate_type *m_lb;
  coordinate_type *m_ub;
  coordinate_type *m_tmp;
};

template <typename coordinate_type = default_types::dcoordinate_type>
class cpu_kernel_region : kernel_region<coordinate_type> {
public:
  using base_type = kernel_region<coordinate_type>;
  using self_type = cpu_kernel_region<coordinate_type>;
  using size_type = typename base_type::size_type;

public:
  cpu_kernel_region() = delete;
  cpu_kernel_region(
      RegionType::Type type,
      size_type coordinate_size,
      size_type const *tensor_stride,
      size_type const *kernel_size,
      size_type const *dilation,
      size_type const volume = 0,
      coordinate_type const *p_offset = nullptr,
      uint32_t n_offset = 0,
      bool is_transpose = false)
      : base_type{type,   coordinate_size, tensor_stride, kernel_size,
                  dilation, volume,        p_offset,      n_offset,
                  is_transpose} {}

  using base_type::begin;
  using base_type::cbegin;
  using base_type::end;

  using base_type::coordinate_at;
  using base_type::coordinate_size;
  using base_type::is_transpose;
  using base_type::num_offset;
  using base_type::offset;
  using base_type::region_type;
  using base_type::tensor_stride;
  using base_type::volume;

#ifndef CPU_ONLY
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *
  device_tensor_stride() const {
    return m_d_tensor_stride;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *device_kernel_size() const {
    return m_d_kernel_size;
  }
  RIEMANN_CUDA_HOST_DEVICE inline size_type const *device_dilation() const {
    return m_d_dilation;
  }
  RIEMANN_CUDA_HOST_DEVICE inline coordinate_type const *device_offset() const {
    return m_d_offset;
  }

  // Upload the stride/size/dilation (and custom offsets) arrays into one
  // contiguous device block. Idempotent.
  self_type const to_gpu() {
    LOG_DEBUG("to_gpu");
    if (m_on_gpu) {
      return *this;
    }

    size_type num_bytes = (m_coordinate_size - 1) * 3 * sizeof(size_type);
    if (m_region_type == RegionType::CUSTOM)
      num_bytes +=
          (m_coordinate_size - 1) * m_num_offset * sizeof(coordinate_type);

    void *p_tmp = std::malloc(num_bytes);
    size_type *p_size_type = reinterpret_cast<size_type *>(p_tmp);
    coordinate_type *p_coordinate_type = reinterpret_cast<coordinate_type *>(
        p_size_type + 3 * (m_coordinate_size - 1));

    std::copy_n(m_tensor_stride, m_coordinate_size - 1, &p_size_type[0]);
    std::copy_n(m_kernel_size, m_coordinate_size - 1,
                &p_size_type[m_coordinate_size - 1]);
    std::copy_n(m_dilation, m_coordinate_size - 1,
                &p_size_type[2 * (m_coordinate_size - 1)]);

    if (m_region_type == RegionType::CUSTOM) {
      std::copy_n(m_offset, m_num_offset * (m_coordinate_size - 1),
                  p_coordinate_type);
    }

    auto char_allocator = detail::c10_allocator<char>();
    m_d_space = char_allocator.shared_allocate(num_bytes);

    CUDA_CHECK(
        cudaMemcpy(m_d_space.get(), p_tmp, num_bytes, cudaMemcpyHostToDevice));
    // clang-format off
    size_type *size_d_space = reinterpret_cast<size_type*>(m_d_space.get());
    m_d_tensor_stride = size_d_space + 0 * (m_coordinate_size - 1);
    m_d_kernel_size   = size_d_space + 1 * (m_coordinate_size - 1);
    m_d_dilation      = size_d_space + 2 * (m_coordinate_size - 1);
    m_d_offset        = reinterpret_cast<coordinate_type*>(size_d_space + 3 * (m_coordinate_size - 1));
    // clang-format on

    m_on_gpu = true;
    std::free(p_tmp);
    return *this;
  }

  inline bool on_gpu() const { return m_on_gpu; }

#endif

protected:
  using base_type::m_coordinate_size;
  using base_type::m_num_offset;
  using base_type::m_region_type;
  using base_type::m_volume;

  using base_type::m_dilation;
  using base_type::m_kernel_size;
  using base_type::m_tensor_stride;

  using base_type::m_lb;
  using base_type::m_offset;
  using base_type::m_tmp;
  using base_type::m_ub;

  bool m_on_gpu{false};

  std::shared_ptr<char[]> m_d_space;

  size_type *m_d_tensor_stride;
  size_type *m_d_kernel_size;
  size_type *m_d_dilation;
  coordinate_type *m_d_offset;
};

/*
 * Region whose stride/size/dilation pointers live in device memory. Can be
 * constructed from an uploaded cpu_kernel_region or re-pointed at shared
 * memory copies inside a kernel.
 */
template <typename coordinate_type = default_types::dcoordinate_type>
class gpu_kernel_region : kernel_region<coordinate_type> {
public:
  using base_type = kernel_region<coordinate_type>;
  using size_type = typename base_type::size_type;

public:
  gpu_kernel_region() = delete;
#ifndef CPU_ONLY
  RIEMANN_CUDA_HOST_DEVICE
  gpu_kernel_region(cpu_kernel_region<coordinate_type> const &other)
      : base_type{other.region_type(),          other.coordinate_size(),
                  other.device_tensor_stride(), other.device_kernel_size(),
                  other.device_dilation(),      other.volume(),
                  other.device_offset(),        other.num_offset(),
                  other.is_transpose()} {}
#endif

  RIEMANN_CUDA_HOST_DEVICE gpu_kernel_region(
      gpu_kernel_region<coordinate_type> const &other,
      size_type const *device_tensor_stride,
      size_type const *device_kernel_size,
      size_type const *device_dilation,
      coordinate_type const *device_offset = nullptr)
      : base_type{other.region_type(),  other.coordinate_size(),
                  device_tensor_stride, device_kernel_size,
                  device_dilation,      other.volume(),
                  device_offset ? device_offset : other.offset(),
                  other.num_offset(),
                  other.is_transpose()} {}

  using base_type::begin;
  using base_type::cbegin;
  using base_type::end;

  using base_type::coordinate_at;
  using base_type::coordinate_size;
  using base_type::is_transpose;
  using base_type::num_offset;
  using base_type::offset;
  using base_type::region_type;
  using base_type::volume;

  using base_type::dilation;
  using base_type::kernel_size;
  using base_type::tensor_stride;

protected:
  using base_type::m_coordinate_size;
  using base_type::m_is_transpose;
  using base_type::m_num_offset;
  using base_type::m_region_type;
  using base_type::m_volume;

  using base_type::m_dilation;
  using base_type::m_kernel_size;
  using base_type::m_tensor_stride;

  using base_type::m_lb;
  using base_type::m_offset;
  using base_type::m_tmp;
  using base_type::m_ub;
};

/*
 * Hash of a tensor-stride vector; used to cache per-stride kernel parameters.
 */
struct StrideHash {
  std::size_t operator()(const stride_type &stride) const {
    std::size_t hash = 0;
    for (const auto &s : stride) {
      hash ^= std::hash<int>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

/*
 * Normalizes user-facing kernel hyper-parameters (scalar broadcast, region
 * validation, volume) and produces per-tensor-stride kernel parameters.
 */
class KernelGenerator {
public:
  struct KernelParams {
    stride_type kernel_size;
    stride_type kernel_stride;
    stride_type kernel_dilation;
    RegionType::Type region_type;
    at::Tensor region_offsets;
    bool expand_coordinates;
    int D;
  };

  KernelGenerator(
      torch::IntArrayRef kernel_size,
      torch::IntArrayRef stride = {1},
      torch::IntArrayRef dilation = {1},
      bool is_transpose = false,
      RegionType::Type region_type = RegionType::HYPER_CUBE,
      at::Tensor region_offsets = {},
      bool expand_coordinates = false,
      std::vector<RegionType::Type> axis_types = {},
      int dimension = -1);

  int kernel_volume() const { return kernel_volume_; }
  const stride_type &kernel_size() const { return kernel_size_; }
  const stride_type &kernel_stride() const { return kernel_stride_; }
  const stride_type &kernel_dilation() const { return kernel_dilation_; }
  RegionType::Type region_type() const { return region_type_; }
  at::Tensor region_offsets() const {
    return region_offsets_.defined() ? region_offsets_.clone() : at::Tensor{};
  }
  bool expand_coordinates() const { return expand_coordinates_; }
  bool requires_strided_coordinates() const;
  int dimension() const { return D_; }

  KernelParams get_kernel(const stride_type &tensor_stride,
                          bool is_transpose) const;

private:
  void initialize();
  std::tuple<RegionType::Type, at::Tensor, int> convert_region_type_impl(
      RegionType::Type region_type,
      const stride_type &tensor_stride,
      const stride_type &kernel_size,
      const stride_type &up_stride,
      const stride_type &dilation,
      const at::Tensor &region_offsets,
      const std::vector<RegionType::Type> &axis_types,
      int dimension,
      bool center = true) const;

  stride_type kernel_size_;
  stride_type kernel_stride_;
  stride_type kernel_dilation_;
  bool is_transpose_;
  RegionType::Type region_type_;
  at::Tensor region_offsets_;
  bool expand_coordinates_;
  std::vector<RegionType::Type> axis_types_;
  int D_;
  int kernel_volume_;
  bool requires_strided_coordinates_;

  mutable std::array<
      std::unordered_map<stride_type, KernelParams, StrideHash>, 2> cache_;
};

} // namespace riemann

#endif // RIEMANN_ENGINE_GRID_STENCIL_HPP
