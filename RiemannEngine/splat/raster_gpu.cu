#include "splat/gpu.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace riemann {
namespace gaussian {
namespace gpu {
namespace {

constexpr int kSide = kTileSide;
constexpr int kBatch = kSide * kSide; // Gaussians staged per shared batch
constexpr int kColumns = 10;          // mx my ca cb cc opacity depth r g b
constexpr float kAlphaFloor = 1.0f / 255.0f;
constexpr float kAlphaCeil = 0.99f;
constexpr float kTransmittanceFloor = 1e-4f;

// One depth-ordered batch of Gaussian attributes staged in shared memory.
struct Staged {
  float2 center[kBatch];
  float4 conic_alpha[kBatch]; // ca cb cc opacity
  float4 rgbd[kBatch];        // r g b depth
};

__device__ __forceinline__ void stage_batch(Staged &staged, const float *attrs,
                                            const int32_t *ids, int base,
                                            int end) {
  const int slot = threadIdx.y * kSide + threadIdx.x;
  const int item = base + slot;
  if (item < end) {
    const float *a = attrs + static_cast<int64_t>(ids[item]) * kColumns;
    staged.center[slot] = make_float2(a[0], a[1]);
    staged.conic_alpha[slot] = make_float4(a[2], a[3], a[4], a[5]);
    staged.rgbd[slot] = make_float4(a[7], a[8], a[9], a[6]);
  }
}

// Alpha of one pixel/Gaussian pair, or -1 if it does not contribute.
__device__ __forceinline__ float pair_alpha(const Staged &staged, int slot,
                                            float px, float py, float *dx,
                                            float *dy, float *falloff) {
  const float2 center = staged.center[slot];
  const float4 conic = staged.conic_alpha[slot];
  *dx = px - center.x;
  *dy = py - center.y;
  const float power =
      -0.5f * (conic.x * *dx * *dx + conic.z * *dy * *dy) - conic.y * *dx * *dy;
  if (power > 0.0f)
    return -1.0f;
  *falloff = __expf(power);
  const float alpha = fminf(kAlphaCeil, conic.w * *falloff);
  return alpha < kAlphaFloor ? -1.0f : alpha;
}

__global__ void forward_kernel(const float *attrs, const int32_t *ids,
                               const int32_t *ranges, const float *background,
                               int height, int width, int tiles_x, float *out,
                               float *final_t, int32_t *last_index) {
  __shared__ Staged staged;
  const int tile = blockIdx.y * tiles_x + blockIdx.x;
  const int px_i = blockIdx.x * kSide + threadIdx.x;
  const int py_i = blockIdx.y * kSide + threadIdx.y;
  const bool inside = px_i < width && py_i < height;
  const float px = px_i + 0.5f, py = py_i + 0.5f;
  const int begin = ranges[tile * 2], end = ranges[tile * 2 + 1];
  const int count = end - begin;

  float T = 1.0f, red = 0, green = 0, blue = 0, depth = 0;
  int last = 0;
  bool done = !inside;

  for (int base = 0; base < count; base += kBatch) {
    if (__syncthreads_count(done) == kBatch)
      break;
    stage_batch(staged, attrs, ids + begin, base, count);
    __syncthreads();
    const int n = min(kBatch, count - base);
    for (int j = 0; j < n && !done; ++j) {
      float dx, dy, falloff;
      const float alpha = pair_alpha(staged, j, px, py, &dx, &dy, &falloff);
      if (alpha < 0.0f)
        continue;
      const float next = T * (1.0f - alpha);
      if (next < kTransmittanceFloor) {
        done = true;
        break;
      }
      const float4 c = staged.rgbd[j];
      const float w = alpha * T;
      red += c.x * w;
      green += c.y * w;
      blue += c.z * w;
      depth += c.w * w;
      T = next;
      last = base + j + 1;
    }
    __syncthreads();
  }

  if (inside) {
    const int pixel = py_i * width + px_i;
    float *o = out + static_cast<int64_t>(pixel) * 5;
    o[0] = red + T * background[0];
    o[1] = green + T * background[1];
    o[2] = blue + T * background[2];
    o[3] = depth;
    o[4] = 1.0f - T;
    final_t[pixel] = T;
    last_index[pixel] = last;
  }
}

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

__global__ void backward_kernel(const float *attrs, const int32_t *ids,
                                const int32_t *ranges, const float *background,
                                int height, int width, int tiles_x,
                                const float *grad_out, const float *final_t,
                                const int32_t *last_index, float *grad_attrs) {
  __shared__ Staged staged;
  __shared__ int block_last;
  const int slot = threadIdx.y * kSide + threadIdx.x;
  const int lane = slot & 31;
  const int tile = blockIdx.y * tiles_x + blockIdx.x;
  const int px_i = blockIdx.x * kSide + threadIdx.x;
  const int py_i = blockIdx.y * kSide + threadIdx.y;
  const bool inside = px_i < width && py_i < height;
  const float px = px_i + 0.5f, py = py_i + 0.5f;
  const int begin = ranges[tile * 2];

  const int pixel = py_i * width + px_i;
  const int my_last = inside ? last_index[pixel] : 0;
  float T = inside ? final_t[pixel] : 1.0f;
  const float T_final = T;
  float go[5] = {0, 0, 0, 0, 0};
  if (inside)
    for (int k = 0; k < 5; ++k)
      go[k] = grad_out[static_cast<int64_t>(pixel) * 5 + k];
  const float bg_dot = background[0] * go[0] + background[1] * go[1] +
                       background[2] * go[2];

  if (slot == 0)
    block_last = 0;
  __syncthreads();
  atomicMax(&block_last, my_last);
  __syncthreads();
  const int total = block_last;

  // suffix[] composites everything behind the Gaussian being processed.
  float suffix[4] = {0, 0, 0, 0};
  float prev_alpha = 0.0f;
  float prev_channel[4] = {0, 0, 0, 0};

  for (int base = ((total + kBatch - 1) / kBatch - 1) * kBatch; base >= 0;
       base -= kBatch) {
    __syncthreads();
    stage_batch(staged, attrs, ids + begin, base, total);
    __syncthreads();
    const int n = min(kBatch, total - base);
    for (int j = n - 1; j >= 0; --j) {
      float v[kColumns];
#pragma unroll
      for (int k = 0; k < kColumns; ++k)
        v[k] = 0.0f;

      float dx, dy, falloff;
      float alpha = -1.0f;
      if (inside && base + j < my_last)
        alpha = pair_alpha(staged, j, px, py, &dx, &dy, &falloff);
      if (alpha >= 0.0f) {
        T = T / (1.0f - alpha);
        const float4 conic = staged.conic_alpha[j];
        const float4 cd = staged.rgbd[j];
        const float channel[4] = {cd.x, cd.y, cd.z, cd.w};
        const float weight = alpha * T;
        float dalpha = 0.0f;
        for (int k = 0; k < 4; ++k) {
          suffix[k] =
              prev_alpha * prev_channel[k] + (1.0f - prev_alpha) * suffix[k];
          prev_channel[k] = channel[k];
          dalpha += (channel[k] - suffix[k]) * go[k == 3 ? 3 : k];
        }
        dalpha *= T;
        const float inv = T_final / (1.0f - alpha);
        dalpha += go[4] * inv - bg_dot * inv;
        prev_alpha = alpha;

        v[6] = weight * go[3];
        v[7] = weight * go[0];
        v[8] = weight * go[1];
        v[9] = weight * go[2];
        v[5] = falloff * dalpha;
        // Only the unclamped branch passes gradient through min(alpha, 0.99).
        const float unclamped = conic.w * falloff;
        if (unclamped <= kAlphaCeil) {
          const float dpower = unclamped * dalpha;
          v[0] = dpower * (conic.x * dx + conic.y * dy);
          v[1] = dpower * (conic.z * dy + conic.y * dx);
          v[2] = -0.5f * dx * dx * dpower;
          v[3] = -dx * dy * dpower;
          v[4] = -0.5f * dy * dy * dpower;
        } else {
          v[5] = 0.0f;
        }
      }

      // Whole warp reduces, then one lane issues the atomics.
#pragma unroll
      for (int k = 0; k < kColumns; ++k)
        v[k] = warp_reduce(v[k]);
      if (lane == 0) {
        float *dst = grad_attrs +
                     static_cast<int64_t>(ids[begin + base + j]) * kColumns;
#pragma unroll
        for (int k = 0; k < kColumns; ++k)
          if (v[k] != 0.0f)
            atomicAdd(dst + k, v[k]);
      }
    }
  }
}

} // namespace

void rasterize_forward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                            const at::Tensor &ranges,
                            const at::Tensor &background, int64_t height,
                            int64_t width, at::Tensor &out, at::Tensor &final_t,
                            at::Tensor &last_index) {
  const c10::cuda::CUDAGuard guard(attrs.device());
  const int tiles_x = static_cast<int>((width + kSide - 1) / kSide);
  const int tiles_y = static_cast<int>((height + kSide - 1) / kSide);
  const dim3 grid(tiles_x, tiles_y), block(kSide, kSide);
  forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      attrs.data_ptr<float>(), ids.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>(), background.data_ptr<float>(),
      static_cast<int>(height), static_cast<int>(width), tiles_x,
      out.data_ptr<float>(), final_t.data_ptr<float>(),
      last_index.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rasterize_backward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                             const at::Tensor &ranges,
                             const at::Tensor &background,
                             const at::Tensor &grad_out,
                             const at::Tensor &final_t,
                             const at::Tensor &last_index, int64_t height,
                             int64_t width, at::Tensor &grad_attrs) {
  const c10::cuda::CUDAGuard guard(attrs.device());
  const int tiles_x = static_cast<int>((width + kSide - 1) / kSide);
  const int tiles_y = static_cast<int>((height + kSide - 1) / kSide);
  const dim3 grid(tiles_x, tiles_y), block(kSide, kSide);
  backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      attrs.data_ptr<float>(), ids.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>(), background.data_ptr<float>(),
      static_cast<int>(height), static_cast<int>(width), tiles_x,
      grad_out.data_ptr<float>(), final_t.data_ptr<float>(),
      last_index.data_ptr<int32_t>(), grad_attrs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gpu
} // namespace gaussian
} // namespace riemann
