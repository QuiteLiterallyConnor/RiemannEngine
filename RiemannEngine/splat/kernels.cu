#include "splat/gpu.hpp"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace riemann {
namespace gaussian {
namespace gpu {
namespace {

constexpr int kThreadsPerBlock = 256;

template <typename scalar_t>
__device__ __forceinline__ void quaternion_to_matrix(const scalar_t *q,
                                                     scalar_t *r) {
  const scalar_t norm =
      sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const scalar_t w = q[0] / norm, x = q[1] / norm;
  const scalar_t y = q[2] / norm, z = q[3] / norm;
  r[0] = 1 - 2 * (y * y + z * z);
  r[1] = 2 * (x * y - z * w);
  r[2] = 2 * (x * z + y * w);
  r[3] = 2 * (x * y + z * w);
  r[4] = 1 - 2 * (x * x + z * z);
  r[5] = 2 * (y * z - x * w);
  r[6] = 2 * (x * z - y * w);
  r[7] = 2 * (y * z + x * w);
  r[8] = 1 - 2 * (x * x + y * y);
}

template <typename scalar_t>
__global__ void covariance_kernel(const scalar_t *quaternions,
                                  const scalar_t *scales, scalar_t *output,
                                  int64_t count) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count)
    return;

  scalar_t r[9];
  quaternion_to_matrix(quaternions + row * 4, r);
  const scalar_t *s = scales + row * 3;
  const scalar_t v0 = s[0] * s[0], v1 = s[1] * s[1], v2 = s[2] * s[2];
  scalar_t *sigma = output + row * 9;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      sigma[i * 3 + j] = r[i * 3] * v0 * r[j * 3] +
                         r[i * 3 + 1] * v1 * r[j * 3 + 1] +
                         r[i * 3 + 2] * v2 * r[j * 3 + 2];
}

template <typename scalar_t>
__device__ __forceinline__ void rotate_by(const scalar_t *q, const scalar_t *v,
                                          scalar_t *out) {
  scalar_t r[9];
  quaternion_to_matrix(q, r);
  out[0] = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
  out[1] = r[3] * v[0] + r[4] * v[1] + r[5] * v[2];
  out[2] = r[6] * v[0] + r[7] * v[1] + r[8] * v[2];
}

template <typename scalar_t>
__global__ void transform_kernel(
    const scalar_t *positions, const scalar_t *gaussian_rotations,
    const scalar_t *transform_rotations, const scalar_t *translations,
    const scalar_t *velocities, scalar_t *output_positions,
    scalar_t *output_rotations, scalar_t *output_velocities, int64_t count) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count)
    return;

  const scalar_t *q = transform_rotations + row * 4;
  scalar_t moved[3];
  rotate_by(q, positions + row * 3, moved);
  for (int axis = 0; axis < 3; ++axis)
    output_positions[row * 3 + axis] =
        moved[axis] + translations[row * 3 + axis];

  const scalar_t *g = gaussian_rotations + row * 4;
  const scalar_t qn = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] +
                           q[3] * q[3]);
  const scalar_t gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2] +
                           g[3] * g[3]);
  const scalar_t aw = q[0] / qn, ax = q[1] / qn;
  const scalar_t ay = q[2] / qn, az = q[3] / qn;
  const scalar_t bw = g[0] / gn, bx = g[1] / gn;
  const scalar_t by = g[2] / gn, bz = g[3] / gn;
  scalar_t *composed = output_rotations + row * 4;
  composed[0] = aw * bw - ax * bx - ay * by - az * bz;
  composed[1] = aw * bx + ax * bw + ay * bz - az * by;
  composed[2] = aw * by - ax * bz + ay * bw + az * bx;
  composed[3] = aw * bz + ax * by - ay * bx + az * bw;

  if (velocities != nullptr)
    rotate_by(q, velocities + row * 3, output_velocities + row * 3);
}

} // namespace

void covariance_cuda(const at::Tensor &rotations, const at::Tensor &scales,
                     at::Tensor &output) {
  const int64_t count = rotations.size(0);
  if (count == 0)
    return;
  c10::cuda::CUDAGuard guard(rotations.device());
  const int blocks =
      static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      rotations.scalar_type(), "riemann_gaussian_covariance", [&] {
        covariance_kernel<scalar_t>
            <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
                rotations.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transform_cuda(const at::Tensor &positions,
                    const at::Tensor &gaussian_rotations,
                    const at::Tensor &transform_rotations,
                    const at::Tensor &translations,
                    const at::Tensor &velocities, at::Tensor &output_positions,
                    at::Tensor &output_rotations,
                    at::Tensor &output_velocities) {
  const int64_t count = positions.size(0);
  if (count == 0)
    return;
  c10::cuda::CUDAGuard guard(positions.device());
  const int blocks =
      static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      positions.scalar_type(), "riemann_gaussian_transform", [&] {
        transform_kernel<scalar_t>
            <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
                positions.data_ptr<scalar_t>(),
                gaussian_rotations.data_ptr<scalar_t>(),
                transform_rotations.data_ptr<scalar_t>(),
                translations.data_ptr<scalar_t>(),
                velocities.defined() ? velocities.data_ptr<scalar_t>()
                                     : nullptr,
                output_positions.data_ptr<scalar_t>(),
                output_rotations.data_ptr<scalar_t>(),
                output_velocities.defined()
                    ? output_velocities.data_ptr<scalar_t>()
                    : nullptr,
                count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gpu
} // namespace gaussian
} // namespace riemann
