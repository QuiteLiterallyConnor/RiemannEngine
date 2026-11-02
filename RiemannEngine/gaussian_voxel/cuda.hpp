#ifndef RIEMANN_GAUSSIAN_VOXEL_CUDA_HPP
#define RIEMANN_GAUSSIAN_VOXEL_CUDA_HPP

#include "gaussian_voxel/voxelize.hpp"

namespace riemann::gaussian_voxel {

// Only topology metadata is uploaded; support lookup never downloads Gaussians.
std::pair<at::Tensor, at::Tensor> cuda_associations(
    const at::Tensor &positions, const at::Tensor &covariances,
    const adaptive::AdaptiveSparseTensor &reference,
    const GaussianVoxelOptions &options);

std::pair<at::Tensor, at::Tensor> cuda_accumulate(
    const at::Tensor &positions, const at::Tensor &precision,
    const at::Tensor &normalization, const at::Tensor &amplitudes,
    const at::Tensor &features, const at::Tensor &gaussian_ids,
    const at::Tensor &voxel_ids, const at::Tensor &centers,
    const at::Tensor &volumes);

} // namespace riemann::gaussian_voxel

#endif
