#ifndef RIEMANN_OCTREE_OPS_HPP
#define RIEMANN_OCTREE_OPS_HPP

#include "stencil.hpp"
#include "tensor.hpp"

namespace riemann {
namespace adaptive {

enum class InterpMode { Nearest, Multilinear };

// y_o = sum_k sum_i w(o,i,k) * x_i W_k with weight [K^D, Cin, Cout].
// Differentiable w.r.t. features and weight. Output leaves equal the input
// leaves unless config.output_level_shift != 0.
AdaptiveSparseTensor adaptive_convolution(const AdaptiveSparseTensor &input,
                                          const at::Tensor &weight,
                                          const AdaptiveKernelConfig &config);

// Transpose of a (possibly strided) convolution. `reference` supplies the
// output leaves and must be the tensor the forward convolution ran on; `input`
// must carry the topology that forward convolution produces.
AdaptiveSparseTensor adaptive_convolution_transpose(const AdaptiveSparseTensor &input,
                                                    const at::Tensor &weight,
                                                    const AdaptiveKernelConfig &config,
                                                    const AdaptiveSparseTensor &reference);

// [C] tensor; volume-weighted by default so the value is resolution independent.
at::Tensor adaptive_global_avg_pool(const AdaptiveSparseTensor &input,
                                    bool volume_weighted = true);

// Coarsens every leaf finer than target_level up to target_level.
AdaptiveSparseTensor adaptive_pool(const AdaptiveSparseTensor &input, uint32_t target_level,
                                   Reduction reduction = Reduction::Mean);

// points: [M, D]. Points outside every active leaf give zeros. Nearest reads
// the containing leaf; Multilinear weights leaves by overlap with a box of the
// containing leaf's size centered on the point (exact multilinear interpolation
// between leaf centers on uniform grids, a partition of unity everywhere).
at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode = InterpMode::Nearest);

} // namespace adaptive
} // namespace riemann

#endif
