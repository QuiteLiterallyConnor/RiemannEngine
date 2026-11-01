#ifndef RIEMANN_ENGINE_PUBLIC_RIEMANN_HPP
#define RIEMANN_ENGINE_PUBLIC_RIEMANN_HPP

#include "datatypes.hpp"
#include "position_manager.hpp"
#include "stencil_generator.hpp"
#include "sparse_array.hpp"
#include "array_field.hpp"
#include "persistence.hpp"
#include "functions.hpp"
#include "autodiff/conv_func.hpp"
#include "autodiff/aggregation_func.hpp"
#include "autodiff/distribute_func.hpp"
#include "autodiff/trim_func.hpp"
#include "autodiff/resample_func.hpp"
#include "autodiff/instance_norm_func.hpp"
#include "autodiff/sparse_matmul_func.hpp"
#include "core/host_sparse_conv_stencils.hpp"
#include "engine/nn/conv.hpp"
#include "engine/nn/activation.hpp"
#include "engine/nn/normalization.hpp"
#include "engine/nn/linear.hpp"
#include "engine/nn/pooling.hpp"
#include "engine/nn/operations.hpp"
#include "engine/nn/residual.hpp"
#include "engine/nn/squeeze_excite.hpp"
#include "engine/nn/resample.hpp"
#include "layers/distribute.hpp"
#include "layers/resample.hpp"
#include "layers/trim.hpp"
#include "layers/aggregation.hpp"
#include "layers/operations.hpp"
#include "layers/merge_op.hpp"
#include "layers/activation.hpp"
#include "layers/standardization.hpp"
#include "layers/dense.hpp"
#include "layers/conv.hpp"
#include "layers/residual_block.hpp"
#include "layers/squeeze_excite_block.hpp"
#include "dynamic/dynamic.hpp"
#include "helpers/ball.hpp"
#include "helpers/batching.hpp"
#include "helpers/discretization.hpp"
#include "helpers/setup.hpp"
#include "learning/learning.hpp"

#ifdef RIEMANN_ENABLE_GAUSSIAN
#include "gaussian_voxel/occupancy.hpp"
#include "gaussian_voxel/voxelize.hpp"
#include "normal/normal3d.hpp"
#include "normal/normal_tensor.hpp"
#include "normal/normal_optimizer.hpp"
#include "normal/normal_ply.hpp"
#include "normal/normal_camera.hpp"
#include "normal/normal_sh.hpp"
#include "normal/normal_model.hpp"
#include "normal/normal_render.hpp"
#include "normal/normal_trainer.hpp"
#endif

namespace riemann {

inline constexpr const char* VERSION = "1.0.0";
inline constexpr int VERSION_MAJOR = 1;
inline constexpr int VERSION_MINOR = 0;
inline constexpr int VERSION_PATCH = 0;

} // namespace riemann

#endif
