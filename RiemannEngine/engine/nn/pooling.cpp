#include "engine/nn/pooling.hpp"

namespace riemann {
namespace {

enum class Reduction { Sum, Average, Maximum };

struct PoolingMaps {
  std::shared_ptr<CoordinateManager> manager;
  CoordinateMapKey input_key;
  CoordinateMapKey output_key;
  CoordinateMapKey result_key;
  at::Tensor input_features;
  at::Tensor output_row_remap;
};

PoolingMaps prepare_pooling_maps(const SparseTensor& input,
                                 const KernelGenerator::KernelParams& params,
                                 bool transpose, bool expand_coordinates) {
  const auto source_manager = input.coordinate_manager();
  auto manager = std::make_shared<CoordinateManager>(
      input.dimension(), CoordinateMapBackend::CPU);
  const auto input_key_value = input.coordinate_map_key().get_key();
  auto inserted_input = manager->insert_and_map(
      source_manager->get_coordinates(input.coordinate_map_key())
          .to(torch::kCPU)
          .contiguous(),
      input_key_value.first, input_key_value.second);
  auto input_features = input.F().index_select(
      0, inserted_input.second.first.to(input.F().device(), torch::kLong));

  CoordinateMapKey output_key(input.dimension() + 1);
  CoordinateMapKey result_key(input.dimension() + 1);
  at::Tensor output_row_remap;
  if (!transpose) {
    const auto source_output_key =
        source_manager->stride(input.coordinate_map_key(),
                               params.kernel_stride);
    result_key = source_output_key;
    const auto output_key_value = source_output_key.get_key();
    auto inserted_output = manager->insert_and_map(
        source_manager->get_coordinates(source_output_key)
            .to(torch::kCPU)
            .contiguous(),
        output_key_value.first, output_key_value.second);
    output_key = inserted_output.first;
    output_row_remap = inserted_output.second.second;
  } else {
    const auto output_stride = detail::stride_tensor_stride(
        input_key_value.first, params.kernel_stride, true);
    auto offsets = params.region_offsets;
    if (!offsets.defined())
      offsets = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32));
    offsets = offsets.to(torch::kCPU).contiguous();
    cpu_kernel_region<int32_t> region(
        params.region_type, input.dimension() + 1, output_stride.data(),
        params.kernel_size.data(), params.kernel_dilation.data(), 0,
        offsets.numel() ? offsets.data_ptr<int32_t>() : nullptr,
        offsets.numel() ? static_cast<uint32_t>(offsets.size(0)) : 0, true);
    auto* cpu_manager = manager->manager_ptr<cpu_manager_type<int32_t>>();
    const auto generated = cpu_manager->stride_region(
        input_key_value, region, output_stride, expand_coordinates);
    output_key = CoordinateMapKey(input.dimension() + 1, generated.first);
    const auto generated_coords =
        manager->get_coordinates(output_key).contiguous();
    auto inserted_output = source_manager->insert_and_map(
        generated_coords.to(input.C().device()), output_stride,
        generated.first.second);
    output_row_remap = inserted_output.second.first;
    result_key = inserted_output.first;
  }
  return {std::move(manager), std::move(inserted_input.first),
          std::move(output_key), std::move(result_key),
          std::move(input_features),
          std::move(output_row_remap)};
}

SparseTensor evaluate_pooling(const SparseTensor& input,
                              const KernelGenerator::KernelParams& params,
                              PoolingMode::Type mode, bool transpose,
                              bool expand_coordinates) {
  TORCH_CHECK(input.F().is_floating_point(),
              "Sparse pooling requires floating-point features");
  auto maps = prepare_pooling_maps(input, params, transpose,
                                   expand_coordinates);
  auto* cpu_manager =
      maps.manager->manager_ptr<cpu_manager_type<int32_t>>();
  auto offsets = params.region_offsets;
  if (!offsets.defined())
    offsets = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32));
  offsets = offsets.to(torch::kCPU).contiguous();
  const auto& kernel_map = cpu_manager->kernel_map(
      &maps.input_key, &maps.output_key, params.kernel_size,
      params.kernel_stride, params.kernel_dilation, params.region_type, offsets,
      transpose, true);

  const auto output_rows =
      static_cast<int64_t>(maps.manager->size(maps.output_key));
  std::vector<std::vector<int64_t>> inputs_by_output(
      static_cast<size_t>(output_rows));
  for (size_t offset = 0; offset < kernel_map.first.size(); ++offset) {
    const auto& input_rows = kernel_map.first[offset];
    const auto& output_indices = kernel_map.second[offset];
    TORCH_CHECK(input_rows.size() == output_indices.size(),
                "Pooling kernel map index arrays have different lengths");
    for (size_t pair = 0; pair < input_rows.size(); ++pair)
      inputs_by_output[output_indices[pair]].push_back(input_rows[pair]);
  }

  std::vector<at::Tensor> pooled;
  pooled.reserve(static_cast<size_t>(output_rows));
  for (const auto& source_rows : inputs_by_output) {
    if (source_rows.empty()) {
      pooled.push_back(torch::zeros({input.F().size(1)}, input.F().options()));
      continue;
    }
    auto indexes = torch::from_blob(
        const_cast<int64_t*>(source_rows.data()),
        {static_cast<int64_t>(source_rows.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto values = maps.input_features.index_select(0, indexes.to(
        maps.input_features.device(), torch::kLong));
    if (mode == PoolingMode::LOCAL_SUM_POOLING)
      pooled.push_back(values.sum(0));
    else if (mode == PoolingMode::LOCAL_AVG_POOLING)
      pooled.push_back(values.mean(0));
    else
      pooled.push_back(values.amax({0}));
  }
  auto output = output_rows == 0
                    ? torch::empty({0, input.F().size(1)}, input.F().options())
                    : torch::stack(pooled, 0);
  if (output_rows > 0) {
    output = output.index_select(
        0, maps.output_row_remap.to(output.device(), torch::kLong));
  }
  TORCH_CHECK(output.size(0) ==
                  input.coordinate_manager()->size(maps.result_key),
              "Pooling feature/map row count mismatch: features=",
              output.size(0), " map=",
              input.coordinate_manager()->size(maps.result_key));
  return SparseTensor(std::move(output), std::move(maps.result_key),
                      input.coordinate_manager());
}

Reduction reduction_for(PoolingMode::Type mode) {
  switch (mode) {
  case PoolingMode::GLOBAL_SUM_POOLING_DEFAULT:
  case PoolingMode::GLOBAL_SUM_POOLING_KERNEL:
  case PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX:
    return Reduction::Sum;
  case PoolingMode::GLOBAL_AVG_POOLING_DEFAULT:
  case PoolingMode::GLOBAL_AVG_POOLING_KERNEL:
  case PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX:
    return Reduction::Average;
  case PoolingMode::GLOBAL_MAX_POOLING_DEFAULT:
  case PoolingMode::GLOBAL_MAX_POOLING_KERNEL:
  case PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX:
    return Reduction::Maximum;
  default:
    TORCH_CHECK(false, "Global pooling requires a global PoolingMode");
  }
}

} // namespace

RiemannPoolingBaseImpl::RiemannPoolingBaseImpl(
    int kernel_size, int stride, int dilation, PoolingMode::Type mode,
    int dimension)
    : kernel_generator_({kernel_size}, {stride}, {dilation}, false,
                        RegionType::HYPER_CUBE, {}, false, {}, dimension),
      mode_(mode), dimension_(dimension) {
  TORCH_CHECK(mode == PoolingMode::LOCAL_SUM_POOLING ||
                  mode == PoolingMode::LOCAL_AVG_POOLING ||
                  mode == PoolingMode::LOCAL_MAX_POOLING,
              "Local pooling requires a local pooling mode");
}

RiemannPoolingBaseImpl::RiemannPoolingBaseImpl(
    KernelGenerator kernel_generator, PoolingMode::Type mode, int dimension)
    : kernel_generator_(std::move(kernel_generator)), mode_(mode),
      dimension_(dimension) {
  TORCH_CHECK(mode == PoolingMode::LOCAL_SUM_POOLING ||
                  mode == PoolingMode::LOCAL_AVG_POOLING ||
                  mode == PoolingMode::LOCAL_MAX_POOLING,
              "Local pooling requires a local pooling mode");
}

SparseTensor RiemannPoolingBaseImpl::forward(const SparseTensor& input) {
  TORCH_CHECK(input.dimension() == dimension_, "Input dimension mismatch");
  const auto params = kernel_generator_.get_kernel(input.tensor_stride(), false);
  return evaluate_pooling(input, params, mode_, false, false);
}

RiemannPoolingTransposeImpl::RiemannPoolingTransposeImpl(
    int kernel_size, int stride, int dilation, bool expand_coordinates,
    int dimension)
    : kernel_generator_({kernel_size}, {stride}, {dilation}, true,
                        RegionType::HYPER_CUBE, {}, expand_coordinates, {},
                        dimension),
      dimension_(dimension) {}

RiemannPoolingTransposeImpl::RiemannPoolingTransposeImpl(
    KernelGenerator kernel_generator, int dimension, PoolingMode::Type mode)
    : kernel_generator_(std::move(kernel_generator)), dimension_(dimension),
      mode_(mode) {
  TORCH_CHECK(mode_ == PoolingMode::LOCAL_SUM_POOLING ||
                  mode_ == PoolingMode::LOCAL_AVG_POOLING ||
                  mode_ == PoolingMode::LOCAL_MAX_POOLING,
              "Transpose pooling requires a local pooling mode");
}

SparseTensor RiemannPoolingTransposeImpl::forward(const SparseTensor& input) {
  TORCH_CHECK(input.dimension() == dimension_, "Input dimension mismatch");
  const auto params = kernel_generator_.get_kernel(input.tensor_stride(), true);
  return evaluate_pooling(input, params, mode_, true,
                          kernel_generator_.expand_coordinates());
}

SparseTensor RiemannGlobalPoolingImpl::forward(const SparseTensor& input) {
  const auto reduction = reduction_for(mode_);
  const auto coordinates = input.C();
  const auto rows = coordinates.size(0);
  const auto dimension = input.dimension();
  int64_t batches = 0;
  if (rows > 0)
    batches = coordinates.select(1, 0).max().item<int64_t>() + 1;

  auto output_coordinates = torch::zeros(
      {batches, dimension + 1}, coordinates.options());
  if (batches > 0) {
    output_coordinates.select(1, 0).copy_(
        torch::arange(batches, coordinates.options()));
  }

  std::vector<at::Tensor> pooled;
  pooled.reserve(static_cast<size_t>(batches));
  for (int64_t batch = 0; batch < batches; ++batch) {
    const auto indexes = at::nonzero(coordinates.select(1, 0) == batch)
                             .reshape({-1})
                             .to(torch::kLong);
    if (indexes.numel() == 0) {
      pooled.push_back(torch::zeros({input.F().size(1)}, input.F().options()));
      continue;
    }
    const auto values = input.F().index_select(0, indexes);
    switch (reduction) {
    case Reduction::Sum:
      pooled.push_back(values.sum(0));
      break;
    case Reduction::Average:
      pooled.push_back(values.mean(0));
      break;
    case Reduction::Maximum:
      pooled.push_back(values.amax({0}));
      break;
    }
  }

  auto features = batches == 0
                      ? torch::empty({0, input.F().size(1)}, input.F().options())
                      : torch::stack(pooled, 0);
  auto inserted = input.coordinate_manager()->insert_and_map(
      output_coordinates, input.tensor_stride());
  if (batches > 0) {
    TORCH_CHECK(inserted.second.first.numel() == batches,
                "Global pooled coordinate map unexpectedly collapsed batches");
    features = features.index_select(
        0, inserted.second.first.to(features.device(), torch::kLong));
  }
  return SparseTensor(std::move(features), std::move(inserted.first),
                      input.coordinate_manager());
}

} // namespace riemann
