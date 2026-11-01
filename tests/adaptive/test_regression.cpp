// Regression tests: the existing uniform sparse tensor APIs must keep working unchanged
// once adaptive support is added.
#include "RiemannEngine/dynamic/dynamic.hpp"
#include "RiemannEngine/helpers/ball_geometry.hpp"
#include "RiemannEngine/normal/normal_device.hpp"
#include "RiemannEngine/riemann.hpp"
#include "RiemannEngine/octree/tree.hpp"
#include "RiemannEngine/helpers/batching.hpp"
#include "RiemannEngine/helpers/discretization.hpp"
#include "utils/test_framework.hpp"

#include <filesystem>
#include <map>

namespace {

riemann::SparseTensor make_legacy(int64_t n, int64_t channels) {
    torch::manual_seed(3);
    auto coords = std::get<0>(torch::unique_dim(torch::randint(0, 6, {n, 3}, torch::kInt32), 0));
    auto batch = torch::zeros({coords.size(0), 1}, torch::kInt32);
    return riemann::SparseTensor(torch::randn({coords.size(0), channels}),
                                   torch::cat({batch, coords}, 1));
}

}  // namespace

TEST(regression_sparse_tensor_roundtrip) {
    auto t = make_legacy(100, 4);
    CHECK_EQ(t.dimension(), 3);
    CHECK_EQ(t.F().size(0), t.C().size(0));
    CHECK_EQ(t.F().size(1), int64_t(4));
}

TEST(regression_uniform_convolution_shapes_and_determinism) {
    auto t = make_legacy(150, 3);
    riemann::RiemannConvolution conv(3, 5, 3, 1, 1, false, false,
                                         riemann::ConvolutionMode::DEFAULT, 3);
    auto a = conv->forward(t);
    auto b = conv->forward(t);
    CHECK_EQ(a.F().size(1), int64_t(5));
    CHECK_EQ(a.F().size(0), t.F().size(0));  // stride 1, no coordinate expansion
    CHECK_ALLCLOSE(a.F(), b.F(), 1e-6, 1e-6);
}

TEST(regression_uniform_tensor_arithmetic) {
    auto t = make_legacy(80, 2);
    auto doubled = t + t;
    CHECK_ALLCLOSE(doubled.F(), t.F() * 2, 1e-6, 1e-6);
}

TEST(regression_sparse_dense_transfer_and_cross_map_arithmetic) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        2, riemann::CoordinateMapBackend::CPU);
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto coords_a = torch::tensor({{0, 0, 1}, {1, 2, 0}}, options);
    auto coords_b = torch::tensor({{0, 0, 1}, {0, 1, 1}}, options);
    auto features_a = torch::tensor(
        {{1.0f, 10.0f}, {2.0f, 20.0f}},
        torch::TensorOptions().requires_grad(true));
    auto features_b = torch::tensor(
        {{3.0f, 30.0f}, {4.0f, 40.0f}},
        torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor a(features_a, coords_a, manager);
    riemann::SparseTensor b(features_b, coords_b, manager);

    auto dense = a.dense();
    CHECK(dense.sizes() == torch::IntArrayRef({2, 2, 3, 2}));
    CHECK_ALLCLOSE(dense.index({0, torch::indexing::Slice(), 0, 1}),
                   features_a.index({0}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(dense.index({1, torch::indexing::Slice(), 2, 0}),
                   features_a.index({1}), 1e-6, 1e-6);
    CHECK_EQ(dense.index({0, torch::indexing::Slice(), 0, 0}).abs().sum().item<float>(),
             0.0f);
    auto sized_dense = a.dense(std::vector<int64_t>{2, 2, 4, 3});
    CHECK(sized_dense.sizes() == torch::IntArrayRef({2, 2, 4, 3}));
    CHECK_ALLCLOSE(sized_dense.index({1, torch::indexing::Slice(), 2, 0}),
                   features_a.index({1}), 1e-6, 1e-6);
    dense.sum().backward();
    CHECK_ALLCLOSE(features_a.grad(), torch::ones_like(features_a), 1e-6, 1e-6);

    auto sum = a + b;
    CHECK_EQ(sum.F().size(0), int64_t(3));
    auto coordinates = sum.C();
    for (int64_t row = 0; row < coordinates.size(0); ++row) {
        const auto coord = coordinates[row];
        auto value = sum.F()[row];
        if (coord[0].item<int>() == 0 && coord[1].item<int>() == 0) {
            CHECK_ALLCLOSE(value, torch::tensor({4.0f, 40.0f}), 1e-6, 1e-6);
        } else if (coord[0].item<int>() == 0 && coord[1].item<int>() == 1) {
            CHECK_ALLCLOSE(value, torch::tensor({4.0f, 40.0f}), 1e-6, 1e-6);
        } else {
            CHECK_ALLCLOSE(value, torch::tensor({2.0f, 20.0f}), 1e-6, 1e-6);
        }
    }
    sum.F().sum().backward();
    CHECK_ALLCLOSE(features_a.grad(), torch::full_like(features_a, 2.0f), 1e-6, 1e-6);
    CHECK_ALLCLOSE(features_b.grad(), torch::ones_like(features_b), 1e-6, 1e-6);

    auto transferred = a.to(torch::kCPU);
    CHECK(torch::equal(transferred.C(), a.C()));
    CHECK_ALLCLOSE(transferred.F(), a.F(), 1e-6, 1e-6);
    CHECK(transferred.tensor_stride() == a.tensor_stride());
}

#if !defined(CPU_ONLY)
TEST(regression_sparse_cuda_device_transfer) {
    if (!torch::cuda::is_available())
        throw riemann_test::Skip("CUDA device not available");

    auto manager = std::make_shared<riemann::CoordinateManager>(
        2, riemann::CoordinateMapBackend::CPU);
    auto coordinates = torch::tensor(
        {{0, 0, 1}, {1, 2, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features = torch::tensor(
        {{1.0f, 10.0f}, {2.0f, 20.0f}},
        torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor input(features, coordinates, manager);
    auto gpu = input.to(torch::kCUDA);
    CHECK(gpu.device().is_cuda());
    CHECK(torch::equal(gpu.C().cpu(), input.C()));
    CHECK_ALLCLOSE(gpu.F().cpu(), input.F(), 1e-6, 1e-6);
    gpu.F().sum().backward();
    CHECK_ALLCLOSE(features.grad(), torch::ones_like(features), 1e-6, 1e-6);
}
#endif

TEST(regression_custom_coordinate_region) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        1, riemann::CoordinateMapBackend::CPU);
    auto coords = torch::tensor({{0, 0}, {0, 1}, {0, 2}},
                                torch::TensorOptions().dtype(torch::kInt32));
    riemann::SparseTensor input(torch::tensor({{1.0f}, {2.0f}, {4.0f}}),
                                coords, manager);
    auto offsets = torch::tensor({{-1}, {0}}, torch::kInt32);
    riemann::KernelGenerator generator({2}, {1}, {1}, false,
                                       riemann::RegionType::CUSTOM, offsets,
                                       false, {}, 1);
    riemann::RiemannPoolingBaseImpl pooling(
        generator, riemann::PoolingMode::LOCAL_SUM_POOLING, 1);
    auto output = pooling.forward(input);
    auto values_by_coordinate = std::map<int, float>();
    auto output_coordinates = output.C();
    for (int64_t row = 0; row < output_coordinates.size(0); ++row)
        values_by_coordinate[output_coordinates[row][1].item<int>()] =
            output.F()[row][0].item<float>();
    CHECK_NEAR(values_by_coordinate[0], 1.0f, 1e-6);
    CHECK_NEAR(values_by_coordinate[1], 3.0f, 1e-6);
    CHECK_NEAR(values_by_coordinate[2], 6.0f, 1e-6);

    auto other_offsets = torch::tensor({{0}, {1}}, torch::kInt32);
    riemann::KernelGenerator other_generator(
        {2}, {1}, {1}, false, riemann::RegionType::CUSTOM, other_offsets,
        false, {}, 1);
    riemann::RiemannPoolingBaseImpl other_pooling(
        other_generator, riemann::PoolingMode::LOCAL_SUM_POOLING, 1);
    auto other_output = other_pooling.forward(input);
    auto other_coordinates = other_output.C();
    for (int64_t row = 0; row < other_coordinates.size(0); ++row)
        values_by_coordinate[other_coordinates[row][1].item<int>()] =
            other_output.F()[row][0].item<float>();
    CHECK_NEAR(values_by_coordinate[0], 3.0f, 1e-6);
    CHECK_NEAR(values_by_coordinate[1], 6.0f, 1e-6);
    CHECK_NEAR(values_by_coordinate[2], 4.0f, 1e-6);

#if !defined(CPU_ONLY)
    if (torch::cuda::is_available()) {
        riemann::SparseTensor gpu_input(input.F().to(torch::kCUDA),
                                        input.C().to(torch::kCUDA));
        auto gpu_output = pooling.forward(gpu_input);
        CHECK(torch::equal(gpu_output.C().cpu(), output.C()));
        CHECK_ALLCLOSE(gpu_output.F().cpu(), output.F(), 1e-6, 1e-6);
    }
#endif
}

TEST(regression_sparse_batchnorm_and_relu) {
    auto t = make_legacy(80, 3);
    riemann::RiemannBatchNorm normalization(3);
    riemann::RiemannReLU relu;
    auto output = relu->forward(normalization->forward(t));
    CHECK(output.F().sizes() == t.F().sizes());
    CHECK((output.F() >= 0).all().item<bool>());
    riemann::RiemannGELU gelu;
    riemann::RiemannDropout dropout(0.25);
    dropout->eval();
    auto activated = dropout->forward(gelu->forward(t));
    CHECK(activated.F().sizes() == t.F().sizes());
    riemann::RiemannLinear linear(3, 5);
    auto projected = linear->forward(t);
    CHECK_EQ(projected.F().size(1), int64_t(5));
    CHECK(torch::equal(projected.C(), t.C()));
}

TEST(regression_sparse_matrix_multiply_and_gradients) {
    auto rows = torch::tensor({0, 0, 2}, torch::kInt32);
    auto cols = torch::tensor({1, 0, 1}, torch::kInt32);
    auto vals = torch::tensor({2.0f, 3.0f, 4.0f});
    auto mat = torch::tensor(
        {{1.0f, 10.0f}, {2.0f, 20.0f}},
        torch::TensorOptions().requires_grad(true));

    auto product =
        riemann::SPMMFunction::apply(rows, cols, vals, 3, 2, mat);
    CHECK_ALLCLOSE(
        product,
        torch::tensor({{7.0f, 70.0f}, {0.0f, 0.0f}, {8.0f, 80.0f}}),
        1e-6, 1e-6);
    product.sum().backward();
    CHECK_ALLCLOSE(mat.grad(),
                   torch::tensor({{3.0f, 3.0f}, {6.0f, 6.0f}}),
                   1e-6, 1e-6);

    auto average_mat = torch::tensor(
        {{1.0f, 10.0f}, {2.0f, 20.0f}},
        torch::TensorOptions().requires_grad(true));
    auto average =
        riemann::SPMMAverageFunction::apply(rows, cols, 3, 2, average_mat);
    CHECK_ALLCLOSE(
        average,
        torch::tensor({{1.5f, 15.0f}, {0.0f, 0.0f}, {2.0f, 20.0f}}),
        1e-6, 1e-6);
    average.sum().backward();
    CHECK_ALLCLOSE(average_mat.grad(),
                   torch::tensor({{0.5f, 0.5f}, {1.5f, 1.5f}}),
                   1e-6, 1e-6);

#if !defined(CPU_ONLY)
    if (torch::cuda::is_available()) {
        auto cuda_mat = mat.detach().to(torch::kCUDA).set_requires_grad(true);
        auto cuda_product = riemann::SPMMFunction::apply(
            rows, cols, vals, 3, 2, cuda_mat);
        CHECK_ALLCLOSE(cuda_product.cpu(), product.detach(), 1e-6, 1e-6);
        cuda_product.sum().backward();
        CHECK_ALLCLOSE(cuda_mat.grad().cpu(), mat.grad(), 1e-6, 1e-6);

        auto cuda_average_mat =
            average_mat.detach().to(torch::kCUDA).set_requires_grad(true);
        auto cuda_average = riemann::SPMMAverageFunction::apply(
            rows, cols, 3, 2, cuda_average_mat);
        CHECK_ALLCLOSE(cuda_average.cpu(), average.detach(), 1e-6, 1e-6);
        cuda_average.sum().backward();
        CHECK_ALLCLOSE(cuda_average_mat.grad().cpu(), average_mat.grad(),
                       1e-6, 1e-6);
    }
#endif
}

TEST(regression_legacy_autodiff_entry_points) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        2, riemann::CoordinateMapBackend::CPU);
    auto coordinates = torch::tensor(
        {{0, 0, 0}, {0, 1, 0}, {0, 2, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features =
        torch::tensor({{1.0f}, {2.0f}, {3.0f}},
                      torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor input(features, coordinates, manager);
    auto input_key = std::make_shared<riemann::CoordinateMapKey>(
        input.coordinate_map_key());
    auto output_key =
        std::make_shared<riemann::CoordinateMapKey>(3);

    auto kernel =
        torch::ones({9, 1, 1}, torch::TensorOptions().requires_grad(true));
    auto convolved = riemann::ConvolutionFunction::apply(
        features, kernel, {3}, {1}, {1}, riemann::RegionType::HYPER_CUBE,
        at::Tensor(), false, riemann::ConvolutionMode::DEFAULT, input_key,
        output_key, manager);
    CHECK_EQ(convolved.size(1), int64_t(1));
    convolved.sum().backward();
    CHECK(features.grad().defined());
    CHECK(kernel.grad().defined());

    auto pooled_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto pooled = riemann::LocalPoolingFunction::apply(
        features.detach(), riemann::PoolingMode::LOCAL_AVG_POOLING, {2}, {2},
        {1}, riemann::RegionType::HYPER_CUBE, at::Tensor(), input_key,
        pooled_key, manager);
    CHECK_EQ(pooled.size(1), int64_t(1));
    CHECK(pooled_key->is_key_set());
    auto upsampled_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto upsampled = riemann::LocalPoolingTransposeFunction::apply(
        pooled, riemann::PoolingMode::LOCAL_SUM_POOLING, {2}, {2}, {1},
        riemann::RegionType::HYPER_CUBE, at::Tensor(), true, pooled_key,
        upsampled_key, manager);
    CHECK(upsampled_key->is_key_set());
    CHECK_EQ(upsampled.size(1), int64_t(1));
    auto transpose_conv_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto transposed_convolution =
        riemann::ConvolutionTransposeFunction::apply(
            pooled, torch::ones({9, 1, 1}), {3}, {2}, {1},
            riemann::RegionType::HYPER_CUBE, at::Tensor(), true,
            riemann::ConvolutionMode::DEFAULT, pooled_key,
            transpose_conv_key, manager);
    CHECK(transpose_conv_key->is_key_set());
    CHECK_EQ(transposed_convolution.size(1), int64_t(1));

    auto global_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto global = riemann::GlobalPoolingFunction::apply(
        features.detach(), riemann::PoolingMode::GLOBAL_AVG_POOLING_DEFAULT,
        input_key, global_key, manager);
    CHECK_EQ(global.size(0), int64_t(1));
    auto broadcast = riemann::BroadcastFunction::apply(
        features.detach(), global, riemann::BroadcastMode::ELEMENTWISE_ADDITON,
        input_key, global_key, manager);
    CHECK_EQ(broadcast.size(0), features.size(0));
    auto normalized = riemann::InstanceNormFunction::apply(
        features.detach(), input_key, global_key, manager);
    CHECK_EQ(normalized.size(0), features.size(0));
    const auto expected_normalized =
        (features.detach() - 2.0) / std::sqrt(2.0 / 3.0 + 1e-8);
    CHECK_ALLCLOSE(normalized, expected_normalized, 1e-6, 1e-6);
    auto sum_global_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto sum_normalized = riemann::InstanceNormFunction::apply(
        features.detach(), input_key, sum_global_key, manager,
        riemann::PoolingMode::GLOBAL_SUM_POOLING_KERNEL);
    CHECK_ALLCLOSE(
        sum_normalized,
        (features.detach() - 6.0) / std::sqrt(50.0 + 1e-8), 1e-6,
        1e-6);

    auto pruned_key =
        std::make_shared<riemann::CoordinateMapKey>(3);
    auto pruned = riemann::PruningFunction::apply(
        features.detach(), torch::tensor({true, false, true}), input_key,
        pruned_key, manager);
    CHECK_EQ(pruned.size(0), int64_t(2));
    CHECK(pruned_key->is_key_set());

    auto interpolation = riemann::InterpolationFunction::apply(
        features.detach(),
        torch::tensor({{0.0f, 1.0f, 0.0f}}), input_key, manager);
    CHECK_EQ(interpolation.size(), size_t(4));
    CHECK_EQ(interpolation[0].size(0), int64_t(1));

    auto maps_in = torch::tensor({0, 1, 2}, torch::kInt32);
    auto maps_out = torch::tensor({0, 0, 1}, torch::kInt32);
    auto [max_values, max_mask] =
        riemann::max_pool_fw(maps_in, maps_out, features.detach(), 2, false);
    CHECK_ALLCLOSE(max_values, torch::tensor({{2.0f}, {3.0f}}), 1e-6,
                   1e-6);
    auto max_grad = riemann::max_pool_bw(torch::ones_like(max_values),
                                          max_mask, 3);
    CHECK_ALLCLOSE(max_grad, torch::tensor({{0.0f}, {1.0f}, {1.0f}}),
                   1e-6, 1e-6);
    auto direct_max = riemann::DirectMaxPoolingFunction::apply(
        maps_in, maps_out, features.detach(), 2, false);
    CHECK_ALLCLOSE(direct_max, max_values, 1e-6, 1e-6);
}

TEST(regression_sparse_functional_and_batch_collation) {
    auto sample_a = torch::tensor({{0, 1, 2}, {0, 3, 4}}, torch::kInt32);
    auto sample_b = torch::tensor({{5, 6, 7}}, torch::kInt32);
    auto features_a = torch::tensor({{-1.0f}, {2.0f}});
    auto features_b = torch::tensor({{3.0f}});
    const auto [coords, features, labels] = riemann::sparse_collate(
        {sample_a, sample_b}, {features_a, features_b});
    CHECK_EQ(coords.size(0), int64_t(3));
    CHECK_EQ(coords.size(1), int64_t(4));
    CHECK(!labels.has_value());
    CHECK_EQ(coords.select(1, 0).max().item<int64_t>(), int64_t(1));

    riemann::SparseTensor input(features, coords);
    auto output = riemann::functional::relu(input);
    CHECK((output.F() >= 0).all().item<bool>());
    CHECK(output.C().sizes() == input.C().sizes());
}

TEST(regression_sparse_quantize_and_persistence) {
    auto coords = torch::tensor(
        {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto feats = torch::tensor({{1.0f}, {3.0f}, {5.0f}});
    auto quantized = riemann::sparse_quantize(coords, feats);
    CHECK_EQ(quantized.coordinates.size(0), int64_t(2));
    CHECK_EQ(quantized.features->size(0), int64_t(2));
    CHECK_EQ(quantized.inverse_mapping.size(0), int64_t(3));

    riemann::SparseTensor tensor(feats, coords);
    const auto archive =
        std::filesystem::temp_directory_path() / "riemann_sparse_roundtrip.pt";
    riemann::save_sparse_tensor(tensor, archive.string());
    auto loaded = riemann::load_sparse_tensor(archive.string());
    std::filesystem::remove(archive);
    CHECK(torch::equal(tensor.C(), loaded.C()));
    CHECK_ALLCLOSE(tensor.F(), loaded.F(), 1e-6, 1e-6);

    const auto tensor_archive =
        std::filesystem::temp_directory_path() / "riemann_tensor_roundtrip.pt";
    riemann::save_tensors({{"weights", feats}}, tensor_archive.string());
    auto tensors = riemann::load_tensors(tensor_archive.string());
    std::filesystem::remove(tensor_archive);
    CHECK_EQ(tensors.size(), size_t(1));
    CHECK(tensors[0].first == "weights");
    CHECK(torch::equal(tensors[0].second, feats));
}

TEST(regression_label_quantization_marks_conflicts) {
    auto coords = torch::tensor(
        {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto labels = torch::tensor({4, 2, 3}, torch::kInt32);
    auto quantized =
        riemann::sparse_quantize(coords, c10::nullopt, labels, -100);
    CHECK_EQ(quantized.coordinates.size(0), int64_t(2));
    CHECK_EQ(quantized.labels->index({0}).item<int64_t>(), int64_t(4));
    CHECK_EQ(quantized.labels->index({1}).item<int64_t>(), int64_t(-100));
}

TEST(regression_training_helpers_and_gradient_accumulation) {
    riemann::training::SparseSample sample(
        torch::tensor({{1, 2, 3}}, torch::kInt32),
        torch::tensor({{0.5f, 1.0f}}), torch::tensor({1}, torch::kInt64));
    auto batch = riemann::training::sparse_collate_samples({sample, sample});
    CHECK_EQ(batch.batch_size, int64_t(2));
    CHECK_EQ(batch.coordinates.size(0), int64_t(2));
    CHECK_EQ(batch.features.size(0), int64_t(2));
    CHECK_EQ(batch.labels.size(0), int64_t(2));

    riemann::training::SparseDataset dataset({sample});
    CHECK_EQ(dataset.size().value(), size_t(1));
    CHECK_EQ(dataset.get(0).coordinates.size(0), int64_t(1));

    auto parameter = torch::ones(
        {1}, torch::TensorOptions().requires_grad(true));
    riemann::training::OptimizerConfig config;
    config.type = riemann::training::OptimizerType::SGD;
    config.lr = 0.1;
    config.momentum = 0.0;
    auto optimizer =
        riemann::training::create_optimizer({parameter}, config);
    riemann::training::GradientAccumulator accumulator(*optimizer, 2);
    const auto before = parameter.clone();
    CHECK(!accumulator.backward_and_maybe_step(parameter.square().sum()));
    CHECK(torch::equal(parameter, before));
    CHECK(accumulator.backward_and_maybe_step(parameter.square().sum()));
    CHECK(!torch::equal(parameter, before));

    riemann::training::StepLRScheduler scheduler(*optimizer, 1, 0.5);
    const auto old_lr = scheduler.current_lr();
    scheduler.step();
    CHECK_NEAR(scheduler.current_lr(), old_lr * 0.5, 1e-8);
}

TEST(regression_sparse_global_pooling_reductions) {
    auto coordinates = torch::tensor(
        {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features = torch::tensor({{1.0f, 4.0f}, {3.0f, -1.0f}, {5.0f, 2.0f}});
    riemann::SparseTensor input(features, coordinates);
    auto average = riemann::RiemannGlobalAvgPooling()->forward(input);
    auto sum = riemann::RiemannGlobalSumPooling()->forward(input);
    auto maximum = riemann::RiemannGlobalMaxPooling()->forward(input);
    CHECK_ALLCLOSE(average.F(),
                   torch::tensor({{2.0f, 1.5f}, {5.0f, 2.0f}}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(sum.F(),
                   torch::tensor({{4.0f, 3.0f}, {5.0f, 2.0f}}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(maximum.F(),
                   torch::tensor({{3.0f, 4.0f}, {5.0f, 2.0f}}), 1e-6, 1e-6);
    CHECK_EQ(average.C().size(0), int64_t(2));
}

TEST(regression_sparse_local_pooling_and_transpose) {
    std::vector<int32_t> coordinate_values;
    std::vector<float> feature_values;
    for (int32_t x = 0; x < 2; ++x)
      for (int32_t y = 0; y < 2; ++y)
        for (int32_t z = 0; z < 2; ++z) {
          coordinate_values.insert(coordinate_values.end(), {0, x, y, z});
          feature_values.push_back(static_cast<float>(feature_values.size() + 1));
        }
    auto coordinates = torch::from_blob(
        coordinate_values.data(), {8, 4},
        torch::TensorOptions().dtype(torch::kInt32)).clone();
    auto features = torch::from_blob(
        feature_values.data(), {8, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    features.set_requires_grad(true);
    riemann::SparseTensor input(features, coordinates);
    auto average = riemann::RiemannAvgPooling(2, 2, 1, 3)->forward(input);
    auto sum = riemann::RiemannSumPooling(2, 2, 1, 3)->forward(input);
    auto maximum = riemann::RiemannMaxPooling(2, 2, 1, 3)->forward(input);
    CHECK_EQ(average.F().size(0), int64_t(1));
    CHECK_ALLCLOSE(average.F(), torch::tensor({{4.5f}}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(sum.F(), torch::tensor({{36.0f}}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(maximum.F(), torch::tensor({{8.0f}}), 1e-6, 1e-6);
    auto expanded = riemann::RiemannPoolingTranspose(2, 2, 1, true, 3)
                        ->forward(sum);
    CHECK_EQ(expanded.F().size(0), int64_t(8));
    sum.F().sum().backward();
    CHECK(features.grad().defined());
#if !defined(CPU_ONLY)
    if (torch::cuda::is_available()) {
      auto cuda_input = riemann::SparseTensor(
          features.detach().to(torch::kCUDA),
          coordinates.to(torch::kCUDA));
      auto cuda_average =
          riemann::RiemannAvgPooling(2, 2, 1, 3)->forward(cuda_input);
      CHECK_ALLCLOSE(cuda_average.F().cpu(), average.F(), 1e-6, 1e-6);
      auto cuda_expanded =
          riemann::RiemannPoolingTranspose(2, 2, 1, true, 3)
              ->forward(cuda_average);
      CHECK_EQ(cuda_expanded.F().size(0), int64_t(8));
    }
#endif
}

TEST(regression_sparse_union_and_dense_conversion) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        2, riemann::CoordinateMapBackend::CPU);
    auto coords_a = torch::tensor(
        {{0, 0, 0}, {0, 1, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto coords_b = torch::tensor(
        {{0, 1, 0}, {0, 2, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    riemann::SparseTensor a(torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}),
                            coords_a, manager);
    riemann::SparseTensor b(torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}}),
                            coords_b, manager);
    auto concatenated = riemann::cat({a, a});
    CHECK_EQ(concatenated.F().size(1), int64_t(4));
    auto united = riemann::RiemannUnion()->forward({a, b});
    CHECK_EQ(united.F().size(0), int64_t(3));
    CHECK_EQ(united.F().size(1), int64_t(2));
    auto coordinates = riemann::dense_coordinates({1, 2, 2, 2});
    CHECK_EQ(coordinates.size(0), int64_t(4));
    auto dense = torch::zeros({1, 2, 2, 2});
    dense.index_put_({0, 0, 1, 0}, 3.0);
    dense.index_put_({0, 1, 1, 0}, 4.0);
    auto sparse = riemann::to_sparse(dense);
    CHECK_EQ(sparse.F().size(0), int64_t(1));
    CHECK_ALLCLOSE(sparse.F(), torch::tensor({{3.0f, 4.0f}}), 1e-6, 1e-6);
}

TEST(regression_sparse_broadcast_and_pruning) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        2, riemann::CoordinateMapBackend::CPU);
    auto input_coordinates = torch::tensor(
        {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto global_coordinates = torch::tensor(
        {{0, 0, 0}, {1, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    riemann::SparseTensor input(
        torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}),
        input_coordinates, manager);
    riemann::SparseTensor global(
        torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}}),
        global_coordinates, manager);

    auto added = riemann::RiemannBroadcastAddition()->forward(input, global);
    CHECK_ALLCLOSE(
        added.F(),
        torch::tensor({{11.0f, 22.0f}, {13.0f, 24.0f}, {35.0f, 46.0f}}),
        1e-6, 1e-6);
    auto repeated = riemann::RiemannBroadcast()->forward(input, global);
    CHECK_ALLCLOSE(
        repeated.F(),
        torch::tensor({{10.0f, 20.0f}, {10.0f, 20.0f}, {30.0f, 40.0f}}),
        1e-6, 1e-6);
    auto pruned = riemann::RiemannPruning()->forward(
        input, torch::tensor({true, false, true}));
    CHECK_EQ(pruned.F().size(0), int64_t(2));
    CHECK_ALLCLOSE(pruned.F(), torch::tensor({{1.0f, 2.0f}, {5.0f, 6.0f}}),
                   1e-6, 1e-6);
}

TEST(regression_sparse_instance_normalization) {
    auto coordinates = torch::tensor(
        {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features = torch::tensor(
        {{1.0f, 4.0f}, {3.0f, 8.0f}, {5.0f, 2.0f}, {9.0f, 6.0f}},
        torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor input(features, coordinates);
    auto normalized = riemann::RiemannInstanceNorm(2)->forward(input);
    auto stable = riemann::RiemannStableInstanceNorm(2)->forward(input);
    auto expected = torch::tensor(
        {{-1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, 1.0f}});
    CHECK_ALLCLOSE(normalized.F(), expected, 1e-3, 1e-3);
    CHECK_ALLCLOSE(stable.F(), expected, 1e-3, 1e-3);
    normalized.F().square().sum().backward();
    CHECK(features.grad().defined());
}

TEST(regression_residual_and_squeeze_excite_blocks) {
    auto coordinates = torch::tensor(
        {{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features = torch::randn({4, 4}, torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor input(features, coordinates);
    auto block = riemann::BasicBlock(4, 4, 1, 1,
                                     torch::nn::AnyModule(), 0.1, 3);
    auto output = block->forward(input);
    CHECK_EQ(output.F().size(0), int64_t(4));
    CHECK_EQ(output.F().size(1), int64_t(4));
    output.F().sum().backward();
    CHECK(features.grad().defined());

    auto se = riemann::SELayer(4, 2, 3);
    auto scaled = se->forward(input);
    CHECK_EQ(scaled.F().size(0), int64_t(4));
    CHECK_EQ(scaled.F().size(1), int64_t(4));
    auto se_block = riemann::SEBasicBlock(4, 4, 1, 1,
                                          torch::nn::AnyModule(), 2, 0.1, 3);
    CHECK_EQ(se_block->forward(input).F().size(1), int64_t(4));
    auto bottleneck = riemann::Bottleneck(4, 1, 1, 1,
                                          torch::nn::AnyModule(), 0.1, 3);
    CHECK_EQ(bottleneck->forward(input).F().size(1), int64_t(4));
    auto se_bottleneck = riemann::SEBottleneck(
        4, 1, 1, 1, torch::nn::AnyModule(), 2, 0.1, 3);
    CHECK_EQ(se_bottleneck->forward(input).F().size(1), int64_t(4));
    auto sinusoidal = riemann::RiemannSinusoidal(4, 3);
    CHECK_EQ(sinusoidal->forward(input).F().size(1), int64_t(3));
    auto relu = riemann::RiemannReLU(
        torch::nn::ReLUOptions().inplace(false));
    CHECK_EQ(relu->forward(input).F().size(1), int64_t(4));
    CHECK_EQ(riemann::RiemannHardswish()->forward(input).F().size(1),
             int64_t(4));
    CHECK_EQ(riemann::RiemannHardsigmoid()->forward(input).F().size(1),
             int64_t(4));
}

TEST(regression_sparse_multilinear_interpolation) {
    auto coordinates = torch::tensor(
        {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
         {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}},
        torch::TensorOptions().dtype(torch::kInt32));
    auto features = torch::tensor(
        {{0.0f}, {3.0f}, {2.0f}, {5.0f}, {1.0f}, {4.0f}, {3.0f}, {6.0f}},
        torch::TensorOptions().requires_grad(true));
    riemann::SparseTensor input(features, coordinates);
    auto field = torch::tensor({{0.0f, 0.25f, 0.5f, 0.75f}});
    auto interpolation = riemann::RiemannInterpolation();
    auto result = interpolation->forward_full(input, field);
    CHECK_ALLCLOSE(result.output, torch::tensor({{3.5f}}), 1e-6, 1e-6);
    CHECK_EQ(result.in_map.numel(), int64_t(8));
    CHECK_EQ(result.out_map.numel(), int64_t(8));
    result.output.sum().backward();
    CHECK(features.grad().defined());
}

TEST(regression_sparse_transpose_convolution_restores_stride) {
    std::vector<int32_t> coordinates;
    for (int32_t x = 0; x < 4; ++x)
        for (int32_t y = 0; y < 4; ++y)
            for (int32_t z = 0; z < 4; ++z) {
                coordinates.push_back(0);
                coordinates.push_back(x);
                coordinates.push_back(y);
                coordinates.push_back(z);
            }
    auto coords = torch::from_blob(
        coordinates.data(), {64, 4},
        torch::TensorOptions().dtype(torch::kInt32)).clone();
    auto input = riemann::SparseTensor(torch::ones({64, 1}), coords);
    riemann::RiemannConvolution down(1, 1, 2, 2, 1, false, false,
                                     riemann::ConvolutionMode::DEFAULT, 3);
    riemann::RiemannConvolutionTranspose up(
        1, 1, 2, 2, 1, false, false,
        riemann::ConvolutionMode::DEFAULT, 3);

    auto coarse = down->forward(input);
    auto restored = up->forward(coarse);
    CHECK(coarse.tensor_stride() == riemann::stride_type({2, 2, 2}));
    CHECK(restored.tensor_stride() == riemann::stride_type({1, 1, 1}));
    CHECK_EQ(restored.F().size(0), input.F().size(0));
}

TEST(regression_adaptive_is_opt_in) {
    // Constructing an adaptive tensor must not perturb the uniform coordinate manager state.
    auto t = make_legacy(60, 2);
    auto before = t.C().clone();
    auto c = riemann::adaptive::AdaptiveVoxelConfig();
    c.base_voxel_size = {1.0, 1.0, 1.0};
    c.max_depth = 0;
    auto a = riemann::adaptive::AdaptiveSparseTensor::from_voxels(
        torch::zeros({1, 3}, torch::kInt64), torch::zeros({1}, torch::kInt32),
        torch::ones({1, 2}, torch::kFloat64), c);
    CHECK_EQ(a.size(), int64_t(1));
    CHECK(torch::equal(before, t.C()));
}

TEST(regression_adaptive_convolution_refines_output_and_backpropagates) {
    using namespace riemann::adaptive;
    AdaptiveVoxelConfig config;
    config.base_voxel_size = {2.0, 2.0, 2.0};
    config.min_voxel_size = {1.0, 1.0, 1.0};
    config.max_depth = 1;
    auto features = torch::tensor(
        {{2.0}}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    auto input = AdaptiveSparseTensor::from_voxels(
        torch::zeros({1, 3}, torch::kInt64),
        torch::zeros({1}, torch::kInt32), features, config);
    AdaptiveKernelConfig kernel;
    kernel.kernel_size = 1;
    kernel.output_level_shift = 1;
    auto weight = torch::tensor(
        {{{3.0}}}, torch::TensorOptions().dtype(torch::kFloat64)
                        .requires_grad(true));
    auto output = adaptive_convolution(input, weight, kernel);
    CHECK_EQ(output.size(), int64_t(8));
    CHECK(torch::equal(output.levels(), torch::ones({8}, torch::kInt32)));
    CHECK_ALLCLOSE(output.features(), torch::full({8, 1}, 6.0, torch::kFloat64),
                   1e-10, 1e-10);
    output.features().sum().backward();
    CHECK_ALLCLOSE(features.grad(), torch::tensor({{24.0}}, torch::kFloat64),
                   1e-10, 1e-10);
    CHECK_ALLCLOSE(weight.grad(), torch::tensor({{{16.0}}}, torch::kFloat64),
                   1e-10, 1e-10);
}

#ifdef USE_CUDA
TEST(regression_cuda_topology_sort_matches_canonical_order) {
    using namespace riemann::adaptive::detail;
    if (!cuda_topology_available())
        throw riemann_test::Skip("CUDA device not available");

    constexpr int64_t count = (int64_t(1) << 14) + 32;
    std::vector<Key> keys(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const auto key_index = i / 2;
        keys[static_cast<size_t>(i)].level =
            static_cast<uint32_t>(key_index % 5);
        keys[static_cast<size_t>(i)].coordinates = {
            count - key_index, key_index % 17, -key_index, 0};
    }
    const auto original = keys;

    std::vector<int64_t> permutation;
    cuda_sort_keys(keys, permutation);
    CHECK_EQ(static_cast<int64_t>(permutation.size()), count);
    for (int64_t i = 0; i < count; ++i) {
        CHECK(permutation[static_cast<size_t>(i)] >= 0);
        CHECK(permutation[static_cast<size_t>(i)] < count);
        CHECK(keys[static_cast<size_t>(i)] ==
              original[static_cast<size_t>(permutation[static_cast<size_t>(i)])]);
        if (i == 0) continue;
        CHECK(!key_before(keys[static_cast<size_t>(i)],
                          keys[static_cast<size_t>(i - 1)]));
        if (keys[static_cast<size_t>(i)] ==
            keys[static_cast<size_t>(i - 1)])
            CHECK(permutation[static_cast<size_t>(i)] >
                  permutation[static_cast<size_t>(i - 1)]);
    }
}

TEST(regression_cuda_topology_ancestor_detection) {
    using namespace riemann::adaptive::detail;
    if (!cuda_topology_available())
        throw riemann_test::Skip("CUDA device not available");

    constexpr int64_t count = (int64_t(1) << 14) + 32;
    std::vector<Key> keys(static_cast<size_t>(count));
    keys[0].level = 0;
    keys[0].coordinates = {0, 0, 0, 0};
    for (int64_t i = 1; i < count; ++i) {
        const auto child = i - 1;
        keys[static_cast<size_t>(i)].level = 2;
        keys[static_cast<size_t>(i)].coordinates = {
            child / (32 * 32), (child / 32) % 32, child % 32, 0};
    }

    const std::vector<int64_t> offsets{0, 1, 1, count};
    CHECK(cuda_has_active_ancestor(keys, offsets, 1));
    keys[0].coordinates = {100, 0, 0, 0};
    CHECK(!cuda_has_active_ancestor(keys, offsets, 1));
}

TEST(regression_cuda_sparse_convolution_matches_cpu) {
    if (!torch::cuda::is_available())
        throw riemann_test::Skip("CUDA device not available");

    auto cpu_input = make_legacy(96, 2);
    auto gpu_input = riemann::SparseTensor(
        cpu_input.F().to(torch::kCUDA), cpu_input.C().to(torch::kCUDA));
    riemann::RiemannConvolution cpu_conv(
        2, 3, 3, 1, 1, true, false,
        riemann::ConvolutionMode::DEFAULT, 3);
    riemann::RiemannConvolution gpu_conv(
        2, 3, 3, 1, 1, true, false,
        riemann::ConvolutionMode::DEFAULT, 3);
    gpu_conv->to(torch::kCUDA);
    {
        torch::NoGradGuard guard;
        gpu_conv->base()->kernel_.copy_(
            cpu_conv->base()->kernel_.to(torch::kCUDA));
        gpu_conv->base()->bias_.copy_(cpu_conv->base()->bias_.to(torch::kCUDA));
    }

    auto cpu_output = cpu_conv->forward(cpu_input);
    auto gpu_output = gpu_conv->forward(gpu_input);
    CHECK(torch::equal(cpu_output.C(), gpu_output.C().to(torch::kCPU)));
    CHECK_ALLCLOSE(cpu_output.F(), gpu_output.F().to(torch::kCPU), 1e-5, 1e-5);
}
#endif
