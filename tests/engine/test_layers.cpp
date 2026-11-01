// High-level behavioral tests for the public neural-network layers operating
// on sparse tensors: pooling, activations, normalization, pruning, broadcast,
// dense/sparse conversion and residual blocks.
#include "RiemannEngine/riemann.hpp"

#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <vector>

namespace {

riemann::SparseTensor make_random(int64_t n, int64_t channels, int seed = 7,
                                  bool requires_grad = false) {
    torch::manual_seed(seed);
    auto coords = std::get<0>(
        torch::unique_dim(torch::randint(0, 6, {n, 3}, torch::kInt32), 0));
    auto batch = torch::zeros({coords.size(0), 1}, torch::kInt32);
    auto features = torch::randn({coords.size(0), channels});
    if (requires_grad) features.requires_grad_(true);
    return riemann::SparseTensor(features, torch::cat({batch, coords}, 1));
}

riemann::SparseTensor make_two_batches() {
    auto coords = torch::tensor({{0, 0, 0, 0},
                                 {0, 1, 0, 0},
                                 {1, 0, 0, 0},
                                 {1, 2, 2, 2}},
                                torch::kInt32);
    auto features = torch::tensor({{1.0f, 2.0f},
                                   {3.0f, 4.0f},
                                   {10.0f, 20.0f},
                                   {30.0f, 40.0f}});
    return riemann::SparseTensor(features, coords);
}

int64_t row_for_batch(const riemann::SparseTensor& tensor, int batch) {
    auto coords = tensor.C();
    for (int64_t row = 0; row < coords.size(0); ++row)
        if (coords[row][0].item<int>() == batch) return row;
    throw riemann_test::Failure("no row found for requested batch index");
}

int64_t row_for_coordinate(const riemann::SparseTensor& tensor, int x, int y,
                           int z) {
    auto coords = tensor.C();
    for (int64_t row = 0; row < coords.size(0); ++row)
        if (coords[row][1].item<int>() == x && coords[row][2].item<int>() == y &&
            coords[row][3].item<int>() == z)
            return row;
    throw riemann_test::Failure("no row found for requested coordinate");
}

}  // namespace

TEST(layers_global_pooling_reduces_per_batch) {
    auto input = make_two_batches();

    riemann::RiemannGlobalAvgPooling avg;
    auto averaged = avg->forward(input);
    CHECK_EQ(averaged.F().size(0), int64_t(2));
    CHECK_ALLCLOSE(averaged.F()[row_for_batch(averaged, 0)],
                   torch::tensor({2.0f, 3.0f}), 1e-6, 1e-6);
    CHECK_ALLCLOSE(averaged.F()[row_for_batch(averaged, 1)],
                   torch::tensor({20.0f, 30.0f}), 1e-6, 1e-6);

    riemann::RiemannGlobalMaxPooling max;
    auto maxed = max->forward(input);
    CHECK_ALLCLOSE(maxed.F()[row_for_batch(maxed, 0)],
                   torch::tensor({3.0f, 4.0f}), 1e-6, 1e-6);

    riemann::RiemannGlobalSumPooling sum;
    auto summed = sum->forward(input);
    CHECK_ALLCLOSE(summed.F()[row_for_batch(summed, 1)],
                   torch::tensor({40.0f, 60.0f}), 1e-6, 1e-6);
}

TEST(layers_strided_pooling_downsamples_cells) {
    auto coords = torch::tensor({{0, 0, 0, 0},
                                 {0, 1, 0, 0},
                                 {0, 0, 1, 0},
                                 {0, 2, 0, 0},
                                 {0, 3, 1, 1}},
                                torch::kInt32);
    auto features =
        torch::tensor({{1.0f}, {2.0f}, {3.0f}, {10.0f}, {20.0f}});
    riemann::SparseTensor input(features, coords);

    riemann::RiemannAvgPooling avg(2, 2);
    auto averaged = avg->forward(input);
    CHECK_EQ(averaged.F().size(0), int64_t(2));
    auto stride = averaged.tensor_stride();
    CHECK_EQ(stride.size(), size_t(3));
    for (auto value : stride) CHECK_EQ(value, 2);
    CHECK_NEAR(
        averaged.F()[row_for_coordinate(averaged, 0, 0, 0)][0].item<float>(),
        2.0f, 1e-5);
    CHECK_NEAR(
        averaged.F()[row_for_coordinate(averaged, 2, 0, 0)][0].item<float>(),
        15.0f, 1e-5);

    riemann::RiemannMaxPooling max(2, 2);
    auto maxed = max->forward(input);
    CHECK_NEAR(maxed.F()[row_for_coordinate(maxed, 0, 0, 0)][0].item<float>(),
               3.0f, 1e-5);
    CHECK_NEAR(maxed.F()[row_for_coordinate(maxed, 2, 0, 0)][0].item<float>(),
               20.0f, 1e-5);

    riemann::RiemannSumPooling sum(2, 2);
    auto summed = sum->forward(input);
    CHECK_NEAR(summed.F()[row_for_coordinate(summed, 0, 0, 0)][0].item<float>(),
               6.0f, 1e-5);
}

TEST(layers_pooling_transpose_restores_tensor_stride) {
    auto input = make_random(60, 2);
    riemann::RiemannAvgPooling down(2, 2);
    auto pooled = down->forward(input);

    riemann::RiemannPoolingTranspose up(2, 2);
    auto restored = up->forward(pooled);

    auto stride = restored.tensor_stride();
    CHECK_EQ(stride.size(), size_t(3));
    for (auto value : stride) CHECK_EQ(value, 1);
    CHECK(restored.F().size(0) >= pooled.F().size(0));
    CHECK(restored.F().isfinite().all().item<bool>());
}

TEST(layers_conv_downsample_then_transpose_upsample) {
    auto input = make_random(80, 3);
    riemann::RiemannConvolution down(3, 4, 2, 2, 1, false, false,
                                     riemann::ConvolutionMode::DEFAULT, 3);
    auto coarse = down->forward(input);
    CHECK_EQ(coarse.F().size(1), int64_t(4));
    CHECK(coarse.F().size(0) < input.F().size(0));
    for (auto value : coarse.tensor_stride()) CHECK_EQ(value, 2);

    riemann::RiemannConvolutionTranspose up(
        4, 3, 2, 2, 1, false, false, riemann::ConvolutionMode::DEFAULT, 3);
    auto fine = up->forward(coarse);
    CHECK_EQ(fine.F().size(1), int64_t(3));
    for (auto value : fine.tensor_stride()) CHECK_EQ(value, 1);
    CHECK(fine.F().size(0) >= coarse.F().size(0));
    CHECK(fine.F().isfinite().all().item<bool>());
}

TEST(layers_linear_norm_relu_preserve_geometry) {
    auto input = make_random(100, 3);
    riemann::RiemannLinear linear(3, 8);
    riemann::RiemannBatchNorm norm(8);
    riemann::RiemannReLU relu;

    auto output = relu->forward(norm->forward(linear->forward(input)));

    CHECK_EQ(output.F().size(0), input.F().size(0));
    CHECK_EQ(output.F().size(1), int64_t(8));
    CHECK(torch::equal(output.C(), input.C()));
    CHECK(output.F().min().item<float>() >= 0.0f);
}

TEST(layers_feature_activations_match_dense_math) {
    auto input = make_random(50, 4);

    riemann::RiemannTanh tanh_layer;
    auto tanh_out = tanh_layer->forward(input);
    CHECK_ALLCLOSE(tanh_out.F(), torch::tanh(input.F()), 1e-6, 1e-6);
    CHECK(torch::equal(tanh_out.C(), input.C()));

    riemann::RiemannSigmoid sigmoid_layer;
    auto sigmoid_out = sigmoid_layer->forward(input);
    CHECK_ALLCLOSE(sigmoid_out.F(), torch::sigmoid(input.F()), 1e-6, 1e-6);

    riemann::RiemannHardswish hardswish_layer;
    auto hardswish_out = hardswish_layer->forward(input);
    CHECK_ALLCLOSE(hardswish_out.F(), at::hardswish(input.F()), 1e-6, 1e-6);
}

TEST(layers_pruning_keeps_masked_rows) {
    auto input = make_two_batches();
    auto mask = torch::tensor({true, false, true, false});

    riemann::RiemannPruning pruning;
    auto pruned = pruning->forward(input, mask);

    CHECK_EQ(pruned.F().size(0), int64_t(2));
    // Each surviving row must correspond to a kept input row.
    auto input_coords = input.C();
    auto input_features = input.F();
    auto pruned_coords = pruned.C();
    for (int64_t row = 0; row < pruned_coords.size(0); ++row) {
        bool found = false;
        for (int64_t source : {int64_t(0), int64_t(2)}) {
            if (torch::equal(pruned_coords[row], input_coords[source])) {
                CHECK_ALLCLOSE(pruned.F()[row], input_features[source], 1e-6,
                               1e-6);
                found = true;
            }
        }
        CHECK(found);
    }
}

TEST(layers_cat_sum_mean_var_on_shared_map) {
    auto first = make_random(70, 2);
    riemann::SparseTensor second(first.F() * 2, first.coordinate_map_key(),
                                 first.coordinate_manager());

    auto concatenated = riemann::cat({first, second});
    CHECK_EQ(concatenated.F().size(1), int64_t(4));
    CHECK_ALLCLOSE(concatenated.F().narrow(1, 0, 2), first.F(), 1e-6, 1e-6);

    auto summed = riemann::sum({first, second});
    CHECK_ALLCLOSE(summed.F(), first.F() * 3, 1e-6, 1e-6);

    auto averaged = riemann::mean({first, second});
    CHECK_ALLCLOSE(averaged.F(), first.F() * 1.5, 1e-6, 1e-6);

    auto variance = riemann::var({first, second});
    CHECK_ALLCLOSE(variance.F(), first.F().square() * 0.25, 1e-5, 1e-6);
}

TEST(layers_broadcast_combines_global_statistics) {
    auto input = make_two_batches();
    riemann::RiemannGlobalAvgPooling avg;
    auto global = avg->forward(input);

    riemann::RiemannBroadcastAddition add;
    auto shifted = add->forward(input, global);
    CHECK_EQ(shifted.F().size(0), input.F().size(0));
    auto coords = input.C();
    for (int64_t row = 0; row < coords.size(0); ++row) {
        auto expected = coords[row][0].item<int>() == 0
                            ? torch::tensor({2.0f, 3.0f})
                            : torch::tensor({20.0f, 30.0f});
        CHECK_ALLCLOSE(shifted.F()[row], input.F()[row] + expected, 1e-5, 1e-5);
    }

    riemann::RiemannBroadcastConcatenation concat;
    auto widened = concat->forward(input, global);
    CHECK_EQ(widened.F().size(1), int64_t(4));
    CHECK_ALLCLOSE(widened.F().narrow(1, 0, 2), input.F(), 1e-6, 1e-6);
}

TEST(layers_to_sparse_dense_roundtrip) {
    auto dense = torch::zeros({2, 3, 4, 5});
    dense.index_put_({0, torch::indexing::Slice(), 1, 2},
                     torch::tensor({1.0f, 2.0f, 3.0f}));
    dense.index_put_({1, torch::indexing::Slice(), 3, 4},
                     torch::tensor({4.0f, 5.0f, 6.0f}));

    auto sparse = riemann::to_sparse(dense);
    CHECK_EQ(sparse.F().size(0), int64_t(2));
    CHECK_EQ(sparse.F().size(1), int64_t(3));
    CHECK_EQ(sparse.dimension(), 2);

    // Anchor reconstruction at the origin: dense() otherwise shifts by the
    // minimum occupied coordinate.
    auto restored = sparse.dense(std::vector<int64_t>{2, 3, 4, 5},
                                 torch::zeros({2}, torch::kInt32));
    CHECK_ALLCLOSE(restored, dense, 1e-6, 1e-6);
}

TEST(layers_union_merges_coordinate_sets) {
    auto manager = std::make_shared<riemann::CoordinateManager>(
        3, riemann::CoordinateMapBackend::CPU);
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    riemann::SparseTensor first(
        torch::tensor({{1.0f}, {2.0f}}),
        torch::tensor({{0, 0, 0, 0}, {0, 1, 0, 0}}, options), manager);
    riemann::SparseTensor second(
        torch::tensor({{10.0f}, {20.0f}}),
        torch::tensor({{0, 1, 0, 0}, {0, 2, 0, 0}}, options), manager);

    riemann::RiemannUnion unite;
    auto merged = unite->forward({first, second});

    CHECK_EQ(merged.F().size(0), int64_t(3));  // three distinct coordinates
    CHECK_NEAR(merged.F()[row_for_coordinate(merged, 0, 0, 0)][0].item<float>(),
               1.0f, 1e-5);
    CHECK_NEAR(merged.F()[row_for_coordinate(merged, 1, 0, 0)][0].item<float>(),
               12.0f, 1e-5);  // overlapping coordinate sums contributions
    CHECK_NEAR(merged.F()[row_for_coordinate(merged, 2, 0, 0)][0].item<float>(),
               20.0f, 1e-5);
}

TEST(layers_residual_block_preserves_shape_and_backpropagates) {
    auto input = make_random(90, 8, 11, /*requires_grad=*/true);
    riemann::BasicBlock block(8, 8);

    auto output = block->forward(input);
    CHECK_EQ(output.F().size(0), input.F().size(0));
    CHECK_EQ(output.F().size(1), int64_t(8));

    output.F().sum().backward();
    auto kernel_grad = block->conv1_->base()->kernel_.grad();
    CHECK(kernel_grad.defined());
    CHECK(kernel_grad.abs().sum().item<double>() > 0.0);
}

TEST(layers_residual_block_registers_downsample) {
    auto input = make_random(90, 8, 12);
    riemann::RiemannConvolution projection(8, 16, 1, 2);
    auto projection_kernel = projection->base()->kernel_;
    riemann::BasicBlock block(8, 16, /*stride=*/2, /*dilation=*/1,
                              torch::nn::AnyModule(projection));

    bool found = false;
    for (const auto& p : block->parameters())
        found = found || p.is_same(projection_kernel);
    CHECK(found);

    auto output = block->forward(input);
    CHECK_EQ(output.F().size(1), int64_t(16));
    CHECK_EQ(output.tensor_stride()[0], 2);
}
