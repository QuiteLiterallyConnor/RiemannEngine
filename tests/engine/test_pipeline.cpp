// End-to-end pipeline tests: batched data -> network forward/backward,
// optimization loops, schedulers and model/tensor persistence.
#include "RiemannEngine/riemann.hpp"

#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace {

torch::Tensor unique_cloud(int64_t n, int seed) {
    torch::manual_seed(seed);
    return std::get<0>(
        torch::unique_dim(torch::randint(0, 8, {n, 3}, torch::kInt32), 0));
}

riemann::SparseTensor make_batched_input(int64_t channels,
                                         bool requires_grad = false) {
    auto cloud_a = unique_cloud(40, 21);
    auto cloud_b = unique_cloud(30, 22);
    auto features = torch::randn(
        {cloud_a.size(0) + cloud_b.size(0), channels});
    if (requires_grad) features.requires_grad_(true);
    auto coords = riemann::batched_coordinates({cloud_a, cloud_b});
    return riemann::SparseTensor(features, coords);
}

std::string temp_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace

TEST(pipeline_classification_network_forward_and_backward) {
    auto input = make_batched_input(3, /*requires_grad=*/true);

    riemann::RiemannConvolution conv(3, 8, 3, 1, 1, true, false,
                                     riemann::ConvolutionMode::DEFAULT, 3);
    riemann::RiemannBatchNorm norm(8);
    riemann::RiemannReLU relu;
    riemann::RiemannMaxPooling pool(2, 2);
    riemann::RiemannGlobalAvgPooling global;
    riemann::RiemannLinear classifier(8, 4);

    auto hidden = relu->forward(norm->forward(conv->forward(input)));
    auto pooled = pool->forward(hidden);
    auto embedding = global->forward(pooled);
    auto logits = classifier->forward(embedding);

    CHECK_EQ(logits.F().size(0), int64_t(2));  // one row per batch
    CHECK_EQ(logits.F().size(1), int64_t(4));  // one column per class

    auto targets = torch::tensor({0, 2}, torch::kInt64);
    auto loss = riemann::training::cross_entropy_sparse(logits, targets);
    CHECK(loss.isfinite().item<bool>());
    loss.backward();

    // Gradients must reach both the convolution kernel and the raw features.
    auto kernel_grad = conv->base()->kernel_.grad();
    CHECK(kernel_grad.defined());
    CHECK(kernel_grad.abs().sum().item<double>() > 0.0);
    CHECK(input.F().grad().defined());
    CHECK(input.F().grad().isfinite().all().item<bool>());
}

TEST(pipeline_training_loop_reduces_loss) {
    torch::manual_seed(5);
    auto input = make_batched_input(3);
    auto target = torch::randn({input.F().size(0), 2});

    riemann::RiemannConvolution conv(3, 16, 3, 1, 1, true, false,
                                     riemann::ConvolutionMode::DEFAULT, 3);
    riemann::RiemannReLU relu;
    riemann::RiemannLinear head(16, 2);

    std::vector<torch::Tensor> parameters;
    for (auto& parameter : conv->parameters()) parameters.push_back(parameter);
    for (auto& parameter : head->parameters()) parameters.push_back(parameter);

    riemann::training::OptimizerConfig config;
    config.type = riemann::training::OptimizerType::Adam;
    config.lr = 1e-2;
    config.weight_decay = 0.0;
    auto optimizer = riemann::training::create_optimizer(parameters, config);

    auto evaluate = [&]() {
        auto output = head->forward(relu->forward(conv->forward(input)));
        return torch::mse_loss(output.F(), target);
    };

    const double initial_loss = evaluate().item<double>();
    for (int step = 0; step < 30; ++step) {
        optimizer->zero_grad();
        auto loss = evaluate();
        loss.backward();
        optimizer->step();
    }
    const double final_loss = evaluate().item<double>();

    CHECK(std::isfinite(final_loss));
    CHECK(final_loss < initial_loss);
}

TEST(pipeline_gradient_accumulator_steps_after_accumulation) {
    auto parameter = torch::ones({1}, torch::requires_grad());
    torch::optim::SGD optimizer({parameter}, torch::optim::SGDOptions(0.5));
    riemann::training::GradientAccumulator accumulator(optimizer, 2);

    // d(loss)/d(parameter) = 2, scaled internally by 1/accumulation_steps.
    CHECK(!accumulator.backward_and_maybe_step((parameter * 2.0).sum()));
    CHECK_NEAR(parameter.item<float>(), 1.0f, 1e-6);  // no step yet
    CHECK_EQ(accumulator.current_step(), 1);

    CHECK(accumulator.backward_and_maybe_step((parameter * 2.0).sum()));
    // Accumulated gradient is 1 + 1 = 2; SGD update: 1 - 0.5 * 2 = 0.
    CHECK_NEAR(parameter.item<float>(), 0.0f, 1e-6);
    CHECK_EQ(accumulator.current_step(), 0);
}

TEST(pipeline_step_lr_scheduler_decays_learning_rate) {
    auto parameter = torch::ones({1}, torch::requires_grad());
    torch::optim::SGD optimizer({parameter}, torch::optim::SGDOptions(0.1));
    riemann::training::StepLRScheduler scheduler(optimizer, 2, 0.5);

    scheduler.step();
    CHECK_NEAR(scheduler.current_lr(), 0.1, 1e-9);  // not yet at step boundary
    scheduler.step();
    CHECK_NEAR(scheduler.current_lr(), 0.05, 1e-9);
    scheduler.step();
    scheduler.step();
    CHECK_NEAR(scheduler.current_lr(), 0.025, 1e-9);
    CHECK_EQ(scheduler.current_epoch(), 4);
}

TEST(pipeline_sparse_tensor_persistence_roundtrip) {
    auto coords = torch::tensor({{0, 0, 0, 0},
                                 {0, 1, 0, 0},
                                 {1, 0, 2, 0},
                                 {1, 2, 2, 2}},
                                torch::kInt32);
    auto features = torch::tensor({{1.0f, 2.0f},
                                   {3.0f, 4.0f},
                                   {10.0f, 20.0f},
                                   {30.0f, 40.0f}});
    riemann::SparseTensor original(features, coords);

    const auto path = temp_path("riemann_engine_sparse_roundtrip.pt");
    riemann::save_sparse_tensor(original, path);
    auto restored = riemann::load_sparse_tensor(path);
    std::filesystem::remove(path);

    CHECK_EQ(restored.dimension(), original.dimension());
    CHECK(restored.tensor_stride() == original.tensor_stride());
    CHECK_EQ(restored.F().size(0), original.F().size(0));

    // Every original row must survive with identical features.
    auto restored_coords = restored.C();
    auto original_coords = original.C();
    for (int64_t row = 0; row < original_coords.size(0); ++row) {
        bool found = false;
        for (int64_t other = 0; other < restored_coords.size(0); ++other) {
            if (torch::equal(original_coords[row], restored_coords[other])) {
                CHECK_ALLCLOSE(original.F()[row], restored.F()[other], 1e-6,
                               1e-6);
                found = true;
                break;
            }
        }
        CHECK(found);
    }
}

TEST(pipeline_model_persistence_roundtrip) {
    auto input = make_batched_input(3);
    riemann::RiemannConvolution trained(3, 5, 3, 1, 1, true, false,
                                        riemann::ConvolutionMode::DEFAULT, 3);
    auto expected = trained->forward(input);

    const auto path = temp_path("riemann_engine_model_roundtrip.pt");
    riemann::save_model(trained, path);

    riemann::RiemannConvolution loaded(3, 5, 3, 1, 1, true, false,
                                       riemann::ConvolutionMode::DEFAULT, 3);
    riemann::load_model(loaded, path);
    std::filesystem::remove(path);

    auto actual = loaded->forward(input);
    CHECK_ALLCLOSE(actual.F(), expected.F(), 1e-6, 1e-6);
}

TEST(pipeline_decomposition_splits_collated_batches) {
    auto cloud_a = unique_cloud(25, 31);
    auto cloud_b = unique_cloud(15, 32);
    auto features =
        torch::randn({cloud_a.size(0) + cloud_b.size(0), 4});
    riemann::SparseTensor batch(
        features, riemann::batched_coordinates({cloud_a, cloud_b}));

    auto per_batch_features = batch.decomposed_features();
    CHECK_EQ(per_batch_features.size(), size_t(2));
    CHECK_EQ(per_batch_features[0].size(0), cloud_a.size(0));
    CHECK_EQ(per_batch_features[1].size(0), cloud_b.size(0));

    auto pairs = batch.decomposed_coordinates_and_features();
    CHECK_EQ(pairs.size(), size_t(2));
    for (size_t batch_index = 0; batch_index < pairs.size(); ++batch_index) {
        const auto& [coordinates, values] = pairs[batch_index];
        CHECK_EQ(coordinates.size(0), values.size(0));
        CHECK_EQ(coordinates.size(1), int64_t(4));  // batch column retained
        CHECK_EQ(values.size(1), int64_t(4));
        // All rows must belong to the decomposed batch.
        CHECK((coordinates.select(1, 0) == int64_t(batch_index))
                  .all()
                  .item<bool>());
    }
}
