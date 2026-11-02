#include "splat/dnerf.hpp"
#include "splat/fourdgs.hpp"
#include "splat/harmonics.hpp"
#include "splat/trainer4d.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/Parallel.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace riemann::gaussian;

namespace {

Camera camera4d(int64_t size = 24) {
  return Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0},
                         size, size, 1.0);
}

Gaussian4DModel model4d(int64_t n = 1) {
  return Gaussian4DModel(
      at::zeros({n, 3}),
      (at::tensor({0.8f, 0.3f, 0.1f}).repeat({n, 1}) - 0.5) / kShC0,
      at::zeros({n, 0, 3}), at::full({n, 1}, 2.0),
      at::full({n, 3}, std::log(0.2)),
      at::tensor({1.f, 0.f, 0.f, 0.f}).repeat({n, 1}),
      at::zeros({n, 3}), at::full({n, 1}, 0.5),
      at::full({n, 1}, std::log(0.4)));
}

std::string bouncingballs_path() {
  const char *root = std::getenv("RIEMANN_TEST_DATA_DIR");
  return (std::filesystem::path(root ? root : RIEMANN_TEST_DATA_DIR) /
          "bouncingballs").string();
}

struct ThreadLimit {
  int previous = at::get_num_threads();
  ThreadLimit() { at::set_num_threads(1); }
  ~ThreadLimit() { at::set_num_threads(previous); }
};

} // namespace

TEST(gaussian4d_conditional_mean_opacity_and_snapshot) {
  auto model = model4d();
  model.velocity().fill_(0.4);
  CHECK_NEAR(model.positions(0.75)[0][0].item<float>(), 0.1, 1e-6);
  CHECK_NEAR(model.opacities(0.9)[0][0].item<float>(),
             model.opacities()[0][0].item<float>() * std::exp(-0.5), 1e-6);
  const auto snapshot = model.at_time(0.7);
  CHECK_ALLCLOSE(render(snapshot, camera4d()).image,
                 render(model, camera4d(), 0.7).image, 1e-5, 1e-6);
  CHECK_EQ(model.parameters().size(), 9);
  auto copied = model.clone();
  copied.velocity().zero_();
  CHECK_NEAR(model.velocity()[0][0].item<float>(), 0.4, 1e-6);
  CHECK_EQ(model.select(at::tensor({0}, at::kLong)).size(), 1);
}

TEST(gaussian4d_rejects_invalid_shapes_and_time) {
  auto model = model4d();
  auto parameters = model.parameters();
  parameters[6] = at::zeros({1, 1});
  CHECK_THROWS(model.set_parameters(parameters));
  CHECK_THROWS(model.positions(std::numeric_limits<double>::quiet_NaN()));
  CHECK_THROWS(model.opacities(std::numeric_limits<double>::infinity()));
  CHECK_THROWS(Gaussian4DModel::from_model(model.at_time(0.5), 0.0));
}

TEST(gaussian4d_standard_export) {
  const auto path = std::filesystem::temp_directory_path() /
      ("riemann-4dgs-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) + ".4dgs");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
  } cleanup{path};
  auto model = model4d(5);
  model.velocity().fill_(0.25);
  FourDGSExportOptions options;
  options.chunk_size = 2;
  const auto result = save_four_dgs(model, path.string(), options);
  CHECK_EQ(result.gaussians, 5);
  CHECK_EQ(result.bytes, std::filesystem::file_size(path));
  CHECK_EQ(result.clamped_rgb_components, 0);
  CHECK_NEAR(model.velocity()[0][0].item<float>(), 0.25, 0);
  std::ifstream file(path, std::ios::binary);
  std::string magic(8, '\0');
  file.read(magic.data(), 8);
  CHECK_EQ(magic, std::string("\x89" "4DGS1\r\n", 8));
  file.seekg(-8, std::ios::end);
  std::string tail(8, '\0');
  file.read(tail.data(), 8);
  CHECK_EQ(magic, tail);
  file.close();
  CHECK_THROWS(save_four_dgs(model, path.string()));
  options.cutoff = 1;
  CHECK_THROWS(save_four_dgs(model, path.string() + ".invalid", options));
}

TEST(gaussian4d_temporal_gradients_match_finite_difference) {
  auto model = model4d();
  model.velocity().fill_(0.15);
  model.enable_gradients();
  auto output = render(model, camera4d(), 0.7);
  auto weights = at::arange(24, at::kFloat).reshape({1, 24, 1});
  (output.image * weights).sum().backward();
  for (const int index : {6, 7, 8}) {
    const auto parameter = model.parameters()[index];
    CHECK(parameter.grad().defined());
    CHECK(at::isfinite(parameter.grad()).all().item<bool>());
    const double analytic = parameter.grad().reshape({-1})[0].item<double>();
    CHECK(std::abs(analytic) > 1e-4);
    auto plus = model.clone(), minus = model.clone();
    constexpr double epsilon = 1e-3;
    plus.parameters()[index].reshape({-1})[0].add_(epsilon);
    minus.parameters()[index].reshape({-1})[0].sub_(epsilon);
    const double numerical =
        ((render(plus, camera4d(), 0.7).image * weights).sum().item<double>() -
         (render(minus, camera4d(), 0.7).image * weights).sum().item<double>()) /
        (2 * epsilon);
    CHECK_NEAR(analytic, numerical, 0.03 * std::abs(numerical) + 0.1);
  }
}

TEST(gaussian4d_trainer_learns_motion_across_times) {
  ThreadLimit threads;
  auto truth = model4d();
  truth.velocity()[0][0] = 0.8;
  const auto camera = camera4d();
  const std::vector<double> times{0.1, 0.5, 0.9};
  std::vector<at::Tensor> targets;
  for (const auto time : times)
    targets.push_back(render(truth, camera, time).image.detach());
  auto model = model4d();
  Gaussian4DTrainerConfig config;
  config.densify = false;
  config.lambda_ssim = 0.0;
  config.lr_velocity = 0.025;
  config.lr_position = config.lr_position_final = 0.002;
  Gaussian4DTrainer trainer(model, config);
  auto error = [&]() {
    double loss = 0.0;
    for (size_t i = 0; i < times.size(); ++i)
      loss += riemann::gaussian::l1_loss(trainer.render_image(camera, times[i]), targets[i])
                  .item<double>();
    return loss / times.size();
  };
  const double initial = error();
  for (int step = 0; step < 120; ++step) {
    const auto i = static_cast<size_t>(step) % times.size();
    CHECK(std::isfinite(trainer.step(camera, times[i], targets[i]).loss));
  }
  CHECK(error() < 0.3 * initial);
  CHECK(model.velocity()[0][0].item<float>() > 0.5);
}

TEST(gaussian4d_densification_preserves_temporal_rows_and_trains_again) {
  ThreadLimit threads;
  auto model = model4d(4);
  model.scale_raw().fill_(std::log(0.02));
  model.velocity().fill_(0.25);
  model.opacity_raw()[0][0] = -12;
  Gaussian4DTrainerConfig config;
  config.densify_grad_threshold = 0;
  Gaussian4DTrainer trainer(model, config);
  const auto counts = trainer.densify_and_prune();
  CHECK(counts.first > 0);
  CHECK(counts.second > 0);
  CHECK_EQ(model.size(), 4 + counts.first - counts.second);
  for (const auto &parameter : model.parameters()) {
    CHECK_EQ(parameter.size(0), model.size());
    CHECK(parameter.is_leaf() && parameter.requires_grad());
  }
  CHECK_ALLCLOSE(model.velocity(), at::full_like(model.velocity(), 0.25),
                 0, 0);
  CHECK(std::isfinite(trainer.step(camera4d(), 0.6,
                                   at::zeros({24, 24, 3})).loss));
  trainer.reset_opacity();
  CHECK(model.opacities().max().item<float>() <= 0.01001f);
}

TEST(gaussian4d_bouncingballs_loader_and_training) {
  ThreadLimit threads;
  const auto path = bouncingballs_path();
  if (!std::filesystem::exists(path))
    throw riemann_test::Skip("bouncingballs dataset not found");
  at::manual_seed(42);
  DNerfLoadOptions options;
  options.downscale = 25;
  options.stride = 25;
  options.max_frames = 6;
  const auto views = load_dnerf_views(path, "train", options);
  CHECK_EQ(views.size(), 6);
  CHECK_NEAR(views.front().time, 0.0, 1e-6);
  CHECK(views.back().time > 0.8);
  for (const auto &view : views) {
    CHECK(view.image.sizes() == at::IntArrayRef({32, 32, 3}));
    CHECK(at::isfinite(view.image).all().item<bool>());
    CHECK(view.image.min().item<float>() >= 0);
    CHECK(view.image.max().item<float>() <= 1);
    // The dataset cameras look towards the scene origin.
    const auto origin = view.camera.translation();
    CHECK(origin[2].item<float>() > 3.0f);
    CHECK(origin.slice(0, 0, 2).norm().item<float>() < 0.5f);
  }
  auto model = Gaussian4DModel::from_points(
      at::rand({256, 3}) * 2 - 1, at::full({256, 3}, 0.5), 0, 0.1, 0.5);
  Gaussian4DTrainerConfig config;
  config.densify = false;
  config.lambda_ssim = 0.2;
  config.lr_position = config.lr_position_final = 0.005;
  config.lr_dc = 0.025;
  config.lr_velocity = 0.01;
  config.render.background = at::ones({3});
  Gaussian4DTrainer trainer(model, config);
  auto error = [&]() {
    double loss = 0;
    for (const auto &view : views)
      loss += riemann::gaussian::l1_loss(trainer.render_image(view.camera, view.time), view.image)
                  .item<double>();
    return loss / views.size();
  };
  const double initial = error();
  for (int step = 0; step < 120; ++step) {
    const auto &view = views[static_cast<size_t>(step) % views.size()];
    CHECK(std::isfinite(trainer.step(view.camera, view.time, view.image).loss));
  }
  const double final = error();
  std::cout << "         bouncingballs mean L1: " << initial << " -> "
            << final << " (6 views, 120 steps, 32x32)\n";
  CHECK(final < initial * 0.85);
  CHECK(model.velocity().abs().max().item<float>() > 0.01);
  options.max_frames = 1;
  CHECK_EQ(load_dnerf_views(path, "val", options).size(), 1);
  CHECK_EQ(load_dnerf_views(path, "test", options).size(), 1);
  options.downscale = 0;
  CHECK_THROWS(load_dnerf_views(path, "train", options));
}

TEST(gaussian4d_cuda_forward_backward_matches_cpu) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw riemann_test::Skip("CUDA runtime or device is unavailable");
  ThreadLimit threads;
  auto cpu = model4d(3);
  cpu.xyz().copy_(at::tensor({-.2f, .1f, 0.f, .2f, -.1f, .1f,
                              0.f, .2f, -.1f}).reshape({3, 3}));
  cpu.velocity().fill_(0.15);
  auto gpu = cpu.to(torch::Device(torch::kCUDA, 0));
  cpu.enable_gradients();
  gpu.enable_gradients();
  auto left = render(cpu, camera4d(), 0.7);
  auto right = render(gpu, camera4d(), 0.7);
  CHECK_ALLCLOSE(left.image, right.image.cpu(), 1e-3, 1e-3);
  const auto weights = at::arange(24, at::kFloat).reshape({1, 24, 1});
  (left.image * weights).sum().backward();
  (right.image * weights.to(at::kCUDA)).sum().backward();
  for (size_t i = 0; i < cpu.parameters().size(); ++i) {
    const auto a = cpu.parameters()[i], b = gpu.parameters()[i];
    if (a.numel() == 0)
      continue;
    CHECK(a.grad().defined() && b.grad().defined());
    const double scale = a.grad().abs().max().item<double>();
    CHECK_ALLCLOSE(a.grad(), b.grad().cpu(), 2e-2, 2e-2 * scale + 1e-4);
  }

  Gaussian4DTrainerConfig config;
  config.densify = false;
  Gaussian4DTrainer trainer(gpu, config);
  CHECK(std::isfinite(trainer.step(camera4d(), 0.7, left.image.detach()).loss));
#else
  throw riemann_test::Skip("CUDA kernels are not compiled in this build");
#endif
}

TEST(gaussian4d_checkpoint_resumes_adam_statistics_and_rng) {
  ThreadLimit threads;
  std::vector<torch::Device> devices{torch::Device(torch::kCPU)};
#ifdef USE_CUDA
  if (at::cuda::is_available())
    devices.emplace_back(torch::kCUDA, 0);
#endif
  for (const auto &device : devices) {
    const auto path = std::filesystem::temp_directory_path() /
        ("riemann4d-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".pt");
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::error_code error;
        std::filesystem::remove(path, error);
      }
    } cleanup{path};
    auto model = model4d(3).to(device);
    model.velocity().fill_(0.1);
    Gaussian4DTrainerConfig config;
    config.densify = false;
    config.lambda_ssim = 0;
    config.densify_grad_threshold = 0;
    Gaussian4DTrainer trainer(model, config);
    const auto target = at::zeros({24, 24, 3});
    for (int i = 0; i < 3; ++i)
      trainer.step(camera4d(), 0.7, target);
    trainer.densify_and_prune();
    trainer.save_checkpoint(path.string());
    const auto expected_rng = at::rand({8}, at::TensorOptions().device(device));
    const auto expected = trainer.step(camera4d(), 0.7, target);
    auto restored_model = model4d().to(device);
    Gaussian4DTrainer restored(restored_model, config);
    restored.load_checkpoint(path.string());
    CHECK_EQ(restored.iteration(), 3);
    CHECK_EQ(restored_model.size(), model.size());
    CHECK_ALLCLOSE(at::rand({8}, at::TensorOptions().device(device)),
                   expected_rng, 0, 0);
    const auto actual = restored.step(camera4d(), 0.7, target);
    CHECK_NEAR(actual.loss, expected.loss, 1e-6);
    for (size_t i = 0; i < 9; ++i)
      CHECK_ALLCLOSE(restored_model.parameters()[i], model.parameters()[i],
                     1e-5, 1e-6);
    config.lr_velocity *= 2;
    auto incompatible_model = model4d().to(device);
    Gaussian4DTrainer incompatible(incompatible_model, config);
    CHECK_THROWS(incompatible.load_checkpoint(path.string()));
  }
}
