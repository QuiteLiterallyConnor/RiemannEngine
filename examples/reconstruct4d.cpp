#include "splat/dnerf.hpp"
#include "splat/fourdgs.hpp"
#include "splat/ply.hpp"
#include "splat/trainer4d.hpp"
#include "json.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <ATen/Parallel.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace riemann::gaussian;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

struct Arguments {
  std::string data, output, resume, export_path;
  bool export_only = false;
  double duration_seconds = 1.0;
  int64_t steps = 30000, downscale = 1, points = 20000;
  int64_t max_gaussians = 100000, checkpoint_every = 1000, seed = 42;
  std::string device = "cuda";
};

Arguments parse(int argc, char **argv) {
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--export-only") {
      args.export_only = true;
      continue;
    }
    if (key == "--help") {
      std::cout << "reconstruct4d --data DIR --output DIR [--resume CHECKPOINT]\n"
                << "  --steps 30000 --downscale 1 --points 20000\n"
                << "  --max-gaussians 100000 --checkpoint-every 1000\n"
                << "  --seed 42 --device cuda|cpu\n"
                << "  --export-only --export-path FILE --duration-seconds 1\n"
                << "All training frames are used; val is monitored and test is\n"
                << "evaluated only at completion. Resume requires the same flags.\n";
      std::exit(0);
    }
    TORCH_CHECK(i + 1 < argc, "missing value for ", key);
    const std::string value = argv[++i];
    auto integer = [&]() {
      size_t consumed = 0;
      const auto n = std::stoll(value, &consumed);
      TORCH_CHECK(consumed == value.size(), "invalid integer for ", key);
      return n;
    };
    if (key == "--data") args.data = value;
    else if (key == "--output") args.output = value;
    else if (key == "--resume") args.resume = value;
    else if (key == "--device") args.device = value;
    else if (key == "--export-path") args.export_path = value;
    else if (key == "--duration-seconds") {
      size_t consumed = 0;
      args.duration_seconds = std::stod(value, &consumed);
      TORCH_CHECK(consumed == value.size(), "invalid --duration-seconds");
    }
    else if (key == "--steps") args.steps = integer();
    else if (key == "--downscale") args.downscale = integer();
    else if (key == "--points") args.points = integer();
    else if (key == "--max-gaussians") args.max_gaussians = integer();
    else if (key == "--checkpoint-every") args.checkpoint_every = integer();
    else if (key == "--seed") args.seed = integer();
    else TORCH_CHECK(false, "unknown option: ", key);
  }
  TORCH_CHECK(!args.data.empty() && !args.output.empty(),
              "--data and --output are required");
  TORCH_CHECK(args.steps > 0 && args.downscale > 0 && args.points > 0 &&
                  args.checkpoint_every > 0 && args.seed >= 0 &&
                  args.max_gaussians >= args.points,
              "invalid training budget or Gaussian limit");
  TORCH_CHECK(args.device == "cuda" || args.device == "cpu",
              "--device must be cuda or cpu");
  TORCH_CHECK(std::isfinite(args.duration_seconds) && args.duration_seconds > 0,
              "--duration-seconds must be finite and positive");
  TORCH_CHECK(!args.export_only || !args.resume.empty(),
              "--export-only requires --resume");
  return args;
}

void export_scene(const Gaussian4DModel &model, const Arguments &args,
                   const fs::path &output) {
  const auto path = args.export_path.empty() ? output / "scene.4dgs"
                                             : fs::absolute(args.export_path);
  FourDGSExportOptions options;
  options.seconds_per_time_unit = args.duration_seconds;
  const auto result = save_four_dgs(model, path.string(), options);
  const Json report{{"file", path.string()}, {"gaussians", result.gaussians},
      {"bytes", result.bytes}, {"duration_seconds", args.duration_seconds},
      {"clamped_rgb_components", result.clamped_rgb_components},
      {"clamped_sh_coefficients", result.clamped_sh_coefficients},
      {"max_position_error", result.max_position_error},
      {"max_scale_log_error", result.max_scale_log_error},
      {"max_rotation_error", result.max_rotation_error},
      {"max_rgb_error", result.max_rgb_error},
      {"max_opacity_error", result.max_opacity_error},
      {"max_velocity_error", result.max_velocity_error},
      {"max_time_error", result.max_time_error},
      {"max_sigma_log_error", result.max_sigma_log_error},
      {"max_sh_error", result.max_sh_error}};
  std::ofstream file(path.string() + ".json");
  file << report.dump(2) << '\n';
  file.flush();
  TORCH_CHECK(file.good(), "failed writing 4dgs export report");
  std::cout << "4dgs export: " << report.dump() << std::endl;
}

void write_json(const fs::path &path, const Json &document) {
  std::ofstream file(path);
  file << document.dump(2) << '\n';
  file.flush();
  TORCH_CHECK(file.good(), "failed to write ", path.string());
}

void append_json(const fs::path &path, const Json &document) {
  std::ofstream file(path, std::ios::app);
  file << document.dump() << '\n';
  file.flush();
  TORCH_CHECK(file.good(), "failed to append ", path.string());
}

void save_png(const fs::path &path, const at::Tensor &image) {
  const auto pixels = (image.detach().clamp(0, 1) * 255).round()
                          .to(at::kCPU, at::kByte).contiguous();
  TORCH_CHECK(stbi_write_png(path.string().c_str(), int(pixels.size(1)),
                             int(pixels.size(0)), 3, pixels.data_ptr<uint8_t>(),
                             int(pixels.size(1) * 3)),
              "failed to write PNG ", path.string());
}

std::string numbered(const std::string &prefix, int64_t number,
                      const std::string &extension) {
  std::ostringstream name;
  name << prefix << std::setfill('0') << std::setw(6) << number << extension;
  return name.str();
}

Json evaluate(const Gaussian4DTrainer &trainer,
               const std::vector<TimedView> &views, const fs::path &output,
               const std::string &split, const torch::Device &device) {
  torch::NoGradGuard guard;
  Json per_view = Json::array();
  double psnr_sum = 0, ssim_sum = 0, l1_sum = 0, foreground_sum = 0;
  int64_t foreground_views = 0;
  for (size_t i = 0; i < views.size(); ++i) {
    const auto &view = views[i];
    const auto prediction = trainer.render_image(view.camera, view.time);
    TORCH_CHECK(at::isfinite(prediction).all().item<bool>(),
                "non-finite evaluation image");
    const auto target = view.image.to(device);
    const auto difference = prediction.clamp(0, 1) - target;
    const double mse = difference.square().mean().item<double>();
    const double psnr = -10 * std::log10(std::max(mse, 1e-12));
    const double structural = ssim(prediction.clamp(0, 1), target).item<double>();
    const double l1 = difference.abs().mean().item<double>();
    // This is a target-derived nonwhite mask, not a supplied ground-truth alpha.
    const auto foreground = (target - 1).abs().amax({2}) > 0.02;
    Json fg = nullptr;
    if (foreground.any().item<bool>()) {
      const double foreground_mse =
          difference.square().mean(2).masked_select(foreground).mean().item<double>();
      const double foreground_psnr =
          -10 * std::log10(std::max(foreground_mse, 1e-12));
      foreground_sum += foreground_psnr;
      ++foreground_views;
      fg = foreground_psnr;
    }
    per_view.push_back({{"image", view.image_path}, {"time", view.time},
                        {"psnr", psnr}, {"ssim", structural}, {"l1", l1},
                        {"nonwhite_psnr", fg}});
    psnr_sum += psnr;
    ssim_sum += structural;
    l1_sum += l1;
    if (i == 0 || i == views.size() / 2 || i + 1 == views.size()) {
      const auto comparison = at::cat(
          {target, prediction.clamp(0, 1), difference.abs().clamp(0, 1)}, 1);
      save_png(output / numbered(split + "_", int64_t(i), ".png"), comparison);
    }
  }
  return {{"split", split}, {"iteration", trainer.iteration()},
          {"views", views.size()}, {"psnr", psnr_sum / views.size()},
          {"ssim", ssim_sum / views.size()}, {"l1", l1_sum / views.size()},
          {"nonwhite_psnr", foreground_views
                                ? Json(foreground_sum / foreground_views) : Json(nullptr)},
          {"per_view", per_view}};
}

int run(const Arguments &args) {
  const fs::path output = fs::absolute(args.output);
  fs::create_directories(output);
  at::set_num_threads(2);
  const torch::Device device(args.device);
#ifdef USE_CUDA
  TORCH_CHECK(!device.is_cuda() || at::cuda::is_available(),
              "CUDA runtime or device is unavailable");
#else
  TORCH_CHECK(!device.is_cuda(), "rebuild with CPU_ONLY=OFF for CUDA training");
#endif
  const Json manifest{
      {"schema", 1}, {"data", fs::weakly_canonical(args.data).string()},
      {"steps", args.steps}, {"downscale", args.downscale},
      {"initial_points", args.points}, {"max_gaussians", args.max_gaussians},
      {"checkpoint_every", args.checkpoint_every}, {"seed", args.seed},
      {"device", args.device}, {"background", "white"},
      {"initial_time_scale", 0.15}, {"initial_bounds", {-1.3, 1.3}}};
  const auto manifest_path = output / "run.json";
  if (args.resume.empty()) {
    TORCH_CHECK(!fs::exists(manifest_path),
                "output already contains a run; use --resume or a new directory");
    write_json(manifest_path, manifest);
  } else {
    std::ifstream saved(manifest_path);
    TORCH_CHECK(saved.good(), "resume requires the original run.json");
    TORCH_CHECK(Json::parse(saved) == manifest,
                "resume flags do not match run.json");
  }
  at::manual_seed(args.seed);
  if (args.export_only) {
    auto model = Gaussian4DModel::from_points(
        at::zeros({1, 3}), at::full({1, 3}, 0.5), 3).to(device);
    Gaussian4DTrainerConfig config;
    config.scene_extent = 4;
    config.max_gaussians = args.max_gaussians;
    config.lr_decay_steps = args.steps;
    config.render.background = at::ones({3}, at::TensorOptions().device(device));
    Gaussian4DTrainer trainer(model, config);
    trainer.load_checkpoint(args.resume);
    export_scene(model, args, output);
    return 0;
  }
  DNerfLoadOptions data_options;
  data_options.downscale = args.downscale;
  std::cout << "Loading all train/val/test images at downscale "
            << args.downscale << "..." << std::endl;
  const auto train = load_dnerf_views(args.data, "train", data_options);
  const auto val = load_dnerf_views(args.data, "val", data_options);
  const auto test = load_dnerf_views(args.data, "test", data_options);
  std::cout << "Loaded " << train.size() << " train, " << val.size()
            << " val, " << test.size() << " test; "
            << train.front().camera.width << 'x' << train.front().camera.height
            << std::endl;
  TORCH_CHECK(std::all_of(train.begin(), train.end(), [](const TimedView &view) {
                return view.time >= 0 && view.time <= 1;
              }), "this initialization requires training times in [0, 1]");

  // Initialize on CPU: kNN scale estimation would otherwise consume GPU memory.
  auto model = Gaussian4DModel::from_points(
      at::rand({args.resume.empty() ? args.points : int64_t{1}, 3}) * 2.6 - 1.3,
      at::full({args.resume.empty() ? args.points : int64_t{1}, 3}, 0.5),
      3, 0.1, 0.15).to(device);
  Gaussian4DTrainerConfig config;
  config.scene_extent = 4;
  config.max_gaussians = args.max_gaussians;
  config.lr_decay_steps = args.steps;
  config.render.background = at::ones({3}, at::TensorOptions().device(device));
  Gaussian4DTrainer trainer(model, config);
  if (!args.resume.empty())
    trainer.load_checkpoint(args.resume);
  TORCH_CHECK(trainer.iteration() <= args.steps,
              "checkpoint iteration exceeds requested budget");
  std::cout << "Starting at iteration " << trainer.iteration() << " on "
            << args.device << ", " << model.size() << " Gaussians" << std::endl;

  const auto start = Clock::now();
  const auto initial_iteration = trainer.iteration();
  auto status = [&](const std::string &state, double loss) {
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const int64_t completed = trainer.iteration() - initial_iteration;
    const double speed = completed / std::max(seconds, 1e-6);
    Json progress{{"state", state}, {"iteration", trainer.iteration()},
                  {"target_steps", args.steps}, {"gaussians", model.size()},
                  {"loss", loss}, {"elapsed_seconds", seconds},
                  {"steps_per_second", speed},
                  {"eta_seconds", speed > 0 ? Json((args.steps - trainer.iteration()) / speed)
                                            : Json(nullptr)}};
    write_json(output / "progress.json", progress);
    append_json(output / "training.jsonl", progress);
    std::cout << progress.dump() << std::endl;
  };
  double loss = 0;
  status("running", loss);
  if (trainer.iteration() == 0) {
    const auto baseline = evaluate(trainer, val, output, "baseline_val", device);
    write_json(output / "baseline_val.json", baseline);
    std::cout << "Baseline validation PSNR " << baseline.at("psnr") << std::endl;
  }
  std::vector<size_t> order(train.size());
  int64_t shuffled_epoch = -1;
  while (trainer.iteration() < args.steps) {
    const auto iteration = trainer.iteration();
    const int64_t epoch = iteration / int64_t(train.size());
    if (epoch != shuffled_epoch) {
      std::iota(order.begin(), order.end(), size_t{0});
      std::mt19937 generator(static_cast<uint32_t>(args.seed + epoch));
      std::shuffle(order.begin(), order.end(), generator);
      shuffled_epoch = epoch;
    }
    const auto &view = train[order[size_t(iteration) % order.size()]];
    loss = trainer.step(view.camera, view.time, view.image).loss;
    if (trainer.iteration() % 100 == 0)
      status("running", loss);
    if (trainer.iteration() % args.checkpoint_every == 0 ||
        trainer.iteration() == args.steps) {
      const auto checkpoint = output /
          numbered("checkpoint_", trainer.iteration(), ".pt");
      TORCH_CHECK(!fs::exists(checkpoint), "refusing to overwrite checkpoint ",
                  checkpoint.string());
      trainer.save_checkpoint(checkpoint.string());
      write_json(output / "latest.json",
                 {{"checkpoint", checkpoint.string()}, {"iteration", trainer.iteration()}});
      const auto metrics = evaluate(trainer, val, output, "val", device);
      write_json(output / numbered("val_", trainer.iteration(), ".json"), metrics);
      append_json(output / "validation.jsonl", metrics);
      std::cout << "Validation PSNR " << metrics.at("psnr")
                << " SSIM " << metrics.at("ssim") << std::endl;
      for (int i = 0; i < 3; ++i)
        save_png(output / numbered("time_", i, ".png"),
                  trainer.render_image(train.front().camera, i * 0.5));
      status("running", loss);
    }
  }
  const auto test_metrics = evaluate(trainer, test, output, "test", device);
  write_json(output / "test_metrics.json", test_metrics);
  for (int i = 0; i < 60; ++i) {
    const double time = double(i) / 59;
    save_png(output / numbered("animation_", i, ".png"),
              trainer.render_image(train.front().camera, time));
  }
  for (int i = 0; i < 3; ++i)
    save_gaussian_ply(model.at_time(i * 0.5),
                       (output / numbered("snapshot_", i, ".ply")).string());
  export_scene(model, args, output);
  status("completed", loss);
  std::cout << "Final held-out test PSNR " << test_metrics.at("psnr")
            << " SSIM " << test_metrics.at("ssim") << std::endl;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Arguments args;
  try {
    args = parse(argc, argv);
    return run(args);
  } catch (const std::exception &error) {
    std::cerr << "Reconstruction failed: " << error.what() << std::endl;
    if (!args.output.empty() && fs::exists(fs::path(args.output) / "run.json")) {
      std::ofstream failure(fs::path(args.output) / "failure.txt");
      failure << error.what() << '\n';
    }
    return 1;
  }
}
