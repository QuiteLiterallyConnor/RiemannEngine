#include "splat/trainer4d.hpp"

#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <algorithm>
#include <cmath>

namespace riemann {
namespace gaussian {
namespace {

at::Tensor quaternion_frames(const at::Tensor &q) {
  const auto u = q / q.norm(2, {1}, true).clamp_min(1e-12);
  const auto w = u.select(1, 0), x = u.select(1, 1), y = u.select(1, 2),
             z = u.select(1, 3);
  return at::stack(
      {at::stack({1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y)},
                 1),
       at::stack({2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                  2 * (y * z - w * x)},
                 1),
       at::stack({2 * (x * z - w * y), 2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)},
                 1)},
      1);
}

void require_image(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.dim() == 3 && t.size(2) == 3, name,
              " must have shape [H, W, 3]");
}

constexpr size_t kXyz = 0, kScale = 4, kOpacity = 3;

at::Tensor config_signature(const Gaussian4DTrainerConfig &c) {
  return at::tensor(
      {c.lr_position, c.lr_position_final, double(c.lr_decay_steps), c.lr_dc,
       c.lr_rest, c.lr_opacity, c.lr_scale, c.lr_rotation, c.lr_velocity,
       c.lr_time_center, c.lr_time_scale, c.beta1, c.beta2, c.epsilon,
       c.lambda_ssim, double(c.sh_increase_interval), double(c.densify),
       double(c.densify_start), double(c.densify_interval),
       double(c.densify_stop), c.densify_grad_threshold, c.percent_dense,
       c.min_opacity, c.max_scale_fraction, double(c.opacity_reset_interval),
       double(c.max_gaussians), c.scene_extent, double(c.render.sh_degree),
       double(c.render.tile_size), double(c.render.chunk_size),
       double(c.render.use_cuda_kernels)},
      at::kDouble);
}

} // namespace

Gaussian4DTrainer::Gaussian4DTrainer(Gaussian4DModel &model,
                                     Gaussian4DTrainerConfig config)
    : model_(&model), config_(std::move(config)) {
  TORCH_CHECK(config_.scene_extent > 0.0, "scene_extent must be positive");
  for (auto &parameter : model_->enable_gradients()) {
    moment1_.push_back(at::zeros_like(parameter));
    moment2_.push_back(at::zeros_like(parameter));
  }
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(model_->device());
  gradient_accumulator_ = at::zeros({model_->size()}, options);
  gradient_count_ = at::zeros({model_->size()}, options);
}

int Gaussian4DTrainer::active_sh_degree() const {
  if (config_.sh_increase_interval <= 0)
    return model_->sh_degree();
  const auto grown = static_cast<int64_t>(
      std::min<int64_t>(3, iteration_ / config_.sh_increase_interval));
  return static_cast<int>(std::min<int64_t>(grown, model_->sh_degree()));
}

double Gaussian4DTrainer::learning_rate(size_t parameter) const {
  switch (parameter) {
  case 0: {
    const double t =
        config_.lr_decay_steps > 0
            ? std::min(1.0, static_cast<double>(iteration_) /
                                config_.lr_decay_steps)
            : 1.0;
    const double lr = std::exp((1.0 - t) * std::log(config_.lr_position) +
                               t * std::log(config_.lr_position_final));
    return lr * config_.scene_extent;
  }
  case 1:
    return config_.lr_dc;
  case 2:
    return config_.lr_rest;
  case 3:
    return config_.lr_opacity;
  case 4:
    return config_.lr_scale;
  case 5:
    return config_.lr_rotation;
  case 6:
    return config_.lr_velocity * config_.scene_extent;
  case 7:
    return config_.lr_time_center;
  default:
    return config_.lr_time_scale;
  }
}

void Gaussian4DTrainer::adam_step() {
  torch::NoGradGuard no_grad;
  ++adam_steps_;
  const double bias1 = 1.0 - std::pow(config_.beta1, adam_steps_);
  const double bias2 = 1.0 - std::pow(config_.beta2, adam_steps_);
  const auto parameters = model_->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &parameter = parameters[i];
    const auto &gradient = parameter.grad();
    if (!gradient.defined() || parameter.numel() == 0)
      continue;
    moment1_[i].mul_(config_.beta1).add_(gradient, 1.0 - config_.beta1);
    moment2_[i].mul_(config_.beta2).addcmul_(gradient, gradient,
                                             1.0 - config_.beta2);
    const auto denominator =
        (moment2_[i] / bias2).sqrt().add_(config_.epsilon);
    parameter.addcdiv_(moment1_[i], denominator, -learning_rate(i) / bias1);
  }
}

at::Tensor Gaussian4DTrainer::render_image(const Camera &camera,
                                           double time) const {
  torch::NoGradGuard no_grad;
  RenderOptions options = config_.render;
  options.sh_degree = active_sh_degree();
  return render(*model_, camera, time, options).image;
}

void Gaussian4DTrainer::accumulate_statistics(const RenderOutput &output,
                                              const Camera &camera) {
  torch::NoGradGuard no_grad;
  const auto &gradient = output.means2d.grad();
  if (!gradient.defined())
    return;
  const auto scaled = at::stack(
      {gradient.select(1, 0) * (0.5 * static_cast<double>(camera.width)),
       gradient.select(1, 1) * (0.5 * static_cast<double>(camera.height))},
      1);
  const auto norm = scaled.norm(2, {1});
  const auto visible = output.visible.to(at::kFloat);
  gradient_accumulator_ = gradient_accumulator_ + norm * visible;
  gradient_count_ = gradient_count_ + visible;
}

Gaussian4DTrainer::StepResult
Gaussian4DTrainer::step(const Camera &camera, double time,
                        const at::Tensor &target) {
  require_image(target, "target");
  TORCH_CHECK(target.size(0) == camera.height &&
                  target.size(1) == camera.width,
              "target size must match the camera");
  const auto device = model_->device();
  const auto image_target = target.to(device, at::kFloat);

  RenderOptions options = config_.render;
  options.sh_degree = active_sh_degree();
  auto output = render(*model_, camera, time, options);

  at::Tensor loss = (1.0 - config_.lambda_ssim) *
                    gaussian::l1_loss(output.image, image_target);
  if (config_.lambda_ssim > 0.0)
    loss = loss + config_.lambda_ssim *
                      (1.0 - gaussian::ssim(output.image, image_target));

  TORCH_CHECK(at::isfinite(loss).item<bool>(),
              "4D training loss is not finite at iteration ", iteration_);
  for (auto &parameter : model_->parameters())
    parameter.mutable_grad() = at::Tensor();
  loss.backward();
  accumulate_statistics(output, camera);
  adam_step();
  ++iteration_;

  if (config_.densify && iteration_ < config_.densify_stop &&
      iteration_ > config_.densify_start && config_.densify_interval > 0 &&
      iteration_ % config_.densify_interval == 0)
    densify_and_prune();
  if (config_.densify && config_.opacity_reset_interval > 0 &&
      iteration_ < config_.densify_stop &&
      iteration_ % config_.opacity_reset_interval == 0)
    reset_opacity();

  StepResult result;
  result.loss = loss.item<double>();
  result.gaussians = model_->size();
  result.iteration = iteration_;
  return result;
}

void Gaussian4DTrainer::rebuild_rows(const std::vector<at::Tensor> &new_rows,
                                     const at::Tensor &keep) {
  torch::NoGradGuard no_grad;
  const auto keep_indices = keep.nonzero().squeeze(1);
  const auto parameters = model_->parameters();
  std::vector<at::Tensor> updated;
  for (size_t i = 0; i < parameters.size(); ++i) {
    updated.push_back(at::cat({parameters[i].detach(), new_rows[i]}, 0)
                          .index_select(0, keep_indices)
                          .contiguous());
    moment1_[i] = at::cat({moment1_[i], at::zeros_like(new_rows[i])}, 0)
                      .index_select(0, keep_indices)
                      .contiguous();
    moment2_[i] = at::cat({moment2_[i], at::zeros_like(new_rows[i])}, 0)
                      .index_select(0, keep_indices)
                      .contiguous();
  }
  model_->set_parameters(updated);
  model_->enable_gradients();
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(model_->device());
  gradient_accumulator_ = at::zeros({model_->size()}, options);
  gradient_count_ = at::zeros({model_->size()}, options);
}

std::pair<int64_t, int64_t> Gaussian4DTrainer::densify_and_prune() {
  torch::NoGradGuard no_grad;
  const int64_t n = model_->size();
  const auto parameters = model_->parameters();
  const auto gradients =
      gradient_accumulator_ / gradient_count_.clamp_min(1.0);
  const auto scales = model_->scales().detach();
  const auto max_scale = std::get<0>(scales.max(1));
  const auto selected = gradients >= config_.densify_grad_threshold;
  const double size_threshold = config_.percent_dense * config_.scene_extent;
  auto clone_mask = selected & (max_scale <= size_threshold);
  auto split_mask = selected & (max_scale > size_threshold);

  int64_t clones = clone_mask.sum().item<int64_t>();
  int64_t splits = split_mask.sum().item<int64_t>();
  if (config_.max_gaussians > 0 &&
      n + clones + 2 * splits > config_.max_gaussians) {
    clone_mask = at::zeros_like(clone_mask);
    split_mask = at::zeros_like(split_mask);
    clones = splits = 0;
  }

  const auto clone_indices = clone_mask.nonzero().squeeze(1);
  const auto split_indices = split_mask.nonzero().squeeze(1);
  at::Tensor split_xyz, split_scale_raw;
  if (splits > 0) {
    // Two children sampled from the parent's Gaussian, each 1.6x smaller.
    const auto repeated_scales =
        scales.index_select(0, split_indices).repeat({2, 1});
    const auto frames = quaternion_frames(
        model_->rotation_raw().detach().index_select(0, split_indices)
            .repeat({2, 1}));
    const auto noise = at::randn_like(repeated_scales) * repeated_scales;
    split_xyz = at::bmm(frames, noise.unsqueeze(-1)).squeeze(-1) +
                model_->xyz().detach().index_select(0, split_indices)
                    .repeat({2, 1});
    split_scale_raw = at::log(repeated_scales / 1.6);
  }

  std::vector<at::Tensor> new_rows;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::vector<at::Tensor> parts{
        parameters[i].detach().index_select(0, clone_indices)};
    if (splits > 0) {
      if (i == kXyz) {
        parts.push_back(split_xyz);
      } else if (i == kScale) {
        parts.push_back(split_scale_raw);
      } else {
        const auto rows =
            parameters[i].detach().index_select(0, split_indices);
        parts.push_back(at::cat({rows, rows}, 0));
      }
    }
    new_rows.push_back(at::cat(parts, 0));
  }

  const int64_t added = clones + 2 * splits;
  const auto opacities = model_->opacities().detach().reshape({-1});
  auto prune =
      (opacities < config_.min_opacity) |
      (max_scale > config_.max_scale_fraction * config_.scene_extent) |
      split_mask;
  auto prune_all = at::cat({prune, at::zeros({added}, prune.options())}, 0);
  auto keep = ~prune_all;
  if (keep.sum().item<int64_t>() == 0) {
    keep = at::ones_like(keep);
    prune_all = at::zeros_like(prune_all);
  }
  const int64_t removed = prune_all.sum().item<int64_t>();
  rebuild_rows(new_rows, keep);
  return {added, removed};
}

void Gaussian4DTrainer::reset_opacity() {
  torch::NoGradGuard no_grad;
  const auto &raw = model_->opacity_raw();
  const auto clipped = at::sigmoid(raw).clamp_max(0.01);
  raw.copy_(at::log(clipped / (1.0 - clipped)));
  moment1_[kOpacity].zero_();
  moment2_[kOpacity].zero_();
}

void Gaussian4DTrainer::save_checkpoint(const std::string &path) const {
  torch::serialize::OutputArchive archive;
  archive.write("version", at::tensor(int64_t{1}));
  archive.write("config", config_signature(config_));
  archive.write("background", config_.render.background.defined()
                                  ? config_.render.background.detach().cpu()
                                  : at::empty({0}));
  archive.write("counters", at::tensor({iteration_, adam_steps_}, at::kLong));
  const auto parameters = model_->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto suffix = std::to_string(i);
    archive.write("parameter_" + suffix, parameters[i].detach().cpu());
    archive.write("moment1_" + suffix, moment1_[i].cpu());
    archive.write("moment2_" + suffix, moment2_[i].cpu());
  }
  archive.write("gradient_accumulator", gradient_accumulator_.cpu());
  archive.write("gradient_count", gradient_count_.cpu());
  archive.write("cpu_rng", at::detail::getDefaultCPUGenerator().get_state());
#ifdef USE_CUDA
  if (model_->device().is_cuda())
    archive.write("cuda_rng", at::cuda::detail::getDefaultCUDAGenerator(
                                  model_->device().index()).get_state());
#endif
  archive.save_to(path);
}

void Gaussian4DTrainer::load_checkpoint(const std::string &path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path, torch::Device(torch::kCPU));
  at::Tensor version, signature, background, counters, accumulator, count, rng;
  archive.read("version", version);
  TORCH_CHECK(version.item<int64_t>() == 1, "unsupported 4D checkpoint version");
  archive.read("config", signature);
  TORCH_CHECK(at::equal(signature, config_signature(config_)),
              "checkpoint configuration does not match the trainer");
  archive.read("background", background);
  const auto expected_background = config_.render.background.defined()
                                       ? config_.render.background.detach().cpu()
                                       : at::empty({0});
  TORCH_CHECK(at::equal(background, expected_background),
              "checkpoint background does not match the trainer");
  archive.read("counters", counters);
  TORCH_CHECK(counters.sizes() == at::IntArrayRef({2}) &&
                  counters.scalar_type() == at::kLong &&
                  counters.min().item<int64_t>() >= 0,
              "invalid checkpoint iteration counters");
  std::vector<at::Tensor> parameters, first, second;
  const auto device = model_->device();
  for (size_t i = 0; i < 9; ++i) {
    at::Tensor parameter, m1, m2;
    const auto suffix = std::to_string(i);
    archive.read("parameter_" + suffix, parameter);
    archive.read("moment1_" + suffix, m1);
    archive.read("moment2_" + suffix, m2);
    TORCH_CHECK(m1.sizes() == parameter.sizes() &&
                    m2.sizes() == parameter.sizes() &&
                    m1.scalar_type() == at::kFloat &&
                    m2.scalar_type() == at::kFloat &&
                    at::isfinite(parameter).all().item<bool>() &&
                    at::isfinite(m1).all().item<bool>() &&
                    at::isfinite(m2).all().item<bool>() &&
                    (m2 >= 0).all().item<bool>(),
                "invalid checkpoint parameter/Adam state at index ", i);
    parameters.push_back(parameter.to(device));
    first.push_back(m1.to(device));
    second.push_back(m2.to(device));
  }
  Gaussian4DModel replacement(parameters[0], parameters[1], parameters[2],
                              parameters[3], parameters[4], parameters[5],
                              parameters[6], parameters[7], parameters[8]);
  archive.read("gradient_accumulator", accumulator);
  archive.read("gradient_count", count);
  TORCH_CHECK(accumulator.sizes() == at::IntArrayRef({replacement.size()}) &&
                  count.sizes() == accumulator.sizes() &&
                  at::isfinite(accumulator).all().item<bool>() &&
                  at::isfinite(count).all().item<bool>(),
              "invalid checkpoint densification statistics");
  archive.read("cpu_rng", rng);
#ifdef USE_CUDA
  at::Tensor cuda_rng;
  if (device.is_cuda())
    archive.read("cuda_rng", cuda_rng);
#endif
  *model_ = std::move(replacement);
  model_->enable_gradients();
  moment1_ = std::move(first);
  moment2_ = std::move(second);
  gradient_accumulator_ = accumulator.to(device);
  gradient_count_ = count.to(device);
  iteration_ = counters[0].item<int64_t>();
  adam_steps_ = counters[1].item<int64_t>();
  auto cpu_generator = at::detail::getDefaultCPUGenerator();
  cpu_generator.set_state(rng);
#ifdef USE_CUDA
  if (device.is_cuda()) {
    auto cuda_generator = at::cuda::detail::getDefaultCUDAGenerator(device.index());
    cuda_generator.set_state(cuda_rng);
  }
#endif
}

} // namespace gaussian
} // namespace riemann
