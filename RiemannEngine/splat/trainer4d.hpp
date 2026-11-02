#ifndef RIEMANN_SPLAT_TRAINER4D_HPP
#define RIEMANN_SPLAT_TRAINER4D_HPP

#include "splat/camera.hpp"
#include "splat/model4d.hpp"
#include "splat/render.hpp"
#include "splat/trainer.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace riemann {
namespace gaussian {

struct Gaussian4DTrainerConfig : GaussianTrainerConfig {
  // Velocity is in scene units per unit time; the rate scales by scene_extent.
  double lr_velocity = 1e-3;
  double lr_time_center = 1e-3;
  double lr_time_scale = 5e-3;
};

// Trains a Gaussian4DModel against posed, timed images with the 3DGS recipe
// extended to time: L1 + SSIM loss, Adam with per-parameter learning rates,
// SH-degree warm-up, periodic densification and pruning, and opacity resets.
// The model must outlive the trainer; the trainer replaces the model's
// parameter tensors when the number of Gaussians changes.
class Gaussian4DTrainer {
public:
  using StepResult = GaussianTrainer::StepResult;

  Gaussian4DTrainer(Gaussian4DModel &model, Gaussian4DTrainerConfig config = {});

  // One optimization step on a single posed image ([H, W, 3] float32 target)
  // observed at the given normalized time.
  StepResult step(const Camera &camera, double time, const at::Tensor &target);

  // Renders at the given time without tracking gradients (using the active
  // SH degree).
  at::Tensor render_image(const Camera &camera, double time) const;

  int64_t iteration() const { return iteration_; }
  int active_sh_degree() const;
  const Gaussian4DTrainerConfig &config() const { return config_; }

  // Accumulates the screen-space positional gradient of the visible Gaussians
  // (as in 3DGS, in normalized device units). Called by step(); exposed for
  // custom loops and testing.
  void accumulate_statistics(const RenderOutput &output, const Camera &camera);

  // Clones small and splits large Gaussians with a high accumulated gradient,
  // then prunes transparent or oversized ones. Keeps Adam state consistent.
  // Returns {added, removed}.
  std::pair<int64_t, int64_t> densify_and_prune();

  // Clamps all opacities to at most 0.01 and clears their Adam state.
  void reset_opacity();

  // Includes model, Adam moments, iteration, densification statistics and RNG
  // states. Loading requires identical config; tensors move to this model's
  // device. Save to a new path so an interrupted write cannot damage a prior
  // checkpoint. Loading restores the process-wide CPU/device RNG states.
  void save_checkpoint(const std::string &path) const;
  void load_checkpoint(const std::string &path);

private:
  double learning_rate(size_t parameter) const;
  void adam_step();
  void rebuild_rows(const std::vector<at::Tensor> &new_rows,
                    const at::Tensor &keep);

  Gaussian4DModel *model_;
  Gaussian4DTrainerConfig config_;
  int64_t iteration_ = 0;
  int64_t adam_steps_ = 0;
  std::vector<at::Tensor> moment1_, moment2_;
  at::Tensor gradient_accumulator_; // [N]
  at::Tensor gradient_count_;       // [N]
};

} // namespace gaussian
} // namespace riemann

#endif
