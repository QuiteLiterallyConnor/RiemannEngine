#include "engine/learning/training.hpp"

#include <stdexcept>

namespace riemann {
namespace training {

std::unique_ptr<torch::optim::Optimizer> create_optimizer(
    const std::vector<torch::Tensor>& parameters,
    const OptimizerConfig& config) {
  switch (config.type) {
  case OptimizerType::SGD:
    return std::make_unique<torch::optim::SGD>(
        parameters, torch::optim::SGDOptions(config.lr)
                        .momentum(config.momentum)
                        .weight_decay(config.weight_decay)
                        .nesterov(config.nesterov));
  case OptimizerType::Adam:
    return std::make_unique<torch::optim::Adam>(
        parameters, torch::optim::AdamOptions(config.lr)
                        .betas({config.beta1, config.beta2})
                        .eps(config.eps)
                        .weight_decay(config.weight_decay));
  case OptimizerType::AdamW:
    return std::make_unique<torch::optim::AdamW>(
        parameters, torch::optim::AdamWOptions(config.lr)
                        .betas({config.beta1, config.beta2})
                        .eps(config.eps)
                        .weight_decay(config.weight_decay));
  }
  throw std::invalid_argument("Unknown optimizer type");
}

at::Tensor bce_with_logits_sparse(const SparseTensor& logits,
                                  const at::Tensor& targets,
                                  double pos_weight) {
  TORCH_CHECK(pos_weight > 0.0, "Positive weight must be positive");
  auto weight = torch::tensor({pos_weight}, logits.F().options());
  return torch::nn::functional::binary_cross_entropy_with_logits(
      logits.F(), targets,
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
          .pos_weight(weight));
}

at::Tensor cross_entropy_sparse(
    const SparseTensor& logits, const at::Tensor& targets,
    c10::optional<at::Tensor> weight) {
  auto options = torch::nn::functional::CrossEntropyFuncOptions();
  if (weight.has_value()) options.weight(*weight);
  return torch::nn::functional::cross_entropy(logits.F(), targets, options);
}

SparseBatch sparse_collate_samples(const std::vector<SparseSample>& samples) {
  SparseBatch batch;
  batch.batch_size = static_cast<int64_t>(samples.size());
  if (samples.empty()) return batch;

  std::vector<at::Tensor> coordinates, features, labels;
  bool has_labels = false;
  coordinates.reserve(samples.size());
  features.reserve(samples.size());
  labels.reserve(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    TORCH_CHECK(sample.coordinates.dim() == 2 &&
                    sample.features.dim() == 2 &&
                    sample.coordinates.size(0) == sample.features.size(0),
                "Invalid coordinate/feature shapes in sample ", i);
    auto batch_column = torch::full(
        {sample.coordinates.size(0), 1}, static_cast<int64_t>(i),
        sample.coordinates.options().dtype(torch::kInt32));
    coordinates.push_back(torch::cat(
        {batch_column, sample.coordinates.to(torch::kInt32)}, 1));
    features.push_back(sample.features);
    if (sample.labels.defined() && sample.labels.numel() > 0) {
      TORCH_CHECK(sample.labels.size(0) == sample.coordinates.size(0),
                  "Coordinate and label counts differ in sample ", i);
      has_labels = true;
      labels.push_back(sample.labels);
    } else {
      TORCH_CHECK(!has_labels,
                  "Labels must be supplied consistently for every sample");
    }
  }
  TORCH_CHECK(!has_labels || labels.size() == samples.size(),
              "Labels must be supplied consistently for every sample");
  batch.coordinates = torch::cat(coordinates, 0);
  batch.features = torch::cat(features, 0);
  if (has_labels) batch.labels = torch::cat(labels, 0);
  return batch;
}

SparseDataset::SparseDataset(std::vector<SparseSample> samples)
    : samples_(std::move(samples)) {}

void SparseDataset::add_sample(SparseSample sample) {
  samples_.push_back(std::move(sample));
}

SparseSample SparseDataset::get(size_t index) {
  TORCH_CHECK(index < samples_.size(), "Dataset index out of range");
  const auto& sample = samples_[index];
  return SparseSample(sample.coordinates.clone(), sample.features.clone(),
                      sample.labels.defined() ? sample.labels.clone()
                                              : at::Tensor());
}

torch::optional<size_t> SparseDataset::size() const { return samples_.size(); }

StepResult train_step(
    torch::nn::AnyModule& model, const SparseBatch& batch,
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> criterion,
    torch::optim::Optimizer& optimizer,
    std::shared_ptr<CoordinateManager> manager) {
  TORCH_CHECK(batch.labels.defined(),
              "Training batch must contain labels");
  auto input = SparseTensor(batch.features, batch.coordinates, manager);
  const auto output = model.forward<SparseTensor>(input);
  const auto loss = criterion(output.F(), batch.labels);
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
  return {loss.item<double>(), batch.features.size(0)};
}

EpochResult train_epoch(
    torch::nn::AnyModule& model, const std::vector<SparseBatch>& batches,
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> criterion,
    torch::optim::Optimizer& optimizer, int dimension) {
  EpochResult result;
  for (const auto& batch : batches) {
    auto manager = std::make_shared<CoordinateManager>(
        dimension, CoordinateMapBackend::CPU);
    const auto step = train_step(model, batch, criterion, optimizer, manager);
    result.avg_loss += step.loss;
    result.total_points += step.num_points;
    ++result.num_batches;
  }
  if (result.num_batches > 0)
    result.avg_loss /= static_cast<double>(result.num_batches);
  return result;
}

EpochResult evaluate(
    torch::nn::AnyModule& model, const std::vector<SparseBatch>& batches,
    std::function<at::Tensor(const at::Tensor&, const at::Tensor&)> criterion,
    int dimension) {
  torch::NoGradGuard guard;
  EpochResult result;
  for (const auto& batch : batches) {
    auto manager = std::make_shared<CoordinateManager>(
        dimension, batch.coordinates.is_cuda() ? CoordinateMapBackend::CUDA
                                               : CoordinateMapBackend::CPU);
    auto input = SparseTensor(batch.features, batch.coordinates, manager);
    const auto output = model.forward<SparseTensor>(input);
    const auto loss = criterion(output.F(), batch.labels);
    result.avg_loss += loss.item<double>();
    result.total_points += batch.features.size(0);
    ++result.num_batches;
  }
  if (result.num_batches > 0)
    result.avg_loss /= static_cast<double>(result.num_batches);
  return result;
}

GradientAccumulator::GradientAccumulator(torch::optim::Optimizer& optimizer,
                                         int accumulation_steps)
    : optimizer_(optimizer), accumulation_steps_(accumulation_steps) {
  TORCH_CHECK(accumulation_steps > 0,
              "accumulation_steps must be positive");
}

bool GradientAccumulator::backward_and_maybe_step(at::Tensor loss) {
  (loss / static_cast<double>(accumulation_steps_)).backward();
  if (++current_step_ < accumulation_steps_) return false;
  optimizer_.step();
  optimizer_.zero_grad();
  current_step_ = 0;
  return true;
}

void GradientAccumulator::flush() {
  if (current_step_ == 0) return;
  optimizer_.step();
  optimizer_.zero_grad();
  current_step_ = 0;
}

void GradientAccumulator::reset() { current_step_ = 0; }

StepLRScheduler::StepLRScheduler(torch::optim::Optimizer& optimizer,
                                 int step_size, double gamma)
    : optimizer_(optimizer), step_size_(step_size), gamma_(gamma) {
  TORCH_CHECK(step_size_ > 0 && gamma_ > 0.0,
              "Step size and gamma must be positive");
  if (!optimizer_.param_groups().empty())
    initial_lr_ = optimizer_.param_groups()[0].options().get_lr();
}

void StepLRScheduler::step() {
  ++epoch_;
  if (epoch_ % step_size_ != 0) return;
  const auto lr = current_lr() * gamma_;
  for (auto& group : optimizer_.param_groups())
    static_cast<torch::optim::OptimizerOptions&>(group.options()).set_lr(lr);
}

double StepLRScheduler::current_lr() const {
  return optimizer_.param_groups().empty()
             ? initial_lr_
             : optimizer_.param_groups()[0].options().get_lr();
}

} // namespace training
} // namespace riemann
