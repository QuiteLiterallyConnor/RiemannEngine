#include "bindings.hpp"

#include "engine/nn/activation.hpp"
#include "engine/nn/conv.hpp"
#include "engine/nn/linear.hpp"
#include "engine/nn/normalization.hpp"
#include "engine/nn/operations.hpp"
#include "engine/nn/pooling.hpp"
#include "engine/nn/resample.hpp"
#include "engine/nn/residual.hpp"
#include "engine/nn/squeeze_excite.hpp"

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace riemann;

namespace {

template <typename Impl>
using SparseModuleClass = py::class_<Impl, std::shared_ptr<Impl>>;

// torch_python already registers torch::nn::Module with pybind11, so the
// layers cannot use it as a pybind base class. Attach the shared
// torch::nn::Module surface to each bound class instead.
template <typename Impl>
SparseModuleClass<Impl> bind_module(py::module_ &m, const char *name) {
  SparseModuleClass<Impl> cls(m, name);
  cls.def(
      "parameters",
      [](Impl &self, bool recurse) { return self.parameters(recurse); },
      py::arg("recurse") = true);
  cls.def(
      "named_parameters",
      [](Impl &self, bool recurse) {
        py::dict result;
        for (const auto &pair : self.named_parameters(recurse))
          result[py::str(pair.key())] = pair.value();
        return result;
      },
      py::arg("recurse") = true);
  cls.def(
      "train", [](Impl &self, bool mode) { self.train(mode); },
      py::arg("mode") = true);
  cls.def("eval", [](Impl &self) { self.eval(); });
  cls.def_property_readonly(
      "training", [](const Impl &self) { return self.is_training(); });
  cls.def(
      "to", [](Impl &self, at::Device device) { self.to(device); },
      py::arg("device"));
  cls.def("zero_grad", [](Impl &self) { self.zero_grad(); });
  cls.def("__repr__", [](const Impl &self) {
    return "<riemannengine.nn." + self.name() + ">";
  });
  return cls;
}

// Registers a module whose forward maps SparseTensor -> SparseTensor.
template <typename Impl>
SparseModuleClass<Impl> bind_unary(py::module_ &m, const char *name) {
  auto cls = bind_module<Impl>(m, name);
  cls.def(
      "forward",
      [](Impl &self, const SparseTensor &input) { return self.forward(input); },
      py::arg("input"), py::call_guard<py::gil_scoped_release>());
  cls.def(
      "__call__",
      [](Impl &self, const SparseTensor &input) { return self.forward(input); },
      py::arg("input"), py::call_guard<py::gil_scoped_release>());
  return cls;
}

// Feature-wise activation without constructor arguments.
template <typename Impl>
void bind_simple_activation(py::module_ &m, const char *name) {
  bind_unary<Impl>(m, name).def(
      py::init([]() { return std::make_shared<Impl>(); }));
}

} // namespace

namespace riemann_py {

void bind_nn(py::module_ &m) {

  // --------------------------------------------------------------------
  // Convolutions
  // --------------------------------------------------------------------
  auto conv = bind_unary<RiemannConvolutionImpl>(m, "RiemannConvolution");
  conv.def(py::init([](int in_channels, int out_channels, int kernel_size,
                       int stride, int dilation, bool bias,
                       bool expand_coordinates,
                       ConvolutionMode::Type convolution_mode, int dimension) {
             return std::make_shared<RiemannConvolutionImpl>(
                 in_channels, out_channels, kernel_size, stride, dilation,
                 bias, expand_coordinates, convolution_mode, dimension);
           }),
           py::arg("in_channels"), py::arg("out_channels"),
           py::arg("kernel_size") = 3, py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("bias") = false,
           py::arg("expand_coordinates") = false,
           py::arg("convolution_mode") = ConvolutionMode::DEFAULT,
           py::arg("dimension") = 3);
  conv.def_property_readonly(
      "kernel",
      [](RiemannConvolutionImpl &self) { return self.base()->kernel_; });
  conv.def_property_readonly(
      "bias", [](RiemannConvolutionImpl &self) { return self.base()->bias_; });

  auto convt = bind_unary<RiemannConvolutionTransposeImpl>(
      m, "RiemannConvolutionTranspose");
  convt.def(py::init([](int in_channels, int out_channels, int kernel_size,
                        int stride, int dilation, bool bias,
                        bool expand_coordinates,
                        ConvolutionMode::Type convolution_mode,
                        int dimension) {
              return std::make_shared<RiemannConvolutionTransposeImpl>(
                  in_channels, out_channels, kernel_size, stride, dilation,
                  bias, expand_coordinates, convolution_mode, dimension);
            }),
            py::arg("in_channels"), py::arg("out_channels"),
            py::arg("kernel_size") = 3, py::arg("stride") = 1,
            py::arg("dilation") = 1, py::arg("bias") = false,
            py::arg("expand_coordinates") = false,
            py::arg("convolution_mode") = ConvolutionMode::DEFAULT,
            py::arg("dimension") = 3);
  convt.def_property_readonly(
      "kernel", [](RiemannConvolutionTransposeImpl &self) {
        return self.base()->kernel_;
      });

  auto gconvt = bind_unary<RiemannGenerativeConvolutionTransposeImpl>(
      m, "RiemannGenerativeConvolutionTranspose");
  gconvt.def(py::init([](int in_channels, int out_channels, int kernel_size,
                         int stride, int dilation, bool bias,
                         ConvolutionMode::Type convolution_mode,
                         int dimension) {
               return std::make_shared<
                   RiemannGenerativeConvolutionTransposeImpl>(
                   in_channels, out_channels, kernel_size, stride, dilation,
                   bias, convolution_mode, dimension);
             }),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size") = 3, py::arg("stride") = 1,
             py::arg("dilation") = 1, py::arg("bias") = false,
             py::arg("convolution_mode") = ConvolutionMode::DEFAULT,
             py::arg("dimension") = 3);

  // --------------------------------------------------------------------
  // Linear
  // --------------------------------------------------------------------
  bind_unary<RiemannLinearImpl>(m, "RiemannLinear")
      .def(py::init([](int64_t in_features, int64_t out_features, bool bias) {
             return std::make_shared<RiemannLinearImpl>(in_features,
                                                        out_features, bias);
           }),
           py::arg("in_features"), py::arg("out_features"),
           py::arg("bias") = true);

  // --------------------------------------------------------------------
  // Normalization
  // --------------------------------------------------------------------
  bind_unary<RiemannBatchNormImpl>(m, "RiemannBatchNorm")
      .def(py::init([](int64_t num_features, double eps, double momentum,
                       bool affine, bool track_running_stats) {
             return std::make_shared<RiemannBatchNormImpl>(
                 num_features, eps, momentum, affine, track_running_stats);
           }),
           py::arg("num_features"), py::arg("eps") = 1e-5,
           py::arg("momentum") = 0.1, py::arg("affine") = true,
           py::arg("track_running_stats") = true);

  bind_unary<RiemannInstanceNormImpl>(m, "RiemannInstanceNorm")
      .def(py::init([](int64_t num_features) {
             return std::make_shared<RiemannInstanceNormImpl>(num_features);
           }),
           py::arg("num_features"));

  bind_unary<RiemannStableInstanceNormImpl>(m, "RiemannStableInstanceNorm")
      .def(py::init([](int64_t num_features) {
             return std::make_shared<RiemannStableInstanceNormImpl>(
                 num_features);
           }),
           py::arg("num_features"));

  bind_unary<RiemannLayerNormImpl>(m, "RiemannLayerNorm")
      .def(py::init([](std::vector<int64_t> normalized_shape, double eps) {
             return std::make_shared<RiemannLayerNormImpl>(
                 torch::nn::LayerNormOptions(std::move(normalized_shape))
                     .eps(eps));
           }),
           py::arg("normalized_shape"), py::arg("eps") = 1e-5);

  bind_unary<RiemannGroupNormImpl>(m, "RiemannGroupNorm")
      .def(py::init([](int64_t num_groups, int64_t num_channels, double eps) {
             return std::make_shared<RiemannGroupNormImpl>(
                 torch::nn::GroupNormOptions(num_groups, num_channels)
                     .eps(eps));
           }),
           py::arg("num_groups"), py::arg("num_channels"),
           py::arg("eps") = 1e-5);

  // --------------------------------------------------------------------
  // Activations
  // --------------------------------------------------------------------
  bind_unary<RiemannReLUImpl>(m, "RiemannReLU")
      .def(py::init([](bool inplace) {
             return std::make_shared<RiemannReLUImpl>(inplace);
           }),
           py::arg("inplace") = false);

  bind_simple_activation<RiemannReLU6Impl>(m, "RiemannReLU6");
  bind_simple_activation<RiemannSELUImpl>(m, "RiemannSELU");
  bind_simple_activation<RiemannGELUImpl>(m, "RiemannGELU");
  bind_simple_activation<RiemannSiLUImpl>(m, "RiemannSiLU");
  bind_simple_activation<RiemannSigmoidImpl>(m, "RiemannSigmoid");
  bind_simple_activation<RiemannTanhImpl>(m, "RiemannTanh");
  bind_simple_activation<RiemannLogSigmoidImpl>(m, "RiemannLogSigmoid");
  bind_simple_activation<RiemannSoftsignImpl>(m, "RiemannSoftsign");
  bind_simple_activation<RiemannTanhshrinkImpl>(m, "RiemannTanhshrink");
  bind_simple_activation<RiemannHardsigmoidImpl>(m, "RiemannHardsigmoid");
  bind_simple_activation<RiemannHardswishImpl>(m, "RiemannHardswish");

  bind_unary<RiemannLeakyReLUImpl>(m, "RiemannLeakyReLU")
      .def(py::init([](double negative_slope, bool inplace) {
             return std::make_shared<RiemannLeakyReLUImpl>(
                 torch::nn::LeakyReLUOptions()
                     .negative_slope(negative_slope)
                     .inplace(inplace));
           }),
           py::arg("negative_slope") = 0.01, py::arg("inplace") = false);

  bind_unary<RiemannELUImpl>(m, "RiemannELU")
      .def(py::init([](double alpha, bool inplace) {
             return std::make_shared<RiemannELUImpl>(
                 torch::nn::ELUOptions().alpha(alpha).inplace(inplace));
           }),
           py::arg("alpha") = 1.0, py::arg("inplace") = false);

  bind_unary<RiemannCELUImpl>(m, "RiemannCELU")
      .def(py::init([](double alpha, bool inplace) {
             return std::make_shared<RiemannCELUImpl>(
                 torch::nn::CELUOptions().alpha(alpha).inplace(inplace));
           }),
           py::arg("alpha") = 1.0, py::arg("inplace") = false);

  bind_unary<RiemannHardshrinkImpl>(m, "RiemannHardshrink")
      .def(py::init([](double lambda) {
             return std::make_shared<RiemannHardshrinkImpl>(
                 torch::nn::HardshrinkOptions(lambda));
           }),
           py::arg("lambd") = 0.5);

  bind_unary<RiemannSoftshrinkImpl>(m, "RiemannSoftshrink")
      .def(py::init([](double lambda) {
             return std::make_shared<RiemannSoftshrinkImpl>(
                 torch::nn::SoftshrinkOptions(lambda));
           }),
           py::arg("lambd") = 0.5);

  bind_unary<RiemannHardtanhImpl>(m, "RiemannHardtanh")
      .def(py::init([](double min_val, double max_val) {
             return std::make_shared<RiemannHardtanhImpl>(
                 torch::nn::HardtanhOptions().min_val(min_val).max_val(
                     max_val));
           }),
           py::arg("min_val") = -1.0, py::arg("max_val") = 1.0);

  bind_unary<RiemannPReLUImpl>(m, "RiemannPReLU")
      .def(py::init([](int64_t num_parameters, double init) {
             return std::make_shared<RiemannPReLUImpl>(
                 torch::nn::PReLUOptions()
                     .num_parameters(num_parameters)
                     .init(init));
           }),
           py::arg("num_parameters") = 1, py::arg("init") = 0.25);

  bind_unary<RiemannRReLUImpl>(m, "RiemannRReLU")
      .def(py::init([](double lower, double upper) {
             return std::make_shared<RiemannRReLUImpl>(
                 torch::nn::RReLUOptions().lower(lower).upper(upper));
           }),
           py::arg("lower") = 1.0 / 8.0, py::arg("upper") = 1.0 / 3.0);

  bind_unary<RiemannSoftplusImpl>(m, "RiemannSoftplus")
      .def(py::init([](double beta, double threshold) {
             return std::make_shared<RiemannSoftplusImpl>(
                 torch::nn::SoftplusOptions().beta(beta).threshold(
                     threshold));
           }),
           py::arg("beta") = 1.0, py::arg("threshold") = 20.0);

  bind_unary<RiemannThresholdImpl>(m, "RiemannThreshold")
      .def(py::init([](double threshold, double value) {
             return std::make_shared<RiemannThresholdImpl>(
                 torch::nn::ThresholdOptions(threshold, value));
           }),
           py::arg("threshold"), py::arg("value"));

  bind_unary<RiemannSoftminImpl>(m, "RiemannSoftmin")
      .def(py::init([](int64_t dim) {
             return std::make_shared<RiemannSoftminImpl>(
                 torch::nn::SoftminOptions(dim));
           }),
           py::arg("dim") = 1);

  bind_unary<RiemannSoftmaxImpl>(m, "RiemannSoftmax")
      .def(py::init([](int64_t dim) {
             return std::make_shared<RiemannSoftmaxImpl>(
                 torch::nn::SoftmaxOptions(dim));
           }),
           py::arg("dim") = 1);

  bind_unary<RiemannLogSoftmaxImpl>(m, "RiemannLogSoftmax")
      .def(py::init([](int64_t dim) {
             return std::make_shared<RiemannLogSoftmaxImpl>(
                 torch::nn::LogSoftmaxOptions(dim));
           }),
           py::arg("dim") = 1);

  bind_unary<RiemannDropoutImpl>(m, "RiemannDropout")
      .def(py::init([](double p) {
             return std::make_shared<RiemannDropoutImpl>(
                 torch::nn::DropoutOptions(p));
           }),
           py::arg("p") = 0.5);

  bind_unary<RiemannAlphaDropoutImpl>(m, "RiemannAlphaDropout")
      .def(py::init([](double p) {
             return std::make_shared<RiemannAlphaDropoutImpl>(
                 torch::nn::AlphaDropoutOptions(p));
           }),
           py::arg("p") = 0.5);

  bind_unary<RiemannSinusoidalImpl>(m, "RiemannSinusoidal")
      .def(py::init([](int in_channels, int out_channels) {
             return std::make_shared<RiemannSinusoidalImpl>(in_channels,
                                                            out_channels);
           }),
           py::arg("in_channels"), py::arg("out_channels"));

  // --------------------------------------------------------------------
  // Pooling
  // --------------------------------------------------------------------
  bind_unary<RiemannAvgPoolingImpl>(m, "RiemannAvgPooling")
      .def(py::init([](int kernel_size, int stride, int dilation,
                       int dimension) {
             return std::make_shared<RiemannAvgPoolingImpl>(
                 kernel_size, stride, dilation, dimension);
           }),
           py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("dimension") = 3);

  bind_unary<RiemannSumPoolingImpl>(m, "RiemannSumPooling")
      .def(py::init([](int kernel_size, int stride, int dilation,
                       int dimension) {
             return std::make_shared<RiemannSumPoolingImpl>(
                 kernel_size, stride, dilation, dimension);
           }),
           py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("dimension") = 3);

  bind_unary<RiemannMaxPoolingImpl>(m, "RiemannMaxPooling")
      .def(py::init([](int kernel_size, int stride, int dilation,
                       int dimension) {
             return std::make_shared<RiemannMaxPoolingImpl>(
                 kernel_size, stride, dilation, dimension);
           }),
           py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("dimension") = 3);

  bind_unary<RiemannPoolingTransposeImpl>(m, "RiemannPoolingTranspose")
      .def(py::init([](int kernel_size, int stride, int dilation,
                       bool expand_coordinates, int dimension) {
             return std::make_shared<RiemannPoolingTransposeImpl>(
                 kernel_size, stride, dilation, expand_coordinates,
                 dimension);
           }),
           py::arg("kernel_size"), py::arg("stride"), py::arg("dilation") = 1,
           py::arg("expand_coordinates") = false, py::arg("dimension") = 3);

  bind_unary<RiemannGlobalPoolingImpl>(m, "RiemannGlobalPooling")
      .def(py::init([](PoolingMode::Type mode) {
             return std::make_shared<RiemannGlobalPoolingImpl>(mode);
           }),
           py::arg("mode") = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);

  bind_unary<RiemannGlobalSumPoolingImpl>(m, "RiemannGlobalSumPooling")
      .def(py::init([]() {
        return std::make_shared<RiemannGlobalSumPoolingImpl>();
      }));

  bind_unary<RiemannGlobalAvgPoolingImpl>(m, "RiemannGlobalAvgPooling")
      .def(py::init([]() {
        return std::make_shared<RiemannGlobalAvgPoolingImpl>();
      }));

  bind_unary<RiemannGlobalMaxPoolingImpl>(m, "RiemannGlobalMaxPooling")
      .def(py::init([]() {
        return std::make_shared<RiemannGlobalMaxPoolingImpl>();
      }));

  // --------------------------------------------------------------------
  // Structural operations
  // --------------------------------------------------------------------
  bind_module<RiemannUnionImpl>(m, "RiemannUnion")
      .def(py::init([]() { return std::make_shared<RiemannUnionImpl>(); }))
      .def("forward", &RiemannUnionImpl::forward, py::arg("inputs"))
      .def("__call__", &RiemannUnionImpl::forward, py::arg("inputs"));

  bind_module<RiemannToSparseTensorImpl>(m, "RiemannToSparseTensor")
      .def(py::init([](bool remove_zeros,
                       c10::optional<at::Tensor> coordinates) {
             return std::make_shared<RiemannToSparseTensorImpl>(
                 remove_zeros, std::move(coordinates));
           }),
           py::arg("remove_zeros") = true,
           py::arg("coordinates") = py::none())
      .def("forward", &RiemannToSparseTensorImpl::forward, py::arg("input"))
      .def("__call__", &RiemannToSparseTensorImpl::forward, py::arg("input"));

  bind_module<RiemannToDenseTensorImpl>(m, "RiemannToDenseTensor")
      .def(py::init([](c10::optional<std::vector<int64_t>> shape) {
             return std::make_shared<RiemannToDenseTensorImpl>(
                 std::move(shape));
           }),
           py::arg("shape") = py::none())
      .def("forward", &RiemannToDenseTensorImpl::forward, py::arg("input"))
      .def("__call__", &RiemannToDenseTensorImpl::forward, py::arg("input"));

  bind_module<RiemannToFeatureImpl>(m, "RiemannToFeature")
      .def(py::init([]() { return std::make_shared<RiemannToFeatureImpl>(); }))
      .def("forward", &RiemannToFeatureImpl::forward, py::arg("input"))
      .def("__call__", &RiemannToFeatureImpl::forward, py::arg("input"));

  bind_module<RiemannPruningImpl>(m, "RiemannPruning")
      .def(py::init([]() { return std::make_shared<RiemannPruningImpl>(); }))
      .def("forward", &RiemannPruningImpl::forward, py::arg("input"),
           py::arg("mask"))
      .def("__call__", &RiemannPruningImpl::forward, py::arg("input"),
           py::arg("mask"));

  bind_module<RiemannBroadcastImpl>(m, "RiemannBroadcast")
      .def(py::init([]() { return std::make_shared<RiemannBroadcastImpl>(); }))
      .def("forward", &RiemannBroadcastImpl::forward, py::arg("input"),
           py::arg("input_global"))
      .def("__call__", &RiemannBroadcastImpl::forward, py::arg("input"),
           py::arg("input_global"));

  bind_module<RiemannBroadcastAdditionImpl>(m,
                                            "RiemannBroadcastAddition")
      .def(py::init(
          []() { return std::make_shared<RiemannBroadcastAdditionImpl>(); }))
      .def("forward", &RiemannBroadcastAdditionImpl::forward,
           py::arg("input"), py::arg("input_global"))
      .def("__call__", &RiemannBroadcastAdditionImpl::forward,
           py::arg("input"), py::arg("input_global"));

  bind_module<RiemannBroadcastMultiplicationImpl>(
      m, "RiemannBroadcastMultiplication")
      .def(py::init([]() {
        return std::make_shared<RiemannBroadcastMultiplicationImpl>();
      }))
      .def("forward", &RiemannBroadcastMultiplicationImpl::forward,
           py::arg("input"), py::arg("input_global"))
      .def("__call__", &RiemannBroadcastMultiplicationImpl::forward,
           py::arg("input"), py::arg("input_global"));

  bind_module<RiemannBroadcastConcatenationImpl>(
      m, "RiemannBroadcastConcatenation")
      .def(py::init([]() {
        return std::make_shared<RiemannBroadcastConcatenationImpl>();
      }))
      .def("forward", &RiemannBroadcastConcatenationImpl::forward,
           py::arg("input"), py::arg("input_global"))
      .def("__call__", &RiemannBroadcastConcatenationImpl::forward,
           py::arg("input"), py::arg("input_global"));

  // --------------------------------------------------------------------
  // Interpolation
  // --------------------------------------------------------------------
  py::class_<InterpolationResult>(m, "InterpolationResult")
      .def_readonly("output", &InterpolationResult::output)
      .def_readonly("in_map", &InterpolationResult::in_map)
      .def_readonly("out_map", &InterpolationResult::out_map)
      .def_readonly("weights", &InterpolationResult::weights);

  bind_module<RiemannInterpolationImpl>(m, "RiemannInterpolation")
      .def(py::init([](bool return_kernel_map, bool return_weights) {
             return std::make_shared<RiemannInterpolationImpl>(
                 return_kernel_map, return_weights);
           }),
           py::arg("return_kernel_map") = false,
           py::arg("return_weights") = false)
      .def("forward", &RiemannInterpolationImpl::forward, py::arg("input"),
           py::arg("field"))
      .def("__call__", &RiemannInterpolationImpl::forward, py::arg("input"),
           py::arg("field"))
      .def("forward_full", &RiemannInterpolationImpl::forward_full,
           py::arg("input"), py::arg("field"));

  // --------------------------------------------------------------------
  // Residual and squeeze-excitation blocks
  // --------------------------------------------------------------------
  bind_unary<BasicBlockImpl>(m, "BasicBlock")
      .def(py::init([](int inplanes, int planes, int stride, int dilation,
                       double bn_momentum, int dimension) {
             return std::make_shared<BasicBlockImpl>(
                 inplanes, planes, stride, dilation, torch::nn::AnyModule(),
                 bn_momentum, dimension);
           }),
           py::arg("inplanes"), py::arg("planes"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("bn_momentum") = 0.1,
           py::arg("dimension") = 3)
      .def_readonly_static("expansion", &BasicBlockImpl::expansion);

  bind_unary<BottleneckImpl>(m, "Bottleneck")
      .def(py::init([](int inplanes, int planes, int stride, int dilation,
                       double bn_momentum, int dimension) {
             return std::make_shared<BottleneckImpl>(
                 inplanes, planes, stride, dilation, torch::nn::AnyModule(),
                 bn_momentum, dimension);
           }),
           py::arg("inplanes"), py::arg("planes"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("bn_momentum") = 0.1,
           py::arg("dimension") = 3)
      .def_readonly_static("expansion", &BottleneckImpl::expansion);

  bind_unary<SELayerImpl>(m, "SELayer")
      .def(py::init([](int channel, int reduction, int dimension) {
             return std::make_shared<SELayerImpl>(channel, reduction,
                                                  dimension);
           }),
           py::arg("channel"), py::arg("reduction") = 16,
           py::arg("dimension") = 3);

  bind_unary<SEBasicBlockImpl>(m, "SEBasicBlock")
      .def(py::init([](int inplanes, int planes, int stride, int dilation,
                       int reduction, double bn_momentum, int dimension) {
             return std::make_shared<SEBasicBlockImpl>(
                 inplanes, planes, stride, dilation, torch::nn::AnyModule(),
                 reduction, bn_momentum, dimension);
           }),
           py::arg("inplanes"), py::arg("planes"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("reduction") = 16,
           py::arg("bn_momentum") = 0.1, py::arg("dimension") = 3);

  bind_unary<SEBottleneckImpl>(m, "SEBottleneck")
      .def(py::init([](int inplanes, int planes, int stride, int dilation,
                       int reduction, double bn_momentum, int dimension) {
             return std::make_shared<SEBottleneckImpl>(
                 inplanes, planes, stride, dilation, torch::nn::AnyModule(),
                 reduction, bn_momentum, dimension);
           }),
           py::arg("inplanes"), py::arg("planes"), py::arg("stride") = 1,
           py::arg("dilation") = 1, py::arg("reduction") = 16,
           py::arg("bn_momentum") = 0.1, py::arg("dimension") = 3);
}

} // namespace riemann_py
