#include "core/host_sparse_conv_stencils.hpp"
#include "utils/test_framework.hpp"

#include <array>

namespace {

void check_values(const float *actual, const std::array<float, 6> &expected) {
  for (size_t i = 0; i < expected.size(); ++i)
    CHECK_NEAR(actual[i], expected[i], 1e-5);
}

} // namespace

TEST(sparse_cpu_pointwise_forward_identity_mapping_matches_expected) {
  const float input[] = {1, 2, 3, 4, 5, 6};
  const float kernel[] = {1, 2, 3, 4};
  float output[6] = {};
  const riemann::cpu_in_maps in_maps{{0, 1, 2}};
  const riemann::cpu_out_maps out_maps{{0, 1, 2}};

  riemann::sparse_convolution_forward_cpu_kernel<float, int>(
      input, 2, output, 2, 3, 3, kernel, in_maps, out_maps);

  check_values(output, {7, 10, 15, 22, 23, 34});
}

TEST(sparse_cpu_pointwise_backward_identity_mapping_matches_expected) {
  const float input[] = {1, 2, 3, 4, 5, 6};
  const float grad_output[] = {1, 2, 3, 4, 5, 6};
  const float kernel[] = {1, 2, 3, 4};
  float grad_input[6] = {};
  float grad_kernel[4] = {};
  const riemann::cpu_in_maps in_maps{{0, 1, 2}};
  const riemann::cpu_out_maps out_maps{{0, 1, 2}};

  riemann::sparse_convolution_backward_cpu_kernel<float, int>(
      input, grad_input, 2, grad_output, 2, 3, 3, kernel, grad_kernel,
      in_maps, out_maps);

  check_values(grad_input, {5, 11, 11, 25, 17, 39});
  CHECK_NEAR(grad_kernel[0], 35, 1e-5);
  CHECK_NEAR(grad_kernel[1], 44, 1e-5);
  CHECK_NEAR(grad_kernel[2], 44, 1e-5);
  CHECK_NEAR(grad_kernel[3], 56, 1e-5);
}

TEST(sparse_cpu_pointwise_forward_nonidentity_mapping_uses_fallback) {
  const float input[] = {1, 3, 5};
  const float kernel[] = {2};
  float output[3] = {};
  const riemann::cpu_in_maps in_maps{{2, 0, 1}};
  const riemann::cpu_out_maps out_maps{{1, 2, 0}};

  riemann::sparse_convolution_forward_cpu_kernel<float, int>(
      input, 1, output, 1, 3, 3, kernel, in_maps, out_maps);

  CHECK_NEAR(output[0], 6, 1e-5);
  CHECK_NEAR(output[1], 10, 1e-5);
  CHECK_NEAR(output[2], 2, 1e-5);
}
