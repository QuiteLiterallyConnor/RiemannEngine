// High-level tests for the batching and discretization helpers that feed
// point-cloud data into the engine.
#include "RiemannEngine/riemann.hpp"

#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <tuple>
#include <vector>

namespace {

torch::Tensor int_coords(std::initializer_list<std::initializer_list<int>> rows) {
    std::vector<int> flat;
    int64_t columns = 0;
    for (const auto& row : rows) {
        columns = static_cast<int64_t>(row.size());
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return torch::tensor(flat, torch::kInt32)
        .reshape({static_cast<int64_t>(rows.size()), columns});
}

}  // namespace

TEST(batching_batched_coordinates_prepends_batch_index) {
    auto a = int_coords({{0, 0, 0}, {1, 2, 3}});
    auto b = int_coords({{4, 5, 6}});
    auto batched = riemann::batched_coordinates({a, b});

    CHECK_EQ(batched.size(0), int64_t(3));
    CHECK_EQ(batched.size(1), int64_t(4));
    CHECK_EQ(batched[0][0].item<int>(), 0);
    CHECK_EQ(batched[1][0].item<int>(), 0);
    CHECK_EQ(batched[2][0].item<int>(), 1);
    CHECK_EQ(batched[1][3].item<int>(), 3);
    CHECK_EQ(batched[2][1].item<int>(), 4);
}

TEST(batching_batched_coordinates_floors_float_input) {
    auto points = torch::tensor({{0.9f, -0.1f, 1.5f}});
    auto batched = riemann::batched_coordinates({points});
    CHECK_EQ(batched.scalar_type(), torch::kInt32);
    CHECK_EQ(batched[0][1].item<int>(), 0);
    CHECK_EQ(batched[0][2].item<int>(), -1);
    CHECK_EQ(batched[0][3].item<int>(), 1);
}

TEST(batching_sparse_collate_concatenates_samples_and_labels) {
    auto coords_a = int_coords({{0, 0, 0}, {1, 1, 1}});
    auto coords_b = int_coords({{2, 2, 2}});
    auto feats_a = torch::ones({2, 4});
    auto feats_b = torch::full({1, 4}, 2.0f);
    auto labels_a = torch::tensor({1, 2}, torch::kInt64);
    auto labels_b = torch::tensor({3}, torch::kInt64);

    auto [coords, feats, labels] = riemann::sparse_collate(
        {coords_a, coords_b}, {feats_a, feats_b}, {labels_a, labels_b});

    CHECK_EQ(coords.size(0), int64_t(3));
    CHECK_EQ(coords.size(1), int64_t(4));
    CHECK_EQ(feats.size(0), int64_t(3));
    CHECK(labels.has_value());
    CHECK_EQ(labels->size(0), int64_t(3));
    CHECK_EQ(coords[2][0].item<int>(), 1);  // second sample's batch index
    CHECK_NEAR(feats[2][0].item<float>(), 2.0f, 1e-6);
    CHECK_EQ((*labels)[2].item<int64_t>(), int64_t(3));

    // The collated batch must form a valid sparse tensor.
    riemann::SparseTensor tensor(feats, coords);
    CHECK_EQ(tensor.dimension(), 3);
    CHECK_EQ(tensor.F().size(0), tensor.C().size(0));
}

TEST(batching_sparse_collation_respects_point_limit) {
    auto sample = [](int offset, int64_t count) {
        auto coords = torch::arange(count, torch::kInt32)
                          .add(offset)
                          .reshape({count, 1})
                          .repeat({1, 3});
        auto feats = torch::ones({count, 2});
        auto labels = torch::zeros({count}, torch::kInt64);
        return std::make_tuple(coords, feats, labels);
    };

    riemann::SparseCollation collate(3);
    auto [coords, feats, labels] = collate({sample(0, 2), sample(10, 2)});

    // The second sample would exceed the 3-point budget and must be dropped.
    CHECK_EQ(coords.size(0), int64_t(2));
    CHECK_EQ(feats.size(0), int64_t(2));
    CHECK_EQ(labels.size(0), int64_t(2));
    CHECK_EQ(coords.select(1, 0).max().item<int>(), 0);
}

TEST(batching_sparse_quantize_dedupes_voxels) {
    auto points = torch::tensor({{0.1, 0.4, 0.9},
                                 {0.3, 0.2, 0.7},
                                 {2.0, 3.0, 4.0}},
                                torch::kFloat64);
    auto features = torch::tensor({{1.0f}, {2.0f}, {3.0f}});

    auto result = riemann::sparse_quantize(points, features);

    CHECK_EQ(result.coordinates.size(0), int64_t(2));
    CHECK(result.features.has_value());
    CHECK_EQ(result.features->size(0), int64_t(2));
    CHECK_EQ(result.inverse_mapping.size(0), int64_t(3));
    // The first two points fall into the same voxel.
    CHECK_EQ(result.inverse_mapping[0].item<int64_t>(),
             result.inverse_mapping[1].item<int64_t>());
    CHECK(result.inverse_mapping[0].item<int64_t>() !=
          result.inverse_mapping[2].item<int64_t>());
}

TEST(batching_sparse_quantize_applies_quantization_size) {
    auto points = torch::tensor({{0.0, 0.0},
                                 {1.9, 1.9},
                                 {2.1, 0.0}},
                                torch::kFloat64);
    auto result = riemann::sparse_quantize(points, c10::nullopt, c10::nullopt,
                                           -100, 2.0);
    // With a voxel size of 2 the first two points share a cell.
    CHECK_EQ(result.coordinates.size(0), int64_t(2));
    CHECK_EQ(result.inverse_mapping[0].item<int64_t>(),
             result.inverse_mapping[1].item<int64_t>());
}

TEST(batching_sparse_quantize_coalesces_conflicting_labels) {
    auto points = torch::tensor({{0.0, 0.0}, {0.4, 0.4}, {3.0, 3.0}},
                                torch::kFloat64);
    auto features = torch::ones({3, 1});
    auto labels = torch::tensor({1, 2, 5}, torch::kInt32);

    auto result = riemann::sparse_quantize(points, features, labels, -100);

    CHECK(result.labels.has_value());
    CHECK_EQ(result.labels->size(0), int64_t(2));
    // Conflicting labels in a shared voxel collapse to the ignore label.
    CHECK_EQ((*result.labels)[0].item<int64_t>(), int64_t(-100));
    CHECK_EQ((*result.labels)[1].item<int64_t>(), int64_t(5));
}

TEST(batching_unique_coordinate_map_groups_duplicates) {
    auto coords = int_coords({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 1, 1}});
    auto [unique_rows, inverse] = riemann::unique_coordinate_map(coords);

    CHECK_EQ(unique_rows.size(0), int64_t(2));
    CHECK_EQ(inverse.size(0), int64_t(3));
    CHECK_EQ(inverse[0].item<int64_t>(), inverse[1].item<int64_t>());
    CHECK(inverse[0].item<int64_t>() != inverse[2].item<int64_t>());
}

TEST(batching_training_collate_samples_builds_batch) {
    using riemann::training::SparseSample;
    std::vector<SparseSample> samples;
    samples.emplace_back(int_coords({{0, 0, 0}, {1, 1, 1}}),
                         torch::ones({2, 3}),
                         torch::tensor({0, 1}, torch::kInt64));
    samples.emplace_back(int_coords({{2, 2, 2}}),
                         torch::full({1, 3}, 5.0f),
                         torch::tensor({2}, torch::kInt64));

    auto batch = riemann::training::sparse_collate_samples(samples);

    CHECK_EQ(batch.batch_size, int64_t(2));
    CHECK_EQ(batch.coordinates.size(0), int64_t(3));
    CHECK_EQ(batch.features.size(0), int64_t(3));
    CHECK_EQ(batch.labels.size(0), int64_t(3));
    CHECK_EQ(batch.coordinates[2][0].item<int>(), 1);
}

TEST(batching_tensor_field_sparse_inverse_mapping_and_splat) {
    // Three points; the first two share voxel (0, 0, 0).
    auto coords = torch::tensor({0.f, 0.2f, 0.3f, 0.1f,
                                 0.f, 0.7f, 0.1f, 0.9f,
                                 0.f, 2.5f, 1.5f, 0.5f}).reshape({3, 4});
    auto feats = torch::tensor({1.f, 3.f, 10.f}).reshape({3, 1});
    riemann::TensorField field(feats, coords);

    auto voxels = field.sparse();
    CHECK_EQ(voxels.F().size(0), int64_t(2));

    auto inverse = field.inverse_mapping(voxels.coordinate_map_key()).to(torch::kLong);
    CHECK_EQ(inverse.size(0), int64_t(3));
    CHECK_EQ(inverse[0].item<int64_t>(), inverse[1].item<int64_t>());
    CHECK(inverse[0].item<int64_t>() != inverse[2].item<int64_t>());
    CHECK(std::abs(voxels.F()[inverse[0].item<int64_t>()][0].item<float>() - 2.f) < 1e-5f);

    // C() must keep the continuous coordinates even after sparse() was called.
    CHECK(torch::allclose(field.C(), coords));

    // Trilinear splat conserves the feature sum.
    auto splatted = field.splat();
    CHECK(std::abs(splatted.F().sum().item<float>() - 14.f) < 1e-4f);
}
