import inspect
import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("riemannengine.gaussian_voxel")

from riemannengine.camera_occupancy import (
    CameraOccupancyModel,
    GridSpec,
    OccupancyMetrics,
    RefinementSpec,
    limit_samples,
    load_checkpoint,
    masked_occupancy_loss,
    project_image_features,
    save_checkpoint,
    scene_split,
    seed_everything,
)


def small_problem():
    spec = GridSpec((6, 6, 3), (-3.0, -3.0, 0.5), (1.0, 1.0, 1.0))
    reference = spec.reference()
    model = CameraOccupancyModel(spec, channels=8)
    camera = torch.eye(4).unsqueeze(0)
    intrinsic = torch.tensor([[[6.0, 0.0, 16.0],
                               [0.0, 6.0, 16.0],
                               [0.0, 0.0, 1.0]]])
    return model, reference, camera, intrinsic


def target_masks(occupied, observed=None):
    observed = torch.ones_like(occupied) if observed is None else observed
    return SimpleNamespace(
        occupied=occupied & observed, free=(~occupied) & observed,
        observed=observed,
    )


def test_scene_split_is_disjoint_reproducible_and_limits_cover_scenes():
    samples = [{"scene_token": f"scene-{scene}"} for scene in range(5)
               for _ in range(4)]
    split = scene_split(samples, seed=3)
    assert split == scene_split(samples, seed=3)
    assert not set(split["train_scenes"]) & set(split["validation_scenes"])
    assert sorted(split["train"] + split["validation"]) == list(range(20))
    selected = limit_samples(split["train"], samples, 4)
    assert len({samples[i]["scene_token"] for i in selected}) == 4
    with pytest.raises(ValueError):
        scene_split([{"scene_token": "only"}])


def test_masked_loss_and_metrics_ignore_unknown_cells():
    targets = target_masks(
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[True, True, False, False]]),
    )
    logits = torch.tensor([[2.0, -2.0, -100.0, 100.0]], requires_grad=True)
    loss = masked_occupancy_loss(logits, targets)
    loss.backward()
    assert torch.equal(logits.grad[0, 2:], torch.zeros(2))
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    metrics = OccupancyMetrics()
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["occupied_iou"] == result["free_iou"] == 1.0
    assert result["observed"] == 2
    empty = target_masks(targets.occupied, torch.zeros_like(targets.observed))
    with pytest.raises(ValueError, match="no observed"):
        masked_occupancy_loss(logits, empty)


def column_targets():
    # One 2x1 x-row of columns, four z levels; voxel index order follows
    # reference.coordinates(), so masks are built from the coordinates.
    spec = GridSpec((2, 1, 4), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    reference = spec.reference()
    coords = reference.coordinates().long()
    x, z = coords[:, 0], coords[:, 2]
    occupied = ((x == 0) & (z == 1)).unsqueeze(0)
    free = ((x == 0) & (z == 0)).unsqueeze(0) | ((x == 1) & (z == 0)).unsqueeze(0)
    targets = SimpleNamespace(occupied=occupied, free=free,
                              observed=occupied | free)
    return reference, targets, x, z


def test_column_ceiling_prior_marks_only_unknown_cells_above_returns():
    from riemannengine.camera_occupancy import column_ceiling_free_prior

    reference, targets, x, z = column_targets()
    prior = column_ceiling_free_prior(targets, reference)[0]
    # Column 0 has a return at z=1; column 1 has none and uses the sample top.
    assert torch.equal(prior, (z >= 2) & ~targets.observed[0])
    margin = column_ceiling_free_prior(targets, reference, margin_levels=1)[0]
    assert torch.equal(margin, z == 3)
    assert not (prior & targets.observed[0]).any()
    no_returns = SimpleNamespace(
        occupied=torch.zeros_like(targets.occupied), free=targets.free,
        observed=targets.free,
    )
    assert not column_ceiling_free_prior(no_returns, reference).any()
    with pytest.raises(ValueError):
        column_ceiling_free_prior(targets, reference, margin_levels=-1)


def test_ceiling_prior_loss_pushes_only_prior_cells_toward_free():
    reference, targets, _, _ = column_targets()
    from riemannengine.camera_occupancy import column_ceiling_free_prior

    prior = column_ceiling_free_prior(targets, reference)
    logits = torch.zeros(targets.occupied.shape, requires_grad=True)
    masked_occupancy_loss(logits, targets, prior, 0.0).backward()
    unconstrained = ~targets.observed & ~prior
    assert torch.equal(logits.grad[prior], torch.zeros(int(prior.sum())))
    logits.grad = None
    masked_occupancy_loss(logits, targets, prior, 0.5).backward()
    assert (logits.grad[prior] > 0).all()
    assert torch.equal(logits.grad[unconstrained],
                       torch.zeros(int(unconstrained.sum())))
    with pytest.raises(ValueError, match="unobserved"):
        masked_occupancy_loss(logits, targets, targets.observed, 0.5)
    metrics = OccupancyMetrics()
    metrics.update(torch.full_like(logits, 3.0), targets, prior)
    result = metrics.compute()
    assert result["above_ceiling_predicted_occupied"] == int(prior.sum())
    assert result["unknown_predicted_occupied"] == int((~targets.observed).sum())


def test_metrics_confusion_counts_match_expected_iou():
    targets = target_masks(torch.tensor([[True, True, False, False]]))
    metrics = OccupancyMetrics()
    metrics.update(torch.tensor([[3.0, -3.0, 3.0, -3.0]]), targets)
    result = metrics.compute()
    assert result["occupied_iou"] == pytest.approx(1 / 3)
    assert result["free_iou"] == pytest.approx(1 / 3)
    assert result["precision"] == result["recall"] == 0.5
    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 1


def test_projection_excludes_behind_camera_and_out_of_frame_points():
    features = torch.full((1, 2, 4, 4), 3.0)
    positions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                              [1e20, 0.0, 1.0]])
    camera = torch.eye(4).unsqueeze(0)
    intrinsic = torch.tensor([[[1.0, 0.0, 2.0],
                               [0.0, 1.0, 2.0],
                               [0.0, 0.0, 1.0]]])
    fused, visible = project_image_features(
        features, positions, camera, intrinsic, (4, 4)
    )
    torch.testing.assert_close(fused[0], torch.full((2,), 3.0))
    assert torch.equal(fused[1:], torch.zeros(2, 2))
    assert torch.equal(visible[:, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_model_rejects_invalid_feature_channel_counts():
    for channels in (0, -1, 0.5):
        with pytest.raises(ValueError, match="positive integer"):
            CameraOccupancyModel(channels=channels)


def test_adaptive_camera_integration_preserves_decoder_shape_and_gradients(tmp_path):
    seed_everything(11)
    model, _, camera, intrinsic = small_problem()
    model.refinement = RefinementSpec(
        max_depth=2, max_voxels=122, max_splits_per_iteration=1,
        max_iterations=2, score_threshold=0, min_density=0,
    )
    reference = model.spec.reference(max_depth=2)
    images = torch.rand(1, 3, 32, 32, requires_grad=True)
    logits, rasterized, _ = model(images, camera, intrinsic, reference)
    assert logits.shape == (1, 108)
    assert rasterized.density.shape == (108, 1)
    fine = rasterized.adaptive.rasterized
    assert rasterized.adaptive.leaf_counts == [108, 115, 122]
    assert not fine.voxels.has_overlapping_leaves()
    torch.testing.assert_close(rasterized.density.sum(), fine.density.sum())
    targets = target_masks((reference.centers()[:, 0] < 0).unsqueeze(0))
    masked_occupancy_loss(logits, targets).backward()
    assert images.grad.abs().max() > 0
    assert model.gaussian_head[-1].weight.grad.abs().max() > 0
    path = tmp_path / "adaptive.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, optimizer, 0, {}, [], (32, 32), 11, 0, [])
    restored, _ = load_checkpoint(path)
    assert restored.refinement == model.refinement
    with torch.no_grad():
        actual, _, _ = restored(images, camera, intrinsic, reference)
    torch.testing.assert_close(actual, logits.detach(), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA-enabled Python PyTorch"
)
def test_cuda_camera_adaptive_forward_backward():
    from riemannengine import gaussian_voxel
    if not gaussian_voxel.cuda_available():
        pytest.skip("extension was built without CUDA kernels")
    model, _, camera, intrinsic = small_problem()
    model.refinement = RefinementSpec(
        max_depth=1, max_voxels=115, max_splits_per_iteration=1,
        max_iterations=1, score_threshold=0, min_density=0,
    )
    model.cuda()
    reference = model.spec.reference(max_depth=1)
    images = torch.rand(1, 3, 32, 32, device="cuda", requires_grad=True)
    logits, rasterized, _ = model(images, camera.cuda(), intrinsic.cuda(), reference)
    targets = target_masks((reference.centers()[:, 0] < 0).unsqueeze(0))
    masked_occupancy_loss(logits, targets).backward()
    assert rasterized.adaptive.leaf_counts == [108, 115]
    assert images.grad.abs().max() > 0
    assert model.gaussian_head[-1].weight.grad.abs().max() > 0


def test_forward_is_image_only_and_gradients_reach_all_learning_stages():
    seed_everything(11)
    model, reference, camera, intrinsic = small_problem()
    assert list(inspect.signature(model.forward).parameters) == [
        "images", "camera_from_reference", "intrinsics", "reference"
    ]
    images = torch.rand(1, 3, 32, 32, requires_grad=True)
    logits, rasterized, gaussians = model(images, camera, intrinsic, reference)
    gaussians.positions.retain_grad()
    gaussians.scales.retain_grad()
    gaussians.opacities.retain_grad()
    targets = target_masks((reference.centers()[:, 0] < 0).unsqueeze(0))
    masked_occupancy_loss(logits, targets).backward()
    assert rasterized.density.shape == (len(reference), 1)
    for gradient in (
        images.grad, model.encoder[0].weight.grad,
        model.gaussian_head[-1].weight.grad, model.decoder[0].weight.grad,
        gaussians.positions.grad, gaussians.scales.grad, gaussians.opacities.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max().item() > 0


def test_height_range_encoding_has_no_horizontal_cell_identity(tmp_path):
    from riemannengine.camera_occupancy import position_encoding

    spec = GridSpec((4, 4, 2), (-4.0, -4.0, 0.0), (2.0, 2.0, 2.0))
    centers = spec.reference().centers().float()
    rotated = centers[:, [1, 0, 2]] * torch.tensor([-1.0, 1.0, 1.0])
    encoded = position_encoding(centers, spec)
    assert torch.allclose(encoded, position_encoding(rotated, spec))
    assert encoded.shape[1] == 18
    assert position_encoding(centers, spec, "absolute").shape[1] == 27
    with pytest.raises(ValueError):
        position_encoding(centers, spec, "unknown")

    model = CameraOccupancyModel(spec, channels=4, encoding="absolute")
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, torch.optim.AdamW(model.parameters()), 0,
                    {}, [], (8, 8), 0, 0.0, [])
    restored, checkpoint = load_checkpoint(path)
    assert restored.encoding_kind == "absolute"
    del checkpoint["position_encoding"]
    torch.save(checkpoint, path)
    assert load_checkpoint(path)[0].encoding_kind == "absolute"


def test_checkpoint_restores_optimizer_rng_and_identical_next_update(tmp_path):
    seed_everything(9)
    model, reference, camera, intrinsic = small_problem()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(1, 3, 32, 32)
    targets = target_masks((reference.centers()[:, 0] < 0).unsqueeze(0))

    def update(network, optim):
        optim.zero_grad(set_to_none=True)
        logits, _, _ = network(images, camera, intrinsic, reference)
        masked_occupancy_loss(logits, targets).backward()
        optim.step()

    update(model, optimizer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, 1,
        {"train": [0], "validation": [1]}, ["a", "b"], (32, 32),
        9, 0.5, [{"step": 1, "occupied_iou": 0.5}],
    )
    random_value, torch_value = random.random(), torch.rand(1)
    update(model, optimizer)
    restored, checkpoint = load_checkpoint(path)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    assert random.random() == random_value
    assert torch.equal(torch.rand(1), torch_value)
    update(restored, restored_optimizer)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_conflicting_scenes_require_and_learn_image_conditioning():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        seed_everything(13)
        model, reference, camera, intrinsic = small_problem()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        images = [torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)]
        left = (reference.centers()[:, 0] < 0).unsqueeze(0)
        targets = [target_masks(left), target_masks(~left)]
        initial_losses = [
            float(masked_occupancy_loss(
                model(image, camera, intrinsic, reference)[0], target
            ).detach())
            for image, target in zip(images, targets)
        ]
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            for image, target in zip(images, targets):
                logits, _, _ = model(image, camera, intrinsic, reference)
                (masked_occupancy_loss(logits, target) / 2).backward()
            optimizer.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            for image, target, initial_loss in zip(images, targets, initial_losses):
                logits, _, _ = model(image, camera, intrinsic, reference)
                metrics = OccupancyMetrics()
                metrics.update(logits, target)
                assert metrics.compute()["occupied_iou"] >= 0.95
                assert float(masked_occupancy_loss(logits, target)) < initial_loss * 0.2
                predictions.append(logits.sigmoid())
        assert (predictions[0] - predictions[1]).abs().mean().item() > 0.8
    finally:
        torch.set_num_threads(old_threads)


def test_old_smoke_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"encoder.weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_checkpoint(path)
