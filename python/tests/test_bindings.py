"""Smoke tests for the riemannengine Python bindings (CPU)."""

import math

import pytest
import torch

import riemannengine as re_


def make_sparse_tensor(n=64, channels=4, dimension=3, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.randint(0, 10, (n, dimension), generator=g,
                           dtype=torch.int32)
    batch_idx = torch.randint(0, batch, (n, 1), generator=g,
                              dtype=torch.int32)
    coords = torch.cat([batch_idx, coords], dim=1)
    feats = torch.randn(n, channels, generator=g)
    return re_.SparseTensor(feats, coords)


class TestSparseTensor:
    def test_construction(self):
        x = make_sparse_tensor()
        assert x.F.shape[1] == 4
        assert x.C.shape[1] == 4  # batch + 3 spatial
        assert x.F.shape[0] == x.C.shape[0]
        assert len(x) == x.F.shape[0]
        assert x.dimension == 3
        assert x.tensor_stride == [1, 1, 1]
        assert "SparseTensor" in repr(x)

    def test_quantization_modes(self):
        coords = torch.zeros(4, 4, dtype=torch.int32)  # all duplicates
        feats = torch.ones(4, 2)
        x = re_.SparseTensor(
            feats, coords,
            quantization_mode=re_.QuantizationMode.UNWEIGHTED_SUM)
        assert len(x) == 1
        assert torch.allclose(x.F, torch.full((1, 2), 4.0))

    def test_arithmetic(self):
        x = make_sparse_tensor()
        y = (x + x) * 2.0
        assert torch.allclose(y.F, x.F * 4.0)

    def test_dense_roundtrip(self):
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.int32)
        feats = torch.tensor([[1.0], [2.0]])
        x = re_.SparseTensor(feats, coords)
        dense = x.dense()
        assert dense[0, 0, 0, 0, 0] == 1.0
        assert dense[0, 0, 1, 2, 3] == 2.0

    def test_batched_coordinates(self):
        a = torch.randint(0, 5, (7, 3), dtype=torch.int32)
        b = torch.randint(0, 5, (5, 3), dtype=torch.int32)
        coords = re_.batched_coordinates([a, b])
        assert coords.shape == (12, 4)
        assert coords[:7, 0].eq(0).all()
        assert coords[7:, 0].eq(1).all()

    def test_coordinate_manager_shared(self):
        x = make_sparse_tensor()
        manager = x.coordinate_manager
        assert manager is not None
        y = re_.SparseTensor(torch.randn(len(x), 2), x.C,
                             coordinate_manager=manager)
        assert y.coordinate_manager is not None


class TestNN:
    def test_convolution_forward_backward(self):
        x = make_sparse_tensor(channels=4)
        conv = re_.nn.RiemannConvolution(4, 8, kernel_size=3, dimension=3)
        y = conv(x)
        assert y.F.shape[1] == 8
        y.F.sum().backward()
        grads = [p.grad for p in conv.parameters()]
        assert grads and all(g is not None for g in grads)

    def test_strided_convolution(self):
        x = make_sparse_tensor(channels=4)
        conv = re_.nn.RiemannConvolution(4, 8, kernel_size=2, stride=2)
        y = conv(x)
        assert y.tensor_stride == [2, 2, 2]

    def test_activation_and_norm(self):
        x = make_sparse_tensor(channels=4)
        relu = re_.nn.RiemannReLU()
        bn = re_.nn.RiemannBatchNorm(4)
        y = relu(bn(x))
        assert (y.F >= 0).all()
        bn.eval()
        assert not bn.training

    def test_global_pooling(self):
        x = make_sparse_tensor(channels=4, batch=2)
        pool = re_.nn.RiemannGlobalAvgPooling()
        y = pool(x)
        assert y.F.shape == (2, 4)

    def test_linear(self):
        x = make_sparse_tensor(channels=4)
        linear = re_.nn.RiemannLinear(4, 16)
        y = linear(x)
        assert y.F.shape[1] == 16

    def test_basic_block(self):
        x = make_sparse_tensor(channels=8)
        block = re_.nn.BasicBlock(8, 8)
        y = block(x)
        assert y.F.shape[1] == 8

    def test_optimizer_integration(self):
        x = make_sparse_tensor(channels=4)
        conv = re_.nn.RiemannConvolution(4, 4, bias=True)
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.1)
        before = conv.kernel.detach().clone()
        loss = conv(x).F.pow(2).sum()
        loss.backward()
        optimizer.step()
        assert not torch.allclose(before, conv.kernel.detach())


class TestAdaptive:
    def make_tensor(self, max_depth=2):
        g = torch.Generator().manual_seed(1)
        points = torch.rand(256, 3, generator=g, dtype=torch.float64) * 4.0
        features = torch.randn(256, 2, generator=g)
        config = re_.adaptive.AdaptiveVoxelConfig(
            base_voxel_size=[1.0, 1.0, 1.0], max_depth=max_depth)
        return re_.adaptive.AdaptiveSparseTensor.from_points(
            points, features, config)

    def test_from_points(self):
        t = self.make_tensor()
        assert len(t) > 0
        assert t.dimension == 3
        assert t.channels == 2
        assert t.coordinates().shape[1] == 3
        assert sum(t.counts_by_level()) == len(t)

    def test_refine_and_pool(self):
        t = self.make_tensor()
        before = len(t)
        region = re_.adaptive.AxisAlignedRegion(
            lo=[0.0, 0.0, 0.0], hi=[2.0, 2.0, 2.0])
        t.refine(region, 1)
        assert len(t) > before
        pooled = re_.adaptive.pool(t, 0)
        assert max(pooled.levels().tolist()) == 0

    def test_refine_by_density(self):
        t = self.make_tensor()
        t.refine_by_density(8)
        assert t.has_source_points()

    def test_convolution(self):
        t = self.make_tensor()
        weight = torch.randn(27, 2, 4, dtype=torch.float64) * 0.1
        config = re_.adaptive.AdaptiveKernelConfig(kernel_size=3)
        out = re_.adaptive.convolution(
            t.with_features(t.features.double()), weight, config)
        assert out.channels == 4

    def test_interpolate(self):
        t = self.make_tensor()
        points = torch.rand(10, 3, dtype=torch.float64) * 4.0
        values = re_.adaptive.interpolate(t, points)
        assert values.shape == (10, 2)

    def test_global_avg_pool(self):
        t = self.make_tensor()
        pooled = re_.adaptive.global_avg_pool(t)
        assert pooled.shape == (2,)


gaussian = pytest.importorskip("riemannengine.gaussian",
                               reason="built without gaussian support")


class TestGaussian:
    def make_gaussians(self, n=32):
        g = torch.Generator().manual_seed(2)
        positions = torch.randn(n, 3, generator=g)
        rotations = torch.zeros(n, 4)
        rotations[:, 0] = 1.0
        scales = torch.rand(n, 3, generator=g) * 0.1 + 0.05
        opacities = torch.rand(n, 1, generator=g) * 0.5 + 0.25
        colors = torch.rand(n, 3, generator=g)
        return gaussian.GaussianTensor(positions, rotations, scales,
                                       opacities, colors)

    def test_tensor_basics(self):
        t = self.make_gaussians()
        assert len(t) == 32
        assert t.covariances().shape == (32, 3, 3)
        kept = t.filter_opacity(0.5)
        assert len(kept) <= len(t)
        moved = t.transform(torch.tensor([1.0, 0.0, 0.0, 0.0]),
                            torch.tensor([1.0, 0.0, 0.0]))
        assert torch.allclose(moved.positions,
                              t.positions + torch.tensor([1.0, 0.0, 0.0]))

    def test_model_from_points(self):
        g = torch.Generator().manual_seed(3)
        pts = torch.randn(64, 3, generator=g)
        colors = torch.rand(64, 3, generator=g)
        model = gaussian.GaussianModel.from_points(pts, colors, sh_degree=1)
        assert len(model) == 64
        assert model.sh_degree == 1
        assert len(model.parameters()) == 6

    def test_render(self):
        t = self.make_gaussians()
        camera = gaussian.Camera.look_at(
            eye=[0.0, 0.0, -5.0], target=[0.0, 0.0, 0.0],
            up=[0.0, -1.0, 0.0], width=32, height=32,
            fov_x_radians=math.radians(60.0))
        out = gaussian.render(t, camera)
        assert out.image.shape == (32, 32, 3)
        assert out.alpha.shape == (32, 32)

    def test_trainer_step(self):
        g = torch.Generator().manual_seed(4)
        pts = torch.randn(32, 3, generator=g) * 0.5
        colors = torch.rand(32, 3, generator=g)
        model = gaussian.GaussianModel.from_points(pts, colors, sh_degree=0)
        config = gaussian.GaussianTrainerConfig()
        config.densify = False
        trainer = gaussian.GaussianTrainer(model, config)
        camera = gaussian.Camera.look_at(
            eye=[0.0, 0.0, -4.0], target=[0.0, 0.0, 0.0],
            up=[0.0, -1.0, 0.0], width=16, height=16,
            fov_x_radians=math.radians(60.0))
        target = torch.rand(16, 16, 3, generator=g)
        result = trainer.step(camera, target)
        assert result.iteration == 1
        assert result.gaussians == 32
        assert math.isfinite(result.loss)

    def test_losses(self):
        g = torch.Generator().manual_seed(5)
        a = torch.rand(16, 16, 3, generator=g)
        b = torch.rand(16, 16, 3, generator=g)
        assert gaussian.l1_loss(a, a).item() == 0.0
        assert gaussian.l1_loss(a, b).item() > 0.0
        assert gaussian.ssim(a, a).item() == pytest.approx(1.0, abs=1e-4)


gaussian_voxel = pytest.importorskip(
    "riemannengine.gaussian_voxel", reason="built without Gaussian voxel support")
occupancy = pytest.importorskip(
    "riemannengine.occupancy", reason="built without occupancy support")


def make_voxel_grid():
    coordinates = torch.tensor(
        [[x, y, z] for x in range(5) for y in range(5) for z in range(5)],
        dtype=torch.int64)
    levels = torch.zeros(len(coordinates), dtype=torch.int32)
    config = re_.adaptive.AdaptiveVoxelConfig(
        [1.0, 1.0, 1.0], origin=[-2.0, -2.0, -2.0])
    return re_.adaptive.AdaptiveSparseTensor.from_voxels(
        coordinates, levels, torch.zeros(len(coordinates), 1), config)


class TestGaussianVoxel:
    def test_rasterize_and_backward(self):
        positions = torch.zeros(1, 3, requires_grad=True)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
        scales = torch.full((1, 3), 0.5, requires_grad=True)
        opacities = torch.full((1, 1), 0.8, requires_grad=True)
        features = torch.tensor([[2.0, 3.0]], requires_grad=True)
        gaussians = gaussian.GaussianTensor(
            positions, rotations, scales, opacities,
            torch.zeros(1, 3), features=features)
        output = gaussian_voxel.rasterize(gaussians, make_voxel_grid())

        assert output.density.shape == (125, 1)
        assert output.occupancy.shape == (125, 1)
        assert output.voxels.features.shape == (125, 2)
        assert output.density.sum().item() > 0.0
        (output.density.sum() + output.voxels.features.sum()).backward()
        assert positions.grad is not None
        assert scales.grad is not None
        assert opacities.grad is not None
        assert features.grad is not None


class TestLidarOccupancy:
    def test_ray_targets_and_masked_loss(self):
        grid = make_voxel_grid()
        points = torch.tensor([[1.2, -1.5, -1.5]])
        origins = torch.tensor([[-1.9, -1.5, -1.5]])
        targets = occupancy.from_lidar(
            points, origins, torch.zeros(1, dtype=torch.int64), 1, grid)

        assert targets.occupied.shape == (1, 125)
        assert targets.observed.sum().item() == 4
        assert targets.labels().sum().item() == 1
        prediction = torch.full((1, 125), 0.4, requires_grad=True)
        loss = occupancy.masked_binary_cross_entropy(prediction, targets)
        loss.backward()
        assert loss.item() > 0.0
        assert prediction.grad is not None
        assert prediction.grad[0, 124].item() == 0.0
