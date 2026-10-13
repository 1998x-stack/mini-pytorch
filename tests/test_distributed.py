import unittest

import numpy as np

from mini_pytorch import Tensor, manual_seed, nn, optim
from mini_pytorch.distributed import (
    CollectiveError,
    DistributedDataParallel,
    DistributedError,
    DistributedSampler,
    ReducerError,
    broadcast_module_state,
    create_process_group,
    synchronize_ddp,
)


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class TestProcessGroup(unittest.TestCase):
    def test_rank_world_size_validation(self):
        with self.assertRaises(ValueError):
            create_process_group(0)
        groups = create_process_group(3)
        self.assertEqual([group.rank for group in groups], [0, 1, 2])
        self.assertTrue(all(group.world_size == 3 for group in groups))
        self.assertTrue(groups[0].same_world(groups[2]))

    def test_all_reduce_mean_completes_after_all_ranks_submit(self):
        g0, g1 = create_process_group(2)
        w0 = g0.all_reduce(np.array([1.0, 3.0]), op="mean")
        self.assertFalse(w0.is_completed)
        with self.assertRaises(CollectiveError):
            w0.result()
        w1 = g1.all_reduce(np.array([3.0, 5.0]), op="mean")
        self.assertTrue(w0.is_completed)
        self.assertTrue(w1.is_completed)
        np.testing.assert_allclose(w0.result(), np.array([2.0, 4.0]))
        np.testing.assert_allclose(w1.result(), np.array([2.0, 4.0]))

    def test_all_reduce_sum(self):
        groups = create_process_group(3)
        works = [
            group.all_reduce(np.array([float(group.rank + 1)]), op="sum")
            for group in groups
        ]
        for work in works:
            np.testing.assert_allclose(work.result(), np.array([6.0]))

    def test_collective_operation_mismatch_fails(self):
        g0, g1 = create_process_group(2)
        g0.all_reduce(np.ones(2), op="sum")
        with self.assertRaises(CollectiveError):
            g1.all_reduce(np.ones(2), op="mean")

    def test_collective_shape_mismatch_fails(self):
        g0, g1 = create_process_group(2)
        g0.all_reduce(np.ones(2))
        with self.assertRaises(CollectiveError):
            g1.all_reduce(np.ones(3))

    def test_collective_dtype_mismatch_fails(self):
        g0, g1 = create_process_group(2)
        g0.all_reduce(np.ones(2, dtype=np.float32))
        with self.assertRaises(CollectiveError):
            g1.all_reduce(np.ones(2, dtype=np.float64))


class TestDistributedSampler(unittest.TestCase):
    def test_divisible_partition_is_disjoint_and_complete(self):
        dataset = _Sized(12)
        samplers = [
            DistributedSampler(dataset, num_replicas=3, rank=rank, shuffle=False)
            for rank in range(3)
        ]
        partitions = [list(sampler) for sampler in samplers]
        self.assertEqual(partitions[0], [0, 3, 6, 9])
        self.assertEqual(partitions[1], [1, 4, 7, 10])
        self.assertEqual(partitions[2], [2, 5, 8, 11])
        self.assertEqual(sorted(sum(partitions, [])), list(range(12)))

    def test_padding_and_drop_last(self):
        dataset = _Sized(10)
        padded = [
            DistributedSampler(dataset, num_replicas=3, rank=rank, shuffle=False)
            for rank in range(3)
        ]
        padded_indices = [list(sampler) for sampler in padded]
        self.assertTrue(all(len(indices) == 4 for indices in padded_indices))
        flat = sum(padded_indices, [])
        self.assertEqual(len(flat), 12)
        self.assertTrue(set(range(10)).issubset(set(flat)))

        dropped = [
            DistributedSampler(
                dataset,
                num_replicas=3,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            for rank in range(3)
        ]
        dropped_flat = sum((list(sampler) for sampler in dropped), [])
        self.assertEqual(len(dropped_flat), 9)
        self.assertEqual(len(set(dropped_flat)), 9)

    def test_set_epoch_is_deterministic_and_changes_shuffle(self):
        dataset = _Sized(20)
        a = DistributedSampler(dataset, num_replicas=2, rank=0, seed=11)
        b = DistributedSampler(dataset, num_replicas=2, rank=0, seed=11)
        self.assertEqual(list(a), list(b))
        epoch0 = list(a)
        a.set_epoch(1)
        b.set_epoch(1)
        self.assertEqual(list(a), list(b))
        self.assertNotEqual(epoch0, list(a))

    def test_invalid_rank_fails(self):
        with self.assertRaises(ValueError):
            DistributedSampler(_Sized(4), num_replicas=2, rank=2)


class TestReducerAndDDP(unittest.TestCase):
    @staticmethod
    def _two_linear_replicas(*, bucket_cap_numel=2, device=None):
        manual_seed(123)
        source = nn.Linear(2, 1)
        state = source.state_dict()
        groups = create_process_group(2)
        models = [nn.Linear(2, 1), nn.Linear(2, 1)]
        for model in models:
            model.load_state_dict(state)
            if device is not None:
                model.to(device)
        ddps = [
            DistributedDataParallel(
                model,
                group,
                bucket_cap_numel=bucket_cap_numel,
            )
            for model, group in zip(models, groups)
        ]
        return source, models, ddps

    def test_bucket_layout_is_deterministic_and_reversed(self):
        _, _, ddps = self._two_linear_replicas(bucket_cap_numel=2)
        names0 = [bucket.names for bucket in ddps[0].reducer.buckets]
        names1 = [bucket.names for bucket in ddps[1].reducer.buckets]
        self.assertEqual(names0, names1)
        self.assertEqual(names0, [("bias",), ("weight",)])

    def test_backward_hooks_stage_complete_local_grads(self):
        _, _, ddps = self._two_linear_replicas()
        ddp = ddps[0]
        x = Tensor(np.array([[1.0, 2.0], [2.0, -1.0]]))
        y = Tensor(np.array([[1.0], [0.0]]))
        loss = nn.MSELoss()(ddp(x), y)
        loss.backward()
        self.assertTrue(ddp.reducer.all_ready)
        self.assertEqual(set(ddp.reducer.ready_parameter_names), {"weight", "bias"})
        self.assertIsNotNone(ddp.module.weight.grad)
        self.assertIsNotNone(ddp.module.bias.grad)

    def test_synchronize_averages_local_gradients(self):
        _, models, ddps = self._two_linear_replicas()
        batches = [
            (np.array([[1.0, 2.0]]), np.array([[1.0]])),
            (np.array([[3.0, -1.0]]), np.array([[-2.0]])),
        ]
        local_grads = []
        for ddp, (x, y) in zip(ddps, batches):
            loss = nn.MSELoss()(ddp(Tensor(x)), Tensor(y))
            loss.backward()
            local_grads.append(
                (ddp.module.weight.grad.copy(), ddp.module.bias.grad.copy())
            )
        expected_w = (local_grads[0][0] + local_grads[1][0]) / 2.0
        expected_b = (local_grads[0][1] + local_grads[1][1]) / 2.0
        synchronize_ddp(ddps)
        for model in models:
            np.testing.assert_allclose(model.weight.grad, expected_w)
            np.testing.assert_allclose(model.bias.grad, expected_b)

    def test_missing_rank_fails_at_explicit_barrier(self):
        _, _, ddps = self._two_linear_replicas()
        loss = nn.MSELoss()(ddps[0](Tensor([[1.0, 2.0]])), Tensor([[1.0]]))
        loss.backward()
        with self.assertRaises(DistributedError):
            synchronize_ddp([ddps[0]])

    def test_unused_parameter_fails_clearly(self):
        class HasUnused(nn.Module):
            def __init__(self):
                super().__init__()
                self.used = nn.Linear(2, 1)
                self.unused = nn.Parameter(np.array([1.0]))

            def forward(self, x):
                return self.used(x)

        manual_seed(5)
        state = HasUnused().state_dict()
        groups = create_process_group(2)
        ddps = []
        for group in groups:
            model = HasUnused()
            model.load_state_dict(state)
            ddps.append(DistributedDataParallel(model, group))
        for ddp in ddps:
            nn.MSELoss()(ddp(Tensor([[1.0, 2.0]])), Tensor([[0.0]])).backward()
        self.assertIn("unused", ddps[0].reducer.missing_parameter_names)
        with self.assertRaises(DistributedError):
            synchronize_ddp(ddps)

    def test_stale_grad_requires_zero_grad_before_next_forward(self):
        _, _, ddps = self._two_linear_replicas()
        for ddp in ddps:
            nn.MSELoss()(ddp(Tensor([[1.0, 2.0]])), Tensor([[0.0]])).backward()
        synchronize_ddp(ddps)
        with self.assertRaises(ReducerError):
            ddps[0](Tensor([[1.0, 2.0]]))
        for ddp in ddps:
            ddp.zero_grad()
        ddps[0](Tensor([[1.0, 2.0]]))

    def test_broadcast_module_state_copies_source_values(self):
        groups = create_process_group(2)
        manual_seed(1)
        m0 = nn.Linear(2, 1)
        manual_seed(2)
        m1 = nn.Linear(2, 1)
        self.assertFalse(np.allclose(m0.weight.data, m1.weight.data))
        ddps = [
            DistributedDataParallel(m0, groups[0]),
            DistributedDataParallel(m1, groups[1]),
        ]
        broadcast_module_state(ddps, src=0)
        np.testing.assert_array_equal(m0.weight.data, m1.weight.data)
        np.testing.assert_array_equal(m0.bias.data, m1.bias.data)

    def test_edu_parameter_placement_survives_sync(self):
        _, models, ddps = self._two_linear_replicas(device="edu")
        for ddp in ddps:
            x = Tensor([[1.0, 2.0]], device="edu")
            y = Tensor([[0.0]], device="edu")
            nn.MSELoss()(ddp(x), y).backward()
        synchronize_ddp(ddps)
        for model in models:
            self.assertEqual(str(model.weight.device), "edu")
            self.assertEqual(str(model.bias.device), "edu")
            self.assertIsNotNone(model.weight.grad)

    def test_global_batch_reference_equivalence_after_optimizer_step(self):
        manual_seed(77)
        initial = nn.Linear(2, 1)
        initial_state = initial.state_dict()

        ref = nn.Linear(2, 1)
        ref.load_state_dict(initial_state)
        groups = create_process_group(2)
        replicas = [nn.Linear(2, 1), nn.Linear(2, 1)]
        for replica in replicas:
            replica.load_state_dict(initial_state)
        ddps = [
            DistributedDataParallel(replica, group, bucket_cap_numel=2)
            for replica, group in zip(replicas, groups)
        ]

        x = np.array(
            [[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [0.5, 0.25]],
            dtype=np.float64,
        )
        y = np.array([[1.0], [-2.0], [0.5], [1.5]], dtype=np.float64)
        criterion = nn.MSELoss(reduction="mean")
        local_opts = [optim.SGD(ddp.parameters(), lr=0.05) for ddp in ddps]
        ref_opt = optim.SGD(ref.parameters(), lr=0.05)

        for rank, (ddp, optimizer) in enumerate(zip(ddps, local_opts)):
            optimizer.zero_grad()
            start = rank * 2
            stop = start + 2
            loss = criterion(ddp(Tensor(x[start:stop])), Tensor(y[start:stop]))
            loss.backward()

        ref_opt.zero_grad()
        ref_loss = criterion(ref(Tensor(x)), Tensor(y))
        ref_loss.backward()

        synchronize_ddp(ddps)
        for optimizer in local_opts:
            optimizer.step()
        ref_opt.step()

        for replica in replicas:
            np.testing.assert_allclose(replica.weight.data, ref.weight.data, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(replica.bias.data, ref.bias.data, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(replicas[0].weight.data, replicas[1].weight.data)
        np.testing.assert_array_equal(replicas[0].bias.data, replicas[1].bias.data)


if __name__ == "__main__":
    unittest.main()
