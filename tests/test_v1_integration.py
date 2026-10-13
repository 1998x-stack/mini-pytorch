import unittest

import numpy as np

import mini_pytorch as mp
from mini_pytorch import Tensor, TrainingState, manual_seed, nn, optim
from mini_pytorch.compiler import (
    PlannedExecutor,
    aot_compile,
    build_execution_plan,
    compile_graph,
    lower_graph,
)
from mini_pytorch.runtime import collect_checkpoint_state, restore_checkpoint_state


class TestV1VersionAndRuntimeState(unittest.TestCase):
    def test_package_and_checkpoint_framework_versions_agree(self):
        manual_seed(100)
        model = nn.Linear(2, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        checkpoint = collect_checkpoint_state(model, optimizer)

        self.assertEqual(mp.__version__, "1.0.0")
        self.assertEqual(checkpoint.framework_version, mp.__version__)

    def test_training_objects_restore_across_module_optimizer_runtime_boundary(self):
        manual_seed(1234)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = optim.StepLR(optimizer, step_size=1, gamma=0.5)

        x = Tensor(
            np.array(
                [
                    [0.2, -1.0, 0.5],
                    [1.2, 0.3, -0.7],
                    [-0.4, 0.8, 1.1],
                ]
            )
        )
        target = Tensor(np.array([[0.4], [-0.8], [1.1]]))

        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        training_state = TrainingState(
            epoch=1,
            global_step=1,
            batch_in_epoch=0,
            notes="v1 integration boundary",
        )
        checkpoint = collect_checkpoint_state(
            model,
            optimizer,
            scheduler,
            training_state=training_state,
        )

        # Constructors intentionally produce different parameter values; restore
        # must replace them and preserve the target objects' identities.
        manual_seed(9999)
        restored_model = nn.Sequential(
            nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)
        )
        restored_optimizer = optim.SGD(
            restored_model.parameters(), lr=0.01, momentum=0.9
        )
        restored_scheduler = optim.StepLR(
            restored_optimizer, step_size=1, gamma=0.5
        )

        restored_state = restore_checkpoint_state(
            checkpoint,
            restored_model,
            restored_optimizer,
            restored_scheduler,
            restore_rng=False,
        )

        for name, expected in checkpoint.model_state.items():
            np.testing.assert_allclose(restored_model.state_dict()[name], expected)
        self.assertEqual(restored_state.to_dict(), training_state.to_dict())
        self.assertEqual(restored_optimizer.learning_rates, optimizer.learning_rates)
        self.assertEqual(
            restored_scheduler.state_dict(),
            checkpoint.scheduler_state,
        )


class TestV1CoreComposition(unittest.TestCase):
    def test_tensor_dispatch_backend_autograd_composition_matches_cpu_and_edu(self):
        outputs = {}
        grads = {}
        for device in ("cpu", "edu"):
            x = Tensor(
                np.array([[-1.0, 2.0], [3.0, -4.0]]),
                requires_grad=True,
                device=device,
            )
            w = Tensor(
                np.array([[2.0, -1.0], [0.5, 3.0]]),
                requires_grad=True,
                device=device,
            )
            out = ((x * w) + x).relu().sum()
            out.backward()

            outputs[device] = np.array(out.data, copy=True)
            grads[device] = (x.grad.copy(), w.grad.copy())
            self.assertEqual(str(out.device), device)

        np.testing.assert_allclose(outputs["edu"], outputs["cpu"])
        np.testing.assert_allclose(grads["edu"][0], grads["cpu"][0])
        np.testing.assert_allclose(grads["edu"][1], grads["cpu"][1])

    def test_capture_cache_aot_lowering_and_planned_backward_compose(self):
        def fn(x, w, b):
            hidden = ((x @ w) + b).relu()
            return (hidden * hidden).sum()

        x_data = np.array(
            [
                [1.0, -2.0, 0.5],
                [0.25, 1.5, -1.0],
            ]
        )
        w_data = np.array(
            [
                [0.5, -1.0],
                [1.5, 0.25],
                [-0.75, 2.0],
            ]
        )
        b_data = np.array([0.1, -0.2])

        compiled = compile_graph(fn)
        first = compiled(
            Tensor(x_data, requires_grad=True),
            Tensor(w_data, requires_grad=True),
            Tensor(b_data, requires_grad=True),
        )
        second = compiled(
            Tensor(x_data * 0.5, requires_grad=True),
            Tensor(w_data, requires_grad=True),
            Tensor(b_data, requires_grad=True),
        )
        self.assertEqual(compiled.stats().compile_count, 1)
        self.assertEqual(compiled.stats().hit_count, 1)
        self.assertEqual(first.shape, second.shape)

        eager_x = Tensor(x_data, requires_grad=True)
        eager_w = Tensor(w_data, requires_grad=True)
        eager_b = Tensor(b_data, requires_grad=True)
        eager_out = fn(eager_x, eager_w, eager_b)
        eager_out.backward()
        eager_grads = (eager_x.grad.copy(), eager_w.grad.copy(), eager_b.grad.copy())

        aot = aot_compile(
            fn,
            Tensor(x_data, requires_grad=True),
            Tensor(w_data, requires_grad=True),
            Tensor(b_data, requires_grad=True),
        )
        forward_lowered = lower_graph(aot.forward_graph)
        retained_ids = [
            forward_lowered.mapped_value_id(source_id)
            for source_id in aot.spec.saved_value_ids
        ]
        forward_plan = build_execution_plan(
            forward_lowered.program,
            retained_value_ids=retained_ids,
        )
        planned_out, saved = PlannedExecutor(forward_plan).run_with_values(
            Tensor(x_data, requires_grad=True),
            Tensor(w_data, requires_grad=True),
            Tensor(b_data, requires_grad=True),
            value_ids=retained_ids,
        )

        backward_lowered = lower_graph(aot.backward_graph)
        backward_plan = build_execution_plan(backward_lowered.program)
        seed = Tensor(np.ones_like(planned_out.data), device=planned_out.device)
        planned_grads = PlannedExecutor(backward_plan)(*saved, seed)

        np.testing.assert_allclose(planned_out.data, eager_out.data)
        self.assertIsInstance(planned_grads, tuple)
        self.assertEqual(len(planned_grads), 3)
        for got, expected in zip(planned_grads, eager_grads):
            np.testing.assert_allclose(got.data, expected)


if __name__ == "__main__":
    unittest.main()
