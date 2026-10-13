import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mini_pytorch import (
    CheckpointError,
    Tensor,
    TrainingState,
    __version__,
    get_rng_state,
    load_checkpoint,
    load_checkpoint_state,
    manual_seed,
    nn,
    optim,
    runtime,
    save_checkpoint,
    set_rng_state,
)


def assert_nested_equal(testcase, left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        testcase.assertIsInstance(left, np.ndarray)
        testcase.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        testcase.assertIsInstance(left, dict)
        testcase.assertIsInstance(right, dict)
        testcase.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(testcase, left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        testcase.assertEqual(type(left), type(right))
        testcase.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(testcase, a, b)
        return
    testcase.assertEqual(left, right)


def build_model():
    return nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(6, 1),
    )


def build_training_objects():
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.7)
    return model, optimizer, scheduler


def training_batch():
    x = Tensor(
        np.array(
            [
                [0.2, -1.0, 0.5],
                [1.2, 0.3, -0.7],
                [-0.4, 0.8, 1.1],
                [0.6, -0.2, 0.9],
            ],
            dtype=np.float64,
        )
    )
    target = Tensor(np.array([[0.4], [-0.8], [1.1], [0.2]], dtype=np.float64))
    return x, target


def train_one_step(model, optimizer, scheduler, state):
    x, target = training_batch()
    optimizer.zero_grad()
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    state.global_step += 1
    state.batch_in_epoch += 1
    return float(loss.item())


class TestFrameworkRNG(unittest.TestCase):
    def test_manual_seed_restarts_same_stream(self):
        manual_seed(1234)
        a = runtime.random(8)
        manual_seed(1234)
        b = runtime.random(8)
        np.testing.assert_array_equal(a, b)

    def test_rng_state_restores_continuation(self):
        manual_seed(99)
        runtime.random(3)
        state = get_rng_state()
        expected = runtime.random(7)
        runtime.random(5)
        set_rng_state(state)
        actual = runtime.random(7)
        np.testing.assert_array_equal(actual, expected)

    def test_parameter_initialization_uses_framework_rng(self):
        manual_seed(11)
        a = nn.Linear(4, 3)
        manual_seed(11)
        b = nn.Linear(4, 3)
        np.testing.assert_array_equal(a.weight.data, b.weight.data)
        np.testing.assert_array_equal(a.bias.data, b.bias.data)

    def test_default_dropout_uses_framework_rng(self):
        x = Tensor(np.ones((12, 12)))
        manual_seed(17)
        dropout = nn.Dropout(0.5)
        dropout(x)
        state = get_rng_state()
        expected = dropout(x).data.copy()
        set_rng_state(state)
        actual = dropout(x).data.copy()
        np.testing.assert_array_equal(actual, expected)

    def test_default_random_sampler_uses_framework_rng(self):
        data = list(range(12))
        sampler = runtime_sampler = None
        from mini_pytorch.data import RandomSampler

        manual_seed(31)
        sampler = RandomSampler(data)
        list(sampler)
        state = get_rng_state()
        expected = list(sampler)
        set_rng_state(state)
        actual = list(sampler)
        self.assertEqual(actual, expected)
        self.assertIsNone(runtime_sampler)


class TestTrainingState(unittest.TestCase):
    def test_round_trip(self):
        state = TrainingState(
            epoch=3,
            global_step=17,
            batch_in_epoch=4,
            best_metric=0.25,
            run_id="demo",
            notes="checkpoint",
        )
        self.assertEqual(TrainingState.from_dict(state.to_dict()), state)

    def test_invalid_counters_fail(self):
        with self.assertRaises(ValueError):
            TrainingState(epoch=-1)
        with self.assertRaises(TypeError):
            TrainingState(global_step=1.5)


class TestCheckpointSerialization(unittest.TestCase):
    def _populated_objects(self):
        manual_seed(101)
        model, optimizer, scheduler = build_training_objects()
        state = TrainingState(epoch=2, global_step=0, batch_in_epoch=0)
        train_one_step(model, optimizer, scheduler, state)
        return model, optimizer, scheduler, state

    def test_json_npz_round_trip_is_inspectable(self):
        model, optimizer, scheduler, training_state = self._populated_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(
                root,
                model,
                optimizer,
                scheduler,
                training_state=training_state,
                metadata={"experiment": "unit-test", "tags": ["v0.9", "resume"]},
            )

            self.assertTrue((root / "metadata.json").is_file())
            self.assertTrue((root / "tensors.npz").is_file())
            document = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(document["format"], "mini-pytorch-checkpoint")
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["framework_version"], __version__)
            self.assertEqual(document["training_state"]["global_step"], 1)

            loaded = load_checkpoint_state(root)
            assert_nested_equal(self, loaded.model_state, model.state_dict())
            assert_nested_equal(self, loaded.optimizer_state, optimizer.state_dict())
            assert_nested_equal(self, loaded.scheduler_state, scheduler.state_dict())
            self.assertEqual(loaded.training_state, training_state)
            self.assertEqual(loaded.metadata["experiment"], "unit-test")

    def test_unknown_schema_is_rejected(self):
        model, optimizer, scheduler, training_state = self._populated_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer, scheduler, training_state=training_state)
            path = root / "metadata.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["schema_version"] = 999
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(CheckpointError):
                load_checkpoint_state(root)

    def test_missing_and_corrupt_files_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            root.mkdir()
            with self.assertRaises(CheckpointError):
                load_checkpoint_state(root)

            (root / "metadata.json").write_text("{}", encoding="utf-8")
            (root / "tensors.npz").write_bytes(b"not-an-npz")
            with self.assertRaises(CheckpointError):
                load_checkpoint_state(root)

    def test_overwrite_false_rejects_existing_checkpoint(self):
        model, optimizer, scheduler, training_state = self._populated_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer, scheduler, training_state=training_state)
            with self.assertRaises(FileExistsError):
                save_checkpoint(
                    root,
                    model,
                    optimizer,
                    scheduler,
                    training_state=training_state,
                    overwrite=False,
                )


class TestCheckpointRestore(unittest.TestCase):
    def test_model_optimizer_scheduler_and_progress_restore(self):
        manual_seed(202)
        model, optimizer, scheduler = build_training_objects()
        state = TrainingState(epoch=1)
        for _ in range(3):
            train_one_step(model, optimizer, scheduler, state)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer, scheduler, training_state=state)

            fresh_model, fresh_optimizer, fresh_scheduler = build_training_objects()
            restored = load_checkpoint(
                root,
                fresh_model,
                fresh_optimizer,
                fresh_scheduler,
            )

            assert_nested_equal(self, fresh_model.state_dict(), model.state_dict())
            assert_nested_equal(self, fresh_optimizer.state_dict(), optimizer.state_dict())
            assert_nested_equal(self, fresh_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(restored, state)

    def test_restore_preserves_target_device(self):
        manual_seed(303)
        source = nn.Linear(3, 2)
        source_optimizer = optim.SGD(source.parameters(), lr=0.1)
        state = TrainingState()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, source, source_optimizer, training_state=state)

            target = nn.Linear(3, 2).to("edu")
            target_optimizer = optim.SGD(target.parameters(), lr=0.1)
            load_checkpoint(root, target, target_optimizer)

            for parameter in target.parameters():
                self.assertEqual(str(parameter.device), "edu")
            assert_nested_equal(self, target.state_dict(), source.state_dict())

    def test_wrong_model_shape_fails(self):
        manual_seed(404)
        model = nn.Linear(3, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer)
            target = nn.Linear(4, 2)
            target_optimizer = optim.SGD(target.parameters(), lr=0.1)
            with self.assertRaises(ValueError):
                load_checkpoint(root, target, target_optimizer)

    def test_optimizer_group_mismatch_fails(self):
        manual_seed(505)
        model = nn.Linear(3, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer)

            target = nn.Linear(3, 2)
            params = list(target.parameters())
            target_optimizer = optim.SGD(
                [
                    {"params": [params[0]], "lr": 0.1},
                    {"params": [params[1]], "lr": 0.1},
                ],
                lr=0.1,
            )
            with self.assertRaises(ValueError):
                load_checkpoint(root, target, target_optimizer)

    def test_scheduler_presence_is_strict(self):
        manual_seed(606)
        model, optimizer, scheduler = build_training_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(root, model, optimizer, scheduler)
            target_model = build_model()
            target_optimizer = optim.Adam(target_model.parameters(), lr=0.02)
            with self.assertRaises(CheckpointError):
                load_checkpoint(root, target_model, target_optimizer)


class TestDeterministicResume(unittest.TestCase):
    def test_uninterrupted_equals_save_reload_continue(self):
        total_steps = 7
        split_step = 3
        seed = 20261012

        manual_seed(seed)
        model_a, optimizer_a, scheduler_a = build_training_objects()
        state_a = TrainingState(epoch=0)
        losses_a = [
            train_one_step(model_a, optimizer_a, scheduler_a, state_a)
            for _ in range(total_steps)
        ]
        model_state_a = model_a.state_dict()
        optimizer_state_a = optimizer_a.state_dict()
        scheduler_state_a = scheduler_a.state_dict()
        future_rng_a = runtime.random(10)

        manual_seed(seed)
        model_b, optimizer_b, scheduler_b = build_training_objects()
        state_b = TrainingState(epoch=0)
        losses_b = [
            train_one_step(model_b, optimizer_b, scheduler_b, state_b)
            for _ in range(split_step)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            save_checkpoint(
                root,
                model_b,
                optimizer_b,
                scheduler_b,
                training_state=state_b,
                metadata={"resume_boundary": "step"},
            )

            # Fresh construction intentionally consumes framework RNG. The load
            # path must restore the saved stream *after* object construction.
            resumed_model, resumed_optimizer, resumed_scheduler = build_training_objects()
            resumed_state = load_checkpoint(
                root,
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
            )

            for _ in range(split_step, total_steps):
                losses_b.append(
                    train_one_step(
                        resumed_model,
                        resumed_optimizer,
                        resumed_scheduler,
                        resumed_state,
                    )
                )

            future_rng_b = runtime.random(10)

        np.testing.assert_array_equal(np.asarray(losses_b), np.asarray(losses_a))
        assert_nested_equal(self, resumed_model.state_dict(), model_state_a)
        assert_nested_equal(self, resumed_optimizer.state_dict(), optimizer_state_a)
        assert_nested_equal(self, resumed_scheduler.state_dict(), scheduler_state_a)
        self.assertEqual(resumed_state, state_a)
        np.testing.assert_array_equal(future_rng_b, future_rng_a)


if __name__ == "__main__":
    unittest.main()
