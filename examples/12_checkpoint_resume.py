"""v0.9 walkthrough: save, inspect, reconstruct, restore, and resume exactly."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from loguru import logger
import numpy as np

from mini_pytorch import Tensor, TrainingState, manual_seed, nn, optim, runtime
from mini_pytorch.runtime import load_checkpoint, save_checkpoint


def build_training_objects():
    model = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(6, 1),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.7)
    return model, optimizer, scheduler


def batch():
    x = Tensor(
        np.array(
            [
                [0.2, -1.0, 0.5],
                [1.2, 0.3, -0.7],
                [-0.4, 0.8, 1.1],
                [0.6, -0.2, 0.9],
            ]
        )
    )
    target = Tensor(np.array([[0.4], [-0.8], [1.1], [0.2]]))
    return x, target


def train_step(model, optimizer, scheduler, state):
    x, target = batch()
    optimizer.zero_grad()
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    state.global_step += 1
    state.batch_in_epoch += 1
    return float(loss.item())


def state_arrays(model):
    return {name: value.copy() for name, value in model.state_dict().items()}


def main():
    seed = 20261012
    total_steps = 7
    split_step = 3

    # Reference: uninterrupted execution.
    manual_seed(seed)
    ref_model, ref_optimizer, ref_scheduler = build_training_objects()
    ref_state = TrainingState()
    ref_losses = [
        train_step(ref_model, ref_optimizer, ref_scheduler, ref_state)
        for _ in range(total_steps)
    ]
    ref_params = state_arrays(ref_model)
    ref_future_rng = runtime.random(5)

    # Interrupted execution.
    manual_seed(seed)
    model, optimizer, scheduler = build_training_objects()
    state = TrainingState()
    resumed_losses = [
        train_step(model, optimizer, scheduler, state)
        for _ in range(split_step)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "training.ckpt"
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            training_state=state,
            metadata={"purpose": "v0.9 deterministic-resume demo"},
        )

        metadata = json.loads((checkpoint / "metadata.json").read_text())
        logger.info(
            "checkpoint schema={} framework={} step={}",
            metadata["schema_version"],
            metadata["framework_version"],
            metadata["training_state"]["global_step"],
        )
        logger.info("checkpoint files={}", sorted(p.name for p in checkpoint.iterdir()))

        # These constructors intentionally consume RNG. load_checkpoint restores
        # the saved stream last, so future Dropout masks still match Run A.
        model2, optimizer2, scheduler2 = build_training_objects()
        state2 = load_checkpoint(
            checkpoint,
            model2,
            optimizer2,
            scheduler2,
        )

        for _ in range(split_step, total_steps):
            resumed_losses.append(
                train_step(model2, optimizer2, scheduler2, state2)
            )

        resumed_future_rng = runtime.random(5)

    np.testing.assert_array_equal(ref_losses, resumed_losses)
    for name, expected in ref_params.items():
        np.testing.assert_array_equal(model2.state_dict()[name], expected)
    np.testing.assert_array_equal(resumed_future_rng, ref_future_rng)

    logger.info("reference losses={}", [round(v, 6) for v in ref_losses])
    logger.info("resumed losses={}", [round(v, 6) for v in resumed_losses])
    logger.info("final lr={}", optimizer2.learning_rates)
    logger.info("global_step={}", state2.global_step)
    logger.info("deterministic resume: PASS")


if __name__ == "__main__":
    main()
