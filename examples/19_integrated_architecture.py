"""v1.0: one coherent walk through eager training, runtime state and compiler/runtime layers.

This example intentionally separates phases. It does not pretend checkpointing,
DDP and compilation all belong inside one production execution loop.
"""
from __future__ import annotations

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


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build_training_stack():
    model = TinyMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = optim.StepLR(optimizer, step_size=1, gamma=0.8)
    return model, optimizer, scheduler


def training_batch():
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


def train_step(model, optimizer, scheduler) -> float:
    x, target = training_batch()
    optimizer.zero_grad()
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def compiler_subgraph(x, weight_t, bias):
    """Supported Tensor-only subgraph extracted from the trained first layer.

    `Linear.forward` performs a transpose view before matmul. v1.0 capture is
    intentionally Dispatcher-op based and does not claim Python/module/layout
    capture parity, so the transposed weight is made an explicit graph input.
    """

    hidden = ((x @ weight_t) + bias).relu()
    return (hidden * hidden).sum()


def main():
    print(f"mini-pytorch {mp.__version__}")

    # ------------------------------------------------------------------
    # Phase 1: ordinary eager training.
    # ------------------------------------------------------------------
    print("\n=== phase 1: eager training ===")
    manual_seed(20261013)
    model, optimizer, scheduler = build_training_stack()
    training_state = TrainingState(run_id="v1-integrated-example")

    losses = []
    for _ in range(2):
        losses.append(train_step(model, optimizer, scheduler))
        training_state.global_step += 1
        training_state.batch_in_epoch += 1

    print("losses:", [round(value, 6) for value in losses])
    print("learning rates:", optimizer.learning_rates)

    # ------------------------------------------------------------------
    # Phase 2: logical runtime/checkpoint state.
    # ------------------------------------------------------------------
    print("\n=== phase 2: runtime state round-trip ===")
    checkpoint = collect_checkpoint_state(
        model,
        optimizer,
        scheduler,
        training_state=training_state,
        metadata={"example": "v1.0 integrated architecture"},
    )
    print("checkpoint summary:", checkpoint.summary())

    # New constructors may consume RNG and start with unrelated parameters.
    # restore_checkpoint_state rehydrates model/optimizer/scheduler state and
    # restores the framework RNG last when requested.
    manual_seed(7)
    restored_model, restored_optimizer, restored_scheduler = build_training_stack()
    restored_state = restore_checkpoint_state(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restore_rng=True,
    )

    for name, expected in checkpoint.model_state.items():
        np.testing.assert_allclose(restored_model.state_dict()[name], expected)
    assert restored_state.to_dict() == training_state.to_dict()
    print("restored global_step:", restored_state.global_step)

    # ------------------------------------------------------------------
    # Phase 3: guarded graph specialization/cache.
    # ------------------------------------------------------------------
    print("\n=== phase 3: graph capture + guards/cache ===")
    x_data = training_batch()[0].data.copy()
    weight_t_data = restored_model.fc1.weight.data.T.copy()
    bias_data = restored_model.fc1.bias.data.copy()

    compiled = compile_graph(compiler_subgraph)
    compiled(
        Tensor(x_data, requires_grad=True),
        Tensor(weight_t_data, requires_grad=True),
        Tensor(bias_data, requires_grad=True),
    )
    compiled(
        Tensor(x_data * 0.5, requires_grad=True),
        Tensor(weight_t_data, requires_grad=True),
        Tensor(bias_data, requires_grad=True),
    )
    print("cache stats:", compiled.stats())

    # ------------------------------------------------------------------
    # Phase 4: eager gradient reference.
    # ------------------------------------------------------------------
    print("\n=== phase 4: eager gradient reference ===")
    eager_x = Tensor(x_data, requires_grad=True)
    eager_w = Tensor(weight_t_data, requires_grad=True)
    eager_b = Tensor(bias_data, requires_grad=True)
    eager_out = compiler_subgraph(eager_x, eager_w, eager_b)
    eager_out.backward()
    eager_grads = (eager_x.grad.copy(), eager_w.grad.copy(), eager_b.grad.copy())
    print("eager scalar output:", float(eager_out.item()))

    # ------------------------------------------------------------------
    # Phase 5: AOT forward/backward graphs → lowering → ExecutionPlan.
    # ------------------------------------------------------------------
    print("\n=== phase 5: AOT + lowering + planned runtime ===")
    aot = aot_compile(
        compiler_subgraph,
        Tensor(x_data, requires_grad=True),
        Tensor(weight_t_data, requires_grad=True),
        Tensor(bias_data, requires_grad=True),
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
        Tensor(weight_t_data, requires_grad=True),
        Tensor(bias_data, requires_grad=True),
        value_ids=retained_ids,
    )

    backward_lowered = lower_graph(aot.backward_graph)
    backward_plan = build_execution_plan(backward_lowered.program)
    grad_seed = Tensor(np.ones_like(planned_out.data), device=planned_out.device)
    planned_grads = PlannedExecutor(backward_plan)(*saved, grad_seed)

    np.testing.assert_allclose(planned_out.data, eager_out.data)
    for got, expected in zip(planned_grads, eager_grads):
        np.testing.assert_allclose(got.data, expected)

    print("forward lowering stats:", forward_lowered.stats())
    print("forward memory plan:")
    print(forward_plan.memory_plan.format())
    print("planned output matches eager: PASS")
    print("planned gradients match eager: PASS")

    # DDP is intentionally not forced into this compiler subgraph. It is a
    # sibling training-runtime concern: local backward → Reducer/ProcessGroup →
    # synchronized gradients → ordinary Optimizer.step. See example 13.
    print("\n=== architecture boundary note ===")
    print(
        "DDP remains a training synchronization layer, while graph/AOT/lowering "
        "operate on a supported Tensor function. v1.0 integrates the concepts "
        "by making their ownership boundaries explicit rather than mixing them."
    )


if __name__ == "__main__":
    main()
