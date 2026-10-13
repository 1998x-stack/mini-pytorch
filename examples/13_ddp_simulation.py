"""v0.10: two-rank DDP simulation and global-batch reference equivalence."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Tensor, manual_seed, nn, optim
from mini_pytorch.distributed import (
    DistributedDataParallel,
    DistributedSampler,
    broadcast_module_state,
    create_process_group,
    synchronize_ddp,
)


def main() -> None:
    manual_seed(2026)
    groups = create_process_group(2)

    replicas = [nn.Linear(2, 1), nn.Linear(2, 1)]
    ddps = [
        DistributedDataParallel(model, group, bucket_cap_numel=2)
        for model, group in zip(replicas, groups)
    ]
    broadcast_module_state(ddps, src=0)

    reference = nn.Linear(2, 1)
    reference.load_state_dict(replicas[0].state_dict())

    x = np.array(
        [[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [0.5, 0.25]],
        dtype=np.float64,
    )
    y = np.array([[1.0], [-2.0], [0.5], [1.5]], dtype=np.float64)

    criterion = nn.MSELoss(reduction="mean")
    optimizers = [optim.SGD(ddp.parameters(), lr=0.05) for ddp in ddps]
    reference_optimizer = optim.SGD(reference.parameters(), lr=0.05)

    print("bucket layout:")
    for bucket in ddps[0].reducer.buckets:
        print(f"  bucket {bucket.index}: names={bucket.names}, numel={bucket.total_numel}")

    local_before_sync = []
    for rank, (ddp, optimizer) in enumerate(zip(ddps, optimizers)):
        optimizer.zero_grad()
        start, stop = rank * 2, rank * 2 + 2
        loss = criterion(ddp(Tensor(x[start:stop])), Tensor(y[start:stop]))
        loss.backward()
        local_before_sync.append(ddp.module.weight.grad.copy())
        print(f"rank {rank}: local loss={loss.item():.8f}")
        print(f"rank {rank}: local weight grad={ddp.module.weight.grad}")

    print("\nlocal gradients differ:", not np.allclose(*local_before_sync))

    synchronize_ddp(ddps)
    print("\nafter explicit synchronization barrier:")
    for rank, ddp in enumerate(ddps):
        print(f"rank {rank}: synchronized weight grad={ddp.module.weight.grad}")

    reference_optimizer.zero_grad()
    reference_loss = criterion(reference(Tensor(x)), Tensor(y))
    reference_loss.backward()

    np.testing.assert_allclose(
        replicas[0].weight.grad,
        reference.weight.grad,
        atol=1e-12,
        rtol=1e-12,
    )

    for optimizer in optimizers:
        optimizer.step()
    reference_optimizer.step()

    for replica in replicas:
        np.testing.assert_allclose(replica.weight.data, reference.weight.data, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(replica.bias.data, reference.bias.data, atol=1e-12, rtol=1e-12)

    print("\nDDP update == global-batch reference update: OK")

    print("\nDistributedSampler example:")
    dataset = list(range(12))
    for rank in range(2):
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=False,
        )
        print(f"rank {rank}: {list(sampler)}")


if __name__ == "__main__":
    main()
