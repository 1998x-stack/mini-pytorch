# Deep Review Findings

This document records the post-v1.0 maintenance review. It distinguishes correctness fixes from intentional educational limits so future work does not confuse the two.

## Correctness issues fixed during review

- Adam / AdamW finite-epsilon bias correction now matches the bias-corrected reference update.
- CrossEntropyLoss uses stable log-sum-exp loss and a mathematically matching backward path; fractional/non-finite class indices are rejected.
- Completed in-process collective slots are reclaimed after all ranks consume the result.
- Step-style LR scheduler state restores configuration as well as epoch/LR state, and rejects type/group mismatches.
- Tensor leaf semantics follow graph provenance rather than `requires_grad` alone.
- `x ** 0` backward avoids `0 * x**-1` NaNs at zero.
- transpose validates dimension bounds instead of wrapping arbitrary positive/negative indices modulo ndim.
- Optimizer rejects a Parameter duplicated inside one parameter group instead of updating it twice and producing ambiguous checkpoint indices.
- default_collate rejects heterogeneous or ragged tuple/list batches instead of silently truncating them through `zip`.
- Zero-size Tensor/view operations preserve explicit Storage identity; empty transpose/slice/view semantics no longer fall through NumPy's `shares_memory=False` behavior for zero-byte arrays.
- Optimizer checkpoint loading is transactional and fail-fast: unknown parameter-state indices and wrong-shaped per-parameter ndarray state are rejected before current optimizer options/state are mutated.

## Review candidates investigated but not treated as bugs

- MatMul backward is intentionally 2-D: the backend kernel rejects non-2-D inputs before Autograd, so forward/backward domains are consistent.
- GraphCache is deliberately a small ordered teaching cache. It has no eviction policy and lookup is linear; this is a documented scalability limit, not a correctness defect for v1.0.
- DataLoader remains deliberately single-process; workers, prefetch and pinned memory are outside the integrated v1.0 contract.
- Tensor/Storage expose low-level NumPy views. Direct user mutation can bypass tracked inplace APIs; this remains an explicit educational escape hatch rather than a production safety guarantee.
- Autograd grad-mode state is process-global rather than thread-local; concurrency parity with PyTorch is outside the v1.0 scope.

## Historical branch policy

Historical version branches remain learning snapshots. CI-only maintenance is permitted so their current HEADs can be revalidated on current GitHub Actions. Correctness fixes are not mechanically backported if they depend on mechanisms introduced by later versions; only fixes inside a branch's own stated contract should be considered for backporting.

## Test policy

For `main`, every hardening change must pass:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

For historical snapshots, the branch-local test suite is the oracle. A green newer branch does not substitute for exact-head validation of an older branch.
