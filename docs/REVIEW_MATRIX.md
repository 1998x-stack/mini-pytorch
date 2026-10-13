# Branch Validation Matrix

This matrix records the completed post-v1.0 repository review. Every historical version branch was validated on its **current exact HEAD** with its own branch-local GitHub Actions workflow; a newer branch passing was never used as a substitute for an older snapshot.

For `main`, the table pins the exact **reviewed implementation HEAD** where the post-v1.0 correctness hardening reached 292 tests. Later documentation-only commits are separately required to pass the same exact-main CI gate, but they do not alter that reviewed implementation tree.

CI maintenance performed during the review:

- `v0.1`–`v0.3` received branch-local test workflows because their current snapshot HEADs previously had no runnable Actions gate;
- existing workflows on later branches were refreshed to current Node-based GitHub Actions (`actions/checkout@v7`, `actions/setup-python@v7`);
- `v0.3` governance was corrected from `active/finalizing` to a frozen historical learning snapshot;
- framework scope on frozen branches was not expanded.

## Exact-head validation results

| Branch | Validated snapshot / implementation HEAD | Milestone | Result |
| --- | --- | --- | --- |
| `v0.1` | `8d40cc8a` | Tensor / Autograd / Module / basic Optimizer | ✅ exact-head CI success |
| `v0.2` | `5c0819c2` | Optimizer family + scheduler | ✅ exact-head CI success |
| `v0.3` | `ab14d9f6` | Data pipeline + classification | ✅ exact-head CI success |
| `v0.4` | `d5571a25` | Storage / stride / views | ✅ exact-head CI success |
| `v0.5` | `4a6a545b` | Dispatcher | ✅ exact-head CI success |
| `v0.6` | `498b15c2` | CNN stack | ✅ exact-head CI success |
| `v0.7` | `aaa51f1a` | Device / backend | ✅ exact-head CI success |
| `v0.8` | `f07652a6` | Advanced Autograd | ✅ exact-head CI success |
| `v0.9` | `cb2fe548` | Checkpoint / RNG / runtime | ✅ exact-head CI success |
| `v0.10` | `ba5a846c` | Distributed / DDP concepts | ✅ exact-head CI success |
| `v0.11` | `79686191` | Graph IR / capture | ✅ exact-head CI success |
| `v0.12` | `4fb602ef` | Guards / graph cache | ✅ exact-head CI success |
| `v0.13` | `2d4fab6c` | AOT Autograd | ✅ exact-head CI success |
| `v0.14` | `93dbc6d6` | Analysis / lowering / fusion | ✅ exact-head CI success |
| `v0.15` | `d0994168` | Runtime / performance planning | ✅ exact-head CI success |
| `v1.0` | `d28ad884` | Frozen integrated architecture | ✅ exact-head CI success |
| `main` | `28a9071d` | Post-v1.0 reviewed implementation | ✅ **292/292 tests + integrated example PASS** |

The full main validation commands are:

```bash
python -m unittest discover -s tests -v
python examples/19_integrated_architecture.py
```

The first documentation-only closeout HEAD after the reviewed implementation, `61a55457`, was also run on exact `main` and passed the same **292/292 + integrated-example** gate. Future documentation changes must continue to satisfy the exact-head CI rule rather than relying on this recorded result.

## Interpretation

A green historical row means the snapshot represented by that branch remains internally coherent under its own historical contract. It does **not** mean older branches contain later fixes or later subsystems. Correctness hardening discovered after v1.0 is integrated into `main`; frozen branches receive only scope-preserving maintenance when justified.

For `main`, `28a9071d` is intentionally the reviewed **framework implementation** boundary. Documentation-only commits may advance the branch HEAD while leaving that implementation tree unchanged; those commits still require their own CI run.

The detailed correctness findings and intentionally retained limits are documented in [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md).
