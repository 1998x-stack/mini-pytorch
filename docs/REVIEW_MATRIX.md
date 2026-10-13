# Branch Validation Matrix

This matrix is maintained during the post-v1.0 deep review. The exact branch-local GitHub Actions run remains the source of truth for each snapshot.

| Branch | Role | Validation expectation |
| --- | --- | --- |
| `v0.1` | Tensor / Autograd / Module / basic Optimizer | branch-local unit suite |
| `v0.2` | Optimizer family + scheduler | branch-local unit suite |
| `v0.3` | Data pipeline + classification | branch-local unit suite |
| `v0.4` | Storage / stride / views | branch-local unit suite |
| `v0.5` | Dispatcher | branch-local unit suite |
| `v0.6` | CNN stack | branch-local unit suite |
| `v0.7` | Device/backend | branch-local unit suite |
| `v0.8` | Advanced Autograd | branch-local unit suite |
| `v0.9` | Checkpoint/RNG/runtime | branch-local unit suite |
| `v0.10` | Distributed/DDP concepts | branch-local unit suite |
| `v0.11` | Graph IR/capture | branch-local unit suite |
| `v0.12` | Guards/cache | branch-local unit suite |
| `v0.13` | AOT Autograd | branch-local unit suite |
| `v0.14` | Lowering/fusion | branch-local unit suite |
| `v0.15` | Runtime/performance planning | branch-local unit suite |
| `v1.0` | Integrated architecture frozen snapshot | unit suite + integrated example |
| `main` | newest validated integrated state | unit suite + integrated example |

Historical branches may receive CI-only maintenance so their current HEAD can be revalidated on current Actions runtimes. Their framework scope must remain frozen.
