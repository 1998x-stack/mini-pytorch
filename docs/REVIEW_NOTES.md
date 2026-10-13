# Review Notes

Second hardening batch:

- reject duplicate Parameters inside one optimizer parameter group;
- reject ragged tuple/list batches in `default_collate`;
- reject heterogeneous Tensor/ndarray/scalar sequence batches explicitly;
- add focused regression coverage for these invariants.

These changes are maintenance-only and do not introduce a new subsystem or version milestone.
