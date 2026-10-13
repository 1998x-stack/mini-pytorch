# Review Scope

The post-v1.0 review evaluates correctness, invariant coverage, historical-branch reproducibility, and maintainability without expanding the framework into a new feature release.

Priority order:

1. silent numerical or state corruption;
2. forward/backward contract mismatch;
3. checkpoint/resume inconsistency;
4. lifetime/resource leaks;
5. silent data loss;
6. branch/documentation drift;
7. scalability limits that should be documented rather than disguised as correctness fixes.
