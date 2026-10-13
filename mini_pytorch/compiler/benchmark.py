"""Reproducible educational benchmarking primitives for mini-pytorch v0.15."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median
from time import perf_counter_ns
from typing import Any, Callable


def _require_int(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        relation = ">= 0" if minimum == 0 else f">= {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    """Timing policy for a small steady-state benchmark."""

    warmup: int = 3
    repeats: int = 10
    iterations_per_repeat: int = 1

    def __post_init__(self) -> None:
        _require_int("warmup", self.warmup, minimum=0)
        _require_int("repeats", self.repeats, minimum=1)
        _require_int("iterations_per_repeat", self.iterations_per_repeat, minimum=1)


@dataclass(frozen=True)
class BenchmarkReport:
    """Per-iteration latency samples plus simple descriptive statistics."""

    config: BenchmarkConfig
    samples_ns: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.samples_ns) != self.config.repeats:
            raise ValueError(
                "samples_ns length must equal BenchmarkConfig.repeats "
                f"({len(self.samples_ns)} != {self.config.repeats})"
            )
        if any(sample < 0 for sample in self.samples_ns):
            raise ValueError("benchmark samples must be non-negative")

    @property
    def median_ns(self) -> float:
        return float(median(self.samples_ns))

    @property
    def mean_ns(self) -> float:
        return float(fmean(self.samples_ns))

    @property
    def min_ns(self) -> float:
        return float(min(self.samples_ns))

    @property
    def max_ns(self) -> float:
        return float(max(self.samples_ns))

    @property
    def median_ms(self) -> float:
        return self.median_ns / 1_000_000.0

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / 1_000_000.0

    def format(self) -> str:
        return (
            "BenchmarkReport("
            f"warmup={self.config.warmup}, "
            f"repeats={self.config.repeats}, "
            f"iterations_per_repeat={self.config.iterations_per_repeat}, "
            f"median_ms={self.median_ms:.6f}, "
            f"mean_ms={self.mean_ms:.6f}, "
            f"min_ms={self.min_ns / 1_000_000.0:.6f}, "
            f"max_ms={self.max_ns / 1_000_000.0:.6f})"
        )


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    config: BenchmarkConfig | None = None,
    **kwargs: Any,
) -> BenchmarkReport:
    """Benchmark ``fn`` without making any claim about what should be faster.

    Warmup calls are intentionally excluded from the measured samples. Each
    sample is normalized to per-iteration nanoseconds so different
    ``iterations_per_repeat`` settings remain directly comparable.
    """

    if not callable(fn):
        raise TypeError("benchmark expects a callable")
    config = config or BenchmarkConfig()
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")

    for _ in range(config.warmup):
        fn(*args, **kwargs)

    samples: list[float] = []
    for _ in range(config.repeats):
        start = perf_counter_ns()
        for _ in range(config.iterations_per_repeat):
            fn(*args, **kwargs)
        elapsed = perf_counter_ns() - start
        samples.append(elapsed / config.iterations_per_repeat)

    return BenchmarkReport(config=config, samples_ns=tuple(samples))
