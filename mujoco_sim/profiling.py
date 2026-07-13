"""Low-overhead hierarchical profiling for planning and execution.

The profiler is deliberately independent of MuJoCo.  Algorithm stages use
``span`` blocks and the resulting report distinguishes inclusive time from
self time.  That distinction matters in this project: a slow direct-search
stage may actually spend nearly all of its time in IK or motion planning.

The implementation uses ``contextvars`` and a small lock, so the same API can
also profile future parallel-arm planners without mixing nested call stacks.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import threading
import time
from typing import Any, Callable, Iterator, Mapping


@dataclass(frozen=True)
class ProfileMetric:
    """Aggregated measurements for one hierarchical span path."""

    path: str
    name: str
    parent: str | None
    calls: int
    failures: int
    wall_total_s: float
    wall_self_s: float
    wall_max_s: float
    cpu_total_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Aggregate:
    name: str
    parent: str | None
    calls: int = 0
    failures: int = 0
    wall_total_ns: int = 0
    wall_self_ns: int = 0
    wall_max_ns: int = 0
    cpu_total_ns: int = 0


@dataclass
class _Frame:
    path: str
    child_wall_ns: int = 0


class HierarchicalProfiler:
    """Aggregate nested wall/CPU timings under stable stage names.

    ``wall_total_s`` is inclusive. ``wall_self_s`` removes time spent in
    directly or indirectly nested spans on the same context.  Sorting by self
    time therefore points to the code that is actually consuming the query.
    """

    def __init__(
        self,
        namespace: str = "",
        *,
        wall_clock_ns: Callable[[], int] = time.perf_counter_ns,
        cpu_clock_ns: Callable[[], int] = time.process_time_ns,
    ):
        self.namespace = self._validate_name(namespace, allow_empty=True)
        if not callable(wall_clock_ns) or not callable(cpu_clock_ns):
            raise TypeError("profile clocks must be callable")
        self._wall_clock_ns = wall_clock_ns
        self._cpu_clock_ns = cpu_clock_ns
        self._stack: ContextVar[tuple[_Frame, ...]] = ContextVar(
            f"profile_stack_{id(self)}", default=())
        self._metrics: dict[str, _Aggregate] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _validate_name(value: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            raise TypeError("profile span names must be strings")
        name = value.strip().strip(".")
        if not name:
            if allow_empty:
                return ""
            raise ValueError("profile span name cannot be empty")
        if any(not part for part in name.split(".")):
            raise ValueError("profile span names cannot contain empty components")
        return name

    def reset(self) -> None:
        """Discard completed metrics.

        Resetting while a span is active is a programming error because its
        eventual exit would otherwise create a partial report.
        """
        if self._stack.get():
            raise RuntimeError("cannot reset profiler while a span is active")
        with self._lock:
            self._metrics.clear()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Measure one named operation, preserving its hierarchical parent."""
        leaf = self._validate_name(name)
        stack = self._stack.get()
        parent = stack[-1].path if stack else None
        prefix = parent or self.namespace
        path = f"{prefix}.{leaf}" if prefix else leaf
        frame = _Frame(path)
        token = self._stack.set(stack + (frame,))
        wall_start = self._wall_clock_ns()
        cpu_start = self._cpu_clock_ns()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            wall_ns = max(0, self._wall_clock_ns() - wall_start)
            cpu_ns = max(0, self._cpu_clock_ns() - cpu_start)
            self_ns = max(0, wall_ns - frame.child_wall_ns)
            self._stack.reset(token)
            if stack:
                stack[-1].child_wall_ns += wall_ns
            with self._lock:
                metric = self._metrics.get(path)
                if metric is None:
                    metric = _Aggregate(name=leaf, parent=parent)
                    self._metrics[path] = metric
                metric.calls += 1
                metric.failures += int(failed)
                metric.wall_total_ns += wall_ns
                metric.wall_self_ns += self_ns
                metric.wall_max_ns = max(metric.wall_max_ns, wall_ns)
                metric.cpu_total_ns += cpu_ns

    def metrics(self) -> tuple[ProfileMetric, ...]:
        """Return an immutable, deterministically ordered snapshot."""
        with self._lock:
            values = tuple((path, aggregate) for path, aggregate
                           in self._metrics.items())
        result = []
        for path, item in values:
            result.append(ProfileMetric(
                path=path,
                name=item.name,
                parent=item.parent,
                calls=item.calls,
                failures=item.failures,
                wall_total_s=item.wall_total_ns / 1e9,
                wall_self_s=item.wall_self_ns / 1e9,
                wall_max_s=item.wall_max_ns / 1e9,
                cpu_total_s=item.cpu_total_ns / 1e9,
            ))
        return tuple(sorted(result, key=lambda value: value.path))

    def report(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.to_dict() for item in self.metrics())

    def bottlenecks(self, limit: int = 8) -> tuple[dict[str, Any], ...]:
        """Return the highest self-time stages for optimization work."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("bottleneck limit must be a non-negative integer")
        ranked = sorted(
            self.metrics(),
            key=lambda item: (-item.wall_self_s, -item.wall_total_s, item.path),
        )
        return tuple(item.to_dict() for item in ranked[:limit])


def profile_lookup(
    report: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index a serialized report by path; useful in tests and CLI tooling."""
    output: dict[str, Mapping[str, Any]] = {}
    for item in report:
        path = str(item["path"])
        if path in output:
            raise ValueError(f"duplicate profile path: {path}")
        output[path] = item
    return output


__all__ = ["HierarchicalProfiler", "ProfileMetric", "profile_lookup"]
