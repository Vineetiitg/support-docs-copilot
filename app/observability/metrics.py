import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("support_docs_copilot.metrics")


@dataclass
class RequestMetrics:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)


@contextmanager
def timed_stage(metrics: RequestMetrics, stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics.stages[stage] = round((time.perf_counter() - started) * 1000, 2)


def log_request_metrics(metrics: RequestMetrics, **extra) -> None:
    logger.info(
        "request_id=%s total_ms=%s stages=%s extra=%s",
        metrics.request_id,
        metrics.total_ms(),
        metrics.stages,
        extra,
    )
