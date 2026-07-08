import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager


LOGGER_ROOT = "reflect_real_world"
DEFAULT_LOG_LEVEL = os.environ.get("REFLECT_LOG_LEVEL", "INFO").upper()


def configure_logging(level=None):
    level_name = (level or os.environ.get("REFLECT_LOG_LEVEL", DEFAULT_LOG_LEVEL)).upper()
    os.environ["REFLECT_LOG_LEVEL"] = level_name

    logger = logging.getLogger(LOGGER_ROOT)
    if not getattr(configure_logging, "_configured", False):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        configure_logging._configured = True

    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


def get_logger(name):
    root_logger = logging.getLogger(LOGGER_ROOT)
    if not root_logger.handlers:
        configure_logging()
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")


class TaskProfiler:
    def __init__(self, enabled=False, output_path=None, task_name=None):
        self.enabled = enabled
        self.output_path = output_path
        self.task_name = task_name
        self.events = []
        self.metrics = {}

        if self.enabled and self.output_path and os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                existing = json.load(f)
            self.events = existing.get("events", [])
            self.metrics = existing.get("metrics", {})
            if not self.task_name:
                self.task_name = existing.get("task_name")

    @contextmanager
    def stage(self, stage_name, **metadata):
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record(stage_name, duration, **metadata)

    def record(self, stage_name, duration_sec, **metadata):
        if not self.enabled:
            return
        event = {
            "stage": stage_name,
            "duration_sec": round(float(duration_sec), 6),
        }
        if metadata:
            event["metadata"] = metadata
        self.events.append(event)

    def set_metric(self, key, value):
        if not self.enabled:
            return
        self.metrics[key] = value

    def increment_metric(self, key, value=1):
        if not self.enabled:
            return
        self.metrics[key] = self.metrics.get(key, 0) + value

    def _stage_totals(self):
        totals = defaultdict(float)
        counts = defaultdict(int)
        for event in self.events:
            stage_name = event["stage"]
            totals[stage_name] += float(event["duration_sec"])
            counts[stage_name] += 1
        stage_totals = {key: round(value, 6) for key, value in totals.items()}
        stage_counts = dict(counts)
        return stage_totals, stage_counts

    def to_dict(self):
        stage_totals, stage_counts = self._stage_totals()
        total_wall_time_sec = round(sum(stage_totals.values()), 6)
        return {
            "task_name": self.task_name,
            "total_wall_time_sec": total_wall_time_sec,
            "stage_totals_sec": stage_totals,
            "stage_counts": stage_counts,
            "metrics": self.metrics,
            "events": self.events,
        }

    def save(self):
        if not self.enabled or not self.output_path:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
