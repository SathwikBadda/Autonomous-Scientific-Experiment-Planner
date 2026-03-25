"""
utils/tracer.py - Langfuse integration for agent-level tracing.
Every agent call is wrapped as a Langfuse span for full observability.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

_langfuse_client = None


def get_langfuse():
    """Lazy-init Langfuse client (only if enabled in config)."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    lf_cfg = config.get("langfuse", {})
    enabled = lf_cfg.get("enabled", False)
    
    if not enabled:
        logger.info("langfuse_disabled_in_config")
        return None

    if not lf_cfg.get("public_key") or "YOUR" in lf_cfg.get("public_key", ""):
        logger.warning("langfuse_missing_public_key")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=lf_cfg["public_key"],
            secret_key=lf_cfg["secret_key"],
            host=lf_cfg["host"],
            flush_at=lf_cfg.get("flush_at", 15),
            flush_interval=lf_cfg.get("flush_interval", 60),
        )
        logger.info("langfuse_initialized", host=lf_cfg["host"], public_key=lf_cfg["public_key"][:10] + "...")
    except Exception as e:
        logger.error("langfuse_init_failed", error=str(e))
        _langfuse_client = None

    return _langfuse_client


class NoOpTrace:
    """Fallback when Langfuse is disabled or unavailable."""

    def __init__(self, *args, **kwargs):
        self.id = "no-op"

    def span(self, *args, **kwargs):
        return NoOpSpan()

    def generation(self, *args, **kwargs):
        return NoOpSpan()

    def update(self, *args, **kwargs):
        pass

    def flush(self):
        pass


class NoOpSpan:
    def __init__(self):
        self.id = "no-op"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def generation(self, *args, **kwargs):
        return NoOpSpan()

    def span(self, *args, **kwargs):
        return NoOpSpan()


def create_trace(name: str, metadata: dict | None = None):
    """Create a top-level Langfuse trace for a pipeline run."""
    lf = get_langfuse()
    if lf is None:
        return NoOpTrace()
    try:
        return lf.trace(name=name, metadata=metadata or {})
    except Exception as e:
        logger.warning("trace_creation_failed", error=str(e))
        return NoOpTrace()


@contextmanager
def agent_span(
    trace,
    agent_name: str,
    input_data: Any = None,
    metadata: dict | None = None,
) -> Generator[Any, None, None]:
    """Context manager that wraps an agent call in a Langfuse span."""
    span = None
    start = time.time()
    try:
        span = trace.span(
            name=agent_name,
            input=str(input_data)[:2000] if input_data else None,
            metadata=metadata or {},
        )
        logger.info("agent_span_start", agent=agent_name)
        yield span
    except Exception as exc:
        if span:
            try:
                span.update(status_message=f"ERROR: {exc}", level="ERROR")
            except Exception:
                pass
        logger.error("agent_span_error", agent=agent_name, error=str(exc))
        raise
    finally:
        elapsed = round(time.time() - start, 3)
        if span:
            try:
                span.end()
            except Exception:
                pass
        logger.info("agent_span_end", agent=agent_name, elapsed_s=elapsed)


def log_llm_call(
    trace_or_span,
    model: str,
    prompt: str,
    response: str,
    agent_name: str,
    metadata: dict | None = None,
):
    """Log an individual LLM call as a Langfuse generation."""
    try:
        trace_or_span.generation(
            name=f"{agent_name}_llm_call",
            model=model,
            prompt=prompt[:4000],
            completion=response[:4000],
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning("langfuse_generation_log_failed", error=str(e))


def flush_traces():
    """Flush all pending Langfuse events."""
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
