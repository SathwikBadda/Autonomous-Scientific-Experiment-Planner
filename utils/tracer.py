"""
utils/tracer.py - Langfuse integration for agent-level tracing.
Keys loaded directly from os.getenv() to avoid lru_cache stale reads.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_langfuse_client = None


def reset_langfuse():
    """Force re-initialization (useful after env changes)."""
    global _langfuse_client
    _langfuse_client = None


def get_langfuse():
    """
    Lazy-init Langfuse client.
    Reads keys DIRECTLY from os.getenv to bypass any lru_cache config stale reads.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    # Read directly from env — bypasses any cached config
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    enabled_str = os.getenv("LANGFUSE_ENABLED", "false").lower()
    enabled = enabled_str in ("true", "1", "yes")

    logger.info(
        "langfuse_config_check",
        enabled=enabled,
        has_public_key=bool(public_key),
        has_secret_key=bool(secret_key),
        host=host,
    )

    if not enabled:
        logger.info("langfuse_disabled_by_env")
        return None

    if not public_key or not secret_key:
        logger.warning("langfuse_missing_keys", public_key_set=bool(public_key), secret_key_set=bool(secret_key))
        return None

    if "YOUR" in public_key or "YOUR" in secret_key:
        logger.warning("langfuse_placeholder_keys_detected")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            flush_at=15,
            flush_interval=60,
        )
        logger.info(
            "langfuse_initialized_successfully",
            host=host,
            public_key_prefix=public_key[:12] + "...",
        )
        # Verify connectivity
        _langfuse_client.auth_check()
        logger.info("langfuse_auth_check_passed")
    except Exception as e:
        logger.error("langfuse_init_failed", error=str(e))
        _langfuse_client = None

    return _langfuse_client


class NoOpTrace:
    """Fallback when Langfuse is disabled or unavailable."""
    def __init__(self, *args, **kwargs):
        self.id = "no-op"

    def span(self, *args, **kwargs): return NoOpSpan()
    def generation(self, *args, **kwargs): return NoOpSpan()
    def update(self, *args, **kwargs): pass
    def flush(self): pass


class NoOpSpan:
    def __init__(self):
        self.id = "no-op"

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass
    def generation(self, *args, **kwargs): return NoOpSpan()
    def span(self, *args, **kwargs): return NoOpSpan()


def create_trace(name: str, metadata: Optional[dict] = None):
    """Create a top-level Langfuse trace for a pipeline run."""
    lf = get_langfuse()
    if lf is None:
        return NoOpTrace()
    try:
        trace = lf.trace(name=name, metadata=metadata or {})
        logger.info("langfuse_trace_created", name=name, trace_id=getattr(trace, "id", "?"))
        return trace
    except Exception as e:
        logger.warning("trace_creation_failed", error=str(e))
        return NoOpTrace()


@contextmanager
def agent_span(
    trace,
    agent_name: str,
    input_data: Any = None,
    metadata: Optional[dict] = None,
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
    metadata: Optional[dict] = None,
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
            logger.info("langfuse_traces_flushed")
        except Exception as e:
            logger.warning("langfuse_flush_failed", error=str(e))
