"""
config/settings.py - Centralized configuration loader.
Merges config.yaml with environment variable overrides.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "agent_prompts.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache the YAML config, applying env-var overrides."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Override from environment
    cfg["llm"]["model"] = os.getenv("CLAUDE_MODEL", cfg["llm"]["model"])
    cfg["langfuse"]["public_key"] = os.getenv(
        "LANGFUSE_PUBLIC_KEY", cfg["langfuse"]["public_key"]
    )
    cfg["langfuse"]["secret_key"] = os.getenv(
        "LANGFUSE_SECRET_KEY", cfg["langfuse"]["secret_key"]
    )
    cfg["langfuse"]["host"] = os.getenv("LANGFUSE_HOST", cfg["langfuse"]["host"])
    cfg["rag"]["embedding_model_path"] = os.getenv(
        "EMBEDDING_MODEL_PATH", cfg["rag"]["embedding_model_path"]
    )
    cfg["app"]["log_level"] = os.getenv("LOG_LEVEL", cfg["app"]["log_level"])
    return cfg


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    """Load and cache all agent prompt templates."""
    with open(PROMPTS_PATH, "r") as f:
        return yaml.safe_load(f)


def get_anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return key


# Convenience accessors
config = load_config()
prompts = load_prompts()
