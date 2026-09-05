from __future__ import annotations

import os
from pathlib import Path


def load_project_env() -> Path | None:
    """Load .env from the project root when python-dotenv is installed.

    Existing process environment variables take precedence over values in .env.
    Returns the loaded .env path, or None when no file exists / loader unavailable.
    """
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    load_dotenv(env_path, override=False)
    return env_path


def require_env(name: str) -> str:
    load_project_env()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Put it in the project .env or environment; never commit .env.")
    return value
