"""Load local secrets without requiring a manual export.

Import this module early in any entrypoint that talks to an LLM provider.
Missing `.env.local` is fine — CI and offline runs have no key.
"""

from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent / ".env.local"

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)  # no-op / silent if the file is absent
except ModuleNotFoundError:
    # python-dotenv is optional: local (Ollama) runs need no key, and CI/offline
    # runs have none. Fall back to a minimal KEY=VALUE parser so an existing
    # .env.local still populates os.environ without the dependency.
    import os
    if _ENV_PATH.exists():
        for _line in _ENV_PATH.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
