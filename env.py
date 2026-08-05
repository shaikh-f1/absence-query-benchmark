"""Load local secrets without requiring a manual export.

Import this module early in any entrypoint that talks to an LLM provider.
Missing `.env.local` is fine — CI and offline runs have no key.
"""

from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env.local"
load_dotenv(_ENV_PATH)  # no-op / silent if the file is absent
