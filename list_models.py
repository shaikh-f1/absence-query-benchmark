"""List OpenRouter models and live per-million-token prices.

  python list_models.py              # all models, cheapest first
  python list_models.py claude       # substring filter
"""

from __future__ import annotations

import argparse
import sys

import env  # noqa: F401 — load .env.local if present
from llm_extractor import list_models


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List OpenRouter models by price")
    ap.add_argument("filter", nargs="?", default="",
                    help="optional substring filter (e.g. claude, gemini)")
    args = ap.parse_args(argv)

    rows = list_models(args.filter)
    if not rows:
        print(f"no models matched {args.filter!r}", file=sys.stderr)
        return 1

    print(f"{'model':<52}{'in $/M':>10}{'out $/M':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['id']:<52}{r['in_per_mtok']:>10.3f}{r['out_per_mtok']:>10.3f}")
    print(f"\n{len(rows)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
