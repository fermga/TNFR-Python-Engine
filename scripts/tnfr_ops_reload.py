#!/usr/bin/env python
"""TNFR Operator Reload Helper

Provides interactive CLI for operator registry reload with telemetry.

Usage:
  python scripts/tnfr_ops_reload.py            # soft reload
  python scripts/tnfr_ops_reload.py --hard     # hard reload
  python scripts/tnfr_ops_reload.py --stats    # show stats only

Options:
  --hard    Perform hard invalidation (drop registry then rediscover)
  --stats   Print cache + grammar stats without reloading

Physics-neutral: does not alter operator semantics or glyph activation.
Safe for interactive development workflows.
"""
from __future__ import annotations

import argparse
import json
import pprint
from datetime import datetime

from tnfr.operators.registry import (
    invalidate_operator_cache,
    get_operator_cache_stats,
    OPERATORS,
)
from tnfr.operators.grammar import get_grammar_cache_stats


def format_ts(ts: float | None) -> str:
    if ts is None:
        return "<never>"
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TNFR operator reload with telemetry", add_help=True
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Hard invalidation (drop registry first)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show stats only, no reload"
    )
    args = parser.parse_args()

    if args.stats:
        stats = get_operator_cache_stats()
        grammar_stats = get_grammar_cache_stats()
        print("Operator Cache Stats:")
        pprint.pprint(stats)
        print("Grammar Cache Stats:")
        pprint.pprint(grammar_stats)
        return 0

    stats_before = get_operator_cache_stats()
    reload_stats = invalidate_operator_cache(hard=args.hard)
    stats_after = get_operator_cache_stats()

    grammar_stats = get_grammar_cache_stats()

    report = {
        "mode": "hard" if args.hard else "soft",
        "registry_before": stats_before,
        "reload_result": reload_stats,
        "registry_after": stats_after,
        "last_invalidation_iso": format_ts(
            stats_after.get("last_invalidation_ts")
        ),
        "operator_names": sorted(OPERATORS.keys()),
        "grammar_caches": grammar_stats,
    }

    print(json.dumps(report, indent=2))
    print(f"Total operators: {stats_after['current_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
