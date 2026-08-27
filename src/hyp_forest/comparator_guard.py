"""One-time, loud warning when the answer comparator silently degrades.

`math_verify` is an optional dependency. Every scoring path in this
repository falls back to a normalized string match when it is absent
(`except ImportError: ...`). That fallback is *conservative*: it
under-counts correct answers, uniformly, by several absolute points.

That is harmless when a single run is scored end-to-end under one
configuration, and actively wrong when two runs are compared across
configurations -- a `metrics.json` written while `math_verify` was
importable is NOT comparable to one recomputed while it was not.

This exact mismatch produced an incorrect number in the ARR May 2026
response period (a prune-immunity "lift" reported as +6.96 pp that is
+0.70 pp when both arms are scored under one comparator; see README
"A correction, and the trap that caused it"). The fallback was silent,
so nothing in the pipeline flagged it.

Import `warn_if_degraded()` from any scoring path and call it once.
"""

from __future__ import annotations

import sys

_WARNED = False


def math_verify_available() -> bool:
    try:
        import math_verify  # noqa: F401
        return True
    except ImportError:
        return False


def warn_if_degraded() -> bool:
    """Return True if the comparator is degraded; warn once to stderr."""
    global _WARNED
    if math_verify_available():
        return False
    if not _WARNED:
        _WARNED = True
        print(
            "\n" + "=" * 72 + "\n"
            "WARNING: `math_verify` is not importable. Answer comparison has\n"
            "fallen back to normalized string matching, which UNDER-COUNTS\n"
            "correct answers by several absolute points.\n"
            "\n"
            "  * Safe:   comparing runs all scored in THIS environment.\n"
            "  * UNSAFE: comparing against a metrics.json written elsewhere,\n"
            "            or against numbers quoted in the paper.\n"
            "\n"
            "Install with `pip install math-verify`, or re-score every arm\n"
            "of your comparison in this same environment.\n"
            + "=" * 72 + "\n",
            file=sys.stderr,
        )
    return True
