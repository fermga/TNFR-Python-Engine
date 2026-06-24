"""Compatibility shim forwarding to TNFR chess evaluator.

This preserves the legacy import used by `core.evaluation` while routing to
`TNFREvaluatorCore`, which integrates the TNFR engine via the chess adapter.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Ensure tnfr-chess sources are on sys.path so imports below resolve.
_CHESS_SRC = os.path.join(
    os.path.dirname(__file__), "future-research", "tnfr-chess", "src"
)
if _CHESS_SRC not in sys.path:
    sys.path.append(_CHESS_SRC)

from tetrad_fields import TetradFields  # type: ignore
from tnfr_evaluator_core import TNFREvaluatorCore  # type: ignore


class TetradEvaluator(TNFREvaluatorCore):
    """Alias for compatibility with the legacy TetradEvaluator name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["TetradEvaluator", "TetradFields"]
