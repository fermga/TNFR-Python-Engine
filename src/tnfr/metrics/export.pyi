from ..config.constants import GLYPHS_CANONICAL as GLYPHS_CANONICAL
from ..glyph_history import ensure_history as ensure_history
from ..types import Graph as Graph
from ..types import SigmaTrace as SigmaTrace
from ..utils import json_dumps as json_dumps
from ..utils import safe_write as safe_write
from .reporting import glyphogram_series as glyphogram_series

def export_metrics(
    G: Graph,
    base_path: str,
    fmt: str = "csv",
    *,
    output_dir: str | None = None,
) -> None: ...
