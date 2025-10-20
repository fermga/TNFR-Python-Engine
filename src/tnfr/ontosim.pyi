from __future__ import annotations

from .types import TNFRConfigValue, TNFRGraph

__all__: tuple[str, ...]
_PREPARAR_RED_REMOVAL_DATE: str


def prepare_network(
    G: TNFRGraph,
    *,
    init_attrs: bool = True,
    override_defaults: bool = False,
    **overrides: TNFRConfigValue,
) -> TNFRGraph: ...


def preparar_red(
    G: TNFRGraph,
    *,
    init_attrs: bool = True,
    override_defaults: bool = False,
    **overrides: TNFRConfigValue,
) -> TNFRGraph: ...


def step(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None: ...


def run(
    G: TNFRGraph,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None: ...
