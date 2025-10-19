from typing import Literal

from tnfr.types import TNFRGraph

__all__: tuple[str, ...]

def prepare_integration_params(
    G: TNFRGraph,
    dt: float | None = ...,
    t: float | None = ...,
    method: Literal["euler", "rk4"] | None = ...,
) -> tuple[float, int, float, Literal["euler", "rk4"]]: ...

def update_epi_via_nodal_equation(
    G: TNFRGraph,
    *,
    dt: float | None = ...,
    t: float | None = ...,
    method: Literal["euler", "rk4"] | None = ...,
    n_jobs: int | None = ...,
) -> None: ...
