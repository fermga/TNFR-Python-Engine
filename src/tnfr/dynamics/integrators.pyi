from __future__ import annotations

from typing import Literal

from tnfr.types import TNFRGraph

__all__: tuple[str, ...]

class AbstractIntegrator:
    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None = ...,
        t: float | None = ...,
        method: str | None = ...,
        n_jobs: int | None = ...,
    ) -> None: ...

class DefaultIntegrator(AbstractIntegrator):
    def __init__(self) -> None: ...

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
