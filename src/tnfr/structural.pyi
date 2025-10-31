from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Sequence

from .operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from .validation import ValidationOutcome

if TYPE_CHECKING:
    import networkx as nx
    from .mathematics import (
        BasicStateProjector,
        CoherenceOperator,
        FrequencyOperator,
        HilbertSpace,
        MathematicalDynamicsEngine,
        NFRValidator,
    )

__all__: tuple[str, ...]

def __getattr__(name: str) -> Any: ...
def create_nfr(
    name: str,
    *,
    epi: float = ...,
    vf: float = ...,
    theta: float = ...,
    graph: "nx.Graph" | None = ...,
    dnfr_hook: Callable[..., None] = ...,
) -> tuple["nx.Graph", str]: ...
def create_math_nfr(
    name: str,
    *,
    epi: float = ...,
    vf: float = ...,
    theta: float = ...,
    graph: "nx.Graph" | None = ...,
    dnfr_hook: Callable[..., None] = ...,
    dimension: int | None = ...,
    hilbert_space: "HilbertSpace" | None = ...,
    coherence_operator: "CoherenceOperator" | None = ...,
    coherence_spectrum: Sequence[float] | None = ...,
    coherence_c_min: float | None = ...,
    coherence_threshold: float | None = ...,
    frequency_operator: "FrequencyOperator" | None = ...,
    frequency_diagonal: Sequence[float] | None = ...,
    generator_diagonal: Sequence[float] | None = ...,
    state_projector: "BasicStateProjector" | None = ...,
    dynamics_engine: "MathematicalDynamicsEngine" | None = ...,
    validator: "NFRValidator" | None = ...,
) -> tuple["nx.Graph", str]: ...

OPERATORS: dict[str, Operator]

def validate_sequence(names: Iterable[str]) -> ValidationOutcome[tuple[str, ...]]: ...
def run_sequence(G: "nx.Graph", node: Hashable, ops: Iterable[Operator]) -> None: ...
