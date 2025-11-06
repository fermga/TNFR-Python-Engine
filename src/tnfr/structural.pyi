from __future__ import annotations

from .mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
)
from .operators.definitions import (
    Coherence as Coherence,
    Contraction as Contraction,
    Coupling as Coupling,
    Dissonance as Dissonance,
    Emission as Emission,
    Expansion as Expansion,
    Mutation as Mutation,
    Operator as Operator,
    Reception as Reception,
    Recursivity as Recursivity,
    Resonance as Resonance,
    SelfOrganization as SelfOrganization,
    Silence as Silence,
    Transition as Transition,
)
from .operators.registry import OPERATORS as OPERATORS
from .types import DeltaNFRHook, NodeId, TNFRGraph
from tnfr.validation import NFRValidator, validate_sequence as validate_sequence
from typing import Iterable, Sequence

__all__ = [
    "create_nfr",
    "create_math_nfr",
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
    "OPERATORS",
    "validate_sequence",
    "run_sequence",
]

def create_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = ...,
) -> tuple[TNFRGraph, str]: ...
def create_math_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = ...,
    dimension: int | None = None,
    hilbert_space: HilbertSpace | None = None,
    coherence_operator: CoherenceOperator | None = None,
    coherence_spectrum: Sequence[float] | None = None,
    coherence_c_min: float | None = None,
    coherence_threshold: float | None = None,
    frequency_operator: FrequencyOperator | None = None,
    frequency_diagonal: Sequence[float] | None = None,
    generator_diagonal: Sequence[float] | None = None,
    state_projector: BasicStateProjector | None = None,
    dynamics_engine: MathematicalDynamicsEngine | None = None,
    validator: NFRValidator | None = None,
) -> tuple[TNFRGraph, str]: ...
def run_sequence(G: TNFRGraph, node: NodeId, ops: Iterable[Operator]) -> None: ...
