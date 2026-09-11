"""B10 — U-Rules Consistency Signature (URC) diagnostic.

Type-hygiene closure for the unified-grammar rule checkers (U1-U6).

Question (B10 of Catalog Type-Hygiene Programme, §13quinquaginta-octava
of TNFR_RIEMANN_RESEARCH_NOTES.md): do the rule checkers in
``src/tnfr/operators/grammar*.py`` read only operator-name sequences plus
scalar telemetry, and return only scalar/string verdicts — or do they
silently introduce a richer canonical envelope (callable kernel, measure,
operator-valued intermediate, matrix lift, Banach-derivative apparatus)
along the way?

Eleventh CDM (Canonical Discharging Method): **URC = U-Rules Consistency
Discipline**, acting on the **U-rules type-hygiene surface**. Distinct
from the ten preceding CDMs (B0 Pontryagin/measure-ν_f, B1 TMEP,
B2 PWDP, B3 BSAD, B4 DITS, B5 STD, B6 SWD, B7 TRC, B8 CCC, B9 ACD), which
act on field-measure / element-projection / phase-wrap / scalar-aggregation
/ temporal-sampling / coupling-verdict / mixing-aggregation /
tetrad-reduction-closure / currents-reduction-closure /
aggregates-reduction-closure surfaces respectively.

Closure proceeds in two phases:

* Phase a (this module): empirical type-witness across all U1-U5 rule
  checkers in ``grammar_core.py`` and the U6 checker in ``grammar_u6.py``.
  Each checker is invoked with synthetic inputs derived only from canonical
  scalar slots, and its argument and return types are inspected for any
  non-scalar/non-string/non-bool/non-tuple leakage.
* Phase c (research notes §13quinquaginta-octava + §13quinquaginta-nona):
  per-rule source-code closure trace plus NEGATIVE verdict, promotion of
  URC to canonical status.

Scope guard: this module does NOT advance G4 = RH (Conjecture T-HP,
§13septies). It does NOT modify any canonical implementation. It is a
methodological diagnostic, not a physical or spectral statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx

from ..operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
)
from ..operators.grammar_core import GrammarValidator
from ..operators.grammar_u6 import validate_structural_potential_confinement

__all__ = [
    "URulesConsistencySignatureCertificate",
    "compute_urules_consistency_signature",
]


# ---------------------------------------------------------------------------
# Type-leakage classifier
# ---------------------------------------------------------------------------

_SCALAR_INPUT_TYPES = (int, float, bool, str)
_SCALAR_OUTPUT_TYPES = (int, float, bool, str, type(None))


def _classify_input(value: Any) -> str:
    """Classify a U-rule checker input as scalar-admissible or leaking."""
    if isinstance(value, _SCALAR_INPUT_TYPES):
        return "scalar"
    if isinstance(value, list):
        # Sequence-of-operator-names is admissible (operator instances carry
        # only a canonical_name string + class identity).
        if all(_is_operator_instance(item) for item in value):
            return "operator_name_sequence"
        return "leaking_list"
    if isinstance(value, dict):
        # Per-node scalar telemetry dicts are admissible if every value is
        # a scalar (U6 reads phi_s_before / phi_s_after as dict[node, float]).
        if all(isinstance(v, _SCALAR_INPUT_TYPES) for v in value.values()):
            return "scalar_dict"
        return "leaking_dict"
    if isinstance(value, nx.Graph):
        # The graph metric itself is the canonical substrate (not a richer
        # envelope); it is the same graph object operated on by every
        # canonical operator. Admissible.
        return "graph_metric"
    return "leaking_other"


def _is_operator_instance(item: Any) -> bool:
    """Check if item carries the canonical Operator surface (name + class)."""
    return hasattr(item, "name") or hasattr(item, "canonical_name")


def _classify_output(value: Any) -> str:
    """Classify a U-rule checker output as scalar-admissible or leaking."""
    if isinstance(value, _SCALAR_OUTPUT_TYPES):
        return "scalar"
    if isinstance(value, tuple):
        # Tuple-of-scalars (e.g. (bool, str) or (bool, float, str)) is
        # admissible. Anything else leaks.
        if all(isinstance(v, _SCALAR_OUTPUT_TYPES) for v in value):
            return "scalar_tuple"
        return "leaking_tuple"
    return "leaking_other"


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


@dataclass
class URulesConsistencySignatureCertificate:
    """Frozen diagnostic certificate for B10 (URC) Phase a.

    Attributes
    ----------
    rules_probed : list[str]
        Names of U-rule checkers probed.
    input_classifications : dict[str, list[str]]
        Per-rule input classification list (one entry per argument).
    output_classifications : dict[str, str]
        Per-rule output classification.
    input_scalar_fraction : float
        Fraction of inputs classified as admissible (scalar / operator-name
        sequence / scalar_dict / graph_metric).
    output_scalar_fraction : float
        Fraction of outputs classified as admissible (scalar / scalar_tuple).
    leaking_inputs : int
        Total count of leaking inputs (any non-admissible classification).
    leaking_outputs : int
        Total count of leaking outputs.
    S_UR : float
        U-Rules Consistency signature := leakage rate over all probes.
        Computed as ``(leaking_inputs + leaking_outputs) / total_probes``.
    verdict : str
        "TYPE_HYGIENE_ADEQUATE" if S_UR == 0 (Phase a passes); otherwise
        "TYPE_HYGIENE_VIOLATION".
    notes : str
        Free-form descriptive notes.
    """

    rules_probed: list[str] = field(default_factory=list)
    input_classifications: dict[str, list[str]] = field(default_factory=dict)
    output_classifications: dict[str, str] = field(default_factory=dict)
    input_scalar_fraction: float = 0.0
    output_scalar_fraction: float = 0.0
    leaking_inputs: int = 0
    leaking_outputs: int = 0
    S_UR: float = 0.0
    verdict: str = ""
    notes: str = ""

    def summary(self) -> str:
        """Return a single-line human-readable summary."""
        return (
            f"URC[rules={len(self.rules_probed)}, "
            f"S_UR={self.S_UR:.6f}, "
            f"input_scalar={self.input_scalar_fraction:.3f}, "
            f"output_scalar={self.output_scalar_fraction:.3f}, "
            f"verdict={self.verdict}]"
        )


# ---------------------------------------------------------------------------
# Synthetic input fixtures (built ONLY from canonical scalar slots)
# ---------------------------------------------------------------------------


def _make_canonical_sequence() -> list[Any]:
    """Build a grammar-valid synthetic operator sequence.

    Pattern: generator -> coupling -> destabilizer -> stabilizer -> handler
    -> transformer -> recursivity -> closure. Touches every U-rule branch.
    """
    return [
        Emission(),  # U1a generator
        Coupling(),  # U3 coupling/resonance trigger
        Resonance(),  # U3 trigger
        Dissonance(),  # U2/U4a destabilizer + bifurcation trigger
        Coherence(),  # U2 stabilizer
        Mutation(),  # U4b transformer
        SelfOrganization(),  # U4a handler + U4b transformer
        Recursivity(),  # U1a generator + U2-REMESH amplifier
        Silence(),  # U1b closure
    ]


def _make_phi_s_snapshot(graph: nx.Graph) -> dict[Any, float]:
    """Synthetic Φ_s per-node snapshot — pure scalar dict."""
    return {n: 0.1 * (i + 1) for i, n in enumerate(graph.nodes())}


# ---------------------------------------------------------------------------
# Probe driver
# ---------------------------------------------------------------------------


def _probe_rule(
    name: str,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[str], str]:
    """Invoke a U-rule checker, classify its inputs and return type."""
    input_class = [_classify_input(a) for a in args]
    input_class.extend(_classify_input(v) for v in kwargs.values())
    result = func(*args, **kwargs)
    output_class = _classify_output(result)
    return input_class, output_class


def compute_urules_consistency_signature(
    n_nodes: int = 24,
    seed: int = 31,
) -> URulesConsistencySignatureCertificate:
    """Compute the URC diagnostic signature (B10 Phase a).

    Parameters
    ----------
    n_nodes : int, default 24
        Size of the synthetic graph fixture used to source per-node Φ_s.
    seed : int, default 31
        Seed for the synthetic graph fixture (passed to ``erdos_renyi_graph``).

    Returns
    -------
    URulesConsistencySignatureCertificate
        Frozen diagnostic with per-rule classifications and S_UR.
    """
    sequence = _make_canonical_sequence()
    graph = nx.erdos_renyi_graph(n_nodes, p=0.3, seed=seed)
    phi_s_before = _make_phi_s_snapshot(graph)
    phi_s_after = {k: v + 0.01 for k, v in phi_s_before.items()}

    probes: list[tuple[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = [
        # U1a: validate_initiation(sequence, epi_initial)
        (
            "U1a_initiation",
            GrammarValidator.validate_initiation,
            (sequence,),
            {"epi_initial": 0.0},
        ),
        # U1b: validate_closure(sequence)
        (
            "U1b_closure",
            GrammarValidator.validate_closure,
            (sequence,),
            {},
        ),
        # U2: validate_convergence(sequence)
        (
            "U2_convergence",
            GrammarValidator.validate_convergence,
            (sequence,),
            {},
        ),
        # U3: validate_resonant_coupling(sequence)
        (
            "U3_resonant_coupling",
            GrammarValidator.validate_resonant_coupling,
            (sequence,),
            {},
        ),
        # U4a: validate_bifurcation_triggers(sequence)
        (
            "U4a_bifurcation_triggers",
            GrammarValidator.validate_bifurcation_triggers,
            (sequence,),
            {},
        ),
        # U4b: validate_transformer_context(sequence)
        (
            "U4b_transformer_context",
            GrammarValidator.validate_transformer_context,
            (sequence,),
            {},
        ),
        # U2-REMESH: validate_remesh_amplification(sequence)
        (
            "U2_remesh_amplification",
            GrammarValidator.validate_remesh_amplification,
            (sequence,),
            {},
        ),
        # U5: validate_multiscale_coherence(sequence)
        (
            "U5_multiscale_coherence",
            GrammarValidator.validate_multiscale_coherence,
            (sequence,),
            {},
        ),
        # Temporal ordering (helper rule supporting U2 ordering refinement)
        (
            "temporal_ordering",
            GrammarValidator.validate_temporal_ordering,
            (sequence,),
            {},
        ),
        # U6: validate_structural_potential_confinement(G, before, after, threshold, strict)
        (
            "U6_structural_potential_confinement",
            validate_structural_potential_confinement,
            (graph, phi_s_before, phi_s_after),
            {"strict": False},
        ),
    ]

    rules_probed: list[str] = []
    input_classifications: dict[str, list[str]] = {}
    output_classifications: dict[str, str] = {}

    leaking_inputs = 0
    leaking_outputs = 0
    total_inputs = 0
    total_outputs = 0
    admissible_inputs = {
        "scalar",
        "operator_name_sequence",
        "scalar_dict",
        "graph_metric",
    }
    admissible_outputs = {"scalar", "scalar_tuple"}

    for name, func, args, kwargs in probes:
        in_cls, out_cls = _probe_rule(name, func, args, kwargs)
        rules_probed.append(name)
        input_classifications[name] = in_cls
        output_classifications[name] = out_cls
        total_inputs += len(in_cls)
        total_outputs += 1
        leaking_inputs += sum(1 for c in in_cls if c not in admissible_inputs)
        leaking_outputs += 0 if out_cls in admissible_outputs else 1

    input_scalar_fraction = (
        (total_inputs - leaking_inputs) / total_inputs if total_inputs > 0 else 1.0
    )
    output_scalar_fraction = (
        (total_outputs - leaking_outputs) / total_outputs if total_outputs > 0 else 1.0
    )
    total_probes = total_inputs + total_outputs
    S_UR = (
        (leaking_inputs + leaking_outputs) / total_probes if total_probes > 0 else 0.0
    )
    verdict = "TYPE_HYGIENE_ADEQUATE" if S_UR == 0.0 else "TYPE_HYGIENE_VIOLATION"

    notes = (
        f"B10 URC Phase a: probed {len(probes)} U-rule checkers (U1a/U1b/U2/"
        f"U2-REMESH/U3/U4a/U4b/U5/temporal_ordering/U6) with synthetic "
        f"canonical-scalar inputs. Total inputs={total_inputs}, "
        f"total outputs={total_outputs}. Admissible input classes: "
        f"{sorted(admissible_inputs)}. Admissible output classes: "
        f"{sorted(admissible_outputs)}."
    )

    return URulesConsistencySignatureCertificate(
        rules_probed=rules_probed,
        input_classifications=input_classifications,
        output_classifications=output_classifications,
        input_scalar_fraction=input_scalar_fraction,
        output_scalar_fraction=output_scalar_fraction,
        leaking_inputs=leaking_inputs,
        leaking_outputs=leaking_outputs,
        S_UR=S_UR,
        verdict=verdict,
        notes=notes,
    )
