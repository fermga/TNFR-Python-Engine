"""B11 (OCD) Phase a — Operator-registry discipline diagnostic.

Empirical schema witness for the declared 13-operator TNFR registry. This
module probes the immutable
registry exposed by ``src/tnfr/operators/registry.py``, the introspection
metadata in ``src/tnfr/operators/introspection.py``, and the public exports
in ``src/tnfr/operators/definitions.py``, and reports a single scalar
"catalog-discipline signature" ``S_OC`` measuring the fraction of probes that
violate the declared registry-schema checks.

Passing these finite introspection checks establishes registry consistency and
reload idempotence only. It does not define an independent space of admissible
TNFR transformations, prove that the 13 registered operators generate that
space, or rule out a future transformation with a distinct contract.

Phase scope (B11 Phase a): methodological diagnostic only. Does NOT modify
any canonical implementation. Does NOT advance G4 = RH (Conjecture T-HP,
``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` Sec 13septies). The Phase c
discharge (NEGATIVE verdict + OCD promotion as twelfth CDM in the L3*
orthogonality ledger) is delivered in a separate commit alongside the
research-notes section.

Declared registry checks:

- ``registry_size`` is the integer 13.
- Every registry value is a subclass of ``Operator``.
- Every registry key is a non-empty lowercase string.
- All registry keys are unique.
- ``OPERATOR_METADATA`` size is the integer 13.
- Every metadata value is an instance of the frozen ``OperatorMeta``
  dataclass.
- Every metadata ``grammar_roles`` and ``contracts`` field is a tuple of
  strings (no callable kernel, no measure, no operator-valued intermediate,
  no matrix lift).
- Every metadata entry's ``name`` matches the class name of a registry
  value (1-to-1 alignment between mnemonic-keyed metadata and lowercase-
  keyed registry).
- ``definitions.__all__`` exposes exactly the 13 canonical operator class
  names (plus the ``Operator`` base and a small known set of
  introspection/grammar-error helpers).

Each probe records the *anomaly count* (0 if invariant holds, 1 if it
breaks). The signature is the leakage rate

    S_OC := total_anomalies / total_probes ,

with the legacy registry-hygiene verdict ``CATALOG_DISCIPLINE_ADEQUATE`` when
``S_OC == 0`` and ``CATALOG_DISCIPLINE_LEAKING`` otherwise. ``ADEQUATE`` here
means internally aligned with the declared schema, not mathematically
complete over admissible dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..operators.definitions import Operator
from ..operators.definitions import __all__ as definitions_all
from ..operators.introspection import OPERATOR_METADATA, OperatorMeta
from ..operators.registry import OPERATORS, _ensure_loaded

# Declared expected size of the current TNFR operator registry.
CANONICAL_CATALOG_SIZE: int = 13

# Lowercase registry keys are sourced from the registry itself; the diagnostic
# only requires that the *count* and *string-ness* are correct.
# These are not hard-coded as a gating set — the diagnostic measures
# structural invariants, not name identity.

# Canonical class names exposed by ``definitions.__all__`` for the 13
# operators. These are the class identifiers a downstream caller would see.
_CANONICAL_OPERATOR_CLASS_NAMES: frozenset[str] = frozenset(
    {
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
    }
)

_DECLARED_NON_OPERATOR_EXPORTS: frozenset[str] = frozenset(
    {
        "Operator",
        "OperatorMeta",
        "OPERATOR_METADATA",
        "get_operator_meta",
        "iter_operator_meta",
        "ExtendedGrammarError",
        "collect_grammar_errors",
        "make_grammar_error",
    }
)


@dataclass(frozen=True, slots=True)
class OperatorCatalogDisciplineSignatureCertificate:
    """Frozen certificate produced by the OCD diagnostic.

    Attributes
    ----------
    probes
        Ordered tuple of probe identifiers.
    probe_results
        Mapping ``probe_id -> "OK" | "ANOMALY: <reason>"``.
    anomalies
        Total number of probes that failed the catalog-discipline invariant.
    total_probes
        Total number of probes executed (currently 10).
    S_OC
        Catalog-discipline signature ``= anomalies / total_probes``.
    registry_size
        Observed size of the ``OPERATORS`` registry.
    metadata_size
        Observed size of the ``OPERATOR_METADATA`` mapping.
    canonical_exports_observed
        Number of ``_CANONICAL_OPERATOR_CLASS_NAMES`` present in
        ``definitions.__all__``.
    verdict
        ``"CATALOG_DISCIPLINE_ADEQUATE"`` when ``S_OC == 0`` (no anomalies);
        ``"CATALOG_DISCIPLINE_LEAKING"`` otherwise.
    notes
        Free-form notes attached by the diagnostic builder.
    """

    probes: tuple[str, ...]
    probe_results: dict[str, str]
    anomalies: int
    total_probes: int
    S_OC: float
    registry_size: int
    metadata_size: int
    canonical_exports_observed: int
    verdict: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"OCD[probes={self.total_probes}, S_OC={self.S_OC:.6f}, "
            f"registry={self.registry_size}, metadata={self.metadata_size}, "
            f"exports={self.canonical_exports_observed}/"
            f"{len(_CANONICAL_OPERATOR_CLASS_NAMES)}, "
            f"verdict={self.verdict}]"
        )


def _probe_registry_size() -> tuple[str, str]:
    actual = len(OPERATORS)
    if actual == CANONICAL_CATALOG_SIZE:
        return "OK", "registry_size == 13"
    return "ANOMALY", f"registry_size = {actual} (expected 13)"


def _probe_registry_entries_are_operator_subclasses() -> tuple[str, str]:
    bad: list[str] = []
    for key, cls in OPERATORS.items():
        if not isinstance(cls, type) or not issubclass(cls, Operator):
            bad.append(key)
    if not bad:
        return "OK", "all registry entries subclass Operator"
    return "ANOMALY", f"non-Operator entries: {bad}"


def _probe_registry_keys_are_lowercase_strings() -> tuple[str, str]:
    bad: list[Any] = []
    for key in OPERATORS:
        if not isinstance(key, str) or not key or key != key.lower():
            bad.append(key)
    if not bad:
        return "OK", "all registry keys non-empty lowercase strings"
    return "ANOMALY", f"non-string/non-lowercase keys: {bad}"


def _probe_registry_keys_unique() -> tuple[str, str]:
    keys = list(OPERATORS.keys())
    if len(keys) == len(set(keys)):
        return "OK", "all registry keys unique"
    return "ANOMALY", "duplicate registry keys observed"


def _probe_metadata_size() -> tuple[str, str]:
    actual = len(OPERATOR_METADATA)
    if actual == CANONICAL_CATALOG_SIZE:
        return "OK", "metadata_size == 13"
    return "ANOMALY", f"metadata_size = {actual} (expected 13)"


def _probe_metadata_values_are_operator_meta() -> tuple[str, str]:
    bad: list[str] = []
    for key, meta in OPERATOR_METADATA.items():
        if not isinstance(meta, OperatorMeta):
            bad.append(key)
    if not bad:
        return "OK", "all metadata values are OperatorMeta"
    return "ANOMALY", f"non-OperatorMeta values: {bad}"


def _probe_metadata_fields_are_string_tuples() -> tuple[str, str]:
    bad: list[str] = []
    for key, meta in OPERATOR_METADATA.items():
        roles = meta.grammar_roles
        contracts = meta.contracts
        if not isinstance(roles, tuple) or not all(isinstance(r, str) for r in roles):
            bad.append(f"{key}:grammar_roles")
        if not isinstance(contracts, tuple) or not all(
            isinstance(c, str) for c in contracts
        ):
            bad.append(f"{key}:contracts")
        if not isinstance(meta.name, str) or not isinstance(meta.mnemonic, str):
            bad.append(f"{key}:name_or_mnemonic")
        if not isinstance(meta.category, str) or not isinstance(meta.doc, str):
            bad.append(f"{key}:category_or_doc")
    if not bad:
        return "OK", "all metadata fields are strings/tuples-of-strings"
    return "ANOMALY", f"non-string/tuple-of-string metadata fields: {bad}"


def _probe_metadata_registry_alignment() -> tuple[str, str]:
    registry_class_names = {cls.__name__ for cls in OPERATORS.values()}
    metadata_names = {meta.name for meta in OPERATOR_METADATA.values()}
    missing = metadata_names - registry_class_names
    extra = registry_class_names - metadata_names
    if not missing and not extra:
        return "OK", "metadata.name 1-to-1 with registry class names"
    return "ANOMALY", f"misaligned (missing={missing}, extra={extra})"


def _probe_definitions_exports_cover_canonical_set() -> tuple[str, str]:
    exported = set(definitions_all)
    expected = _CANONICAL_OPERATOR_CLASS_NAMES | _DECLARED_NON_OPERATOR_EXPORTS
    missing = expected - exported
    extra = exported - expected
    if not missing and not extra:
        return "OK", "exports match the declared operator facade"
    return "ANOMALY", (
        f"export mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
    )


def _probe_registry_reload_idempotence() -> tuple[str, str]:
    """Check that reloading does not perturb the declared registry."""
    # Probe whether re-invoking _ensure_loaded() perturbs the declared mapping.
    before = dict(OPERATORS)
    _ensure_loaded()
    after = dict(OPERATORS)
    if len(after) == CANONICAL_CATALOG_SIZE and set(after.keys()) == set(before.keys()):
        return "OK", "registry idempotent at size 13"
    return "ANOMALY", (f"registry expanded from {len(before)} to {len(after)}")


def compute_operator_catalog_discipline_signature() -> (
    OperatorCatalogDisciplineSignatureCertificate
):
    """Run the OCD diagnostic over the canonical TNFR operator catalog.

    Returns
    -------
    OperatorCatalogDisciplineSignatureCertificate
        Frozen certificate aggregating per-probe results and the scalar
        signature ``S_OC``.

    Notes
    -----
    The probes are pure read-only inspections of module-level mappings;
    no graph state is constructed. The diagnostic is deterministic and
    requires no seed. A zero anomaly score certifies only this finite registry
    surface; it is not evidence of generative completeness or irreducibility.
    """
    _ensure_loaded()

    probes: tuple[tuple[str, Any], ...] = (
        ("registry_size", _probe_registry_size),
        (
            "registry_entries_are_operator_subclasses",
            _probe_registry_entries_are_operator_subclasses,
        ),
        (
            "registry_keys_are_lowercase_strings",
            _probe_registry_keys_are_lowercase_strings,
        ),
        ("registry_keys_unique", _probe_registry_keys_unique),
        ("metadata_size", _probe_metadata_size),
        ("metadata_values_are_operator_meta", _probe_metadata_values_are_operator_meta),
        ("metadata_fields_are_string_tuples", _probe_metadata_fields_are_string_tuples),
        ("metadata_registry_alignment", _probe_metadata_registry_alignment),
        (
            "definitions_exports_cover_canonical_set",
            _probe_definitions_exports_cover_canonical_set,
        ),
        # Historical identifier retained in serialized certificates. The
        # implementation tests reload idempotence, not transformation-space
        # completeness.
        ("no_hidden_fourteenth_operator", _probe_registry_reload_idempotence),
    )

    probe_ids: list[str] = []
    probe_results: dict[str, str] = {}
    anomalies = 0
    for name, fn in probes:
        status, detail = fn()
        probe_ids.append(name)
        if status == "OK":
            probe_results[name] = f"OK: {detail}"
        else:
            probe_results[name] = f"ANOMALY: {detail}"
            anomalies += 1

    total = len(probes)
    s_oc = anomalies / total if total else 0.0
    verdict = (
        "CATALOG_DISCIPLINE_ADEQUATE"
        if anomalies == 0
        else "CATALOG_DISCIPLINE_LEAKING"
    )

    canonical_exports_observed = len(
        _CANONICAL_OPERATOR_CLASS_NAMES & set(definitions_all)
    )

    return OperatorCatalogDisciplineSignatureCertificate(
        probes=tuple(probe_ids),
        probe_results=probe_results,
        anomalies=anomalies,
        total_probes=total,
        S_OC=s_oc,
        registry_size=len(OPERATORS),
        metadata_size=len(OPERATOR_METADATA),
        canonical_exports_observed=canonical_exports_observed,
        verdict=verdict,
        notes=(
            "Diagnostic surface: registry (registry.py), introspection "
            "metadata (introspection.py), public exports "
            "(definitions.__all__).",
            "A zero anomaly score certifies declared registry consistency, "
            "not completeness over admissible TNFR transformations.",
            "No callable kernel, no measure, no operator-valued "
            "intermediate constructed.",
        ),
    )


__all__ = [
    "CANONICAL_CATALOG_SIZE",
    "OperatorCatalogDisciplineSignatureCertificate",
    "compute_operator_catalog_discipline_signature",
]
