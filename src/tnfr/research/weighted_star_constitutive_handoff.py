"""Detached constitutive comparison after the retained AL/UM/SHA word.

The input contains the four previously recorded static graphs. No operator,
refresh callback or trajectory is executed. Re-admission checks primitives and
prefix associations; it does not authenticate that the recorded events occurred.
The fixed cubic-current control is a supplied law, not a new engine default.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..mathematics._rational_interval import I, cos, sin
from ..physics.relational_sine_comparison import _sine_phase_rate_numerators, _sine_work
from ..physics.relational_sine_support import (
    SineFrozenIsolateComparison,
    bound_relational_sine_frozen_isolates,
)
from ..utils.io import json_dumps
from .artifact_io import encode_exact_tree, sha256_bytes
from .retained_graph_state import restore_recorded_graph_state
from .weighted_star_contact import BOOST, MODEL, OLD_EDGES
from .weighted_star_finite_episode import admit_preparation

EPSILON = Q(1, 16)
# Canonical JSON of the already frozen preparation, with sorted mapping keys
# and exact fraction/graph-container tags retained. This is neither the file's
# byte digest nor authentication of the recorded execution or its chronology.
_PREPARATION_RECORD_SHA256 = (
    "8c472d31ad2b99b78c3d8e1a414ad3d5c8bf2fa17c783010bf79a0ecc6209335"
)


@dataclass(frozen=True)
class ConstitutiveFields:
    """Rebuilt instantaneous rows and own storage, with outward residuals."""

    epsilon: Q
    phase_currents: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_storage: Q
    phase_storage: I
    storage: I
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    storage_rate: I
    weighted_form_mean_rate: I
    weighted_phase_mean_rate: I


@dataclass(frozen=True)
class ConstitutivePrefix:
    """One raw prefix and two hypothetical continuous-law evaluations."""

    index: int
    source: SineFrozenIsolateComparison
    stored_pressure: tuple[Q, ...]
    mean_weights: tuple[Q, ...]
    phase_rate_numerators: tuple[Q, ...]
    sine: ConstitutiveFields
    cubic: ConstitutiveFields


@dataclass(frozen=True)
class WeightedStarConstitutiveHandoff:
    """The retained raw preparation and newly derived static comparisons.

    All raw history, configuration and lifecycle records remain associated.
    Their presence is not a proof of execution, chronology or provenance.
    """

    preparation: dict
    preparation_record_sha256: str
    prefixes: tuple[ConstitutivePrefix, ...]
    graph_time: Q
    pressure_difference: tuple[I, ...]
    form_rate_difference: tuple[I, ...]
    sine_event_work: tuple[I, ...]
    cubic_event_work: tuple[I, ...]


def _prefix_fields(graph, index):
    """Use shared graph/scalar admission and the complete reciprocal kernel."""
    source = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    if any(degree <= 0 for degree in source.degrees) or any(
        capacity <= 0 for capacity in source.capacity
    ):
        raise ValueError("the retained prefixes require positive degrees and capacity")
    stored = tuple(
        get_attr(
            graph.nodes[node],
            ALIAS_DNFR,
            strict=True,
            conv=lambda value: exact_or_represented_real(value, "stored pressure"),
        )
        for node in source.nodes
    )
    if any(value is None for value in stored):
        raise ValueError("every raw prefix must retain stored pressure")
    positions = {node: i for i, node in enumerate(source.nodes)}
    correction = [I(0) for _ in source.nodes]
    extra_potential = I(0)
    for left, right in source.edges:
        i, j = positions[left], positions[right]
        gap = I(source.phase[j] - source.phase[i])
        current = sin(gap) ** 3
        correction[i] += current
        correction[j] -= current
        cosine = cos(gap)
        # Integral of sin^3, written without subtracting near-equal constants.
        extra_potential += (1 - cosine) ** 2 * (cosine + 2) / 3
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    total_weight = sum(weights, Q(0))
    currents = tuple(imag for _, imag in source.relative_resultant)

    def fields(epsilon):
        own_currents = tuple(
            value + epsilon * change for value, change in zip(currents, correction)
        )
        work = _sine_work(
            MODEL,
            source.degrees,
            source.form_gradient,
            source.capacity,
            own_currents,
        )
        phase_storage = source.phase_storage + epsilon * extra_potential
        return ConstitutiveFields(
            epsilon=epsilon,
            phase_currents=own_currents,
            pressure=work["pressure"],
            form_rates=work["form_rates"],
            phase_rates=work["phase_rates"],
            form_storage=source.form_storage,
            phase_storage=phase_storage,
            storage=source.form_storage + phase_storage,
            form_work=work["form_work"],
            phase_work=work["phase_work"],
            storage_rate=work["storage_rate"],
            weighted_form_mean_rate=sum(
                (weight * rate for weight, rate in zip(weights, work["form_rates"])),
                I(0),
            )
            / total_weight,
            weighted_phase_mean_rate=sum(
                (weight * rate for weight, rate in zip(weights, work["phase_rates"])),
                I(0),
            )
            / total_weight,
        )

    return ConstitutivePrefix(
        index=index,
        source=source,
        stored_pressure=stored,
        mean_weights=weights,
        phase_rate_numerators=_sine_phase_rate_numerators(
            MODEL, source.degrees, source.form_gradient, source.capacity
        ),
        sine=fields(Q(0)),
        cubic=fields(EPSILON),
    )


def _admit_prefix_association(graph, prefix, final_source):
    """Admit nominal primitives before checking the entire retained record."""
    index, source = prefix.index, prefix.source
    if type(graph.graph.get("_t")) is not int or graph.graph["_t"] != 7:
        raise ValueError("raw prefix structural clock differs")
    if graph.graph.get("GAMMA") != {"type": "none"}:
        raise ValueError("raw prefix requires the declared absent Gamma")
    if "compute_delta_nfr" in graph.graph:
        raise ValueError("the retained static word supplies no pressure callback")
    form = list(final_source.epi)
    capacity = list(final_source.capacity)
    if index == 0:
        form[4] -= BOOST
    capacity[4] = (Q(8), Q(8), Q(117, 16), Q(117, 32))[index]
    edges = OLD_EDGES if index < 2 else final_source.edges
    final_weights = dict(zip(final_source.edges, final_source.conductances))
    if (
        source.nodes != final_source.nodes
        or source.epi != tuple(form)
        or source.phase != final_source.phase
        or source.capacity != tuple(capacity)
        or set(source.edges) != set(edges)
        or any(
            final_weights[edge] != weight
            for edge, weight in zip(source.edges, source.conductances)
        )
        or any(value != 0 for value in prefix.stored_pressure)
    ):
        raise ValueError("raw prefix physical association differs")


def compare_weighted_star_constitutive_handoff(preparation):
    """Re-admit the central raw word and compare the two declared continuations.

    This accepts only the frozen preparation's exact tagged record, after
    primitive admission. Canonical serialization preserves Boolean versus
    integer scalars and the raw graph's tuple/list/deque tags. Its fixed record
    fingerprint associates every retained configuration, history and lifecycle
    field without replacing their owners with another admission catalog.
    Association does not authenticate execution or chronology. No observation,
    identity bound or finite sine trajectory transfers to the cubic law.
    """
    final_source, _, _ = admit_preparation(preparation)
    graphs = tuple(
        restore_recorded_graph_state(record).graph
        for record in preparation["static_states"]
    )
    prefixes = []
    for index, graph in enumerate(graphs):
        prefix = _prefix_fields(graph, index)
        _admit_prefix_association(graph, prefix, final_source)
        prefixes.append(prefix)
    serialized = json_dumps(
        encode_exact_tree(preparation), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    record_sha256 = sha256_bytes(serialized)
    if record_sha256 != _PREPARATION_RECORD_SHA256:
        raise ValueError("full retained preparation record association differs")
    last = prefixes[-1]
    return WeightedStarConstitutiveHandoff(
        preparation=deepcopy(preparation),
        preparation_record_sha256=record_sha256,
        prefixes=tuple(prefixes),
        graph_time=Q(7),
        pressure_difference=tuple(
            cubic - sine for cubic, sine in zip(last.cubic.pressure, last.sine.pressure)
        ),
        form_rate_difference=tuple(
            cubic - sine
            for cubic, sine in zip(last.cubic.form_rates, last.sine.form_rates)
        ),
        sine_event_work=tuple(
            right.sine.storage - left.sine.storage
            for left, right in zip(prefixes, prefixes[1:])
        ),
        cubic_event_work=tuple(
            right.cubic.storage - left.cubic.storage
            for left, right in zip(prefixes, prefixes[1:])
        ),
    )
