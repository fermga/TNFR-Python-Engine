"""Declared finite operator studies and detached observations for SDK and CLI.

Recipes construct the existing scalar baseline and run registered operator words.
They are not arbitrary graph checkpoints, continuous-time solvers, or certificates
of autonomous pattern formation. Diagnostic values never select the next operator.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass, field, fields
from numbers import Integral, Real
from typing import Any, Callable

import networkx as nx
import numpy as np

from .._exact_time import bounded_represented_real, finite_represented_real
from ..config import get_precision_mode
from ..config.parsing import parse_bool
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..physics.fields import (
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)
from ..types import require_finite_real_scalar_epi
from ..utils.io import json_dumps
from ._state import copy_graph_state
from ._topology import nonnegative_integer, numpy_seed, positive_integer, probability
from .simple import TNFR, Network

__all__ = [
    "STUDY_TOPOLOGIES",
    "StudySpec",
    "StudyResult",
    "diagnose_network",
    "list_sequences",
    "run_study",
]

STUDY_TOPOLOGIES = ("ring", "path", "star", "complete", "random")
_SEQUENCE_SCOPE = (
    "Declared registered operator word; live grammar and operator preconditions "
    "remain active. A catalog entry does not certify execution or future stability."
)


def list_sequences(name: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Read detached word definitions from the existing SDK sequence registry."""
    from .fluent import NAMED_SEQUENCES

    def entry(key: str) -> dict[str, Any]:
        return {
            "name": key,
            "operators": list(NAMED_SEQUENCES[key]),
            "scope": _SEQUENCE_SCOPE,
        }

    if name is None:
        return [entry(key) for key in sorted(NAMED_SEQUENCES)]
    if not isinstance(name, str) or name not in NAMED_SEQUENCES:
        raise ValueError(f"Unknown registered sequence: {name!r}")
    return entry(name)


@dataclass(frozen=True)
class StudySpec:
    """Validated construction recipe, with operator-word cycles rather than seconds.

    All topologies use positive node counts and the NumPy topology seed domain
    ``0 <= seed < 2**32``. The same seed is installed in graph metadata for the
    runtime RNG owner. ``probability`` is used only by the random builder.
    Remaining engine configuration is inherited from the running process; this
    recipe is not a complete configuration or cross-version replay guarantee.
    """

    nodes: int = 10
    topology: str = "ring"
    seed: int = 0
    sequence: str = "basic_activation"
    cycles: int = 1
    probability: float = 0.3
    name: str = "study"

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", positive_integer(self.nodes, "nodes"))
        object.__setattr__(self, "cycles", nonnegative_integer(self.cycles, "cycles"))
        if not isinstance(self.topology, str) or self.topology not in STUDY_TOPOLOGIES:
            raise ValueError(f"topology must be one of {STUDY_TOPOLOGIES}")
        seed = numpy_seed(self.seed, allow_none=False)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "probability", probability(self.probability))
        list_sequences(self.sequence)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")

    def to_dict(self) -> dict[str, Any]:
        """Return the recipe only; no runtime state or executable callbacks."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudySpec:
        """Validate a recipe object; reject unknown keys rather than ignoring them."""
        if not isinstance(data, Mapping):
            raise TypeError("A study recipe must be a mapping")
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown study recipe keys: {sorted(map(str, unknown))}")
        return cls(**dict(data))


@dataclass(frozen=True)
class StudyResult:
    """Immutable JSON report carrier; each ``to_dict`` returns detached data.

    Use the existing ``export_to_json(result, path)`` atomic writer to save it.
    All report object keys must already be strings; unsupported keys cannot be
    merged by JSON coercion. Scalars share diagnostic represented-real admission.
    The structural projection excludes history, callbacks, caches and graph
    metadata, so importing this report does not restore a runnable graph.
    """

    report: InitVar[Mapping[str, Any]]
    _payload_json: str = field(init=False, repr=False)

    def __post_init__(self, report: Mapping[str, Any]) -> None:
        if not isinstance(report, Mapping):
            raise TypeError("A study report must be a mapping")
        object.__setattr__(
            self, "_payload_json", json_dumps(_json_value(report), allow_nan=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._payload_json)


def _json_value(value: Any) -> Any:
    """Normalize finite numerical readouts, rejecting unsupported values explicitly."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return finite_represented_real(value, "Diagnostic value")[0]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Diagnostic object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _capture(read: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = read()
        if value is None:
            raise ValueError("Diagnostic owner reported no available value")
        return {"available": True, "value": _json_value(value), "error": None}
    except Exception as error:
        return {
            "available": False,
            "value": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _raw_field(data: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in data:
            return data[alias]
    raise ValueError(f"No stored value for {aliases[0]}")


def _state_projection(graph: nx.Graph) -> dict[str, Any]:
    """Indexed scalar projection; display labels are not serialized node identities."""
    indices = {node: index for index, node in enumerate(graph)}
    rows = []
    for node, data in graph.nodes(data=True):
        row = {"index": indices[node], "label": str(node)}
        for name, aliases in (
            ("epi", ALIAS_EPI),
            ("nu_f", ALIAS_VF),
            ("phase", ALIAS_THETA),
            ("delta_nfr", ALIAS_DNFR),
        ):

            def read_field(aliases=aliases, name=name):
                raw = _raw_field(data, aliases)
                if name == "epi":
                    return require_finite_real_scalar_epi(raw, aliases[0])
                if name == "nu_f":
                    return bounded_represented_real(raw, aliases[0], minimum=0)[0]
                return finite_represented_real(raw, aliases[0])[0]

            row[name] = _capture(read_field)
        rows.append(row)
    return {
        "nodes": rows,
        "edges": [[indices[u], indices[v]] for u, v in graph.edges()],
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "scope": (
            "Indexed scalar projection in graph iteration order; labels are display "
            "only. Edge attributes/keys, history and runtime state are not checkpoints."
        ),
    }


def _observation_options(
    include_coherence_profile, include_phase_evidence, winding_cycles
):
    """Admit optional readout requests before copying or observing graph state."""
    flags = []
    for name, value in (
        ("include_coherence_profile", include_coherence_profile),
        ("include_phase_evidence", include_phase_evidence),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
        flags.append(parse_bool(value))
    if isinstance(winding_cycles, (str, bytes, bytearray)) or not isinstance(
        winding_cycles, Sequence
    ):
        raise TypeError("winding_cycles must be a finite ordered sequence of cycles")
    cycles = []
    for cycle in winding_cycles:
        if isinstance(cycle, (str, bytes, bytearray)) or not isinstance(
            cycle, Sequence
        ):
            raise TypeError(
                "each winding cycle must be a finite ordered sequence of node IDs"
            )
        nodes = tuple(cycle)
        for node in nodes:
            hash(node)  # Node identity must be admissible independently of presence.
        cycles.append(nodes)
    return *flags, tuple(cycles)


def _coherence_profile_projection(profile, indices):
    """Project opaque node identities before copying numerical dataclass fields."""
    data = {
        item.name: getattr(profile, item.name)
        for item in fields(profile)
        if item.name not in {"nodes", "sources", "bins", "fit"}
    }
    data.update(
        nodes=[indices[node] for node in profile.nodes],
        sources=[indices[node] for node in profile.sources],
        bins=[asdict(row) for row in profile.bins],
        fit=asdict(profile.fit),
    )
    return data


def _rational_projection(value):
    return (
        None
        if value is None
        else {"numerator": value.numerator, "denominator": value.denominator}
    )


def _phase_evidence_projection(observation, indices):
    """Retain exact represented sums without coercing rational evidence to floats."""
    rows = []
    for row in observation.rows:
        resultant = row.resultant
        evidence = None
        if resultant is not None:
            evidence = {
                "count": resultant.count,
                "real_sum": _rational_projection(resultant.real_sum),
                "imag_sum": _rational_projection(resultant.imag_sum),
                "joint_zero": resultant.joint_zero,
                "scale": _rational_projection(resultant.scale),
                "scaled_exact": (
                    None
                    if resultant.scaled_exact is None
                    else [
                        _rational_projection(value) for value in resultant.scaled_exact
                    ]
                ),
                "scaled_binary64": resultant.scaled_binary64,
                "scaled_rounding_defect": (
                    None
                    if resultant.scaled_rounding_defect is None
                    else [
                        _rational_projection(value)
                        for value in resultant.scaled_rounding_defect
                    ]
                ),
                "angle": resultant.angle,
            }
        rows.append(
            {
                "node": indices[row.node],
                "neighbors": [indices[node] for node in row.neighbors],
                "gradient": row.gradient,
                "curvature": row.curvature,
                "status": row.status,
                "resultant": evidence,
            }
        )
    return {
        "version": observation.version,
        "nodes": [indices[node] for node in observation.nodes],
        "neighbor_order": [
            [indices[node] for node in neighbors]
            for neighbors in observation.neighbor_order
        ],
        "primitive_phases": observation.primitive_phases,
        "components": observation.components,
        "requested_precision_mode": observation.requested_precision_mode,
        "gradient_accumulator_dtype": observation.gradient_accumulator_dtype,
        "rows": rows,
        "scope": observation.scope,
    }


def _winding_projection(certificate, indices):
    """Snapshot indices preserve known identities; absent nodes remain explicit."""
    data = {
        item.name: getattr(certificate, item.name)
        for item in fields(certificate)
        if item.name != "cycle_nodes"
    }
    data["cycle_nodes"] = [indices.get(node) for node in certificate.cycle_nodes]
    data["cycle_labels"] = [str(node) for node in certificate.cycle_nodes]
    data["scope"] = (
        "Declared oriented cycle; null indices mean absent nodes and labels are display only. "
        "The winding owner's default branch tolerance and pi/2 phase gate apply; "
        "U3 metadata is separate from winding availability and does not resolve a runtime operator gate."
    )
    return data


def diagnose_network(
    network: Network,
    *,
    include_coherence_profile: bool = False,
    include_phase_evidence: bool = False,
    winding_cycles: Sequence[Sequence[Any]] = (),
) -> dict[str, Any]:
    """Observe stored state on a detached graph, preserving independent availability.

    Pressure is not refreshed or inferred from motion. Each failed observation
    retains its exception type/message; other fields remain measurable. Curvature
    uses the canonical represented-resultant observer and xi retains the existing
    fit/fallback provenance. No threshold or telemetry value controls evolution.

    Optional observations preserve their own scopes and failure states. A
    coherence profile reuses its fit for xi provenance; phase evidence reuses
    the tetrad's represented-resultant observation. Winding cycles are explicit
    ordered node sequences, never an inferred orientation. Flags require actual
    booleans. Opaque node IDs project to this snapshot's indices, while rational
    phase evidence retains integer numerator/denominator pairs. With no options,
    the established v1 report shape and computation path remain unchanged.
    """
    if not isinstance(network, Network):
        raise TypeError("diagnose_network requires an SDK Network")
    include_coherence_profile, include_phase_evidence, cycles = _observation_options(
        include_coherence_profile, include_phase_evidence, winding_cycles
    )
    graph = copy_graph_state(network.G)
    observed = Network(graph, name=network.name, seed=network._seed)
    nodes = tuple(graph)
    indices = {node: index for index, node in enumerate(nodes)}
    state = _state_projection(graph)

    def node_report(node):
        value = observed.nodal_state(node).to_dict()
        value["node"] = indices[node]
        return value

    # The phase evidence contains rational resultant carriers. Project only its
    # already-computed finite values/status, not an invented JSON phase kernel.
    phase_observation = None
    try:
        phase_observation = observe_phase_curvature(graph)
        gradient = _capture(lambda: [row.gradient for row in phase_observation.rows])
        curvature = _capture(lambda: [row.curvature for row in phase_observation.rows])
        curvature["node_status"] = [row.status for row in phase_observation.rows]
        curvature["complete"] = all(
            row.curvature is not None for row in phase_observation.rows
        )
        curvature["scope"] = list(phase_observation.scope)
        if not curvature["complete"]:
            curvature["available"] = False
            curvature["error"] = {
                "type": "UndefinedPhaseCurvatureError",
                "message": "Some nodes have an exactly zero represented neighbor resultant",
            }
    except Exception as error:
        failed = {
            "available": False,
            "value": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        gradient = dict(failed)
        curvature = {**failed, "complete": False}

    profile = None
    profile_error = None
    try:
        if include_coherence_profile:
            from ..physics.canonical import _coherence_length_estimate_from_fit
            from ..physics.fields import observe_coherence_profile

            profile = observe_coherence_profile(graph)
            fitted = profile.fit.value
            estimate = _coherence_length_estimate_from_fit(
                graph, fitted if fitted is not None else float("nan")
            )
        else:
            estimate = estimate_coherence_length_with_provenance(graph)
        metadata = asdict(estimate)
        value = metadata.pop("value")
        xi = _capture(lambda: value)
        xi["provenance"] = _json_value(metadata)
    except Exception as error:
        xi = {
            "available": False,
            "value": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        profile_error = xi

    def potential_read():
        potential = compute_structural_potential(graph)
        return [potential[node] for node in nodes]

    report = {
        "schema_version": "tnfr.diagnostics.v1",
        "name": str(network.name),
        "state": state,
        "metrics": {
            "coherence": _capture(observed.coherence),
            "sense_index": _capture(observed.sense_index),
            "avg_phase": _capture(observed.avg_phase),
        },
        "nodal": [
            {"index": indices[node], **_capture(lambda node=node: node_report(node))}
            for node in nodes
        ],
        "tetrad": {
            "phi_s": _capture(potential_read),
            "grad_phi": gradient,
            "k_phi": curvature,
            "xi_c": xi,
        },
        "scope": (
            "Read-only stored-pressure observations on a detached graph. Nodal rates "
            "are model predictions; derivative evidence and field availability retain "
            "their own contracts. No stability, autonomous-law or checkpoint claim."
        ),
    }
    if include_coherence_profile or include_phase_evidence or cycles:
        observations = {
            "request": {
                "include_coherence_profile": include_coherence_profile,
                "include_phase_evidence": include_phase_evidence,
                "winding_cycles": [
                    [indices.get(node) for node in cycle] for cycle in cycles
                ],
            },
            "scope": (
                "Optional task-dependent observations of the same detached stored state. "
                "Channels remain distinct from the historical tetrad and supply no "
                "universal sufficient state, closure or future prediction."
            ),
        }
        if include_coherence_profile:
            observations["coherence_profile"] = (
                _capture(lambda: _coherence_profile_projection(profile, indices))
                if profile is not None
                else dict(profile_error)
            )
        if include_phase_evidence:
            observations["phase_evidence"] = (
                _capture(lambda: _phase_evidence_projection(phase_observation, indices))
                if phase_observation is not None
                else dict(gradient)
            )
        if cycles:
            from ..physics.winding_certificates import certify_phase_winding

            winding = []
            for index, cycle in enumerate(cycles):
                try:
                    certificate = certify_phase_winding(graph, cycle)
                    observation = _capture(
                        lambda: _winding_projection(certificate, indices)
                    )
                    if not certificate.is_defined:
                        observation["available"] = False
                        observation["error"] = {
                            "type": "UndefinedPhaseWinding",
                            "message": certificate.reason,
                        }
                except Exception as error:
                    observation = {
                        "available": False,
                        "value": None,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                winding.append({"index": index, **observation})
            observations["winding_cycles"] = winding
        report["observations"] = observations
    return report


def run_study(spec: StudySpec) -> StudyResult:
    """Construct, execute and observe a declared finite study using current owners.

    Errors from construction, grammar or operators propagate. This wrapper adds
    no repair or selection policy. ``Network.evolve`` validates the complete
    word and rejects blocked steps without substituting another operator.
    The reported word is captured before dispatch; it is not a realized trace
    or a later reading of the mutable sequence registry.
    """
    from .. import __version__

    if not isinstance(spec, StudySpec):
        raise TypeError("run_study requires a validated StudySpec")
    # Revalidate frozen input even if a caller bypassed normal dataclass creation.
    spec = StudySpec.from_dict(spec.to_dict())
    network = TNFR.create(spec.nodes, name=spec.name, seed=spec.seed)
    network.G.graph["RANDOM_SEED"] = spec.seed
    if spec.topology == "random":
        network.random(spec.probability)
    else:
        getattr(network, spec.topology)()
    initial = _state_projection(network.G)
    declared_word = list_sequences(spec.sequence)
    network.evolve(steps=spec.cycles, sequence=spec.sequence)
    return StudyResult(
        {
            "schema_version": "tnfr.study.v1",
            "spec": spec.to_dict(),
            "execution": {
                "versions": {
                    "tnfr": __version__,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "networkx": nx.__version__,
                },
                "precision_mode": str(get_precision_mode()),
                "sequence": declared_word,
                "sequence_scope": "requested word; not a per-node realized operator trace",
                "cycles_completed": spec.cycles,
                "seed": spec.seed,
                "clock": "operator-word cycles; not elapsed physical time",
                "scope": (
                    "Finite declared operator protocol from EPI=0, nu_f=1, phase=0. "
                    "The recipe inherits engine configuration and installed versions; "
                    "same seed alone is not cross-environment replay or a native law."
                ),
            },
            "initial_state": initial,
            "final": diagnose_network(network),
        }
    )
