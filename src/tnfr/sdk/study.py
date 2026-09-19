"""Declared finite operator studies and detached observations for SDK and CLI.

Recipes construct the existing scalar baseline and run registered operator words.
They are not arbitrary graph checkpoints, continuous-time solvers, or certificates
of autonomous pattern formation. Diagnostic values never select the next operator.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field, fields
from numbers import Integral, Real
from typing import Any, Callable

import networkx as nx
import numpy as np

from .._exact_time import finite_represented_real
from ..config import get_precision_mode
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..physics.fields import (
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)
from ..rng import validate_seed
from ..types import require_finite_real_scalar_epi
from ..utils.io import json_dumps
from ._state import copy_graph_state
from ._topology import nonnegative_integer, positive_integer, probability
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
        seed = validate_seed(self.seed, allow_none=False)
        if not 0 <= seed < 2**32:
            raise ValueError("seed must satisfy 0 <= seed < 2**32")
        object.__setattr__(self, "seed", seed)
        if isinstance(self.probability, bool):
            raise ValueError("probability must be a finite real number, not a boolean")
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
    The structural projection excludes history, callbacks, caches and graph
    metadata, so importing this report does not restore a runnable graph.
    """

    report: InitVar[Mapping[str, Any]]
    _payload_json: str = field(init=False, repr=False)

    def __post_init__(self, report: Mapping[str, Any]) -> None:
        if not isinstance(report, Mapping):
            raise TypeError("A study report must be a mapping")
        object.__setattr__(
            self, "_payload_json", json_dumps(dict(report), allow_nan=False)
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
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Diagnostic value is not finite")
        return result
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


def diagnose_network(network: Network) -> dict[str, Any]:
    """Observe stored state on a detached graph, preserving independent availability.

    Pressure is not refreshed or inferred from motion. Each failed observation
    retains its exception type/message; other fields remain measurable. Curvature
    uses the canonical represented-resultant observer and xi retains the existing
    fit/fallback provenance. No threshold or telemetry value controls evolution.
    """
    if not isinstance(network, Network):
        raise TypeError("diagnose_network requires an SDK Network")
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

    try:
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

    def potential_read():
        potential = compute_structural_potential(graph)
        return [potential[node] for node in nodes]

    return {
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


def run_study(spec: StudySpec) -> StudyResult:
    """Construct, execute and observe a declared finite study using current owners.

    Errors from construction, grammar or operators propagate. This wrapper adds
    no repair or selection policy. The engine's existing grammar policy, including
    permitted glyph substitutions, remains active inside ``Network.evolve``.
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
                "sequence": list_sequences(spec.sequence),
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
