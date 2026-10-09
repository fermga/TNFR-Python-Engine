"""Opt-in held-capacity relational exchange in declared phase chambers.

This conditional completion uses native two-channel pressure and a separately
admitted phase law. It does not change runtime dispatch. Evaluation is detached;
one explicit Euler step stages both rows and validates its endpoint before
committing. Graph clipping, controllers, histories and other pressure channels
are not part of this explicitly unclipped model. Continuous storage loss does
not guarantee that a finite Euler step decreases storage.

The default acute executor uses the shared signed atan2 phase chart. The
positive-resultant option certifies a sufficient right-half-plane chamber.
The regular option admits the existing slit-plane law through certified
distance from its excluded nonpositive-real ray. Both use exact represented
raw angles. Every mode mixes its pressure from the same captured relative
source that supplies its phase metric; ordinary pressure dispatch is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as Q
from numbers import Real
from typing import Any

import networkx as nx

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr, set_attr, set_dnfr, set_theta
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..gamma import DEFAULT_GAMMA, _uses_builtin_gamma
from ..mathematics._phase_resultant_chamber import (
    COSINE_ENCLOSURE_METHOD,
    REGULAR_RESULTANT_ENCLOSURE_METHOD,
    regular_resultant_margin,
    relative_resultant_bounds,
    relative_resultant_lower_bounds,
)
from ..mathematics.epi import BEPIElement
from ..types import require_finite_real_scalar_epi
from ..utils.graph import mark_dnfr_prep_dirty
from ..utils.numeric import angle_diff
from ._euler_kernel import euler_update
from .canonical import compute_canonical_nodal_derivative
from .dnfr import (
    _compute_delta_nfr_from_relative_sources,
    _resolve_dnfr_weights,
    _write_dnfr_metadata,
)

__all__ = (
    "RelationalExchangeModel",
    "RelationalWorkBalance",
    "RelationalExchangeField",
    "RelationalUniformTangent",
    "RelationalConsensusTangent",
    "RelationalExchangeStep",
    "evaluate_relational_exchange",
    "evaluate_relational_uniform_tangent",
    "evaluate_relational_consensus_tangent",
    "step_relational_exchange",
)

_SCOPE = (
    "conditional_capacity_separable_exchange; fixed simple connected unit support; "
    "held capacity; acute represented edge gaps; explicit unclipped Euler; "
    "graph clipping/controller/history settings are not consumed"
)
_POSITIVE_RESULTANT_SCOPE = (
    "conditional_capacity_separable_exchange; fixed simple connected unit support; "
    "held capacity; certified positive relative-resultant real parts from exact "
    "represented raw radians; rational cosine enclosures and sufficient "
    "whole straight Euler-segment Lipschitz admission; explicit unclipped Euler; "
    "graph clipping/controller/history settings are not consumed; "
    f"cosine_admission_method={COSINE_ENCLOSURE_METHOD}"
)
_REGULAR_SCOPE = (
    "conditional_capacity_separable_exchange; fixed simple connected unit support; "
    "held capacity; certified relative resultants outside the nonpositive-real ray; "
    "exact represented raw radians and whole straight Euler-segment admission; "
    "native relative phase pressure and positive metric use the same captured resultant; "
    "explicit unclipped Euler; no continuous-trajectory or global-existence certificate; "
    f"regular_admission_method={REGULAR_RESULTANT_ENCLOSURE_METHOD}"
)
_MISSING = object()


def _finite(value, label):
    return finite_represented_real(value, label)[0]


def _phase_edge_storage(gap):
    """Represented cosine cost after the caller admits its phase-gap chart."""
    half = _finite(Q(gap) / 2, "half phase gap")
    return 2 * Q(math.sin(half)) ** 2


@dataclass(frozen=True)
class RelationalExchangeModel:
    """Explicit storage ratio and native-normalized two-channel coefficients.

    ``storage_scale`` is the independently supplied positive beta. Coefficients
    are normalized once by the native owner; phase weight must stay positive.
    No graph pressure defaults are inherited by this model.

    ``phase_domain='acute'`` preserves the default represented acute-edge
    path. ``'positive_resultant'`` instead requires certified positive real
    parts of all relative neighbor resultants, a sufficient regular chamber
    of the same law. ``'regular'`` uses certified distance from the excluded
    nonpositive-real resultant ray; negative real parts with resolved nonzero
    imaginary parts can pass. Bounded numerical work can reject unresolved
    regular states; none of these modes repairs singularities or changes laws.
    """

    storage_scale: float
    epi_weight: float = 0.5
    phase_weight: float = 0.5
    phase_domain: str = "acute"

    def __post_init__(self):
        if not isinstance(self.phase_domain, str) or self.phase_domain not in (
            "acute",
            "positive_resultant",
            "regular",
        ):
            raise ValueError(
                "phase_domain must be 'acute', 'positive_resultant' or 'regular'"
            )
        beta = _finite(self.storage_scale, "storage_scale")
        epi = _finite(self.epi_weight, "epi_weight")
        phase = _finite(self.phase_weight, "phase_weight")
        if beta <= 0 or epi < 0 or phase <= 0:
            raise ValueError("require storage_scale>0, epi_weight>=0, phase_weight>0")
        graph = nx.Graph()
        graph.graph["DNFR_WEIGHTS"] = {
            "epi": epi,
            "phase": phase,
            "vf": 0.0,
            "topo": 0.0,
        }
        weights = _resolve_dnfr_weights(graph)
        normalized = tuple(_finite(weights[key], key) for key in ("epi", "phase"))
        if (epi and not normalized[0]) or not normalized[1]:
            raise ValueError("nonzero model weight is lost during normalization")
        object.__setattr__(self, "storage_scale", beta)
        object.__setattr__(self, "epi_weight", normalized[0])
        object.__setattr__(self, "phase_weight", normalized[1])

    @property
    def effective_weights(self) -> tuple[float, float]:
        """Return effective (EPI, phase) coefficients, in that order."""
        return self.epi_weight, self.phase_weight


@dataclass(frozen=True)
class RelationalWorkBalance:
    """Signed nodal storage work on the field's represented state and rates.

    All tuples follow ``RelationalExchangeField.nodes``. ``form_gradient``
    retains exact ``q=B*x`` before its separate float materialization. With
    degree ``d``, capacity ``nu``, phase source ``g`` and model coefficients
    ``e, w, beta``, the entries are ``D=e*nu*q**2/d``, ``J=w*nu*q*g``,
    ``F=q*xdot`` and ``P=beta*grad(V)*thetadot``. Positive exchange ``J``
    transfers storage toward form in the ideal differential balance.

    ``form_residual=F+D-J`` and ``phase_residual=P+J`` retain the actual
    represented arithmetic; their sum is the nodal balance residual. These
    gradient-work contributions sum to global storage work, but are not
    derivatives of independently assigned nodal storage densities. They
    neither alter rates nor certify a trajectory or physical energy law.
    """

    form_gradient: tuple[Q, ...]
    dissipation: tuple[Q, ...]
    exchange: tuple[Q, ...]
    form_work: tuple[Q, ...]
    phase_work: tuple[Q, ...]
    form_residual: tuple[Q, ...]
    phase_residual: tuple[Q, ...]
    balance_residual: tuple[Q, ...]


@dataclass(frozen=True)
class RelationalExchangeField:
    """Detached state, rates and actual represented-arithmetic work evidence.

    Fractions retain exact arithmetic on materialized scalars, except the
    explicitly certified ``resultant_real_lower_bounds``. Those optional bounds
    enclose mathematical cosine sums at exact represented raw angles; they do
    not bound errors in the separately materialized pressure or rates.
    ``pressure_split_residual`` is
    native pressure minus the separately evaluated relational split. The
    balance residual is measured; it is never assigned its ideal zero value.
    ``phase_storage`` is the unscaled cosine cost V; ``storage`` is E_D+beta*V.
    ``work`` holds signed nodal contributions from the same captured field.
    ``phase_mobility`` retains exact represented capacity divided by the
    captured phase metric. ``phase_rate_rounding_defect`` is the materialized
    phase rate minus ``(w/beta)*phase_mobility*(B*x)``. It measures rate
    rounding, not transcendental error in that metric. These optional defaults
    preserve older manually constructed reports; engine evaluation always
    supplies these observations. ``relative_resultant`` captures the same
    materialized cosine/sine sums used by the phase law, without a second
    trigonometric evaluation or an error enclosure for those sums.
    The regular mode additionally retains ``resultant_bounds`` for the ideal
    complex sums and ``resultant_regular_margin_lower_bounds`` for distance
    from the excluded ray. These certify domain admission, not rate accuracy.
    """

    model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    epi: tuple[float, ...]
    phase: tuple[float, ...]
    capacity: tuple[float, ...]
    pressure: tuple[float, ...]
    phase_source: tuple[float, ...]
    phase_metric: tuple[float, ...]
    form_gradient: tuple[float, ...]
    phase_gradient: tuple[float, ...]
    form_rate: tuple[float, ...]
    phase_rate: tuple[float, ...]
    form_storage: Q
    phase_storage: Q
    storage: Q
    continuous_loss: Q
    storage_rate: Q
    balance_residual: Q
    pressure_split_residual: tuple[Q, ...]
    nodal_rate_rounding_defect: tuple[Q, ...]
    pressure_path: str
    scope: str = _SCOPE
    clipping: str = "none"
    resultant_real_lower_bounds: tuple[Q, ...] | None = None
    work: RelationalWorkBalance | None = None
    phase_mobility: tuple[Q, ...] | None = None
    phase_rate_rounding_defect: tuple[Q, ...] | None = None
    relative_resultant: tuple[tuple[float, float], ...] | None = None
    resultant_bounds: tuple[tuple[tuple[Q, Q], tuple[Q, Q]], ...] | None = None
    resultant_regular_margin_lower_bounds: tuple[Q, ...] | None = None


@dataclass(frozen=True)
class _RelationalTangentObservation:
    """Shared detached joint-derivative representation and rounding evidence."""

    field: RelationalExchangeField
    generator: tuple[tuple[float, ...], ...]
    phase_source_jacobian: tuple[tuple[float, ...], ...]
    common_offset_residuals: tuple[tuple[Q, Q], ...]

    @property
    def phase_source_row_sum_residuals(self) -> tuple[Q, ...]:
        """Exact represented Dg row sums, without a neutrality projection."""
        return tuple(sum(map(Q, row), Q(0)) for row in self.phase_source_jacobian)


@dataclass(frozen=True)
class RelationalUniformTangent(_RelationalTangentObservation):
    """Detached materialized derivative of the ideal uniform-form field.

    ``generator`` acts on all form coordinates followed by all phase
    coordinates, each in ``field.nodes`` order. ``phase_source_jacobian``
    represents Dg, with trigonometry and resultants evaluated in the captured
    field's phase chart. This is not the derivative of binary64 arithmetic or
    a certified transcendental enclosure. Uniform form alone need not be an
    equilibrium: ``field`` retains its actual pressure and nonzero rates.

    ``common_offset_residuals`` has one pair per generator row: its exact
    represented sum over the form columns and over the phase columns. These
    rounding residuals are measured, never removed to force neutral modes.
    """

    scope: str = (
        "materialized_ideal_derivative_at_exactly_uniform_represented_form; "
        "held_support_capacity_and_model; inherited_phase_domain; "
        "baseline_rates_retained; no_equilibrium_or_trajectory_certificate"
    )


@dataclass(frozen=True)
class RelationalConsensusTangent(_RelationalTangentObservation):
    """Joint derivative at exact phase consensus with arbitrary signed form.

    The native inverse phase metric has zero first derivative at consensus,
    even when the supplied form gradient is nonzero. The full Jacobian thus
    has the same block structure as the uniform-form observer, under this
    separate premise. Its baseline field generally moves. This local
    derivative does not turn the nonlinear trajectory into a constant-matrix
    evolution or imply formation, stability, resonance or exact enclosure.
    """

    scope: tuple[str, ...] = (
        "one_detached_native_field_at_exact_raw_phase_consensus",
        "arbitrary_admitted_signed_form_and_held_nonnegative_capacity",
        "inverse_phase_metric_first_derivative_vanishes_at_consensus",
        "materialized_ideal_full_jacobian_not_derivative_of_binary64_arithmetic",
        "baseline_nonzero_rates_retained_without_equilibrium_projection",
        "common_offset_rounding_residuals_measured_not_forced_to_zero",
        "no_constant_generator_nonlinear_evolution_or_future_phase_consensus_claim",
        "no_formation_stability_resonance_solver_event_or_physical_identification",
    )


@dataclass(frozen=True)
class RelationalExchangeStep:
    """One admitted Euler step; defects are not convergence certificates.

    Optional ``segment_resultant_real_lower_bounds`` enclose the real relative
    resultants along the straight represented-endpoint phase proposal only.
    They do not enclose the continuous ODE trajectory or its numerical error.
    ``segment_resultant_regular_margin_lower_bounds`` instead encloses distance
    from the excluded ray in regular mode; unused domain evidence stays None.
    """

    before: RelationalExchangeField
    after: RelationalExchangeField
    dt: float
    t_before: float
    t_after: float
    epi_update_defect: tuple[Q, ...]
    phase_update_defect: tuple[Q, ...]
    clock_defect: Q
    energy_change: Q
    energy_step_defect: Q
    scope: str = _SCOPE
    segment_resultant_real_lower_bounds: tuple[Q, ...] | None = None
    segment_resultant_regular_margin_lower_bounds: tuple[Q, ...] | None = None


def _raw(data, aliases, label):
    value = get_attr(data, aliases, _MISSING, strict=True, conv=lambda raw: raw)
    if value is _MISSING:
        raise ValueError(f"missing consumed {label}")
    return value


def _epi(raw):
    if isinstance(raw, Real):
        return _finite(raw, "EPI")
    if not isinstance(raw, (BEPIElement, Mapping)):
        raise TypeError("EPI must be a real scalar or uniform-real BEPI representation")
    return _finite(require_finite_real_scalar_epi(raw), "EPI")


def _weights(model):
    return {
        "epi": model.epi_weight,
        "phase": model.phase_weight,
        "vf": 0.0,
        "topo": 0.0,
    }


def _relational_model_coefficients(model):
    """Admit authoritative stored coefficients without normalizing them again.

    Construction normalizes the supplied two-channel mix. Later consumers
    validate its current values before arithmetic or graph staging; a frozen
    dataclass or a reconstructed report does not replace scalar admission.
    Exact rational declarations retain their values for detached proof tools.
    Native execution keeps its additional represented-output checks.
    """
    if not isinstance(model, RelationalExchangeModel):
        raise TypeError("model must be a RelationalExchangeModel")
    if not isinstance(model.phase_domain, str) or model.phase_domain not in (
        "acute",
        "positive_resultant",
        "regular",
    ):
        raise ValueError(
            "phase_domain must be 'acute', 'positive_resultant' or 'regular'"
        )
    beta = exact_or_represented_real(model.storage_scale, "storage_scale")
    loss, exchange = (
        exact_or_represented_real(value, label)
        for value, label in zip(model.effective_weights, ("epi_weight", "phase_weight"))
    )
    if beta <= 0 or loss < 0 or exchange <= 0:
        raise ValueError("require positive storage/exchange and nonnegative form loss")
    return loss, exchange, beta


def _admit_graph(graph, model, *, require_connected=True):
    _relational_model_coefficients(model)
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError(
            "relational exchange requires simple undirected loopless support"
        )
    if len(graph) < 2 or (require_connected and not nx.is_connected(graph)):
        raise ValueError(
            "relational exchange requires connected support with at least two nodes"
        )
    if type(graph.graph) is not dict or any(
        type(data) is not dict for _, data in graph.nodes(data=True)
    ):
        raise TypeError(
            "relational exchange requires ordinary graph-owned attribute dictionaries"
        )
    if graph.graph.get("use_extended_dynamics", False) is not False:
        raise ValueError("the independent-pressure extension is outside this model")
    gamma = graph.graph.get("GAMMA")
    if gamma is None:
        gamma = DEFAULT_GAMMA
    if not isinstance(gamma, Mapping) or gamma.get("type", "none") != "none":
        raise ValueError("relational exchange admits only declared absent Gamma")
    if not _uses_builtin_gamma(gamma):
        raise ValueError("the live Gamma 'none' entry is not the built-in zero source")
    for _, _, data in graph.edges(data=True):
        if _finite(data.get("weight", 1.0), "conductance") != 1.0:
            raise ValueError("relational exchange requires unit conductances")


def _stage(graph, model):
    _admit_graph(graph, model)
    nodes = tuple(graph)
    staged = nx.Graph()
    for node in nodes:
        data = graph.nodes[node]
        epi = _epi(_raw(data, ALIAS_EPI, "EPI"))
        capacity = _finite(_raw(data, ALIAS_VF, "capacity"), "capacity")
        phase = _finite(_raw(data, ALIAS_THETA, "phase"), "phase")
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        staged.add_node(
            node, **{ALIAS_EPI[0]: epi, ALIAS_VF[0]: capacity, ALIAS_THETA[0]: phase}
        )
    staged.add_edges_from((a, b, {"weight": 1.0}) for a, b in graph.edges())
    staged.graph["DNFR_WEIGHTS"] = _weights(model)
    # The already-normalized model owns these effective coefficients; avoid a
    # second normalization changing their represented values.
    staged.graph["_dnfr_weights"] = _weights(model)
    return staged


def _relative_phase_gap(left, right, *, wider):
    """Shared represented phase chart for field and detached derivatives."""
    difference = _finite(Q(right) - Q(left), "phase difference")
    # Binary64 tau is not the period of sin/cos at large represented lifts.
    # Reuse the same signed phasor chart as the circular numerical owner.
    return difference if wider else angle_diff(difference, 0.0)


def _field(staged, model):
    nodes = tuple(staged)
    edges = tuple(staged.edges())
    indices = {node: i for i, node in enumerate(nodes)}
    epi = tuple(staged.nodes[node][ALIAS_EPI[0]] for node in nodes)
    phase = tuple(staged.nodes[node][ALIAS_THETA[0]] for node in nodes)
    capacity = tuple(staged.nodes[node][ALIAS_VF[0]] for node in nodes)
    xq = tuple(map(Q, epi))
    neighbors = tuple(tuple(indices[j] for j in staged[node]) for node in nodes)
    wider = model.phase_domain != "acute"
    regular = model.phase_domain == "regular"
    lower_bounds = (
        relative_resultant_lower_bounds(tuple(map(Q, phase)), neighbors)
        if model.phase_domain == "positive_resultant"
        else None
    )
    if lower_bounds is not None and any(value <= 0 for value in lower_bounds):
        raise ValueError(
            "every relative resultant requires a certified positive real part"
        )
    bounds = (
        relative_resultant_bounds(tuple(map(Q, phase)), neighbors) if regular else None
    )
    regular_margins = (
        tuple(regular_resultant_margin(*row) for row in bounds)
        if bounds is not None
        else None
    )
    if regular_margins is not None and any(value <= 0 for value in regular_margins):
        raise ValueError(
            "every relative resultant requires certified separation from the nonpositive-real ray"
        )
    gaps = {}
    form_storage = phase_storage = Q(0)
    for left, right in edges:
        i, j = indices[left], indices[right]
        gap = _relative_phase_gap(phase[i], phase[j], wider=wider)
        if not wider and abs(gap) >= math.pi / 2:
            raise ValueError(
                "every represented wrapped edge gap must be strictly acute"
            )
        gaps[i, j], gaps[j, i] = gap, -gap
        form_storage += (xq[i] - xq[j]) ** 2 / 2
        phase_storage += _phase_edge_storage(gap)
    q = tuple(
        sum((xq[i] - xq[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
    )
    gradient = tuple(_finite(value, "form gradient") for value in q)
    sources, metrics, phase_gradient, phase_rate = [], [], [], []
    phase_mobility, phase_defects = [], []
    resultants = []
    for i, row in enumerate(neighbors):
        cosine = math.fsum(math.cos(gaps[i, j]) for j in row)
        sine = math.fsum(math.sin(gaps[i, j]) for j in row)
        resultants.append((cosine, sine))
        if model.phase_domain == "positive_resultant" and cosine <= 0:
            raise ValueError(
                "the materialized relative resultant must have positive real part"
            )
        displacement = math.atan2(sine, cosine)
        source = _finite(Q(displacement) / Q(math.pi), "phase source")
        if regular:
            real_bounds, imaginary_bounds = bounds[i]
            for value, interval in ((cosine, real_bounds), (sine, imaginary_bounds)):
                if (interval[0] > 0 and value <= 0) or (interval[1] < 0 and value >= 0):
                    raise ValueError(
                        "materialized resultant disagrees with its certified component sign"
                    )
            if (
                (cosine <= 0 and sine == 0)
                or abs(displacement) >= math.pi
                or abs(source) >= 1
            ):
                raise ValueError(
                    "materialized resultant lies on or cannot resolve the nonpositive-real branch"
                )
            if sine == 0:
                metric = _finite(Q(math.pi) * Q(cosine), "phase metric")
            else:
                if displacement == 0:
                    raise ValueError("nonzero phase displacement underflows to zero")
                metric = _finite(Q(math.pi) * Q(sine) / Q(displacement), "phase metric")
        else:
            sinc = 1.0 if displacement == 0.0 else math.sin(displacement) / displacement
            metric = _finite(
                Q(math.pi) * Q(math.hypot(cosine, sine)) * Q(sinc), "phase metric"
            )
        if metric <= 0:
            raise ValueError("the represented phase metric must be positive")
        mobility = Q(capacity[i]) / Q(metric)
        exact_rate = Q(model.phase_weight) / Q(model.storage_scale) * mobility * q[i]
        rate = _finite(exact_rate, "phase rate")
        sources.append(source)
        metrics.append(metric)
        phase_gradient.append(-sine)
        phase_rate.append(rate)
        phase_mobility.append(mobility)
        phase_defects.append(Q(rate) - exact_rate)
    profile = {}
    _compute_delta_nfr_from_relative_sources(
        staged,
        nodes=nodes,
        phase=phase,
        neighbors=neighbors,
        phase_sources=tuple(sources),
        profile=profile,
    )
    pressure = tuple(
        _finite(_raw(staged.nodes[node], ALIAS_DNFR, "pressure"), "pressure")
        for node in nodes
    )
    rates, nodal_defects, split_defects = [], [], []
    for i, p in enumerate(pressure):
        expected = Q(capacity[i]) * Q(p)
        rate = compute_canonical_nodal_derivative(capacity[i], p).derivative
        _finite(expected, "nodal rate")  # Reject a lost nonzero product.
        if expected and not rate:
            raise ValueError("nonzero nodal rate underflows to zero")
        rates.append(rate)
        nodal_defects.append(Q(rate) - expected)
        split = -Q(model.epi_weight) * q[i] / len(neighbors[i]) + Q(
            model.phase_weight
        ) * Q(sources[i])
        split_defects.append(Q(p) - split)
    dissipation = tuple(
        Q(model.epi_weight) * Q(capacity[i]) * q[i] ** 2 / len(row)
        for i, row in enumerate(neighbors)
    )
    exchange = tuple(
        Q(model.phase_weight) * Q(capacity[i]) * q[i] * Q(sources[i])
        for i in range(len(nodes))
    )
    form_work = tuple(q[i] * Q(rates[i]) for i in range(len(nodes)))
    phase_work = tuple(
        Q(model.storage_scale) * Q(phase_gradient[i]) * Q(phase_rate[i])
        for i in range(len(nodes))
    )
    form_residual = tuple(
        form_work[i] + dissipation[i] - exchange[i] for i in range(len(nodes))
    )
    phase_residual = tuple(phase_work[i] + exchange[i] for i in range(len(nodes)))
    work = RelationalWorkBalance(
        q,
        dissipation,
        exchange,
        form_work,
        phase_work,
        form_residual,
        phase_residual,
        tuple(form_residual[i] + phase_residual[i] for i in range(len(nodes))),
    )
    loss = sum(work.dissipation, Q(0))
    storage_rate = sum(work.form_work, Q(0)) + sum(work.phase_work, Q(0))
    balance_residual = sum(work.balance_residual, Q(0))
    storage = form_storage + Q(model.storage_scale) * phase_storage
    return RelationalExchangeField(
        model,
        nodes,
        edges,
        epi,
        phase,
        capacity,
        pressure,
        tuple(sources),
        tuple(metrics),
        gradient,
        tuple(phase_gradient),
        tuple(rates),
        tuple(phase_rate),
        form_storage,
        phase_storage,
        storage,
        loss,
        storage_rate,
        balance_residual,
        tuple(split_defects),
        tuple(nodal_defects),
        str(profile.get("dnfr_path", "relative_resultant_canonical")),
        scope=(
            _REGULAR_SCOPE
            if regular
            else (_POSITIVE_RESULTANT_SCOPE if wider else _SCOPE)
        ),
        resultant_real_lower_bounds=lower_bounds,
        work=work,
        phase_mobility=tuple(phase_mobility),
        phase_rate_rounding_defect=tuple(phase_defects),
        relative_resultant=tuple(resultants),
        resultant_bounds=bounds,
        resultant_regular_margin_lower_bounds=regular_margins,
    )


def evaluate_relational_exchange(
    graph, *, model: RelationalExchangeModel
) -> RelationalExchangeField:
    """Read a detached native-pressure field without changing the live graph."""
    return _field(_stage(graph, model), model)


def evaluate_relational_uniform_tangent(
    graph, *, model: RelationalExchangeModel
) -> RelationalUniformTangent:
    """Materialize the ideal joint Jacobian at exactly uniform signed form.

    With B the unit-support Laplacian, D the degree diagonal and N held
    capacity, its blocks are (-e*N*D^-1*B, w*N*Dg; (w/beta)*N*H^-1*B, 0).
    Uniform form makes B*x exactly zero, so phase derivatives of H^-1
    contribute no lower-right block. No tolerance establishes this premise.

    For relative resultant R+i*S, an adjacent coefficient of Dg is
    (R*cos(delta)+S*sin(delta))/(pi*(R**2+S**2)); its diagonal is -1/pi.
    This is the phase-response owner's ideal Arg derivative, evaluated from
    the captured native resultants rather than a fabricated exact Gram matrix.
    Coefficient arithmetic is rational on materialized values before shared
    finite-real admission; underflow and overflow are not silently accepted.
    """
    field = evaluate_relational_exchange(graph, model=model)
    if any(value != field.epi[0] for value in field.epi):
        raise ValueError("uniform tangent requires exactly uniform represented EPI")
    return RelationalUniformTangent(
        field, *_joint_tangent_without_metric_derivative(field)
    )


def evaluate_relational_consensus_tangent(
    graph, *, model: RelationalExchangeModel
) -> RelationalConsensusTangent:
    """Observe the ideal joint Jacobian at exact initial phase consensus.

    Raw materialized phases must agree exactly; signed form may vary freely.
    At consensus H_i=pi*d_i and D(H_i^-1)=0, because the relative cosine
    sum has zero first derivative and S/atan(S/C) is smooth and even in S.
    Thus the phase row's derivative with respect to phase vanishes even
    when q=L*x is nonzero. With K=diag(nu/d), the blocks are
    (-e*K*L, -(w/pi)*K*L; (w/(beta*pi))*K*L, 0).

    The shared builder retains native coefficient materialization and actual
    row-sum defects. The captured baseline is not assumed stationary; only
    the instantaneous derivative is supplied. Capacity zero is admitted by
    the native owner and freezes its two rows, without a positive-capacity
    spectral or formation conclusion. No graph state or law is changed.
    """
    field = evaluate_relational_exchange(graph, model=model)
    if any(value != field.phase[0] for value in field.phase):
        raise ValueError(
            "consensus tangent requires exactly equal represented raw phases"
        )
    return RelationalConsensusTangent(
        field, *_joint_tangent_without_metric_derivative(field)
    )


def _joint_tangent_without_metric_derivative(field):
    """Materialize shared blocks after a public owner proves the zero block.

    Either uniform form makes q*D(H^-1) vanish, or exact phase consensus
    makes D(H^-1) vanish. This kernel does not admit that missing premise.
    """
    if field.relative_resultant is None or field.phase_mobility is None:
        raise RuntimeError("native field omitted its captured phase coefficients")
    model = field.model
    size = len(field.nodes)
    indices = {node: i for i, node in enumerate(field.nodes)}
    neighbors = [[] for _ in range(size)]
    for left, right in field.edges:
        i, j = indices[left], indices[right]
        neighbors[i].append(j)
        neighbors[j].append(i)
    pi = Q(math.pi)
    wider = model.phase_domain != "acute"
    source_rows, upper_rows, lower_rows = [], [], []
    for i, row in enumerate(neighbors):
        real, imaginary = map(Q, field.relative_resultant[i])
        denominator = pi * (real**2 + imaginary**2)
        source = [Q(0)] * size
        source[i] = -1 / pi
        for j in row:
            gap = _relative_phase_gap(field.phase[i], field.phase[j], wider=wider)
            source[j] = (
                real * Q(math.cos(gap)) + imaginary * Q(math.sin(gap))
            ) / denominator
        materialized_source = tuple(
            _finite(value, "phase source derivative") for value in source
        )
        source_rows.append(materialized_source)
        laplacian = [0] * size
        laplacian[i] = len(row)
        for j in row:
            laplacian[j] = -1
        capacity = Q(field.capacity[i])
        upper_rows.append(
            tuple(
                _finite(
                    -Q(model.epi_weight) * capacity * value / len(row), "form tangent"
                )
                for value in laplacian
            )
            + tuple(
                _finite(
                    Q(model.phase_weight) * capacity * Q(value), "phase-to-form tangent"
                )
                for value in materialized_source
            )
        )
        lower_rows.append(
            tuple(
                _finite(
                    Q(model.phase_weight)
                    / Q(model.storage_scale)
                    * field.phase_mobility[i]
                    * value,
                    "form-to-phase tangent",
                )
                for value in laplacian
            )
            + (0.0,) * size
        )
    generator = tuple(upper_rows + lower_rows)
    residuals = tuple(
        (sum(map(Q, row[:size]), Q(0)), sum(map(Q, row[size:]), Q(0)))
        for row in generator
    )
    return generator, tuple(source_rows), residuals


def _advance(value, rate, dt, label):
    increment = Q(dt) * Q(rate)
    _finite(increment, f"{label} increment")
    updated = _finite(euler_update(value, dt, rate), f"{label} endpoint")
    exact = Q(value) + increment
    if exact and not updated:
        raise ValueError(f"nonzero {label} endpoint underflows to zero")
    return updated, Q(updated) - exact


def _admit_phase_segment(before, staged):
    """Certify the proposal in the selected chamber without evolving it."""
    indices = {node: i for i, node in enumerate(before.nodes)}
    increments = tuple(
        Q(staged.nodes[node][ALIAS_THETA[0]]) - Q(before.phase[i])
        for i, node in enumerate(before.nodes)
    )
    if before.model.phase_domain in ("positive_resultant", "regular"):
        regular = before.model.phase_domain == "regular"
        lower_bounds = (
            before.resultant_regular_margin_lower_bounds
            if regular
            else before.resultant_real_lower_bounds
        )
        if lower_bounds is None:
            raise ValueError("resultant segment requires initial certified bounds")
        segment_bounds = tuple(
            lower_bounds[i]
            - sum(
                (
                    abs(increments[indices[other]] - increments[i])
                    for other in staged[node]
                ),
                Q(0),
            )
            for i, node in enumerate(before.nodes)
        )
        if any(value <= 0 for value in segment_bounds):
            raise ValueError(
                "the whole Euler phase segment requires certified separation from the nonpositive-real ray"
                if regular
                else "the whole Euler phase segment requires certified positive resultant real parts"
            )
        return segment_bounds
    for left, right in before.edges:
        i, j = indices[left], indices[right]
        gap = Q(_relative_phase_gap(before.phase[i], before.phase[j], wider=False))
        endpoint = gap + increments[j] - increments[i]
        if abs(endpoint) >= Q(math.pi) / 2:
            raise ValueError(
                "the Euler phase segment must remain on its initial acute lift"
            )
    return None


def step_relational_exchange(
    graph, *, model: RelationalExchangeModel, dt, t=None
) -> RelationalExchangeStep:
    """Atomically commit one unclipped simultaneous Euler step.

    Time uses the explicit ``t`` or graph ``_t`` (default zero); ``dt`` must
    be strictly positive and advance its represented endpoint. Phases retain
    their real lifts. The default Euler proposal remains on its initial acute
    relative lift. The wider options certify the full straight proposal using
    initial real-part or slit-distance bounds and exact increment differences.
    Endpoint admission alone cannot skip a chart boundary.
    Fresh endpoint pressure/rate and the structural clock are committed;
    stale acceleration is removed and no temporal history is manufactured.
    """
    staged = _stage(graph, model)
    duration = _finite(dt, "dt")
    if duration <= 0:
        raise ValueError("dt must be strictly positive")
    clock = _finite(graph.graph.get("_t", 0.0) if t is None else t, "t")
    next_clock = _finite(euler_update(clock, duration, 1.0), "clock endpoint")
    if next_clock <= clock:
        raise ValueError("dt must advance the represented clock")
    before = _field(staged, model)
    epi_defects, phase_defects = [], []
    for i, node in enumerate(before.nodes):
        epi, epi_defect = _advance(before.epi[i], before.form_rate[i], duration, "EPI")
        phase, phase_defect = _advance(
            before.phase[i], before.phase_rate[i], duration, "phase"
        )
        staged.nodes[node][ALIAS_EPI[0]] = epi
        staged.nodes[node][ALIAS_THETA[0]] = phase
        epi_defects.append(epi_defect)
        phase_defects.append(phase_defect)
    segment_bounds = _admit_phase_segment(before, staged)
    after = _field(_stage(staged, model), model)
    change = after.storage - before.storage
    report = RelationalExchangeStep(
        before,
        after,
        duration,
        clock,
        next_clock,
        tuple(epi_defects),
        tuple(phase_defects),
        Q(next_clock) - Q(clock) - Q(duration),
        change,
        change - Q(duration) * before.storage_rate,
        scope=before.scope,
        segment_resultant_real_lower_bounds=(
            segment_bounds if model.phase_domain == "positive_resultant" else None
        ),
        segment_resultant_regular_margin_lower_bounds=(
            segment_bounds if model.phase_domain == "regular" else None
        ),
    )
    # Execute cache hooks on detached ordinary mappings too, so even malformed
    # old metadata cannot cause a partially committed live graph.
    commit = nx.Graph()
    commit.graph.update(graph.graph)
    commit.graph.pop("_dnfrmax", None)
    commit.graph.pop("_dnfrmax_node", None)
    commit.add_nodes_from((node, dict(graph.nodes[node])) for node in before.nodes)
    for i, node in enumerate(before.nodes):
        # Match the shared scalar nodal integrator: a uniform-real BEPI is
        # admitted by its signed chart, then committed as that real scalar.
        # Existing alias spelling and independent EPI-kind metadata survive.
        set_attr(commit.nodes[node], ALIAS_EPI, after.epi[i])
        set_theta(commit, node, after.phase[i])
        set_dnfr(commit, node, after.pressure[i])
        set_attr(commit.nodes[node], ALIAS_DEPI, after.form_rate[i])
        for key in ALIAS_D2EPI:
            commit.nodes[node].pop(key, None)
    mark_dnfr_prep_dirty(commit)
    _write_dnfr_metadata(
        commit,
        weights=_weights(model),
        hook_name="relative_resultant_canonical",
        note="fresh endpoint of opt-in relational exchange; ambient pressure defaults unchanged",
    )
    commit.graph["_t"] = next_clock
    commit.graph["_relational_exchange"] = report
    for node in before.nodes:
        graph.nodes[node].clear()
        graph.nodes[node].update(commit.nodes[node])
    graph.graph.clear()
    graph.graph.update(commit.graph)
    return report
