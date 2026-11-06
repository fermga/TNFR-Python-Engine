"""New complete-source arrival instruments, without automatic evaluation.

The phase differential inclusion retains the full A2 law through its source
bridge, conserved means, velocity cone and form reconstruction. Import/build
launches no source field, step, target assessment or trajectory. A separately
reviewed finite policy must authorize their evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, cos, sqrt
from . import formed_two_port_reference as reference_owner
from .artifact_io import exact_record

__all__ = (
    "WeightedMeanChart", "FormedTwoPortArrivalModel", "ArrivalSource",
    "ArrivalTarget", "ArrivalTerminal", "SignedGraphGrowth",
    "build_formed_two_port_arrival_model", "admit_formed_two_port_arrival_source",
    "admit_formed_two_port_arrival_target", "assess_formed_two_port_arrival_terminal",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _row(values, size, name):
    _require(isinstance(values, (tuple, list)) and len(values) == size,
             f"{name} requires {size} entries")
    return tuple(values)


def _rational(value, name):
    _require(type(value) is int or isinstance(value, Q), f"{name} must be exact")
    return Q(value)


@dataclass(frozen=True)
class WeightedMeanChart:
    """Exact last-node elimination; no componentwise phase reset or folding."""
    degrees: tuple[int, ...]

    def __post_init__(self):
        _require(isinstance(self.degrees, tuple) and 2 <= len(self.degrees) <= 18,
                 "chart requires two through eighteen nodes")
        _require(all(type(d) is int and d > 0 for d in self.degrees),
                 "chart degrees must be positive integers")

    @property
    def dimension(self):
        return len(self.degrees) - 1

    @property
    def mass(self):
        return sum(self.degrees)

    @property
    def basis(self):
        n = self.dimension
        return tuple(tuple(Q(i == j) for j in range(n)) for i in range(n)) + (
            tuple(-Q(d, self.degrees[-1]) for d in self.degrees[:-1]),)

    @property
    def metric(self):
        return tuple(tuple(Q(self.degrees[i] if i == j else 0)
                           + Q(self.degrees[i] * self.degrees[j], self.degrees[-1])
                           for j in range(self.dimension))
                     for i in range(self.dimension))

    @staticmethod
    def _values(values):
        if any(isinstance(x, Jet) for x in values):
            _require(all(isinstance(x, Jet) for x in values),
                     "chart does not mix Jets and scalars")
            _require(all(x.order == values[0].order for x in values),
                     "chart Jets require one common order")
            return values
        _require(all(isinstance(x, I) or type(x) is int or isinstance(x, Q)
                     for x in values), "chart scalars must be exact or interval")
        return tuple(Q(x) if type(x) is int else x for x in values)

    def lift(self, values):
        values = self._values(_row(values, self.dimension, "relative phase chart"))
        last = -sum(d * x for d, x in zip(self.degrees[:-1], values, strict=True))
        return values + (last / self.degrees[-1],)

    def project(self, values):
        values = self._values(_row(values, len(self.degrees), "full phase projection"))
        mean = sum(d * x for d, x in zip(self.degrees, values, strict=True)) / self.mass
        return tuple(x - mean for x in values[:-1])

    def pullback_flow(self, full_flow):
        def flow(values):
            full = _row(full_flow(self.lift(values)), len(self.degrees), "full field")
            return full[:-1]
        return flow


def _norm_upper(values, degrees):
    values = _row(values, len(degrees), "weighted norm")
    square = sum(d * max(abs(I.coerce(x).lo), abs(I.coerce(x).hi))**2
                 for d, x in zip(degrees, values, strict=True))
    return sqrt(I(square)).hi


@dataclass(frozen=True)
class SignedGraphGrowth:
    tube: tuple[I, ...]
    metric: tuple[tuple[Q, ...], ...]
    certified: bool
    growth_rate_upper_bound: Q | None
    rate_bounds: tuple[Q, Q]
    bisections_requested: int
    bisections_performed: int
    edge_cosine_lower_bounds: tuple[Q, ...]
    stiffness_lower_bound: tuple[tuple[Q, ...], ...]
    slack_matrix: tuple[tuple[Q, ...], ...]
    reason: str | None
    prescribed_time_interval: I | None = None


def _graph_growth(chart, edges, cosine_lower, tube, rate_bounds, bisections):
    """Preserve signed rank-one edge correlations in a relative LMI."""
    _require(type(bisections) is int and 0 <= bisections <= 16,
             "growth bisections must be between zero and sixteen")
    rates = tuple(_rational(x, "growth endpoint") for x in _row(rate_bounds, 2, "rates"))
    _require(rates[0] <= rates[1], "growth endpoints must be ordered")
    weights = tuple(_rational(x, "cosine lower bound")
                    for x in _row(cosine_lower, len(edges), "edge cosines"))
    _require(all(-1 <= x <= 1 for x in weights), "cosine bounds exceed [-1,1]")
    tube = tuple(I.coerce(x) for x in _row(tube, chart.dimension, "growth tube"))
    basis, metric, n = chart.basis, chart.metric, chart.dimension
    vectors = tuple(tuple(basis[j][k] - basis[i][k] for k in range(n)) for i, j in edges)
    lower = tuple(tuple(sum(w * row[i] * row[j]
                            for w, row in zip(weights, vectors, strict=True))
                        for j in range(n)) for i in range(n))

    def slack(rate):
        return tuple(tuple(rate * metric[i][j] + lower[i][j] for j in range(n))
                     for i in range(n))

    low, high = rates
    matrix = slack(high)
    admitted = exact_symmetric_semidefinite(matrix)
    performed = 0
    if admitted:
        low_matrix = slack(low)
        if exact_symmetric_semidefinite(low_matrix):
            high, matrix = low, low_matrix
        else:
            for _ in range(bisections):
                candidate = (low + high) / 2
                candidate_matrix = slack(candidate)
                performed += 1
                if exact_symmetric_semidefinite(candidate_matrix):
                    high, matrix = candidate, candidate_matrix
                else:
                    low = candidate
    return SignedGraphGrowth(
        tube, metric, admitted, high if admitted else None, rates, bisections,
        performed, weights, lower, matrix,
        None if admitted else "signed_graph_growth_upper_endpoint_not_certified")


@dataclass(frozen=True)
class FormedTwoPortArrivalModel:
    reference: reference_owner.FormedTwoPortReference
    chart: WeightedMeanChart

    @property
    def metric(self):
        return self.chart.metric

    def flow(self, values):
        return self.reference.field(self.chart.lift(values))[:-1]

    def domain(self, values):
        """All real continuous lifts; no acute/PSD or principal-cut admission."""
        tuple(I.coerce(x) for x in _row(values, self.chart.dimension, "phase tube"))
        return (Q(1),)

    def full_field(self, values):
        return self.reference.field(self.chart.lift(values))

    def field_norm_upper(self, tube):
        return _norm_upper(self.full_field(tube), self.chart.degrees)

    def forcing_bound(self, tube):
        return min(Q(1, 7000), self.field_norm_upper(tube) / 20000)

    def certify_growth(self, tube, rate_bounds=(Q(-1, 64), Q(2)), bisections=12):
        tube = tuple(I.coerce(x) for x in _row(tube, 17, "growth tube"))
        angles = self.reference.angles(self.chart.lift(tube))
        lower = tuple(max(Q(-1), cos(angle).lo) for angle in angles)
        return _graph_growth(self.chart, self.reference.oriented_edges, lower,
                             tube, rate_bounds, bisections)


def _require_model(model):
    _require(type(model) is FormedTwoPortArrivalModel, "arrival model required")
    reference_owner._require_fixed_reference(model.reference)
    _require(type(model.chart) is WeightedMeanChart
             and model.chart.degrees == model.reference.degrees,
             "different arrival mean chart")


def build_formed_two_port_arrival_model():
    reference = reference_owner.build_formed_two_port_reference()
    return FormedTwoPortArrivalModel(reference, WeightedMeanChart(reference.degrees))


@dataclass(frozen=True)
class ArrivalSource:
    time: Q
    center: tuple[Q, ...]
    radius: Q
    prefix_reference_radius: Q
    complete_source_bridge_radius: Q
    evidence_receipts: tuple[dict, ...]
    reference_center_weighted_mean: Q
    prefix_radius_premise_margin: Q
    scope: tuple[str, ...] = (
        "actual_original_family_at_time_one_not_a_reference_reset",
        "memberwise_means_restored_after_exact_relative_chart",
        "phase_ball_and_velocity_cone_retain_full_A2_state",
    )


def admit_formed_two_port_arrival_source(model, prefix_directory):
    _require_model(model)
    prefix = reference_owner.admit_formed_two_port_acute_exit_prefix(
        model.reference, prefix_directory)
    center18 = tuple(exact_record(x) for x in prefix.endpoint_center)
    radius = exact_record(prefix.endpoint_radius)
    _require(0 <= radius < Q(1, 2**26), "prefix radius exceeds retained source premise")
    bridge = Q(1, 45000)
    # PA2 gives the actual/reference phase discrepancy below this bridge.
    center = model.chart.project(center18)
    mean = sum(d*x for d, x in zip(model.chart.degrees, center18, strict=True)) / model.chart.mass
    return ArrivalSource(Q(1), center, radius + bridge, radius, bridge,
                         tuple(prefix.evidence_receipts), mean, Q(1, 2**26)-radius)


@dataclass(frozen=True)
class ArrivalTarget:
    retained_target: reference_owner.FormedTwoPortTarget
    energy_interval: I
    relative_deviations: tuple[I, ...]
    evidence_directory: str
    evidence_receipts: tuple[dict, ...]
    scope: tuple[str, ...] = (
        "one_retained_correlated_critical_target_not_a_box_of_equilibria",
        "target_receipts_must_remain_bound_to_the_reviewed_frozen_worker_inputs",
    )


def _target_geometry(model, retained):
    _require(type(retained) is reference_owner.FormedTwoPortTarget,
             "retained correlated target required")
    deviations = tuple(I.coerce(x) for x in _row(retained.target_deviations, 18,
                                                "target deviations"))
    # Projection preserves the one true target's mean; boxes remain outer bounds.
    relative = model.chart.lift(model.chart.project(deviations))
    angles = model.reference.angles(deviations)
    energy = sum((I(1) - cos(x) for x in angles), I(0))
    return relative, energy


def admit_formed_two_port_arrival_target(model, target_directory):
    _require_model(model)
    retained = reference_owner.admit_formed_two_port_target(model.reference,
                                                           target_directory)
    relative, energy = _target_geometry(model, retained)
    return ArrivalTarget(retained, energy, relative, str(Path(target_directory).resolve()),
                         tuple(retained.evidence_receipts))


@dataclass(frozen=True)
class ArrivalTerminal:
    status: str
    time: Q
    phase_distance_upper_bound: Q
    target_phase_radius: Q
    capture_form_upper_bound: Q
    capture_excess_upper_bound: Q
    capture_storage_margin: Q
    center_field_norm_upper_bound: Q
    phase_potential_upper_bound: Q
    form_storage_upper_bound: Q
    total_storage_upper_bound: Q
    target_storage_lower_bound: Q
    nonarrival_storage_margin: Q
    reasons: tuple[str, ...]
    scope: tuple[str, ...] = (
        "conditional_on_actual_source_enclosure_and_frozen_target_admission",
        "means_lifts_and_both_full_A2_rows_retained",
        "strict_energy_exclusion_rules_out_target_convergence_and_capture_basin",
        "unavailable_is_neither_arrival_nor_nonarrival",
    )


def assess_formed_two_port_arrival_terminal(model, target, center, radius, time):
    """Assess an admitted actual-source endpoint; never propagate or select it.

    The bounded worker must admit target primitives once and bind their receipts
    to its frozen inputs. Derived cached target bounds are rebuilt here; these
    dataclass labels do not authenticate provenance or establish the endpoint.
    """
    _require_model(model)
    _require(type(target) is ArrivalTarget, "admitted arrival target required")
    center = tuple(_rational(x, "terminal center") for x in _row(center, 17, "center"))
    radius, time = _rational(radius, "terminal radius"), _rational(time, "terminal time")
    _require(radius >= 0 and time >= 1, "terminal requires radius>=0 and time>=1")
    _require(tuple(target.retained_target.evidence_receipts) == target.evidence_receipts,
             "target receipt association differs")
    relative_target, target_energy = _target_geometry(model, target.retained_target)
    full_center = model.chart.lift(center)
    distance = _norm_upper(tuple(I(x) - y for x, y in zip(full_center, relative_target,
                                                         strict=True)),
                           model.chart.degrees) + radius
    field = model.full_field(center)
    field_upper = _norm_upper(field, model.chart.degrees)
    center_potential = sum((I(1) - cos(x)
                            for x in model.reference.angles(full_center)), I(0))
    # H_D<=A<=2I makes this correlated Taylor upper bound global in the lifts.
    phase_upper = center_potential.hi + field_upper*radius + radius**2
    return _terminal_arithmetic(time, distance, radius, field_upper, phase_upper,
                                target_energy.lo)


def _terminal_arithmetic(time, distance, radius, field_upper, phase_upper, target_lower):
    """Exact bound arithmetic; supplied bounds alone do not admit a source."""
    time, distance, radius, field_upper, phase_upper, target_lower = (
        _rational(x, "terminal bound") for x in
        (time, distance, radius, field_upper, phase_upper, target_lower))
    _require(time >= 1 and 0 <= radius <= distance and field_upper >= 0
             and phase_upper >= 0, "invalid terminal bounds")
    capture_form = Q(90, 3069) * (2*distance + Q(1, 7000))
    excess = capture_form**2 + distance**2
    capture_margin = Q(1, 648000) - excess
    capture = (distance <= Q(1, 1024) and capture_form < Q(1, 8192)
               and excess < Q(1, 144) and capture_margin > 0)
    actual_field_upper = field_upper + 2*radius
    lag = min(Q(1, 7000), actual_field_upper / 20000)
    velocity_upper = actual_field_upper + lag
    # eta<1/9000000, lambda>=1/90; x_rel=gamma A^-1 v retains form storage.
    form_storage = Q(1, 200000) * velocity_upper**2
    total_upper = phase_upper + form_storage
    nonarrival_margin = target_lower - total_upper
    excluded = nonarrival_margin > 0
    _require(not (capture and excluded), "contradictory endpoint certificates")
    status = "certified_capture" if capture else "certified_nonarrival" if excluded else "unavailable"
    reasons = () if capture or excluded else ("neither_terminal_inequality_certified",)
    return ArrivalTerminal(status, time, distance, Q(1, 1024), capture_form, excess,
                           capture_margin, field_upper, phase_upper, form_storage,
                           total_upper, target_lower, nonarrival_margin, reasons)
