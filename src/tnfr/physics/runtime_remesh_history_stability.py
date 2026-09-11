r"""Exact bridge from one executed REMESH cycle to its companion model.

The exact companion theorem in :mod:`tnfr.physics.remesh_history_stability`
acts on a newest-first augmented history.  The runtime stores history
oldest-first, evaluates the affine head update in binary64, and then applies
the configured structural clipping map.  This module binds those two views
for one sealed :class:`~tnfr.operators.event_remesh_runtime.EventRemeshCycleResult`.

For the ideal companion head ``y``, runtime raw head ``b`` and committed
bounded head ``z``, the bridge retains the signed exact decomposition

    z = y + r_round + r_clip,

where every value is the rational represented by its binary64 input or output.
It also lifts ``b`` and ``z`` into hypothetical companion histories and gives
exact one-step augmented-energy balances.  The runtime does not append the
post-REMESH head immediately, so these lifted histories are observations, not
claims about the live ``_epi_hist`` or future repeated execution.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from ..dynamics.structural_clip import structural_clip
from ..errors import TNFRValueError
from ..operators.event_remesh_runtime import EventRemeshCycleResult, _proof_value
from ..utils._structural_signature import (
    binary64_vectors_are_identical,
    proof_stamps_are_identical,
)
from .remesh_history_stability import (
    ExactHistory,
    ExactVector,
    UniformRemeshHistoryTransitionObservation,
    _centered_energy,
    _weighted_history_field,
    certify_uniform_remesh_history_stability,
    observe_uniform_remesh_history_transition,
)

__all__ = (
    "RuntimeRemeshHistoryBridgeObservation",
    "observe_runtime_remesh_history_bridge",
)

_PROOF_VERSION = "runtime_remesh_history_bridge_v1"
_SCOPE = (
    "One executor-sealed applied REMESH transition, identified against the "
    "fixed-support uniform exact companion built from the same represented "
    "alpha, delays, history window and positive diagonal metric. Signed exact "
    "rounding and clipping residuals give lifted raw and bounded augmented-"
    "energy balances and a posteriori sufficient one-step lower bounds. The "
    "post-REMESH "
    "head is not immediately appended to the live runtime history. Repeated "
    "runtime stability, companion temporal convergence transfer, changing "
    "support/metric/parameters, solver accuracy, schedule-times-REMESH gain, "
    "full tetrad Lyapunov decrease and future behavior are not certified."
)


def _compact_proof_value(value: Any) -> Any:
    """Represent nested sealed records by their immutable proof stamps."""

    known_types = (
        EventRemeshCycleResult,
        UniformRemeshHistoryTransitionObservation,
        RuntimeRemeshHistoryBridgeObservation,
    )
    if type(value) in known_types:
        try:
            stamp = object.__getattribute__(value, "_proof_stamp")
        except BaseException:
            stamp = None
        if type(stamp) is not tuple:
            stamp = None
        return (
            "tnfr-nested-proof-stamp-v1",
            type(value).__module__,
            type(value).__qualname__,
            stamp,
        )
    if type(value) is tuple:
        return ("tuple", tuple(_compact_proof_value(item) for item in value))
    return _proof_value(value)


def _proof_stamp(
    value: "RuntimeRemeshHistoryBridgeObservation",
) -> tuple[Any, ...]:
    if type(value) is not RuntimeRemeshHistoryBridgeObservation:
        raise TypeError("proof value must have its canonical result type")
    payload = tuple(
        (item.name, object.__getattribute__(value, item.name))
        for item in fields(RuntimeRemeshHistoryBridgeObservation)
        if item.name != "_proof_stamp"
    )
    return (_PROOF_VERSION, _compact_proof_value(payload))


def _seal(
    value: "RuntimeRemeshHistoryBridgeObservation",
) -> "RuntimeRemeshHistoryBridgeObservation":
    return replace(value, _proof_stamp=_proof_stamp(value))


def _sealed(value: Any) -> bool:
    try:
        expected = _proof_stamp(value)
        observed = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return False
    return proof_stamps_are_identical(observed, expected)


def _strict_exact_vector(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == width
        and all(type(item) is Fraction for item in value)
    )


def _strict_exact_history(value: Any, width: int) -> bool:
    return bool(
        type(value) is tuple
        and value
        and all(_strict_exact_vector(row, width) for row in value)
    )


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            for item in value
        )
    )


def _ordered_node_identity(
    left: tuple[Hashable, ...],
    right: tuple[Hashable, ...],
) -> bool:
    return bool(
        type(left) is tuple
        and type(right) is tuple
        and len(left) == len(right)
        and all(a is b for a, b in zip(left, right, strict=True))
    )


def _exact_transition_matches(
    observed: UniformRemeshHistoryTransitionObservation,
    expected: UniformRemeshHistoryTransitionObservation,
) -> bool:
    """Compare exact transition semantics without object-identity seal noise."""

    if type(observed) is not UniformRemeshHistoryTransitionObservation:
        return False
    if not observed.transition_observation_certified:
        return False
    if not _ordered_node_identity(observed.nodes, expected.nodes):
        return False
    observed_certificate = observed.certificate
    expected_certificate = expected.certificate
    certificate_names = tuple(
        item.name
        for item in fields(type(expected_certificate))
        if item.name != "_proof_stamp"
    )
    transition_names = tuple(
        item.name
        for item in fields(UniformRemeshHistoryTransitionObservation)
        if item.name not in {"certificate", "nodes", "_proof_stamp"}
    )
    return bool(
        all(
            proof_stamps_are_identical(
                _proof_value(
                    object.__getattribute__(observed_certificate, name)
                ),
                _proof_value(
                    object.__getattribute__(expected_certificate, name)
                ),
            )
            for name in certificate_names
        )
        and all(
            proof_stamps_are_identical(
                _proof_value(object.__getattribute__(observed, name)),
                _proof_value(object.__getattribute__(expected, name)),
            )
            for name in transition_names
        )
    )


def _vector_subtract(left: ExactVector, right: ExactVector) -> ExactVector:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _vector_add(left: ExactVector, right: ExactVector) -> ExactVector:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _scale_vector(scale: Fraction, value: ExactVector) -> ExactVector:
    return tuple(scale * item for item in value)


def _max_abs(value: ExactVector) -> Fraction:
    return max((abs(item) for item in value), default=Fraction(0))


def _energy_defect_bound(
    base_centered: ExactVector,
    signed_error: ExactVector,
    metric: ExactVector,
) -> Fraction:
    """Bound ``|E(base + error) - E(base)|`` in one fixed metric."""

    error_centered, error_energy = _centered_energy(signed_error, metric)
    cross_bound = sum(
        (
            weight * abs(base_value * error_value)
            for weight, base_value, error_value in zip(
                metric,
                base_centered,
                error_centered,
                strict=True,
            )
        ),
        Fraction(0),
    )
    return cross_bound + error_energy


@dataclass(frozen=True, slots=True)
class RuntimeRemeshHistoryBridgeObservation:
    """One exact runtime/companion identification and perturbation balance."""

    cycle_result: EventRemeshCycleResult = field(repr=False)
    exact_transition: UniformRemeshHistoryTransitionObservation = field(
        repr=False
    )
    nodes: tuple[Hashable, ...]
    exact_metric_weights: ExactVector
    exact_history: ExactHistory
    exact_ideal_next_field: ExactVector
    exact_runtime_raw_next_field: ExactVector
    exact_runtime_bounded_next_field: ExactVector
    exact_rounding_residual: ExactVector
    exact_clipping_residual: ExactVector
    exact_total_residual: ExactVector
    exact_max_abs_rounding_residual: Fraction
    exact_max_abs_clipping_residual: Fraction
    exact_max_abs_total_residual: Fraction
    exact_runtime_raw_next_centered_field: ExactVector
    exact_runtime_bounded_next_centered_field: ExactVector
    exact_runtime_raw_next_energy: Fraction
    exact_runtime_bounded_next_energy: Fraction
    exact_lifted_augmented_energy_before: Fraction
    exact_lifted_ideal_augmented_energy_after: Fraction
    exact_lifted_runtime_raw_augmented_energy_after: Fraction
    exact_lifted_runtime_bounded_augmented_energy_after: Fraction
    exact_jensen_dissipation: Fraction
    exact_rounding_augmented_energy_defect: Fraction
    exact_clipping_augmented_energy_defect: Fraction
    exact_total_augmented_energy_defect: Fraction
    exact_lifted_runtime_raw_energy_drop: Fraction
    exact_lifted_runtime_bounded_energy_drop: Fraction
    exact_rounding_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_clipping_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_total_augmented_energy_defect_absolute_upper_bound: Fraction
    exact_lifted_runtime_raw_energy_drop_lower_bound: Fraction
    exact_lifted_runtime_bounded_energy_drop_lower_bound: Fraction
    exact_lifted_runtime_raw_barycenter_drift: ExactVector
    exact_lifted_runtime_bounded_barycenter_drift: ExactVector
    clipping_intervened: bool
    raw_binary64_replay_identified: bool
    bounded_binary64_replay_identified: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        """Fail closed after mutation of this record or either nested proof."""

        try:
            if not _sealed(self):
                return False
            cycle = self.cycle_result
            transition = self.exact_transition
            if type(cycle) is not EventRemeshCycleResult or type(
                transition
            ) is not UniformRemeshHistoryTransitionObservation:
                return False
            if not EventRemeshCycleResult._proof_fields_are_intact(cycle):
                return False
            if not transition.transition_observation_certified:
                return False
            width = len(self.nodes)
            exact_vectors = (
                self.exact_metric_weights,
                self.exact_ideal_next_field,
                self.exact_runtime_raw_next_field,
                self.exact_runtime_bounded_next_field,
                self.exact_rounding_residual,
                self.exact_clipping_residual,
                self.exact_total_residual,
                self.exact_runtime_raw_next_centered_field,
                self.exact_runtime_bounded_next_centered_field,
                self.exact_lifted_runtime_raw_barycenter_drift,
                self.exact_lifted_runtime_bounded_barycenter_drift,
            )
            exact_scalars = (
                self.exact_max_abs_rounding_residual,
                self.exact_max_abs_clipping_residual,
                self.exact_max_abs_total_residual,
                self.exact_runtime_raw_next_energy,
                self.exact_runtime_bounded_next_energy,
                self.exact_lifted_augmented_energy_before,
                self.exact_lifted_ideal_augmented_energy_after,
                self.exact_lifted_runtime_raw_augmented_energy_after,
                self.exact_lifted_runtime_bounded_augmented_energy_after,
                self.exact_jensen_dissipation,
                self.exact_rounding_augmented_energy_defect,
                self.exact_clipping_augmented_energy_defect,
                self.exact_total_augmented_energy_defect,
                self.exact_lifted_runtime_raw_energy_drop,
                self.exact_lifted_runtime_bounded_energy_drop,
                self.exact_rounding_augmented_energy_defect_absolute_upper_bound,
                self.exact_clipping_augmented_energy_defect_absolute_upper_bound,
                self.exact_total_augmented_energy_defect_absolute_upper_bound,
                self.exact_lifted_runtime_raw_energy_drop_lower_bound,
                self.exact_lifted_runtime_bounded_energy_drop_lower_bound,
            )
            local_shape_is_valid = bool(
                width > 0
                and _ordered_node_identity(self.nodes, cycle.target_nodes)
                and _ordered_node_identity(self.nodes, transition.nodes)
                and _strict_exact_history(self.exact_history, width)
                and all(_strict_exact_vector(item, width) for item in exact_vectors)
                and all(type(item) is Fraction for item in exact_scalars)
                and all(weight > 0 for weight in self.exact_metric_weights)
                and type(self.clipping_intervened) is bool
                and type(self.raw_binary64_replay_identified) is bool
                and type(self.bounded_binary64_replay_identified) is bool
                and _strict_conditions(self.conditions)
                and all(passed for _, passed in self.conditions)
            )
            if not local_shape_is_valid:
                return False
            expected = _build_runtime_remesh_history_bridge(
                cycle,
                verify_result=False,
                exact_transition_override=transition,
            )
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                object.__getattribute__(expected, "_proof_stamp"),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def bridge_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_remesh_history_bridge_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def lifted_runtime_raw_energy_nonincrease_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_lifted_runtime_raw_energy_drop >= 0
        )

    @property
    def lifted_runtime_bounded_energy_nonincrease_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_lifted_runtime_bounded_energy_drop >= 0
        )

    @property
    def lifted_runtime_raw_energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_lifted_runtime_raw_energy_drop_lower_bound >= 0
        )

    @property
    def lifted_runtime_bounded_energy_nonincrease_sufficiently_certified(
        self,
    ) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_lifted_runtime_bounded_energy_drop_lower_bound >= 0
        )

    @property
    def hard_clipping_step_nonexpansive_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.cycle_result.remesh.plan.clip_mode == "hard"
            and self.exact_clipping_augmented_energy_defect <= 0
        )

    @property
    def lifted_runtime_barycenter_preserved_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and all(
                value == 0
                for value in self.exact_lifted_runtime_bounded_barycenter_drift
            )
        )

    @property
    def live_runtime_history_advance_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def companion_temporal_convergence_transferred_to_runtime(self) -> bool:
        return False

    @property
    def schedule_remesh_composition_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False


def _build_runtime_remesh_history_bridge(
    cycle_result: EventRemeshCycleResult,
    *,
    verify_result: bool,
    exact_transition_override: (
        UniformRemeshHistoryTransitionObservation | None
    ) = None,
) -> RuntimeRemeshHistoryBridgeObservation:
    if type(cycle_result) is not EventRemeshCycleResult:
        raise TypeError("cycle_result must be an EventRemeshCycleResult")
    if not EventRemeshCycleResult._proof_fields_are_intact(cycle_result):
        raise TNFRValueError("cycle_result is unsealed, tampered, or inconsistent")
    if not cycle_result.remesh.applied:
        raise TNFRValueError("cycle_result must contain an applied REMESH map")

    plan = cycle_result.remesh.plan
    history_transition = cycle_result.history_transition
    certificate = certify_uniform_remesh_history_stability(
        alpha=plan.alpha,
        tau_local=plan.tau_local,
        tau_global=plan.tau_global,
    )
    required = certificate.active_max_delay + 1
    outgoing = history_transition.outgoing_exact_history
    if len(outgoing) < required:
        raise TNFRValueError(
            "cycle_result does not retain the required exact REMESH history window"
        )
    exact_history = tuple(reversed(outgoing[-required:]))
    canonical_transition = observe_uniform_remesh_history_transition(
        certificate,
        exact_history,
        cycle_result.metric_weights,
        nodes=cycle_result.target_nodes,
    )
    if exact_transition_override is not None:
        if not _exact_transition_matches(
            exact_transition_override,
            canonical_transition,
        ):
            raise TNFRValueError(
                "exact_transition does not match the canonical cycle bridge"
            )
        exact_transition = exact_transition_override
    else:
        exact_transition = canonical_transition

    selected_local = history_transition.selected_local_delayed_epi
    selected_global = history_transition.selected_global_delayed_epi
    selected_inputs_match = bool(
        selected_local == outgoing[-(plan.tau_local + 1)]
        and selected_global == outgoing[-(plan.tau_global + 1)]
        and (
            certificate.gamma == 0
            or exact_history[plan.tau_local] == selected_local
        )
        and (
            certificate.delta == 0
            or exact_history[plan.tau_global] == selected_global
        )
    )
    if not selected_inputs_match:
        raise TNFRValueError(
            "cycle delayed inputs do not match the newest-first companion window"
        )

    proposals = plan.proposals
    runtime_raw_floats = tuple(proposal.raw_epi for proposal in proposals)
    runtime_bounded_floats = tuple(
        proposal.bounded_epi for proposal in proposals
    )
    replayed_raw = tuple(
        (1.0 - plan.alpha)
        * (
            (1.0 - plan.alpha) * proposal.epi_now
            + plan.alpha * proposal.epi_local
        )
        + plan.alpha * proposal.epi_global
        for proposal in proposals
    )
    replayed_bounded = tuple(
        structural_clip(
            proposal.raw_epi,
            lo=plan.epi_min,
            hi=plan.epi_max,
            mode=plan.clip_mode,
            record_stats=False,
        )
        for proposal in proposals
    )
    raw_replay_identified = binary64_vectors_are_identical(
        replayed_raw,
        runtime_raw_floats,
    )
    bounded_replay_identified = binary64_vectors_are_identical(
        replayed_bounded,
        runtime_bounded_floats,
    )
    if not raw_replay_identified or not bounded_replay_identified:
        raise TNFRValueError(
            "cycle REMESH proposals do not replay the canonical binary64 map"
        )

    ideal = exact_transition.exact_next_field
    raw = tuple(Fraction.from_float(value) for value in runtime_raw_floats)
    bounded = tuple(
        Fraction.from_float(value) for value in runtime_bounded_floats
    )
    rounding = _vector_subtract(raw, ideal)
    clipping = _vector_subtract(bounded, raw)
    total = _vector_subtract(bounded, ideal)

    metric = exact_transition.exact_metric_weights
    raw_centered, raw_energy = _centered_energy(raw, metric)
    bounded_centered, bounded_energy = _centered_energy(bounded, metric)
    pi_zero = certificate.stationary_distribution[0]
    tail_energy = sum(
        (
            weight * energy
            for weight, energy in zip(
                certificate.stationary_distribution[1:],
                exact_transition.exact_history_energies[:-1],
                strict=True,
            )
        ),
        Fraction(0),
    )
    before = exact_transition.exact_augmented_energy_before
    ideal_after = exact_transition.exact_augmented_energy_after
    raw_after = pi_zero * raw_energy + tail_energy
    bounded_after = pi_zero * bounded_energy + tail_energy
    rounding_defect = raw_after - ideal_after
    clipping_defect = bounded_after - raw_after
    total_defect = bounded_after - ideal_after
    raw_drop = before - raw_after
    bounded_drop = before - bounded_after

    rounding_bound = pi_zero * _energy_defect_bound(
        exact_transition.exact_next_centered_field,
        rounding,
        metric,
    )
    clipping_bound = pi_zero * _energy_defect_bound(
        raw_centered,
        clipping,
        metric,
    )
    total_bound = pi_zero * _energy_defect_bound(
        exact_transition.exact_next_centered_field,
        total,
        metric,
    )
    raw_drop_lower_bound = exact_transition.exact_energy_drop - rounding_bound
    bounded_drop_lower_bound = exact_transition.exact_energy_drop - total_bound

    raw_post_history = (raw,) + exact_history[:-1]
    bounded_post_history = (bounded,) + exact_history[:-1]
    original_barycenter = exact_transition.exact_stationary_history_barycenter
    raw_barycenter = _weighted_history_field(
        certificate.stationary_distribution,
        raw_post_history,
    )
    bounded_barycenter = _weighted_history_field(
        certificate.stationary_distribution,
        bounded_post_history,
    )
    raw_barycenter_drift = _vector_subtract(raw_barycenter, original_barycenter)
    bounded_barycenter_drift = _vector_subtract(
        bounded_barycenter,
        original_barycenter,
    )

    conditions = (
        ("cycle_proof_fields_intact", True),
        ("applied_remesh_bound_to_exact_history", True),
        ("oldest_first_runtime_history_reversed_once", True),
        ("raw_binary64_replay_identified", raw_replay_identified),
        ("bounded_binary64_replay_identified", bounded_replay_identified),
        (
            "signed_residual_decomposition_exact",
            total == _vector_add(rounding, clipping),
        ),
        (
            "rounding_augmented_energy_defect_identity",
            rounding_defect
            == pi_zero * (raw_energy - exact_transition.exact_next_energy),
        ),
        (
            "clipping_augmented_energy_defect_identity",
            clipping_defect == pi_zero * (bounded_energy - raw_energy),
        ),
        (
            "total_augmented_energy_defect_identity",
            total_defect == rounding_defect + clipping_defect,
        ),
        (
            "runtime_raw_augmented_energy_balance",
            raw_drop
            == exact_transition.exact_jensen_dissipation - rounding_defect,
        ),
        (
            "runtime_bounded_augmented_energy_balance",
            bounded_drop
            == exact_transition.exact_jensen_dissipation - total_defect,
        ),
        (
            "rounding_energy_defect_within_absolute_bound",
            abs(rounding_defect) <= rounding_bound,
        ),
        (
            "clipping_energy_defect_within_absolute_bound",
            abs(clipping_defect) <= clipping_bound,
        ),
        (
            "total_energy_defect_within_absolute_bound",
            abs(total_defect) <= total_bound,
        ),
        (
            "raw_barycenter_drift_equals_weighted_rounding_residual",
            raw_barycenter_drift == _scale_vector(pi_zero, rounding),
        ),
        (
            "bounded_barycenter_drift_equals_weighted_total_residual",
            bounded_barycenter_drift == _scale_vector(pi_zero, total),
        ),
        (
            "hard_clipping_disagreement_nonexpansive",
            plan.clip_mode != "hard" or clipping_defect <= 0,
        ),
    )
    if not all(passed for _, passed in conditions):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise RuntimeError(f"runtime REMESH history bridge failed: {failed}")

    value = RuntimeRemeshHistoryBridgeObservation(
        cycle_result=cycle_result,
        exact_transition=exact_transition,
        nodes=cycle_result.target_nodes,
        exact_metric_weights=metric,
        exact_history=exact_history,
        exact_ideal_next_field=ideal,
        exact_runtime_raw_next_field=raw,
        exact_runtime_bounded_next_field=bounded,
        exact_rounding_residual=rounding,
        exact_clipping_residual=clipping,
        exact_total_residual=total,
        exact_max_abs_rounding_residual=_max_abs(rounding),
        exact_max_abs_clipping_residual=_max_abs(clipping),
        exact_max_abs_total_residual=_max_abs(total),
        exact_runtime_raw_next_centered_field=raw_centered,
        exact_runtime_bounded_next_centered_field=bounded_centered,
        exact_runtime_raw_next_energy=raw_energy,
        exact_runtime_bounded_next_energy=bounded_energy,
        exact_lifted_augmented_energy_before=before,
        exact_lifted_ideal_augmented_energy_after=ideal_after,
        exact_lifted_runtime_raw_augmented_energy_after=raw_after,
        exact_lifted_runtime_bounded_augmented_energy_after=bounded_after,
        exact_jensen_dissipation=exact_transition.exact_jensen_dissipation,
        exact_rounding_augmented_energy_defect=rounding_defect,
        exact_clipping_augmented_energy_defect=clipping_defect,
        exact_total_augmented_energy_defect=total_defect,
        exact_lifted_runtime_raw_energy_drop=raw_drop,
        exact_lifted_runtime_bounded_energy_drop=bounded_drop,
        exact_rounding_augmented_energy_defect_absolute_upper_bound=rounding_bound,
        exact_clipping_augmented_energy_defect_absolute_upper_bound=clipping_bound,
        exact_total_augmented_energy_defect_absolute_upper_bound=total_bound,
        exact_lifted_runtime_raw_energy_drop_lower_bound=raw_drop_lower_bound,
        exact_lifted_runtime_bounded_energy_drop_lower_bound=bounded_drop_lower_bound,
        exact_lifted_runtime_raw_barycenter_drift=raw_barycenter_drift,
        exact_lifted_runtime_bounded_barycenter_drift=bounded_barycenter_drift,
        clipping_intervened=any(
            proposal.clipping_intervened for proposal in proposals
        ),
        raw_binary64_replay_identified=raw_replay_identified,
        bounded_binary64_replay_identified=bounded_replay_identified,
        conditions=conditions,
    )
    result = _seal(value)
    if verify_result and not result.bridge_observation_certified:
        raise RuntimeError("constructed runtime REMESH history bridge is inconsistent")
    return result


def observe_runtime_remesh_history_bridge(
    cycle_result: EventRemeshCycleResult,
) -> RuntimeRemeshHistoryBridgeObservation:
    """Bind one applied executor result to the exact uniform companion step."""

    return _build_runtime_remesh_history_bridge(
        cycle_result,
        verify_result=True,
    )
