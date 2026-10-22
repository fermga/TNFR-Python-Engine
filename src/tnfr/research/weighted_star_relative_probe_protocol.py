"""Prospective finite-read protocol for the retained weighted-star probe.

Only static preparation and analytical bounds are consumed here. The frozen
declaration supplies a later numerical schedule; no response is evaluated.
"""

from __future__ import annotations

from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_BITS, INTERVAL_METHOD
from ..utils.io import json_dumps
from .artifact_io import encode_exact_tree


def frozen_protocol():
    """Return the fixed declaration, with independent post-word error budgets."""
    return {
        "schema": "tnfr.weighted-star-relative-probe.v1",
        "evaluation_status_at_freeze": "not_evaluated",
        "source": "weighted_star_registration_episode/preparation.json",
        "branches": tuple(
            {"donor": donor, "boost": boost}
            for donor in (5, 0)
            for boost in (Q(1, 1024), Q(1, 512))
        ),
        "word": ("emission", "silence"),
        "compatibility_profile": "core",
        "silence_factor": Q(1, 2),
        "receiver": 4,
        "law": "held_zero_loss_weighted_sine_unit_exchange_and_beta",
        "state_order": "all_ten_forms_then_all_ten_continuous_radian_phase_lifts",
        "clock": "original_structural_t; no_laboratory_calibration",
        "start_time": Q(7),
        "duration": Q(1, 256),
        "observation": "q_j=(Y_high(h)-Y_high(0))-(Y_low(h)-Y_low(0)); q_5/q_0",
        "readout": "receiver_phase_in_calibrated_radian_units; branch_constant_offsets",
        "receiver_lift": (Q(-1, 128), Q(1, 128)),
        "post_word_source_form_error": Q(1, 2**40),
        "post_word_input_form_error": Q(1, 2**40),
        "post_word_phase_error": Q(1, 2**40),
        "final_read_time_error": Q(1, 2**40),
        "per_readout_error": Q(1, 2**40),
        "per_numerical_reading_error": Q(1, 2**40),
        "exact_held_inputs": ("support", "conductance", "capacity", "law"),
        "Gamma": "none",
        "form_phase_lipschitz_upper_bound": Q(16, 3),
        "phase_form_lipschitz_upper_bound": Q(48, 5),
        "phase_speed_upper_bound": Q(3, 4),
        "maximum_ratio_error": Q(1, 128),
        "predicted_ratio_interval": (Q(283, 640), Q(65, 128)),
        "comparison_reference": Q(1),
        "comparison_scope": "point_reference_only; not_an_excluded_complete_alternative_law",
        "numerical_protocol": {
            "scope": "nominal_prepared_states; uncertainty_propagated_analytically",
            "field": "tnfr.physics._sine_flow._held_sine_neighbor_field",
            "field_reuse_pair_sines": True,
            "stepper": "tnfr.mathematics._validated_taylor.validated_box_taylor_step",
            "steps_per_branch": 16,
            "step_duration": Q(1, 4096),
            "order": 12,
            "interval_bits": INTERVAL_BITS,
            "arithmetic_method": INTERVAL_METHOD,
            "domain": "finite_full_state; constant_positive_margin_one",
            "readout": "sum_receiver_phase_increments; retain_full_endpoint_boxes",
            "maximum_reading_interval_width": Q(1, 2**39),
            "retries": 0,
            "failure_status": "unavailable_within_frozen_budget",
            "wall_seconds_per_branch": 120,
            "memory_bytes": 512 * 1024**2,
            "maximum_response_bytes": 32 * 1024**2,
            "evaluation_wiring_status": "not_executed; must_match_this_declaration",
        },
        "scope": (
            "post_word_uncertainty_membership_is_a_separate_obligation",
            "no_uncertain_event_selection_or_pre_word_error_transport_claim",
            "no_future_inputs_events_runtime_pressure_refresh_callbacks_or_capacity_evolution",
            "no_trajectory_reserved_response_or_physical_sensor_identification",
        ),
    }


def _admit_protocol(protocol):
    # Admit the serialization domain before exact declaration comparison. Use
    # the freshly normalized declaration, never value-equal caller scalars.
    expected = frozen_protocol()
    if json_dumps(encode_exact_tree(protocol), sort_keys=True) != json_dumps(
        encode_exact_tree(expected), sort_keys=True
    ):
        raise ValueError("relative-probe protocol differs from its fixed declaration")
    return expected


def _error_budget(protocol):
    """Exact conservative arithmetic; the consumer must prove its coefficients."""
    h = protocol["duration"]
    a = protocol["form_phase_lipschitz_upper_bound"]
    b = protocol["phase_form_lipschitz_upper_bound"]
    e = protocol["post_word_source_form_error"] + protocol["post_word_input_form_error"]
    f = protocol["post_word_phase_error"]
    c = protocol["final_read_time_error"]
    m = protocol["per_readout_error"]
    n = protocol["per_numerical_reading_error"]
    speed = protocol["phase_speed_upper_bound"]
    amplitude = Q(1, 1024)
    denominator = 1 - a * b * h**2
    if denominator <= 0:
        raise ValueError("strict comparison margin is unavailable")
    remainder = amplitude * a * b**2 * h**3 / (6 * denominator)
    phase_error = (f + b * h * e) / denominator
    pair_error = remainder + 2 * (phase_error + f) + 2 * speed * c + 4 * m + 4 * n
    # Receiver nu/d=13/32, reference W40=2; pi<22/7. Admission below checks
    # these source-specific values before this exact lower bound is consumed.
    reference_signal_lower = h * Q(13, 32) * 2 * amplitude * Q(7, 22)
    reading_denominator_lower = reference_signal_lower - pair_error
    if reading_denominator_lower <= 0:
        raise ValueError("a nonzero reading denominator is unavailable")
    ratio_error = Q(3, 2) * pair_error / reading_denominator_lower
    return {
        "comparison_denominator": denominator,
        "matched_cubic_remainder": remainder,
        "per_branch_phase_source_error": phase_error,
        "per_pair_complete_reading_error": pair_error,
        "reference_signal_lower_bound": reference_signal_lower,
        "reading_denominator_lower_bound": reading_denominator_lower,
        "ratio_error_upper_bound": ratio_error,
        "receiver_lift_absolute_upper_bound": f + speed * (h + c),
    }


def assess_relative_probe_protocol(preparation, record, protocol):
    """Re-admit complete captures and prove the frozen finite-reading budget.

    Captured records are primitive input, not execution authentication. Every
    consumed field, event association and storage/error bound is rebuilt. The
    future numerical schedule is declared but is not invoked by this function.
    """
    from ..physics.relational_sine_star_retention import (
        assess_sine_weighted_star_support_retention,
    )
    from .weighted_star_relative_probe import admit_weighted_star_relative_probe

    declaration = _admit_protocol(protocol)
    admitted = admit_weighted_star_relative_probe(preparation, record)
    budget = _error_budget(declaration)
    h = declaration["duration"]
    c = declaration["final_read_time_error"]
    e = (
        declaration["post_word_source_form_error"]
        + declaration["post_word_input_form_error"]
    )
    f = declaration["post_word_phase_error"]
    a = declaration["form_phase_lipschitz_upper_bound"]
    b = declaration["phase_form_lipschitz_upper_bound"]
    receiver = declaration["receiver"]
    stars = (tuple(range(5)), tuple(range(5, 10)))
    branch_bounds = []
    pairs = {}
    for branch in admitted.branches:
        source = branch.prefixes[-1]
        if source.nodes != tuple(range(10)):
            raise ValueError("the complete fixed node order is required")
        uncertainty = source.initial_state_perturbation_bounds(
            duration=h,
            form_error_bounds=(e,) * 10,
            phase_error_bounds=(f,) * 10,
        )
        retention = assess_sine_weighted_star_support_retention(source, stars=stars)
        if (
            uncertainty.status != "enclosed"
            or uncertainty.max_form_phase_lipschitz_upper_bound > a
            or uncertainty.max_phase_form_lipschitz_upper_bound > b
            or uncertainty.phase_error_supremum_upper_bound
            > budget["per_branch_phase_source_error"]
            or not retention.all_time_identity_retention_certified
            or uncertainty.initial_storage_upper_bound >= Q(29, 64)
        ):
            raise ValueError(
                "the full-state coefficient or identity margin is unavailable"
            )
        if max(source.capacity) > 8 or source.capacity[receiver] / source.degrees[
            receiver
        ] != Q(13, 32):
            raise ValueError("source capacity coefficients differ")
        # |x_dot_i| <= nu_i/pi < 8/3 at every phase. This also covers the
        # perturbed solution and the interval needed for final timing error.
        span = max(source.epi) - min(source.epi)
        phase_speed = b / 2 * (span + 2 * e + Q(16, 3) * (h + c))
        if phase_speed >= declaration["phase_speed_upper_bound"]:
            raise ValueError("the timing and phase-lift speed bound is unavailable")
        if source.phase[receiver] != 0:
            raise ValueError("receiver lift must start at its declared zero phase")
        branch_bounds.append(
            {
                "donor": branch.donor,
                "boost": branch.boost,
                "initial_storage_upper_bound": uncertainty.initial_storage_upper_bound,
                "identity_storage_margin": Q(35, 64)
                - uncertainty.initial_storage_upper_bound,
                "phase_speed_upper_bound": phase_speed,
                "form_phase_lipschitz_upper_bound": uncertainty.max_form_phase_lipschitz_upper_bound,
                "phase_form_lipschitz_upper_bound": uncertainty.max_phase_form_lipschitz_upper_bound,
                "source_phase_error_upper_bound": uncertainty.phase_error_supremum_upper_bound,
                "event_storage_jumps": branch.event_work,
            }
        )
        pairs.setdefault(branch.donor, []).append(source)
    rate_differences = {}
    for donor in (5, 0):
        low, high = pairs[donor]
        if (
            low.capacity != high.capacity
            or low.phase != high.phase
            or low.edges != high.edges
            or low.conductances != high.conductances
            or low.form_rates != high.form_rates
            or tuple(y - x for x, y in zip(low.epi, high.epi))
            != tuple(Q(1, 1024) if node == donor else Q(0) for node in low.nodes)
        ):
            raise ValueError("matched-pair cancellation premises differ")
        weights = {
            frozenset(edge): weight for edge, weight in zip(low.edges, low.conductances)
        }
        expected = -Q(13, 32) * weights[frozenset((receiver, donor))] / 1024
        observed = (
            high.phase_rate_numerators()[receiver]
            - low.phase_rate_numerators()[receiver]
        )
        if observed != expected or observed >= 0:
            raise ValueError("the rebuilt receiver phase-row identity differs")
        rate_differences[donor] = observed
    if rate_differences[0] != -Q(13, 32) * 2 / 1024:
        raise ValueError("reference conductance must equal two")
    ratio = rate_differences[5] / rate_differences[0]
    if not Q(9, 20) < ratio < Q(1, 2):
        raise ValueError(
            "the retained relative conductance is outside its source bounds"
        )
    if budget["ratio_error_upper_bound"] >= declaration[
        "maximum_ratio_error"
    ] or budget["receiver_lift_absolute_upper_bound"] >= Q(1, 128):
        raise ValueError("the frozen observation margin is unavailable")
    return {
        "schema": declaration["schema"],
        "status": "prospective_finite_reading_admitted",
        "protocol": declaration,
        "ideal_relative_conductance": ratio,
        "phase_rate_difference_numerators": {
            str(donor): value for donor, value in rate_differences.items()
        },
        "error_budget": budget,
        "branch_bounds": tuple(branch_bounds),
        "reserved_response_evaluated": False,
        "scope": "conditional_mathematical_observation; no_physical_bridge_or_alternative_law_exclusion",
    }
