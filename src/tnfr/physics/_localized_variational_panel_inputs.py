"""Outward Chebyshev interpolants of the fixed localized variational inputs.

This owner constructs only the prescribed source and edge multiplier. It
uses the canonical finite-input factory, 97 Chebyshev-Lobatto samples and
the shared outward transform. It does not integrate a variational response,
construct a candidate, or establish nonlinear retention. The interpolation
allowances in LOCALIZED_RETENTION_MECHANISMS.md (96) are separate from the
coefficient arithmetic enclosures retained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from time import perf_counter

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_chebyshev import (
    _norm_upper_admitted,
    interpolate_lobatto_rows,
    lobatto_nodes,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._localized_variational_inputs import (
    _admit_vector,
    _prepare_localized_variational_inputs,
)

_DEGREE = 96
_GENERATOR_INTERPOLATION_ERROR = Q(1, 2**114)
_SOURCE_INTERPOLATION_ERROR = Q(1, 2**109)


def _panel_clock(start, duration):
    """Admit exact/represented affine-clock primitives before preparation."""
    start = _exact(start, "panel start")
    duration = _exact(duration, "panel duration")
    if start < 0:
        raise ValueError("panel start must be nonnegative")
    if not 0 < duration <= 16:
        raise ValueError("panel duration must lie in (0, 16]")
    return start, duration


def _sample_panel_inputs(evaluate, start, duration, nodes, node_count):
    """Sample a locally supplied evaluator; return primitive tables only.

    This private wiring seam assigns no canonical identity or error bounds
    to an evaluator. The canonical entry point supplies its newly prepared
    closure and the shared Lobatto nodes. Tests may use manufactured inputs.
    """
    form = [[] for _ in range(node_count)]
    phase = [[] for _ in range(node_count)]
    edge = [[] for _ in range(node_count)]
    times, elapsed = [], []
    max_source_radius = max_edge_radius = Q(0)
    for node in nodes:
        # One outward construction retains the exact affine node enclosure.
        # In particular, the two exact endpoints cannot spill outside the
        # admitted numerical time rectangle through repeated rounding.
        time = I(
            start + duration * (node.lo + 1) / 2,
            start + duration * (node.hi + 1) / 2,
        )
        started = perf_counter()
        rows = evaluate(time)
        elapsed.append(perf_counter() - started)
        source_form = _admit_vector(rows.source_x, node_count, "sample form source")
        source_phase = _admit_vector(
            rows.source_theta, node_count, "sample phase source"
        )
        coefficient = _admit_vector(
            rows.cosine_coefficient, node_count, "sample edge coefficient"
        )
        for destination, values in (
            (form, source_form),
            (phase, source_phase),
            (edge, coefficient),
        ):
            for target, value in zip(destination, values):
                target.append(value)
        max_source_radius = max(
            max_source_radius, *(value.radius for value in source_form + source_phase)
        )
        max_edge_radius = max(max_edge_radius, *(value.radius for value in coefficient))
        times.append(time)
    return (
        tuple(tuple(row) for row in form),
        tuple(tuple(row) for row in phase),
        tuple(tuple(row) for row in edge),
        tuple(times),
        tuple(elapsed),
        max_source_radius,
        max_edge_radius,
    )


def _coefficient_radius_bounds(form, phase, edge):
    """Report arithmetic radii without replacing analytic interpolation tails."""
    source_radius = sum(
        (
            _norm_upper_admitted(
                I(row[index].width) for rows in (form, phase) for row in rows
            )
            / 2
            for index in range(_DEGREE + 1)
        ),
        Q(0),
    )
    # ||D^T diag(delta C) D/2|| <= 2 ||delta C||_infinity.
    edge_radius = 2 * max(sum((value.radius for value in row), Q(0)) for row in edge)
    return source_radius, edge_radius


@dataclass(frozen=True, slots=True)
class _LocalizedVariationalPanelInputs:
    """Prescribed finite-input interpolants, not a response certificate."""

    start: Q
    duration: Q
    end: Q
    node_order: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    sigma_bounds: I
    horizon_bounds: I
    evaluation_time_bounds: I
    source_form_coefficients: tuple[tuple[I, ...], ...]
    source_phase_coefficients: tuple[tuple[I, ...], ...]
    edge_coefficient_polynomials: tuple[tuple[I, ...], ...]
    sample_time_bounds: tuple[I, ...]
    sample_seconds: tuple[float, ...]
    factory_seconds: float
    sampling_seconds: float
    transform_seconds: float
    max_source_sample_radius: Q
    max_edge_sample_radius: Q
    source_coefficient_radius_norm_sum: Q
    edge_coefficient_radius_operator_upper: Q
    generator_error: Q = _GENERATOR_INTERPOLATION_ERROR
    source_error: Q = _SOURCE_INTERPOLATION_ERROR
    degree: int = _DEGREE
    interval_method: str = INTERVAL_METHOD
    method: str = "canonical_localized_panel_input_lobatto96_dyadic128_v1"
    clock: str = "tau_equals_exchange_times_original_time_over_pi_in_both_rows"
    scope: tuple[str, ...] = (
        "fixed_C1024_m_one_half_finite_profile24_sine17_cosine16",
        "descending_Lobatto_times_from_panel_end_to_panel_start",
        "signed_source_and_edge_multiplier_share_one_affine_clock",
        "analytic_interpolation_tails_separate_from_outward_coefficient_arithmetic",
        "no_candidate_variational_response_or_nonlinear_retention_verdict",
    )


def _build_localized_variational_panel_inputs(*, start=0, duration=16):
    """Construct the fixed degree-96 input interpolants on one admitted panel.

    Only exact/represented scalar clock primitives are accepted. The panel
    must fit in [0,T.hi], the finite-input owner's numerical time rectangle.
    Its possible terminal sliver beyond the uncertain physical T does not
    extend a physical theorem's horizon. No external metadata, sampled
    report, evaluator, preparation, degree or interpolation-error claim is
    consumed. All authoritative constants and the evaluator are prepared
    afresh by the shared fixed-input factory.

    The returned tables enclose the coefficients of the exact 97-node
    interpolants. Their interval radii include sampling and transform
    arithmetic. The separate generator_error and source_error bound the
    analytic difference between those interpolants and the complete finite
    inputs. They do not include the earlier model truncation allowances or
    supply a response bound. A consumer must re-admit primitive tables and
    independently match the complete model and its comparison hypotheses.
    """
    start, duration = _panel_clock(start, duration)
    started = perf_counter()
    metadata, evaluate = _prepare_localized_variational_inputs()
    factory_seconds = perf_counter() - started
    if start + duration > metadata.evaluation_time_bounds.hi:
        raise ValueError("panel must lie wholly in [0,T.hi]")
    nodes = lobatto_nodes(_DEGREE)
    started = perf_counter()
    (
        source_form,
        source_phase,
        edge,
        sample_times,
        sample_seconds,
        source_sample_radius,
        edge_sample_radius,
    ) = _sample_panel_inputs(evaluate, start, duration, nodes, metadata.node_count)
    sampling_seconds = perf_counter() - started
    started = perf_counter()
    coefficients = interpolate_lobatto_rows(
        source_form + source_phase + edge, degree=_DEGREE
    )
    transform_seconds = perf_counter() - started
    size = metadata.node_count
    form_coefficients = coefficients[:size]
    phase_coefficients = coefficients[size : 2 * size]
    edge_coefficients = coefficients[2 * size :]
    source_radius, edge_radius = _coefficient_radius_bounds(
        form_coefficients, phase_coefficients, edge_coefficients
    )
    return _LocalizedVariationalPanelInputs(
        start=start,
        duration=duration,
        end=start + duration,
        node_order=metadata.node_order,
        edges=metadata.edges,
        sigma_bounds=metadata.sigma_bounds,
        horizon_bounds=metadata.horizon_bounds,
        evaluation_time_bounds=metadata.evaluation_time_bounds,
        source_form_coefficients=form_coefficients,
        source_phase_coefficients=phase_coefficients,
        edge_coefficient_polynomials=edge_coefficients,
        sample_time_bounds=sample_times,
        sample_seconds=sample_seconds,
        factory_seconds=factory_seconds,
        sampling_seconds=sampling_seconds,
        transform_seconds=transform_seconds,
        max_source_sample_radius=source_sample_radius,
        max_edge_sample_radius=edge_sample_radius,
        source_coefficient_radius_norm_sum=source_radius,
        edge_coefficient_radius_operator_upper=edge_radius,
    )
