"""Conditional whole-panel defects for a prescribed cycle variational law.

The input polynomials use ordinary Chebyshev coefficients on [-1, 1],
including the full constant coefficient. They specify a candidate, a signed
source, and an edge multiplier. This module bounds their polynomial defect;
it neither generates a candidate nor certifies their relationship to the
complete inputs of a particular model. The caller must establish that
relationship, including the supplied uniform generator and source errors.

LOCALIZED_RETENTION_MECHANISMS.md owns the separate propagator comparison
needed to turn integrated defects into an error bound. No propagator gain,
canonical response, nonlinear conclusion, or retention verdict is supplied
here. Polynomial arithmetic reuses the exact packed-convolution owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_chebyshev import _add_admitted as add
from ..mathematics._interval_chebyshev import _derivative_admitted as derivative
from ..mathematics._interval_chebyshev import _multiply_admitted as multiply
from ..mathematics._interval_chebyshev import _norm_upper_admitted as _norm_upper
from ..mathematics._interval_chebyshev import _scale_admitted as scale
from ..mathematics._interval_chebyshev import _subtract_admitted as subtract
from ..mathematics._interval_chebyshev import admit_coefficients
from ..mathematics._rational_interval import I

_MAX_NODES = 1024
_MAX_INPUT_DEGREE = 96
_ZERO = I(0)
_HALF = I(Q(1, 2))
_INPUT_PREMISE = (
    "On the same complete panel, affine clock, fixed sigma and ordered cycle, "
    "the caller must independently establish that the complete variational "
    "generator differs from J_sigma plus the supplied polynomial edge action "
    "by at most generator_error in Euclidean operator norm, and that its "
    "signed source differs from the supplied source polynomials by at most "
    "source_error in Euclidean norm. These are supplied uniform premises, "
    "not authenticated by this report. Zero errors assert exact polynomial "
    "inputs. No canonical preparation, propagator gain or retention follows."
)


def _interval(value, label):
    """Re-admit primitive scalars and both endpoints of supplied intervals."""
    if isinstance(value, I):
        return I(_exact(value.lo, f"{label} lower"), _exact(value.hi, f"{label} upper"))
    return I(_exact(value, label))


def _ordered_rows(values, label, size=None):
    """Detach bounded node rows without accepting unordered/report inputs."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered finite sequence")
    try:
        count = len(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered finite sequence") from exc
    if size is None:
        if not 3 <= count <= _MAX_NODES:
            raise ValueError(f"{label} requires between 3 and {_MAX_NODES} nodes")
    elif count != size:
        raise ValueError(f"{label} must have exactly {size} node rows")
    try:
        raw = tuple(values[index] for index in range(count))
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(f"{label} must be an indexed finite sequence") from exc
    return tuple(
        admit_coefficients(row, max_degree=_MAX_INPUT_DEGREE, label=f"{label}[{index}]")
        for index, row in enumerate(raw)
    )


def _degree(rows):
    return max(len(row) - 1 for row in rows)


def _cycle_difference(rows):
    return tuple(
        subtract(row, rows[(index + 1) % len(rows)]) for index, row in enumerate(rows)
    )


def _cycle_adjoint(rows):
    return tuple(subtract(row, rows[index - 1]) for index, row in enumerate(rows))


def _cycle_laplacian(rows):
    """A=D^T D/2 retains the spatial constant mode and all other modes."""
    return tuple(
        subtract(row, scale(add(rows[index - 1], rows[(index + 1) % len(rows)]), _HALF))
        for index, row in enumerate(rows)
    )


def _coefficient_norm_sum(form, phase):
    """Use |T_k(s)|<=1 with a full two-row Euclidean bound at every degree."""
    degree = max(_degree(form), _degree(phase))
    return sum(
        (
            _norm_upper(
                row[index] if index < len(row) else _ZERO
                for rows in (form, phase)
                for row in rows
            )
            for index in range(degree + 1)
        ),
        Q(0),
    )


def _endpoint(rows, *, initial):
    """Enclose exact polynomial evaluation at either end of the local chart."""
    result = []
    for row in rows:
        terms = tuple(
            -coefficient if initial and index % 2 else coefficient
            for index, coefficient in enumerate(row)
        )
        result.append(
            I(sum(value.lo for value in terms), sum(value.hi for value in terms))
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _VariationalPanelBounds:
    """Local polynomial bounds conditional on independently supplied inputs."""

    node_count: int
    candidate_degrees: tuple[int, int]
    source_degrees: tuple[int, int]
    edge_degree: int
    residual_degree: int
    duration_bounds: I
    sigma_bounds: I
    generator_error_bounds: I
    source_error_bounds: I
    candidate_norm_upper: Q
    polynomial_defect_norm_upper: Q
    polynomial_integrated_defect_upper: Q
    conditional_integrated_defect_upper: Q
    initial_form: tuple[I, ...]
    initial_phase: tuple[I, ...]
    final_form: tuple[I, ...]
    final_phase: tuple[I, ...]
    required_input_premise: str = _INPUT_PREMISE
    method: str = "cycle_chebyshev_conditional_panel_defect_dyadic128_v1"


def _bound_cycle_variational_panel(
    form_coefficients,
    phase_coefficients,
    source_form_coefficients,
    source_phase_coefficients,
    edge_coefficient_polynomials,
    *,
    sigma,
    duration,
    generator_error=0,
    source_error=0,
):
    """Bound a signed polynomial defect throughout one affine time panel.

    All five coefficient tables are node-major ordered sequences with the
    same 3..1024 node rows; each row is nonempty and has degree at most 96.
    Explicit trailing zeros are preserved. Scalars admit exact/represented
    finite reals or shared intervals, with 0<=sigma<2 and 0<duration<=16
    required for the whole supplied interval. Input errors are nonnegative
    uniform bounds; their default zero states an exact-input premise.

    For D theta_i=theta_i-theta_(i+1), A=D^T D/2 and local coordinate s,
    the returned defect bounds the two rows

      (2/duration) x' + sigma A x + A theta - D^T(C D theta)/2 + r_x,
      (2/duration) theta' - A x + r_theta.

    The conditional integral is duration*(R+source_error+generator_error*P),
    where P bounds the candidate and R bounds this polynomial defect. It
    contains no initial mismatch or jumps between panels. Those must be
    separately retained by any complete comparison that consumes it.

    Work uses O(N d) interval bookkeeping and O(N) packed integer products;
    integer multiplication has backend-dependent bit complexity. Retained
    coefficient storage is O(N d), with d<=192 after multiplication. There
    is no dense matrix or Python quadratic coefficient-product loop.
    """
    form = _ordered_rows(form_coefficients, "form coefficients")
    size = len(form)
    phase = _ordered_rows(phase_coefficients, "phase coefficients", size)
    source_form = _ordered_rows(source_form_coefficients, "form source", size)
    source_phase = _ordered_rows(source_phase_coefficients, "phase source", size)
    coefficient = _ordered_rows(edge_coefficient_polynomials, "edge coefficients", size)
    admitted_sigma = _interval(sigma, "sigma")
    admitted_duration = _interval(duration, "duration")
    admitted_generator_error = _interval(generator_error, "generator error")
    admitted_source_error = _interval(source_error, "source error")
    if admitted_sigma.lo < 0 or admitted_sigma.hi >= 2:
        raise ValueError("sigma must be wholly in [0, 2)")
    if admitted_duration.lo <= 0 or admitted_duration.hi > 16:
        raise ValueError("duration must be wholly in (0, 16]")
    if admitted_generator_error.lo < 0 or admitted_source_error.lo < 0:
        raise ValueError("generator and source errors must be nonnegative")

    # No polynomial action precedes complete primitive admission above. The
    # private algebra kernels consume only these local normalized I tuples
    # and their derived rows, never supplied values or cached report fields.
    time_scale = 2 / admitted_duration
    form_laplacian = _cycle_laplacian(form)
    phase_laplacian = _cycle_laplacian(phase)
    phase_gap = _cycle_difference(phase)
    edge_feedback = tuple(
        multiply(edge, gap) for edge, gap in zip(coefficient, phase_gap)
    )
    feedback = _cycle_adjoint(edge_feedback)
    form_defect = tuple(
        add(
            subtract(
                add(
                    add(scale(derivative(row), time_scale), scale(ax, admitted_sigma)),
                    atheta,
                ),
                scale(edge, _HALF),
            ),
            source,
        )
        for row, ax, atheta, edge, source in zip(
            form, form_laplacian, phase_laplacian, feedback, source_form
        )
    )
    phase_defect = tuple(
        add(subtract(scale(derivative(row), time_scale), ax), source)
        for row, ax, source in zip(phase, form_laplacian, source_phase)
    )
    candidate_bound = _coefficient_norm_sum(form, phase)
    defect_bound = _coefficient_norm_sum(form_defect, phase_defect)
    polynomial_integral = admitted_duration.hi * defect_bound
    conditional_integral = admitted_duration.hi * (
        defect_bound
        + admitted_source_error.hi
        + admitted_generator_error.hi * candidate_bound
    )
    return _VariationalPanelBounds(
        node_count=size,
        candidate_degrees=(_degree(form), _degree(phase)),
        source_degrees=(_degree(source_form), _degree(source_phase)),
        edge_degree=_degree(coefficient),
        residual_degree=max(_degree(form_defect), _degree(phase_defect)),
        duration_bounds=admitted_duration,
        sigma_bounds=admitted_sigma,
        generator_error_bounds=admitted_generator_error,
        source_error_bounds=admitted_source_error,
        candidate_norm_upper=candidate_bound,
        polynomial_defect_norm_upper=defect_bound,
        polynomial_integrated_defect_upper=polynomial_integral,
        conditional_integrated_defect_upper=conditional_integral,
        initial_form=_endpoint(form, initial=True),
        initial_phase=_endpoint(phase, initial=True),
        final_form=_endpoint(form, initial=False),
        final_phase=_endpoint(phase, initial=False),
    )
