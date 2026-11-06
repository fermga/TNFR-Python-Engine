"""Independent finite controls for a genuinely smaller retained predictor.

The analytical oracles use their own exact rational endpoint arithmetic,
alternating exponentials and positive hyperbolic series. No trajectory or
production exponential/interval kernel supplies an expected response.
"""

from dataclasses import FrozenInstanceError, dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics import _comparison_flow as comparison
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as flow
from tnfr.physics import _sine_retained_boundary as owner


@dataclass(frozen=True)
class _B:
    """Small independent exact endpoint oracle, without dyadic rounding."""

    lo: Q
    hi: Q | None = None

    def __post_init__(self):
        object.__setattr__(self, "lo", Q(self.lo))
        object.__setattr__(self, "hi", self.lo if self.hi is None else Q(self.hi))
        assert self.lo <= self.hi

    def __add__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        return _B(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _B(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, _B) else _B(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        endpoints = tuple(
            a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)
        )
        return _B(min(endpoints), max(endpoints))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / Q(scalar))

    def __pow__(self, power):
        assert type(power) is int and power >= 0
        value = _B(1)
        for _ in range(power):
            value = value * self
        return value


def _negative_exp(value):
    """Alternating degree100/101 bounds, independently valid on 0..1."""
    value = Q(value)
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(101)), Q(0))
    return _B(upper - value**101 / factorial(101), upper)


def _sine(value):
    value = Q(value)
    if value < 0:
        return -_sine(-value)
    assert value <= 1
    upper = sum(
        ((-1) ** k * value ** (2 * k + 1) / factorial(2 * k + 1) for k in range(51)),
        Q(0),
    )
    return _B(upper - value**103 / factorial(103), upper)


def _hyperbolic(time):
    """Enclose cosh(z), sinh(z)/sqrt(5), z=sqrt(5)*time/8."""
    time = Q(time)
    assert 0 <= time <= Q(1, 4)
    square = 5 * time**2 / 64
    even = sum((square**k / factorial(2 * k) for k in range(49)), Q(0))
    odd = sum((square**k / factorial(2 * k + 1) for k in range(49)), Q(0))
    even_tail = square**49 / factorial(98) / (1 - square / (99 * 100))
    odd_tail = square**49 / factorial(99) / (1 - square / (100 * 101))
    return _B(even, even + even_tail), time / 8 * _B(odd, odd + odd_tail)


def _zero(bounds, tolerance=Q(1, 10**100)):
    assert bounds.lo <= 0 <= bounds.hi
    assert bounds.hi - bounds.lo < tolerance


def _enclosed(bounds, actual):
    assert -bounds <= actual.lo <= actual.hi <= bounds


def _from_production(bounds):
    return _B(bounds.lo, bounds.hi)


def _contains_oracle(actual, expected):
    assert actual.lo <= expected.lo <= expected.hi <= actual.hi
    assert actual.hi - actual.lo < Q(1, 10**35)


def _display(value):
    value = Q(value)
    with localcontext() as context:
        context.prec = 25
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _arguments(**changes):
    values = dict(
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        conductances=(Q(1), Q(1)),
        capacity=(Q(1),) * 3,
        gamma=Q(0),
        retained_nodes=(0, 1),
        observation_nodes=(0, 1),
        retained_initial_form=(Q(0), Q(0)),
        retained_initial_phase=(Q(0), Q(0)),
        hidden_reference_form=(Q(0),),
        hidden_reference_phase=(Q(0),),
        hidden_form_error_bounds=(Q(1),),
        hidden_phase_error_bounds=(Q(0),),
        elapsed_time=Q(1, 4),
        reading_error_bound=Q(0),
        tolerance=Q(1, 20),
        inverse_capacity_reading_error_bound=Q(0),
        inverse_capacity_tolerance=Q(1, 20),
    )
    values.update(changes)
    return values


def _bound(**changes):
    return owner._prepared_sine_retained_boundary(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("retained predictor controls cannot integrate or execute events")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
            (external, "_derive_organized_external_interaction"),
            (external, "_assess_organized_external_interaction"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _linear_rows(state, conductances=(Q(1), Q(1)), capacity=(Q(1),) * 3):
    result = [_B(0) for _ in state]
    for (a, b), weight in zip(((0, 1), (1, 2)), conductances):
        current = weight * (state[b] - state[a]) / 4
        result[a] += capacity[a] * current
        result[b] -= capacity[b] * current
    return tuple(result)


def _full_solution(time, *, on, source="hidden"):
    """The complete three-node heat states and explicit time derivatives."""
    r, q = _negative_exp(time / 4), _negative_exp(time / 2)
    if not on:
        if source == "visible":
            return (_B(1), _B(0), _B(0)), (_B(0),) * 3
        return (_B(0), (1 - q) / 2, (1 + q) / 2), (_B(0), q / 4, -q / 4)
    cube = _negative_exp(3 * time / 4)
    leaf, middle, source_value = (
        (2 - 3 * r + cube) / 6,
        (1 - cube) / 3,
        (2 + 3 * r + cube) / 6,
    )
    leaf_rate, middle_rate, source_rate = (r - cube) / 8, cube / 4, -(r + cube) / 8
    if source == "visible":
        return (source_value, middle, leaf), (source_rate, middle_rate, leaf_rate)
    return (leaf, middle, source_value), (leaf_rate, middle_rate, source_rate)


def _reduced_solution(time, *, on, mode):
    """Two evolving forms with fixed hidden reference; phases stay zero."""
    if mode == "initialization":
        return (_B(0), _B(0)), (_B(0), _B(0))
    if not on:
        if mode == "feedback":
            return (_B(1), _B(0)), (_B(0), _B(0))
        r = _negative_exp(time / 4)
        return (_B(0), 1 - r), (_B(0), r / 4)
    c, s = _hyperbolic(time)
    decay = _negative_exp(3 * time / 8)
    if mode == "feedback":
        return (decay * (c + s), 2 * decay * s), (
            decay * (-c + s) / 4,
            decay * (c - 3 * s) / 4,
        )
    return (1 - decay * (c + 3 * s), 1 - decay * (c + s)), (
        decay * s / 2,
        decay * (c - s) / 4,
    )


def _check_closed_rows(time, on, mode):
    source = "visible" if mode == "feedback" else "hidden"
    full, full_rate = _full_solution(time, on=on, source=source)
    reduced, reduced_rate = _reduced_solution(time, on=on, mode=mode)
    weights = (Q(int(on)), Q(1))
    for actual, expected in zip(_linear_rows(full, weights), full_rate):
        _zero(actual - expected)
    hidden = _B(1 if mode == "residual" else 0)
    for actual, expected in zip(
        _linear_rows(reduced + (hidden,), weights)[:2], reduced_rate
    ):
        _zero(actual - expected)
    _zero(sum(full) - 1)
    return full, reduced


@pytest.mark.parametrize("mode", ("initialization", "residual", "feedback"))
def test_closed_solutions_satisfy_independent_complete_and_reduced_rows(mode):
    for on in (False, True):
        for time in (Q(0), Q(1, 4)):
            _check_closed_rows(time, on, mode)


def test_hidden_initialization_changes_contact_response_at_identical_retained_data(
    record_property,
):
    reports, errors = {}, {}
    h = Q(1, 4)
    for on in (False, True):
        report = _bound(conductances=(int(on), 1))
        reports[on] = report
        full, reduced = _check_closed_rows(h, on, "initialization")
        error = (full[0] + full[1] - reduced[0] - reduced[1]) / 2
        errors[on] = error
        _enclosed(report.arithmetic_error_upper_bound, error)
        _enclosed(report.arithmetic_error_upper_bound, -error)
        assert report.arithmetic_hidden_drift_error_upper_bound == 0
        assert report.arithmetic_boundary_feedback_error_upper_bound == 0
        assert (
            report.arithmetic_initialization_error_upper_bound
            == report.arithmetic_error_upper_bound
        )
        assert 0 < report.arithmetic_error_upper_bound <= Q(19, 384) < Q(1, 20)
        assert report.tolerance_certified
        assert report.reduced_initial_state == (0, 0, 0, 0)
        assert report.hidden_drift_forcing == (0,) * 3
        assert report.hidden_feedback_forcing == (0,) * 3
        assert (
            report.inverse_capacity_error_upper_bound
            == report.arithmetic_error_upper_bound
        )
        record_property("on" if on else "off", str(report.arithmetic_error_upper_bound))
        record_property(
            "actual_" + ("on" if on else "off") + "_error_display",
            _display((error.lo + error.hi) / 2),
        )
    # Same retained zero state and same reference predict zero on both branches.
    # Hidden +1 and -1 give opposite true contact contrasts. The frozen absolute
    # tolerance is not a sign or relative-response resolution certificate.
    actual_contrast = errors[True] - errors[False]
    expected = (1 - _negative_exp(h / 4)) ** 3 / 12
    _zero(actual_contrast - expected)
    assert actual_contrast.lo > 0 and (-actual_contrast).hi < 0
    allowance = sum(r.arithmetic_observed_error_upper_bound for r in reports.values())
    _enclosed(allowance, actual_contrast)
    assert allowance < Q(1, 10)
    assert actual_contrast.hi < allowance
    assert reports[False].arithmetic_error_upper_bound > 0
    record_property("response_allowance", str(allowance))
    record_property("response_tolerance", "1/10")
    record_property("full_dimension", "6")
    record_property("evolved_dimension", "4")
    record_property(
        "true_response_positive_display",
        _display((actual_contrast.lo + actual_contrast.hi) / 2),
    )
    record_property("hidden_negative_one_reverses_true_response_sign", "true")


@pytest.mark.parametrize("mode", ("residual", "feedback"))
def test_omitted_initial_drift_and_later_feedback_have_separate_error_channels(
    mode, record_property
):
    changes = dict(hidden_form_error_bounds=(0,))
    if mode == "residual":
        changes["hidden_reference_form"] = (1,)
    else:
        changes["retained_initial_form"] = (1, 0)
    h, reports, actual_errors = Q(1, 4), {}, {}
    for on in (False, True):
        report = _bound(**changes, conductances=(int(on), 1))
        reports[on] = report
        full, reduced = _check_closed_rows(h, on, mode)
        error = (full[0] + full[1] - reduced[0] - reduced[1]) / 2
        actual_errors[on] = error
        _enclosed(report.arithmetic_error_upper_bound, error)
        assert report.arithmetic_initialization_error_upper_bound == 0
        assert (
            report.arithmetic_error_upper_bound
            == report.arithmetic_hidden_drift_error_upper_bound
            + report.arithmetic_boundary_feedback_error_upper_bound
        )
        assert report.tolerance_certified and report.arithmetic_error_upper_bound < Q(
            1, 20
        )
        if mode == "feedback":
            assert report.arithmetic_hidden_drift_error_upper_bound == 0
            assert report.initial_rate_upper_bounds[2] == 0
            if on:
                assert report.arithmetic_boundary_feedback_error_upper_bound > 0
                assert error.lo > 0
            else:
                assert report.arithmetic_error_upper_bound == 0
                _zero(error)
        else:
            assert report.arithmetic_hidden_drift_error_upper_bound > 0
            assert report.arithmetic_boundary_feedback_error_upper_bound > 0
            assert error.hi < 0
            # A theorem consequence stronger than the unchanged acceptance
            # tolerance: cH=1/4, KR norm<=16/13, AHR KR cR<=1/13.
            assert report.arithmetic_error_upper_bound <= Q(513, 159744)
        record_property(
            ("on_" if on else "off_") + mode, str(report.arithmetic_error_upper_bound)
        )
        record_property(
            "actual_" + ("on" if on else "off") + "_error_display",
            _display((error.lo + error.hi) / 2),
        )
    actual = actual_errors[True] - actual_errors[False]
    allowance = sum(r.arithmetic_observed_error_upper_bound for r in reports.values())
    _enclosed(allowance, actual)
    assert allowance < Q(1, 10)
    record_property("response_tolerance", "1/10")
    record_property("response_allowance", str(allowance))


def test_original_mean_and_extra_leaf_observation_remain_distinct():
    mean_on = _bound()
    leaf_on = _bound(observation_nodes=(0,))
    leaf_off = _bound(conductances=(0, 1), observation_nodes=(0,))
    assert mean_on.arithmetic_observation_weights == (Q(1, 2), Q(1, 2), 0)
    assert leaf_on.arithmetic_observation_weights == (1, 0, 0)
    assert leaf_on.arithmetic_error_upper_bound > 0
    assert leaf_off.arithmetic_error_upper_bound == 0
    assert leaf_off.exact_observation_zero


def test_disconnected_hidden_block_has_zero_observed_error_despite_hidden_motion():
    report = _bound(
        nodes=(0, 1, 2, 3),
        edges=((0, 1), (2, 3)),
        conductances=(1, 1),
        capacity=(1,) * 4,
        hidden_reference_form=(1, 0),
        hidden_reference_phase=(0, 0),
        hidden_form_error_bounds=(1, 2),
        hidden_phase_error_bounds=(Q(1, 4), Q(1, 2)),
    )
    assert report.hidden_drift_forcing[2] > 0 and report.hidden_drift_forcing[3] > 0
    assert any(report.nodal_error_upper_bounds[i] > 0 for i in (2, 3))
    assert (
        report.arithmetic_error_upper_bound
        == report.inverse_capacity_error_upper_bound
        == 0
    )
    assert (
        report.exact_observation_zero and report.inverse_capacity_exact_observation_zero
    )


def _direct_rows(form, phase, edges, weights, capacity, gamma):
    forms, phases = [_B(0) for _ in form], [_B(0) for _ in form]
    for (a, b), weight in zip(edges, weights):
        current = weight * (form[b] - form[a]) / 4
        sine = weight * _sine(phase[b] - phase[a]) / 4
        forms[a] += capacity[a] * (current + gamma * sine)
        forms[b] -= capacity[b] * (current + gamma * sine)
        phases[a] -= gamma * capacity[a] * current
        phases[b] += gamma * capacity[b] * current
    return tuple(forms), tuple(phases)


@pytest.mark.parametrize("gamma", (Q(0), Q(1), Q(2)))
def test_reduced_field_keeps_actual_capacities_and_both_full_law_rows(gamma):
    capacities = (Q(2), Q(1), Q(1, 2))
    reference_form, reference_phase = (Q(1, 2), Q(-1, 4), Q(1, 8)), (
        Q(1, 4),
        Q(0),
        Q(-1, 4),
    )
    report = _bound(
        capacity=capacities,
        gamma=gamma,
        retained_initial_form=reference_form[:2],
        retained_initial_phase=reference_phase[:2],
        hidden_reference_form=reference_form[2:],
        hidden_reference_phase=reference_phase[2:],
        hidden_form_error_bounds=(0,),
        elapsed_time=Q(1, 100),
    )
    expected_form, expected_phase = _direct_rows(
        reference_form,
        reference_phase,
        ((0, 1), (1, 2)),
        (Q(1), Q(1)),
        capacities,
        gamma,
    )
    for actual, expected in zip(report.initial_form_rate_bounds, expected_form):
        _contains_oracle(actual, expected)
    for actual, expected in zip(report.initial_phase_rate_bounds, expected_phase):
        _contains_oracle(actual, expected)
    field = report.make_reduced_field()
    output = field(report.reduced_initial_state)
    assert len(output) == 4 < 6
    for actual, expected in zip(output, expected_form[:2] + expected_phase[:2]):
        _contains_oracle(actual, expected)
    if gamma:
        assert any(v.lo != 0 for v in output[2:])
    else:
        assert all(v.lo == v.hi == 0 for v in output[2:])
    with pytest.raises((TypeError, ValueError)):
        field(reference_form + reference_phase)


def test_compiled_field_evaluates_only_retained_rows_and_incident_edges(monkeypatch):
    report = _bound(
        nodes=(0, 1, 2, 3),
        edges=((0, 1), (2, 3)),
        conductances=(1, 1),
        capacity=(1,) * 4,
        hidden_reference_form=(1, 0),
        hidden_reference_phase=(Q(1, 4), 0),
        hidden_form_error_bounds=(1, 1),
        hidden_phase_error_bounds=(0, 0),
        gamma=1,
    )
    field = report.make_reduced_field()
    original_rows, original_sine = flow._prepared_quarter_sine_rows, flow.sin
    calls = {"rows": 0, "edges": 0}

    def retained_rows(**kwargs):
        calls["rows"] += 1
        assert (
            len(kwargs["capacity"])
            == len(kwargs["gradient"])
            == len(kwargs["current"])
            == 2
        )
        return original_rows(**kwargs)

    def retained_sine(value):
        calls["edges"] += 1
        return original_sine(value)

    def forbidden(*args, **kwargs):
        pytest.fail("compiled reduced row called preparation or full-field assembly")

    monkeypatch.setattr(flow, "_prepared_quarter_sine_rows", retained_rows)
    monkeypatch.setattr(flow, "sin", retained_sine)
    monkeypatch.setattr(flow, "_held_quarter_sine_field", forbidden)
    monkeypatch.setattr(owner, "_prepared_sine_retained_boundary", forbidden)
    actual = field(report.reduced_initial_state)
    assert len(actual) == 4
    for value in actual:
        _zero(_from_production(value), Q(1, 10**35))
    assert calls == {"rows": 1, "edges": 1}


@pytest.mark.parametrize("coordinate", (0, 2))
def test_reduced_field_supports_independent_form_and_phase_directional_jets(coordinate):
    report = _bound(
        gamma=2,
        capacity=(2, 1, Q(1, 2)),
        hidden_form_error_bounds=(0,),
        elapsed_time=Q(1, 100),
    )
    field = report.make_reduced_field()
    state = tuple(Jet((I(0), I(int(i == coordinate)))) for i in range(4))
    actual = field(state)
    expected = (
        (Q(-1, 2), Q(1, 4), Q(1), Q(-1, 2))
        if coordinate == 0
        else (Q(-1), Q(1, 2), Q(0), Q(0))
    )
    for value, derivative in zip(actual, expected):
        assert value.order == 1
        _zero(_from_production(value.coeffs[0]), Q(1, 10**35))
        assert value.coeffs[1].lo <= derivative <= value.coeffs[1].hi
        assert value.coeffs[1].hi - value.coeffs[1].lo < Q(1, 10**35)
    with pytest.raises(ValueError):
        field((Jet.constant(0, 2),) + state[1:])
    with pytest.raises(ValueError):
        field((I(0),) + state[1:])


def test_reordered_retained_rows_and_full_node_labels_preserve_associations():
    base = _bound(
        capacity=(2, 1, Q(1, 2)),
        gamma=1,
        retained_initial_form=(Q(1, 2), Q(-1, 4)),
        retained_initial_phase=(Q(1, 4), 0),
        hidden_reference_form=(Q(1, 8),),
        hidden_reference_phase=(Q(-1, 4),),
        hidden_form_error_bounds=(Q(1, 10),),
        hidden_phase_error_bounds=(Q(1, 20),),
        elapsed_time=Q(1, 100),
    )
    transformed = _bound(
        nodes=("hidden", "left", "right"),
        edges=(("right", "left"), ("hidden", "right")),
        capacity=(Q(1, 2), 2, 1),
        gamma=1,
        retained_nodes=("right", "left"),
        observation_nodes=("right", "left"),
        retained_initial_form=(Q(-1, 4), Q(1, 2)),
        retained_initial_phase=(0, Q(1, 4)),
        hidden_reference_form=(Q(1, 8),),
        hidden_reference_phase=(Q(-1, 4),),
        hidden_form_error_bounds=(Q(1, 10),),
        hidden_phase_error_bounds=(Q(1, 20),),
        elapsed_time=Q(1, 100),
    )
    order = (2, 0, 1)
    assert transformed.reference_form == tuple(base.reference_form[i] for i in order)
    assert transformed.reference_phase == tuple(base.reference_phase[i] for i in order)
    assert transformed.hidden_nodes == ("hidden",)
    assert transformed.retained_indices == (2, 1) and transformed.hidden_indices == (0,)
    assert transformed.comparison_matrix == tuple(
        tuple(base.comparison_matrix[i][j] for j in order) for i in order
    )
    assert transformed.arithmetic_observation_weights == tuple(
        base.arithmetic_observation_weights[i] for i in order
    )
    assert transformed.inverse_capacity_observation_weights == tuple(
        base.inverse_capacity_observation_weights[i] for i in order
    )
    # Reversing an edge preserves the exact law, but the shared sine
    # enclosure need not be bitwise odd. Compare the independent true rows.
    expected_form, expected_phase = _direct_rows(
        base.reference_form,
        base.reference_phase,
        ((0, 1), (1, 2)),
        (Q(1), Q(1)),
        base.capacity,
        Q(1),
    )
    for report, indices in ((base, (0, 1)), (transformed, (1, 0))):
        expected = tuple(expected_form[i] for i in indices) + tuple(
            expected_phase[i] for i in indices
        )
        for actual, target in zip(
            report.make_reduced_field()(report.reduced_initial_state), expected
        ):
            _contains_oracle(actual, target)
    for actual, expected in zip(
        transformed.nodal_error_upper_bounds,
        (base.nodal_error_upper_bounds[i] for i in order),
    ):
        assert abs(actual - expected) < Q(1, 10**35)


@pytest.mark.parametrize("scale", (Q(1, 4), Q(1), Q(3, 2)))
@pytest.mark.parametrize("gamma", (Q(0), Q(1)))
def test_common_capacity_multiplier_scales_both_rows_and_all_three_budgets(
    scale, gamma
):
    capacities, h = (Q(2), Q(1), Q(1, 2)), Q(1, 100)
    args = dict(
        capacity=capacities,
        gamma=gamma,
        retained_initial_form=(Q(1, 2), Q(-1, 4)),
        hidden_reference_form=(Q(1, 8),),
        hidden_form_error_bounds=(Q(1, 10),),
        hidden_phase_error_bounds=(Q(1, 20),),
    )
    clock = _bound(**args, elapsed_time=scale * h)
    scaled = _bound(
        **dict(args, capacity=tuple(scale * c for c in capacities)), elapsed_time=h
    )
    assert clock.kernel.upper == scaled.kernel.upper
    assert clock.retained_kernel.upper == scaled.retained_kernel.upper
    for name in (
        "initialization_error_upper_bounds",
        "hidden_drift_error_upper_bounds",
        "boundary_feedback_error_upper_bounds",
        "nodal_error_upper_bounds",
    ):
        if not gamma or name == "initialization_error_upper_bounds":
            assert getattr(clock, name) == getattr(scaled, name)
        else:
            # c encloses the initial complete rows. Its grid rounding can
            # change under a common capacity multiplier.
            for left, right in zip(getattr(clock, name), getattr(scaled, name)):
                assert abs(left - right) < Q(1, 10**35)
    assert (
        clock.inverse_capacity_observation_weights
        == scaled.inverse_capacity_observation_weights
    )
    assert abs(
        clock.arithmetic_error_upper_bound - scaled.arithmetic_error_upper_bound
    ) < Q(1, 10**35)
    assert abs(
        clock.inverse_capacity_error_upper_bound
        - scaled.inverse_capacity_error_upper_bound
    ) < Q(1, 10**35)


def test_actual_inverse_capacity_weights_have_their_own_projection_and_error_budget():
    report = _bound(
        capacity=(2, 1, Q(1, 2)),
        reading_error_bound=Q(1, 1000),
        inverse_capacity_reading_error_bound=Q(1, 2000),
    )
    assert report.arithmetic_observation_weights == (Q(1, 2), Q(1, 2), 0)
    assert report.inverse_capacity_observation_weights == (Q(1, 3), Q(2, 3), 0)
    node = report.nodal_error_upper_bounds
    assert report.arithmetic_error_upper_bound == (node[0] + node[1]) / 2
    assert report.inverse_capacity_error_upper_bound == node[0] / 3 + 2 * node[1] / 3
    assert (
        report.arithmetic_error_upper_bound != report.inverse_capacity_error_upper_bound
    )
    assert (
        report.arithmetic_observed_error_upper_bound
        == report.arithmetic_error_upper_bound + Q(1, 1000)
    )
    assert (
        report.inverse_capacity_observed_error_upper_bound
        == report.inverse_capacity_error_upper_bound + Q(1, 2000)
    )


def test_zero_and_tiny_boundary_weights_keep_exact_path_continuity():
    epsilon = Q(1, 1 << 200)
    absent = _bound(edges=((0, 1),), conductances=(1,))
    zero = _bound(conductances=(1, 0))
    tiny = _bound(conductances=(1, epsilon))
    assert absent.adjacency == zero.adjacency
    assert absent.arithmetic_error_upper_bound == zero.arithmetic_error_upper_bound == 0
    assert 0 < tiny.arithmetic_error_upper_bound < epsilon
    assert tiny.initial_discrepancy_upper_bounds == (0, 0, 1)
    assert not tiny.exact_observation_zero


def test_zero_clock_preserves_hidden_uncertainty_but_has_no_retained_error():
    report = _bound(
        elapsed_time=0, hidden_reference_form=(1,), hidden_phase_error_bounds=(Q(1, 3),)
    )
    assert (
        report.initial_discrepancy_upper_bounds
        == report.initialization_error_upper_bounds
        == (0, 0, Q(4, 3))
    )
    assert report.arithmetic_error_upper_bound == 0
    assert (
        report.arithmetic_hidden_drift_error_upper_bound
        == report.arithmetic_boundary_feedback_error_upper_bound
        == 0
    )
    assert report.exact_observation_zero


@pytest.mark.parametrize(
    "field", ("inverse_capacity_reading_error_bound", "inverse_capacity_tolerance")
)
def test_missing_optional_weighted_evidence_leaves_verdict_unavailable(field):
    report = _bound(**{field: None})
    assert report.inverse_capacity_error_upper_bound is not None
    assert report.inverse_capacity_tolerance_certified is None
    if field == "inverse_capacity_reading_error_bound":
        assert report.inverse_capacity_observed_error_upper_bound is None


def test_tolerance_is_nonnegative_absolute_allowance_with_independent_noise_floor():
    report = _bound(
        conductances=(1, 0),
        reading_error_bound=Q(1, 100),
        tolerance=Q(1, 100),
        inverse_capacity_reading_error_bound=Q(1, 50),
        inverse_capacity_tolerance=Q(1, 100),
    )
    assert report.arithmetic_error_upper_bound == 0
    assert report.arithmetic_observed_error_upper_bound == Q(1, 100)
    assert report.tolerance_certified
    assert not report.inverse_capacity_tolerance_certified
    unresolved = _bound(tolerance=0)
    assert not unresolved.tolerance_certified
    # Failure of a sufficient budget does not assert failure of an actual
    # prediction; actual zero hidden preparation still gives exact zero here.
    assert unresolved.arithmetic_error_upper_bound > 0


def test_compiled_report_is_immutable_and_no_cached_source_is_accepted():
    report = _bound()
    with pytest.raises(FrozenInstanceError):
        report.gamma = Q(2)
    with pytest.raises(TypeError):
        _bound(source=report)
    with pytest.raises(TypeError):
        _bound(weighted_degree_upper_bound=1)


def test_compiled_field_rebuilds_derived_coordinates_and_readmits_raw_primitives():
    report = _bound(hidden_reference_form=(1,), hidden_form_error_bounds=(0,))
    forged = replace(
        report,
        retained_indices=(2,),
        hidden_indices=(0, 1),
        index_edges=((9, 9),),
        reference_form=(Q(99),),
        reference_phase=(Q(99),),
        reduced_initial_state=(99,),
        comparison_matrix=((Q(999),),),
        initial_rate_upper_bounds=(Q(999),),
        arithmetic_error_upper_bound=Q(0),
        tolerance_certified=True,
    )
    actual = forged.make_reduced_field()((0, 0, 0, 0))
    assert len(actual) == 4
    for value, expected in zip(actual, (0, Q(1, 4), 0, 0)):
        assert value.lo <= expected <= value.hi
    with pytest.raises((TypeError, ValueError)):
        replace(report, capacity=(True, 1, 1)).make_reduced_field()
    with pytest.raises((TypeError, ValueError)):
        replace(report, hidden_reference_phase=(float("nan"),)).make_reduced_field()


def test_isolated_large_capacity_does_not_import_the_obsolete_scalar_growth_cap(
    monkeypatch,
):
    from tnfr.physics import _sine_preparation as legacy

    def forbidden(*args, **kwargs):
        pytest.fail("retained certificate imported a different scalar growth domain")

    monkeypatch.setattr(legacy, "_prepared_two_contact_composition", forbidden)
    report = _bound(
        nodes=(0, 1, 2, 3),
        capacity=(1, 1, 1, 10**9),
        gamma=1,
        hidden_reference_form=(0, 0),
        hidden_reference_phase=(0, 0),
        hidden_form_error_bounds=(1, 1),
        hidden_phase_error_bounds=(0, 0),
    )
    assert report.comparison_matrix[3] == (0,) * 4
    assert report.kernel.upper[3] == (0, 0, 0, 1)
    assert report.kernel.scaled_row_sum_upper_bound == Q(1, 2)


def test_finite_iterables_normalize_without_changing_declared_coordinate_order():
    baseline = _bound()
    actual = _bound(
        nodes=iter((0, 1, 2)),
        edges=([0, 1], [1, 2]),
        conductances=iter((1, 1)),
        capacity=iter((1, 1, 1)),
        retained_nodes=[0, 1],
        observation_nodes=iter((0, 1)),
        retained_initial_form=iter((0, 0)),
        hidden_reference_form=iter((0,)),
    )
    assert actual == baseline


@pytest.mark.parametrize(
    "changes",
    (
        {"nodes": ()},
        {"nodes": (0,)},
        {"nodes": (0, 0, 2)},
        {"nodes": (0, [], 2)},
        {"nodes": {0, 1, 2}},
        {"nodes": tuple(range(25)), "capacity": (1,) * 25},
        {"edges": ((0, 0),), "conductances": (1,)},
        {"edges": ((0, 3),), "conductances": (1,)},
        {"edges": ((0, 1), (1, 0)), "conductances": (1, 1)},
        {"edges": ({0, 1}, (1, 2))},
        {"conductances": (1,)},
        {"conductances": (1, -1)},
        {"conductances": (1, True)},
        {"conductances": (1, float("nan"))},
        {"capacity": (1, 1, 0)},
        {"capacity": (1, 1, -1)},
        {"capacity": (1, 1, True)},
        {"capacity": (1, 1, float("inf"))},
        {"capacity": (1, 1)},
        {"gamma": -1},
        {"gamma": True},
        {"gamma": float("nan")},
        {"retained_nodes": ()},
        {"retained_nodes": (0, 1, 2)},
        {"retained_nodes": (0, 0)},
        {"retained_nodes": (0, 3)},
        {"retained_nodes": {0, 1}},
        {"observation_nodes": ()},
        {"observation_nodes": (0, 0)},
        {"observation_nodes": (0, 2)},
        {"observation_nodes": {0, 1}},
        {"retained_initial_form": (0,)},
        {"retained_initial_form": (0, True)},
        {"retained_initial_phase": (0, float("inf"))},
        {"hidden_reference_form": ()},
        {"hidden_reference_phase": (True,)},
        {"hidden_form_error_bounds": (-1,)},
        {"hidden_phase_error_bounds": (True,)},
        {"hidden_form_error_bounds": (float("inf"),)},
        {"elapsed_time": -1},
        {"elapsed_time": True},
        {"reading_error_bound": -1},
        {"tolerance": -1},
        {"inverse_capacity_reading_error_bound": True},
        {"inverse_capacity_tolerance": -1},
        {"elapsed_time": Q(32) + Q(1, 1 << 200)},
    ),
)
def test_all_raw_domains_precede_sine_interval_or_matrix_work(changes, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid retained preparation reached numerical work")

    monkeypatch.setattr(comparison, "nonnegative_exponential_upper", forbidden)
    for name in ("nonnegative_exponential_upper", "I", "_held_quarter_sine_field"):
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(flow, "sin", forbidden)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _bound(**changes)


@pytest.mark.parametrize(
    "field",
    ("nodes", "edges", "capacity", "retained_nodes", "hidden_form_error_bounds"),
)
def test_unbounded_primitive_iterables_have_finite_admission_limits(field):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: repeat(1)})


def test_zero_horizon_and_zero_boundary_do_not_hide_invalid_raw_budgets(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid zero-window preparation reached sine")

    monkeypatch.setattr(flow, "sin", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _bound(elapsed_time=0, conductances=(1, 0), hidden_form_error_bounds=(True,))
    with pytest.raises((TypeError, ValueError)):
        _bound(
            elapsed_time=0, conductances=(1, 0), inverse_capacity_tolerance=float("nan")
        )


@pytest.mark.parametrize(
    "state",
    (
        (0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, True, 0),
        (0, 0, float("inf"), 0),
        {0, 1, 2, 3},
    ),
)
def test_reduced_state_admission_precedes_sine(state, monkeypatch):
    field = _bound().make_reduced_field()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid reduced state reached sine")

    monkeypatch.setattr(flow, "sin", forbidden)
    with pytest.raises((TypeError, ValueError)):
        field(state)


def test_matrix_resource_boundary_is_inclusive_without_executing_large_kernel(
    monkeypatch,
):
    class AdmittedBoundary(Exception):
        pass

    def reached(*args, **kwargs):
        raise AdmittedBoundary

    monkeypatch.setattr(owner, "_held_quarter_sine_field", reached)
    with pytest.raises(AdmittedBoundary):
        _bound(elapsed_time=32)
