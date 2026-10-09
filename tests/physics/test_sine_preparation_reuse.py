"""Primitive admission and invocation-local reuse at the preparation boundary."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _sine_preparation as preparation
from tnfr.physics import relational_sine_formation_response as formation
from tnfr.physics import relational_sine_pattern as pattern_owner
from tnfr.physics.relational_sine_comparison import (
    _comparison_from_state,
    _sine_state_from_rows,
)

NODES = tuple(range(5))
EDGES = ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
NEIGHBORS = tuple(
    tuple(j if i == a else i for i, j in EDGES if a in (i, j)) for a in NODES
)
FORMS = tuple(Q(i - 2) for i in NODES)
PHASES = (Q(0),) * 5
CAPACITY = (Q(1),) * 5


def _source(kind="pattern"):
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    if kind == "comparison":
        state = _sine_state_from_rows(NODES, EDGES, FORMS, PHASES, CAPACITY, NEIGHBORS)
        return _comparison_from_state(state, model)
    return pattern_owner._bound_sine_pattern_from_rows(
        reference_node=0,
        reference_model=model,
        nodes=NODES,
        edges=EDGES,
        neighbors=NEIGHBORS,
        capacity=CAPACITY,
        nominal_form=FORMS,
        nominal_phase=PHASES,
        form_error_bounds=(Q(1, 1000),) * 5,
        phase_error_bounds=(Q(1, 2000),) * 5,
    )


@pytest.mark.parametrize("kind", ("comparison", "pattern"))
def test_report_adapter_keeps_its_normalized_primitive_association(kind):
    source = _source(kind)
    form_field = "epi" if kind == "comparison" else "nominal_form"
    # Integral source values are admissible, but the row kernel must receive
    # their normalized rational values rather than the original payload.
    source = replace(source, **{form_field: tuple(range(-2, 3)), "capacity": (1,) * 5})
    result = preparation._sine_preparation(source)
    assert result.admitted is not None
    assert type(result.admitted) is type(source)
    assert result.uncertain is (kind == "pattern")
    assert result.form == FORMS
    assert getattr(result.admitted, form_field) == FORMS
    assert all(type(value) is Q for value in result.form)
    assert all(type(value) is Q for value in result.admitted.capacity)
    assert result.model is result.admitted.reference_model
    assert result.geometry.nodes == source.nodes


@pytest.mark.parametrize(
    "field,bad",
    (
        ("nominal_form", (True,) + FORMS[1:]),
        ("nominal_phase", (0j,) + PHASES[1:]),
        ("phase_error_bounds", (Q(-1),) + (Q(0),) * 4),
        ("capacity", (False,) + CAPACITY[1:]),
        ("edges", EDGES[:-1]),
        ("reference_model", object()),
    ),
)
def test_report_primitives_are_readmitted_before_the_row_kernel(
    monkeypatch, field, bad
):
    source = replace(_source(), **{field: bad})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid report primitives reached the prepared-row kernel")

    monkeypatch.setattr(preparation, "_sine_preparation_from_rows", forbidden)
    with pytest.raises((TypeError, ValueError)):
        preparation._sine_preparation(source)


def test_changed_report_rows_override_old_derived_observations():
    source = _source()
    original = preparation._sine_preparation(source)
    doubled = tuple(2 * value for value in source.nominal_form)
    altered = replace(
        source,
        nominal_form=doubled,
        form_storage_bounds=I(0),
        relative_form_bounds=(I(0),) * 5,
        form_gradient_bounds=(I(0),) * 5,
        form_rate_bounds=(I(0),) * 5,
    )
    rebuilt = preparation._sine_preparation(altered)
    assert rebuilt.admitted is not None
    assert rebuilt.admitted.nominal_form == doubled
    assert rebuilt.form == doubled
    assert rebuilt.nominal_initial_norm.lo > original.nominal_initial_norm.hi
    assert rebuilt.initial_error_norm == original.initial_error_norm
    assert (
        rebuilt.initial_form_storage_bounds.lo > original.initial_form_storage_bounds.hi
    )


def test_formation_shares_one_fresh_domain_without_temporary_observation_reports(
    monkeypatch,
):
    domains, row_results = [], []
    original_domain = formation._sine_domain
    original_rows = formation._sine_preparation_from_rows

    def domain_spy(*args, **kwargs):
        domain = original_domain(*args, **kwargs)
        domains.append(domain)
        return domain

    def rows_spy(domain, **kwargs):
        result = original_rows(domain, **kwargs)
        row_results.append((domain, result))
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("the fixed-row consumer constructed a discarded report")

    monkeypatch.setattr(formation, "_sine_domain", domain_spy)
    monkeypatch.setattr(formation, "_sine_preparation_from_rows", rows_spy)
    monkeypatch.setattr(preparation, "_sine_preparation", forbidden)
    monkeypatch.setattr(pattern_owner, "_bound_sine_pattern_from_rows", forbidden)
    inputs = dict(
        scaled_time=Q(80),
        form_error_bound=Q(2, 10**10),
        phase_error_bound=Q(3, 10**10),
        readout_error_bound=Q(4, 10**10),
        radius=Q(1, 10),
    )
    first = formation.assess_sine_formation_response(**inputs)
    inputs["form_error_bound"] *= 2
    second = formation.assess_sine_formation_response(**inputs)

    assert len(domains) == 2 and len(row_results) == 4
    assert domains[0] is not domains[1]
    for index, (domain, result) in enumerate(row_results):
        assert domain is domains[index // 2]
        assert result.admitted is None
        assert all(type(value) is Q for value in result.form + result.form_errors)
    assert first.form_error_bound == Q(2, 10**10)
    assert second.form_error_bound == Q(4, 10**10)
    assert all(
        later > earlier
        for earlier, later in zip(
            first.initial_scaled_form_error_norm_upper_bounds,
            second.initial_scaled_form_error_norm_upper_bounds,
        )
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"form_error_bound": True},
        {"phase_error_bound": Q(-1)},
        {"scaled_time": Q(6145)},
        {"radius": Q(0)},
    ),
)
def test_fixed_consumer_rejects_original_inputs_before_domain_work(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid public inputs reached shared domain admission")

    monkeypatch.setattr(formation, "_sine_domain", forbidden)
    inputs = dict(
        scaled_time=Q(80),
        form_error_bound=Q(2, 10**10),
        phase_error_bound=Q(3, 10**10),
        readout_error_bound=Q(4, 10**10),
        radius=Q(1, 10),
    )
    with pytest.raises((TypeError, ValueError)):
        formation.assess_sine_formation_response(**(inputs | changes))


@pytest.mark.parametrize("time", (Q(0), Q(1, 10), Q(3), Q(100)))
@pytest.mark.parametrize("forcing_sign", (-1, 1))
def test_transit_kernel_encloses_exact_constant_forcing_flow(time, forcing_sign):
    # Independent scalar solution of z'=-2z+eta*f, theta'=2z. This
    # exercises both transient cancellation and reinforcing forcing without
    # replacing a full nonlinear trajectory by sampled endpoints.
    from tnfr.physics.reversible_eigenmode_reference import _negative_exp_bounds

    decay_upper = _negative_exp_bounds(2 * time)[1]
    form_radius, phase_radius = preparation._prepared_transit_radii(
        time=time,
        decay_upper=decay_upper,
        gap_lower=Q(2),
        initial_norm_upper=Q(3),
        feedback_upper=Q(1, 5),
        forcing_upper=Q(7),
    )

    def decimal(value):
        return Decimal(value.numerator) / Decimal(value.denominator)

    with localcontext() as context:
        context.prec = 90
        elapsed = decimal(time)
        decay = (-2 * elapsed).exp()
        stationary_form = Decimal(forcing_sign * 7) / 10
        form = stationary_form + (3 - stationary_form) * decay
        phase = 2 * stationary_form * elapsed + (3 - stationary_form) * (1 - decay)
        assert abs(form) <= decimal(form_radius)
        assert abs(phase - 3) <= decimal(phase_radius)
