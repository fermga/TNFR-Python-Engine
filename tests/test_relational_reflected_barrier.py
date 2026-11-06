"""Static separating-storage theorem controls, without trajectory execution."""

from fractions import Fraction as Q

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.physics import relational_reflected_transit as owner

MODEL = RelationalExchangeModel(1, phase_domain="regular")


@pytest.fixture(autouse=True)
def no_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a static barrier certificate must not execute a trajectory")

    monkeypatch.setattr(owner, "validated_taylor_step", forbidden)
    monkeypatch.setattr(owner, "certify_relational_reflected_transit", forbidden)


def _state(a, b, capital_a, capital_b, forms=None):
    return tuple((I(0),) * 4 if forms is None else forms) + (a, b, capital_a, capital_b)


def test_consensus_lower_component_has_an_exact_static_future_exclusion():
    report = owner.certify_relational_reflected_barrier((0,) * 8, model=MODEL)
    assert report.obstructed and report.status == "certified"
    assert report.storage.contains(0)
    assert report.barrier_storage == I(7)
    assert report.energy_deficit.lo > 0
    assert report.separator_margin.lo > 0
    assert min(report.domain_lower_bounds) > 0
    assert report.unavailable_reasons == ()
    projection = report.to_dict()
    assert projection["schema"] == "tnfr.relational-reflected-barrier.v1"
    data = projection["report"]
    assert data["status"] == "certified"
    assert Q(**data["energy_deficit"]["lo"]) == report.energy_deficit.lo
    assert (
        "snapshot_theorem_not_new_trajectory_or_physical_identification"
        in data["scope"]
    )


def test_exact_regular_saddle_has_barrier_storage_but_no_strict_obstruction():
    pi = pi_interval()
    state = _state(2 * pi / 3, pi / 3, 2 * pi / 3, pi / 3)
    report = owner.certify_relational_reflected_barrier(state, model=MODEL)
    assert report.storage.contains(7)
    assert report.energy_deficit.contains(0)
    assert report.separator_margin.contains(0)
    assert not report.obstructed
    assert set(report.unavailable_reasons) == {
        "storage_not_strictly_below_separating_barrier",
        "lower_separator_component_not_certified",
    }


@pytest.mark.parametrize("displacement", (Q(-1, 6), Q(-1, 12), Q(0), Q(1, 12), Q(1, 6)))
def test_boundary_storage_uses_both_bridges_and_the_independent_cosine_identity(
    displacement,
):
    pi = pi_interval()
    d = displacement * pi
    a, capital_a = 2 * pi / 3 + d, 2 * pi / 3 - d
    report = owner.certify_relational_reflected_barrier(
        _state(a, a / 2, capital_a, capital_a / 2), model=MODEL
    )
    independent = 12 - cos(2 * d) - 4 * cos(d / 2)
    assert (report.storage - independent).contains(0)
    assert not report.obstructed
    assert report.separator_margin.contains(0)
    if displacement:
        assert independent.lo > 7
    else:
        assert independent.contains(7)


def test_arbitrary_internal_shape_adds_nonnegative_boundary_storage():
    pi = pi_interval()
    d = pi / 12
    a, capital_a = 2 * pi / 3 + d, 2 * pi / 3 - d
    b, capital_b = a / 2 - Q(1, 32), capital_a / 2 + Q(1, 16)
    report = owner.certify_relational_reflected_barrier(
        _state(a, b, capital_a, capital_b), model=MODEL
    )
    minimum = 12 - cos(2 * d) - 4 * cos(d / 2)
    excess = 4 * cos(a / 2) * (1 - cos(a / 2 - b)) + 4 * cos(capital_a / 2) * (
        1 - cos(capital_a / 2 - capital_b)
    )
    assert excess.lo > 0
    assert (report.storage - minimum - excess).contains(0)
    assert report.storage.lo > 7


def test_aligned_acute_twists_below_seven_are_not_excluded_on_the_target_side():
    pi = pi_interval()
    state = _state(4 * pi / 5, 2 * pi / 5, 4 * pi / 5, 2 * pi / 5)
    report = owner.certify_relational_reflected_barrier(state, model=MODEL)
    target = 10 * (1 - cos(2 * pi / 5))
    assert (report.storage - target).contains(0)
    assert report.energy_deficit.lo > 0
    assert report.separator_margin.hi < 0
    assert not report.obstructed
    assert report.unavailable_reasons == ("lower_separator_component_not_certified",)


def test_initial_uniform_receiver_seed_is_not_excluded_by_its_initial_storage():
    pi = pi_interval()
    state = _state(4 * pi / 5, 2 * pi / 5, I(0), I(0))
    report = owner.certify_relational_reflected_barrier(state, model=MODEL)
    independent = 5 * (1 - cos(2 * pi / 5)) + 2 + 2 * cos(pi / 5)
    assert (report.storage - independent).contains(0)
    assert report.energy_deficit.hi < 0
    assert report.separator_margin.lo > 0
    assert not report.obstructed
    assert report.unavailable_reasons == (
        "storage_not_strictly_below_separating_barrier",
    )


@pytest.mark.parametrize("beta", (Q(1, 8), Q(3), Q(8)))
def test_phase_storage_and_barrier_scale_with_the_explicit_beta(beta):
    phase = (I(Q(1, 2)), I(Q(1, 4)), I(Q(1, 4)), I(Q(1, 8)))
    base = owner.certify_relational_reflected_barrier(_state(*phase), model=MODEL)
    scaled = owner.certify_relational_reflected_barrier(
        _state(*phase), model=RelationalExchangeModel(beta, phase_domain="regular")
    )
    assert scaled.obstructed and base.obstructed
    assert scaled.barrier_storage == I(7 * beta)
    assert (scaled.storage - beta * base.storage).contains(0)
    assert (scaled.energy_deficit - beta * base.energy_deficit).contains(0)
    assert scaled.separator == base.separator


def test_nonnegative_form_storage_can_remove_an_otherwise_valid_obstruction():
    state = _state(I(0), I(0), I(0), I(0), forms=(I(2), I(0), I(0), I(0)))
    report = owner.certify_relational_reflected_barrier(state, model=MODEL)
    # The ten-node graph has E_D=2p²+(p-r)²+r²+(p-P)²=16.
    assert report.storage.contains(16)
    assert report.energy_deficit.hi < 0
    assert not report.obstructed


def test_nondissipative_boundary_still_has_the_same_nonincrease_obstruction():
    report = owner.certify_relational_reflected_barrier(
        (0,) * 8, model=RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    )
    assert report.obstructed
    # Storage conservation suffices here; this does not establish recovery.


@pytest.mark.parametrize(
    "model",
    (
        None,
        RelationalExchangeModel(1),
        RelationalExchangeModel(1, phase_domain="positive_resultant"),
    ),
)
def test_a_different_or_missing_model_is_not_silently_replaced(model):
    with pytest.raises(ValueError, match="explicit regular"):
        owner.certify_relational_reflected_barrier((0,) * 8, model=model)


@pytest.mark.parametrize(
    "state",
    (
        (0,) * 7,
        (0,) * 7 + (True,),
        (0,) * 7 + (0.1,),
        (0, 0, 0, 0, 0, 2, 0, 0),
    ),
)
def test_invalid_or_nonregular_state_is_rejected_without_projection(state):
    with pytest.raises((TypeError, ValueError)):
        owner.certify_relational_reflected_barrier(state, model=MODEL)


def test_periodic_field_values_do_not_allow_escape_from_the_declared_continuous_lift():
    state = _state(2 * pi_interval(), I(0), I(0), I(0))
    with pytest.raises(ValueError, match="reflection lift"):
        owner.certify_relational_reflected_barrier(state, model=MODEL)


@pytest.mark.parametrize(
    "phases",
    (
        (Q(5, 2), Q(6, 5), Q(1, 4), Q(-1, 8)),
        (Q(6, 5), Q(3, 10), Q(3, 4), Q(1, 4)),
        (Q(11, 5), Q(11, 10), Q(49, 20), Q(6, 5)),
    ),
)
def test_sharp_midpoint_envelope_matches_independent_full_storage_for_distinct_rings(
    phases,
):
    a, b, capital_a, capital_b = map(I, phases)
    forms = (Q(1, 8), Q(-1, 16), Q(-1, 4), Q(1, 32))
    report = owner.certify_relational_reflected_barrier(
        _state(a, b, capital_a, capital_b, forms=forms), model=MODEL
    )
    m, d = (a + capital_a) / 2, (a - capital_a) / 2
    u, v = a / 2 - b, capital_a / 2 - capital_b
    envelope = 10 - 2 * cos(2 * m) - 8 * cos(m / 2)
    penalties = (
        4 * cos(m) ** 2 * (1 - cos(2 * d)),
        8 * cos(m / 2) * (1 - cos(d / 2)),
        4 * cos(a / 2) * (1 - cos(u)),
        4 * cos(capital_a / 2) * (1 - cos(v)),
    )
    p, r, capital_p, capital_r = forms
    form_storage = (
        2 * p**2
        + (p - r) ** 2
        + r**2
        + 2 * capital_p**2
        + (capital_p - capital_r) ** 2
        + capital_r**2
        + (p - capital_p) ** 2
    )
    assert all(value.lo >= 0 for value in penalties)
    defect = report.storage - form_storage - envelope - sum(penalties, I(0))
    assert defect.contains(0) and defect.width < Q(1, 10**30)


def test_sharp_midpoint_envelope_is_attained_on_the_diagonal_shape():
    a = I(Q(5, 2))
    report = owner.certify_relational_reflected_barrier(
        _state(a, a / 2, a, a / 2), model=MODEL
    )
    envelope = 10 - 2 * cos(2 * a) - 8 * cos(a / 2)
    assert (report.storage - envelope).contains(0)
    assert (report.storage - envelope).width < Q(1, 10**30)


def test_storage_below_the_barrier_does_not_establish_global_resultant_regularity():
    pi = pi_interval()
    a, b, capital_a, capital_b = pi / 6, pi / 2, pi / 6, pi / 12
    phase_storage = (
        12
        - cos(2 * a)
        - 2 * cos(a - b)
        - 2 * cos(b)
        - cos(2 * capital_a)
        - 2 * cos(capital_a - capital_b)
        - 2 * cos(capital_b)
        - 2 * cos(a - capital_a)
    )
    independent = 8 - 4 * cos(pi / 12)
    assert (phase_storage - independent).contains(0)
    assert phase_storage.hi < 7
    assert (2 * cos(b)).contains(0)
    with pytest.raises(ValueError, match="resultant"):
        owner.certify_relational_reflected_barrier(
            _state(a, b, capital_a, capital_b), model=MODEL
        )
